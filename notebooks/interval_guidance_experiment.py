"""
notebooks/interval_guidance_experiment.py
------------------------------------------
DiversityPhi salience-sensitive guidance applied only in the second half
of the reverse diffusion process (steps 250-500 of 500), with the first
half running as standard DDIM. Intermediate noisy samples are decoded and
saved at key timesteps to visualise the trajectory.

Motivated by the symmetry-breaking literature (Raya & Ambrogioni 2023):
the first half of the reverse process commits to a semantic mode; the
second half refines details. Applying salience guidance only in the second
half avoids disturbing semantic commitment while still influencing the
fine-grained detail variation that DiversityPhi was shown to affect at
low guidance scales.

Checkpoints saved at: t=500, t=375, t=250, t=125, t=50, t=0

Run from repo root:
    PYTHONPATH=$(pwd) python notebooks/interval_guidance_experiment.py
"""

import torch
import gc
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import DDIMScheduler
from sd.pipeline import SalienceGradSDPipeline
from sd.phi_sd import DiversityPhiSD

DEVICE     = "cuda"
MODEL_ID   = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = Path("outputs/sd_experiment/interval_guidance")
PROMPT     = "a wooden chair in a blue room"

NUM_IMAGES  = 8
NUM_STEPS   = 500
CFG_SCALE   = 5.0
SAL_SCALE   = 200.0
BATCH_SIZE  = 8
SEED        = 2024

# Timestep checkpoints to save decoded images at
# (in scheduler timestep units, i.e. 0-1000 for SD v1.5)
CHECKPOINT_STEPS = [500, 375, 250, 125, 50, 0]  # as step indices (0 = final)

# Guidance starts at this step index (0-indexed from the beginning of
# the reverse process, i.e. step 250 of 500 = the halfway point)
GUIDANCE_START_STEP = 250

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clear():
    gc.collect()
    torch.cuda.empty_cache()

def decode_latents(pipe, latents):
    """Decode a batch of latents to PIL images."""
    latents_scaled = 1 / pipe.vae.config.scaling_factor * latents
    with torch.no_grad():
        images = pipe.vae.decode(latents_scaled).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype(np.uint8)
    return [Image.fromarray(img) for img in images]

print("Loading pipeline...")
pipe = SalienceGradSDPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
).to(DEVICE)

phi_div = DiversityPhiSD()
pipe.setup_phi(phi_div, context={})
pipe.set_salience_scale(SAL_SCALE)
pipe.set_guidance_frequency(1)

# ---------------------------------------------------------------------------
# Manual denoising loop with interval guidance and checkpoint saving
# ---------------------------------------------------------------------------
generator = torch.Generator(device=DEVICE).manual_seed(SEED)

# Encode prompt
do_cfg = CFG_SCALE > 1.0
prompt_embeds = pipe._encode_prompt(
    PROMPT, DEVICE, NUM_IMAGES, do_cfg, negative_prompt=None,
)

# Prepare latents
pipe.scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
timesteps = pipe.scheduler.timesteps

num_channels = pipe.unet.config.in_channels
height = pipe.unet.config.sample_size * pipe.vae_scale_factor
width  = pipe.unet.config.sample_size * pipe.vae_scale_factor

latents = pipe.prepare_latents(
    NUM_IMAGES, num_channels, height, width,
    prompt_embeds.dtype, DEVICE, generator, None,
)

# Save decoded initial noise (t=500, step index 0)
checkpoint_dirs = {}
for cp in CHECKPOINT_STEPS:
    d = OUTPUT_DIR / f"step_{cp:04d}"
    d.mkdir(parents=True, exist_ok=True)
    checkpoint_dirs[cp] = d

def save_checkpoint(latents, step_idx, step_label):
    if step_label not in checkpoint_dirs:
        return
    imgs = decode_latents(pipe, latents)
    for i, img in enumerate(imgs):
        img.save(checkpoint_dirs[step_label] / f"{i}.png")
    print(f"  Saved checkpoint t={step_label} (step {step_idx})")

save_checkpoint(latents, 0, 500)

print(f"\nRunning denoising loop ({NUM_STEPS} steps)...")
print(f"Salience guidance active from step {GUIDANCE_START_STEP} to {NUM_STEPS}")

for i, t in enumerate(timesteps):

    # CFG: expand latents
    latent_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_input = pipe.scheduler.scale_model_input(latent_input, t)

    # UNet noise prediction
    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_input, t,
            encoder_hidden_states=prompt_embeds,
        ).sample

    # CFG correction
    if do_cfg:
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + CFG_SCALE * (noise_text - noise_uncond)

    # Salience gradient guidance — only in the second half
    if i >= GUIDANCE_START_STEP:
        t_int = int(t)
        context = {
            **pipe.phi_context,
            "unet": pipe.unet,
            "scheduler": pipe.scheduler,
        }
        grad = pipe._compute_salience_gradient(latents, t_int, context)
        sqrt_1m_alpha = (
            1.0 - pipe.scheduler.alphas_cumprod[t]
        ).to(latents.device) ** 0.5
        noise_pred = noise_pred - sqrt_1m_alpha * SAL_SCALE * grad.to(noise_pred.dtype)

    # Scheduler step
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Save checkpoints at key step indices
    # Map step index to our checkpoint labels
    step_label = None
    if i + 1 == 125:
        step_label = 375   # 125 steps in = t~375
    elif i + 1 == GUIDANCE_START_STEP:
        step_label = 250   # guidance kicks in here
    elif i + 1 == 375:
        step_label = 125   # 375 steps in = t~125
    elif i + 1 == 450:
        step_label = 50    # 450 steps in = t~50

    if step_label is not None:
        save_checkpoint(latents, i + 1, step_label)

    if (i + 1) % 50 == 0:
        guidance_status = "guided" if i >= GUIDANCE_START_STEP else "unguided"
        print(f"  Step {i+1}/{NUM_STEPS} ({guidance_status})")

# Final images
save_checkpoint(latents, NUM_STEPS, 0)

# Also save final as standard PIL output via VAE
final_images = decode_latents(pipe, latents)
final_dir = OUTPUT_DIR / "final"
final_dir.mkdir(parents=True, exist_ok=True)
for i, img in enumerate(final_images):
    img.save(final_dir / f"{i}.png")

del pipe
clear()
print(f"\nDone. Results in {OUTPUT_DIR}")
print("Checkpoint directories:")
for cp in CHECKPOINT_STEPS:
    print(f"  t={cp}: {checkpoint_dirs[cp]}")

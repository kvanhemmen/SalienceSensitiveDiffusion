"""
notebooks/interval_guidance_first_half.py
------------------------------------------
DiversityPhi salience-sensitive guidance applied only in the FIRST half
of the reverse diffusion process (steps 0-250 of 500), with the second
half running as standard DDIM.

This tests the hypothesis suggested by the second-half experiment: that
DiversityPhi's semantic variation effect operates primarily during the
early semantic commitment phase of the reverse process, not the late
refinement phase.

Checkpoints saved at: t=500, t=375, t=250, t=125, t=50, t=0

Run from repo root:
    PYTHONPATH=$(pwd) python notebooks/interval_guidance_first_half.py
"""

import torch
import gc
from pathlib import Path
from PIL import Image
import numpy as np
from sd.pipeline import SalienceGradSDPipeline
from sd.phi_sd import DiversityPhiSD

DEVICE     = "cuda"
MODEL_ID   = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = Path("outputs/sd_experiment/interval_guidance_first_half")
PROMPT     = "a wooden chair in a blue room"

NUM_IMAGES = 8
NUM_STEPS  = 500
CFG_SCALE  = 5.0
SAL_SCALE  = 200.0
SEED       = 2024

# Guidance applied only in the first half (steps 0-249)
GUIDANCE_START_STEP = 0
GUIDANCE_END_STEP   = 250  # exclusive

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clear():
    gc.collect()
    torch.cuda.empty_cache()

def decode_latents(pipe, latents):
    latents_scaled = 1 / pipe.vae.config.scaling_factor * latents
    with torch.no_grad():
        images = pipe.vae.decode(latents_scaled).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype(np.uint8)
    return [Image.fromarray(img) for img in images]

def save_checkpoint(pipe, latents, label):
    imgs = decode_latents(pipe, latents)
    d = OUTPUT_DIR / f"step_{label:04d}"
    d.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        img.save(d / f"{i}.png")
    print(f"  Saved checkpoint t={label}")

print("Loading pipeline...")
pipe = SalienceGradSDPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
).to(DEVICE)

phi_div = DiversityPhiSD()
pipe.setup_phi(phi_div, context={})
pipe.set_salience_scale(SAL_SCALE)
pipe.set_guidance_frequency(1)

do_cfg = CFG_SCALE > 1.0
prompt_embeds = pipe._encode_prompt(
    PROMPT, DEVICE, NUM_IMAGES, do_cfg, negative_prompt=None,
)

pipe.scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
timesteps = pipe.scheduler.timesteps

num_channels = pipe.unet.config.in_channels
height = pipe.unet.config.sample_size * pipe.vae_scale_factor
width  = pipe.unet.config.sample_size * pipe.vae_scale_factor

generator = torch.Generator(device=DEVICE).manual_seed(SEED)
latents = pipe.prepare_latents(
    NUM_IMAGES, num_channels, height, width,
    prompt_embeds.dtype, torch.device(DEVICE), generator, None,
)

# Save initial noise
save_checkpoint(pipe, latents.clone(), 500)

print(f"\nRunning denoising loop ({NUM_STEPS} steps)...")
print(f"Salience guidance active from step {GUIDANCE_START_STEP} to {GUIDANCE_END_STEP - 1}")

for i, t in enumerate(timesteps):

    latent_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_input = pipe.scheduler.scale_model_input(latent_input, t)

    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_input, t,
            encoder_hidden_states=prompt_embeds,
        ).sample

    if do_cfg:
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + CFG_SCALE * (noise_text - noise_uncond)

    # Salience guidance — only in the first half
    if GUIDANCE_START_STEP <= i < GUIDANCE_END_STEP:
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

    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Save checkpoints
    step_after = i + 1
    if step_after == 125:
        save_checkpoint(pipe, latents.clone(), 375)
    elif step_after == 250:
        save_checkpoint(pipe, latents.clone(), 250)
    elif step_after == 375:
        save_checkpoint(pipe, latents.clone(), 125)
    elif step_after == 450:
        save_checkpoint(pipe, latents.clone(), 50)

    if (i + 1) % 50 == 0:
        guidance_status = "guided" if GUIDANCE_START_STEP <= i < GUIDANCE_END_STEP else "unguided"
        print(f"  Step {i+1}/{NUM_STEPS} ({guidance_status})")

# Final
save_checkpoint(pipe, latents.clone(), 0)

del pipe
clear()
print(f"\nDone. Results in {OUTPUT_DIR}")
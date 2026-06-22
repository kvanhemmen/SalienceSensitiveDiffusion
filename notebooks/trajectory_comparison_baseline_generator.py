"""
notebooks/trajectory_comparison_experiment.py
----------------------------------------------
Generates intermediate decoded images at key timesteps for:
    1. Baseline (no salience guidance)
    2. Full guidance DiversityPhi (s=200, all 500 steps)

Both use the same 8 seeds and same prompt as interval_guidance_experiment.py,
so all three conditions can be directly compared in a paired figure.

Checkpoints saved at: t=500, t=375, t=250, t=125, t=50, t=0
(matching interval_guidance_experiment.py exactly)

Run from repo root:
    PYTHONPATH=$(pwd) python notebooks/trajectory_comparison_experiment.py
"""

import torch
import gc
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline
from sd.pipeline import SalienceGradSDPipeline
from sd.phi_sd import DiversityPhiSD

DEVICE     = "cuda"
MODEL_ID   = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = Path("outputs/sd_experiment/trajectory_comparison")
PROMPT     = "a wooden chair in a blue room"

NUM_IMAGES = 8
NUM_STEPS  = 500
CFG_SCALE  = 5.0
SAL_SCALE  = 200.0
SEED       = 2024

CHECKPOINT_STEP_INDICES = {
    0:   500,   # step index 0   -> label t=500 (initial noise)
    125: 375,   # step index 125 -> label t=375
    250: 250,   # step index 250 -> label t=250
    375: 125,   # step index 375 -> label t=125
    450: 50,    # step index 450 -> label t=50
    499: 0,     # step index 499 -> label t=0  (final)
}

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

def save_checkpoint(pipe, latents, label, condition_dir):
    imgs = decode_latents(pipe, latents)
    d = condition_dir / f"step_{label:04d}"
    d.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        img.save(d / f"{i}.png")
    print(f"  Saved checkpoint t={label}")

def run_denoising_loop(pipe, prompt_embeds, latents, condition_dir,
                       salience_scale=0.0, guidance_start_step=0):
    pipe.scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    do_cfg = CFG_SCALE > 1.0

    # Save initial noise
    save_checkpoint(pipe, latents.clone(), 500, condition_dir)

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

        if salience_scale > 0 and i >= guidance_start_step:
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
            noise_pred = noise_pred - sqrt_1m_alpha * salience_scale * grad.to(noise_pred.dtype)

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Save at checkpoint step indices (after the step)
        step_after = i + 1
        if step_after in [125, 250, 375, 450]:
            label = CHECKPOINT_STEP_INDICES.get(step_after)
            if label is not None:
                save_checkpoint(pipe, latents.clone(), label, condition_dir)

        if (i + 1) % 50 == 0:
            print(f"  Step {i+1}/{NUM_STEPS}")

    # Final
    save_checkpoint(pipe, latents.clone(), 0, condition_dir)
    return latents

# ---------------------------------------------------------------------------
# 1. Baseline
# ---------------------------------------------------------------------------
print("\n=== Baseline ===")
pipe = SalienceGradSDPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
).to(DEVICE)

phi_div = DiversityPhiSD()
pipe.setup_phi(phi_div, context={})
pipe.set_salience_scale(0.0)
pipe.set_guidance_frequency(1)

do_cfg = CFG_SCALE > 1.0
prompt_embeds = pipe._encode_prompt(
    PROMPT, DEVICE, NUM_IMAGES, do_cfg, negative_prompt=None,
)

generator = torch.Generator(device=DEVICE).manual_seed(SEED)
num_channels = pipe.unet.config.in_channels
height = pipe.unet.config.sample_size * pipe.vae_scale_factor
width  = pipe.unet.config.sample_size * pipe.vae_scale_factor
latents = pipe.prepare_latents(
    NUM_IMAGES, num_channels, height, width,
    prompt_embeds.dtype, torch.device(DEVICE), generator, None,
)

baseline_latents = latents.clone()  # save for reuse in full-guidance run

run_denoising_loop(
    pipe, prompt_embeds, latents,
    condition_dir=OUTPUT_DIR / "baseline",
    salience_scale=0.0,
)

del pipe
clear()
print("Baseline done.")

# ---------------------------------------------------------------------------
# 2. Full guidance DiversityPhi (all 500 steps)
# ---------------------------------------------------------------------------
print("\n=== Full Guidance DiversityPhi (s=200, all steps) ===")
pipe = SalienceGradSDPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
).to(DEVICE)

phi_div = DiversityPhiSD()
pipe.setup_phi(phi_div, context={})
pipe.set_salience_scale(SAL_SCALE)
pipe.set_guidance_frequency(1)

prompt_embeds = pipe._encode_prompt(
    PROMPT, DEVICE, NUM_IMAGES, do_cfg, negative_prompt=None,
)

# Reuse the same initial latents as baseline for direct paired comparison
latents = baseline_latents.to(DEVICE)

run_denoising_loop(
    pipe, prompt_embeds, latents,
    condition_dir=OUTPUT_DIR / "full_guidance",
    salience_scale=SAL_SCALE,
    guidance_start_step=0,
)

del pipe
clear()
print("Full guidance done.")
print(f"\nAll done. Results in {OUTPUT_DIR}")
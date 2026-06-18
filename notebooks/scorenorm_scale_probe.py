"""
notebooks/scorenorm_scale_probe.py
-------------------------------------
Quick visual probe: generate 1 image per scale, per prompt, across a few
candidate guidance scales for ScoreNormPhi, before committing to the full
16-image-per-prompt experiment. ScoreNormPhi's salience landscape is a
different function of the latents than DiversityPhi's, so there's no
reason to assume scale=200 (DiversityPhi's subtle-effect scale) behaves
similarly here -- this could be too weak, too strong, or roughly right.

Run from repo root:
    PYTHONPATH=$(pwd) python notebooks/scorenorm_scale_probe.py
"""

import torch
import gc
from pathlib import Path
from sd.pipeline import SalienceGradSDPipeline
from sd.phi_sd import ScoreNormPhiSD

DEVICE     = "cuda"
MODEL_ID   = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = Path("../outputs/sd_experiment/scorenorm_scale_probe")
PROMPTS    = [
    "a green apple on a brown table",
    "a wolf in the woods",
]  # just 2 prompts for the probe, full 4 in the real experiment

NUM_STEPS  = 500
CFG_SCALE  = 5.0
CANDIDATE_SCALES = [50.0, 200.0, 1000.0, 5000.0]
SEED       = 2024

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clear():
    gc.collect()
    torch.cuda.empty_cache()

print("Loading pipeline...")
pipe = SalienceGradSDPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
).to(DEVICE)
pipe.unet.enable_gradient_checkpointing()

with torch.no_grad():
    null_input = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    null_embeds = pipe.text_encoder(
        null_input.input_ids.to(DEVICE)
    )[0]

phi_norm = ScoreNormPhiSD()
pipe.setup_phi(phi_norm, context={"null_embeds": null_embeds})
pipe.set_guidance_frequency(1)

# Baseline (no salience guidance) for each prompt, for comparison
print("\n=== Baseline (scale=0) ===")
pipe.set_salience_scale(0.0)
for prompt in PROMPTS:
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    image = pipe(
        prompt=prompt,
        num_images_per_prompt=1,
        num_inference_steps=NUM_STEPS,
        guidance_scale=CFG_SCALE,
        generator=generator,
    ).images[0]
    save_dir = OUTPUT_DIR / prompt.replace(" ", "_")
    save_dir.mkdir(parents=True, exist_ok=True)
    image.save(save_dir / "baseline.png")
    print(f"  Saved baseline for: {prompt}")

# One image per candidate scale, per prompt, same seed for paired comparison
for scale in CANDIDATE_SCALES:
    print(f"\n=== ScoreNormPhi scale={scale} ===")
    pipe.set_salience_scale(scale)
    for prompt in PROMPTS:
        generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        image = pipe(
            prompt=prompt,
            num_images_per_prompt=1,
            num_inference_steps=NUM_STEPS,
            guidance_scale=CFG_SCALE,
            generator=generator,
        ).images[0]
        save_dir = OUTPUT_DIR / prompt.replace(" ", "_")
        image.save(save_dir / f"scale_{int(scale)}.png")
        print(f"  Saved scale={scale} for: {prompt}")

del pipe
clear()
print(f"\nProbe done. Results in {OUTPUT_DIR}")
print("Inspect images, pick the scale with the clearest non-degenerate effect,")
print("then update SAL_SCALE in scorenorm_experiment.py before the full run.")
import torch
import gc
from pathlib import Path
from diffusers import StableDiffusionPipeline
from sd.pipeline import SalienceGradSDPipeline
from sd.phi_sd import DiversityPhiSD

DEVICE     = "cuda"
MODEL_ID   = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = Path("../outputs/sd_experiment")
PROMPTS    = [
    "a green apple on a brown table",
    "a wooden chair in a blue room",
    "a wolf in the woods",
    "a piece of toast on a plate",
]

NUM_IMAGES = 50
NUM_STEPS  = 500
CFG_SCALE  = 5.0
SAL_SCALE  = 200.0
BATCH_SIZE = 16
SEED       = 2024

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clear():
    gc.collect()
    torch.cuda.empty_cache()

def generate(pipe, prompt, save_dir, n, batch_size, seed):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    batch_idx = 0
    while generated < n:
        current_batch = min(batch_size, n - generated)
        generator = torch.Generator(device=DEVICE).manual_seed(seed + batch_idx)
        pipe(
            prompt=prompt,
            num_images_per_prompt=current_batch,
            num_inference_steps=NUM_STEPS,
            guidance_scale=CFG_SCALE,
            generator=generator,
            save_dir=save_dir,
            offset=generated,
        )
        generated += current_batch
        batch_idx += 1
        print(f"    {generated}/{n}")

# ---------------------------------------------------------------------------
# 1. Baseline
# ---------------------------------------------------------------------------
print("=== Baseline ===")
baseline_pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
).to(DEVICE)

for prompt in PROMPTS:
    print(f"\nPrompt: {prompt}")
    save_dir = OUTPUT_DIR / "baseline" / prompt.replace(" ", "_")
    save_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    batch_idx = 0
    while generated < NUM_IMAGES:
        current_batch = min(BATCH_SIZE, NUM_IMAGES - generated)
        generator = torch.Generator(device=DEVICE).manual_seed(SEED + batch_idx)
        images = baseline_pipe(
            prompt=prompt,
            num_images_per_prompt=current_batch,
            num_inference_steps=NUM_STEPS,
            guidance_scale=CFG_SCALE,
            generator=generator,
        ).images
        for idx, img in enumerate(images):
            img.save(save_dir / f"{generated + idx}.png")
        generated += current_batch
        batch_idx += 1
        print(f"    {generated}/{NUM_IMAGES}")

del baseline_pipe
clear()
print("\nBaseline done.")

# ---------------------------------------------------------------------------
# 2. DiversityPhi
# ---------------------------------------------------------------------------
print("\n=== DiversityPhi ===")
pipe = SalienceGradSDPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
).to(DEVICE)

phi_div = DiversityPhiSD()
pipe.setup_phi(phi_div, context={})
pipe.set_salience_scale(SAL_SCALE)
pipe.set_guidance_frequency(1)

for prompt in PROMPTS:
    print(f"\nPrompt: {prompt}")
    generate(
        pipe, prompt,
        save_dir=OUTPUT_DIR / "diversity_sal200" / prompt.replace(" ", "_"),
        n=NUM_IMAGES, batch_size=BATCH_SIZE, seed=SEED,
    )

del pipe
clear()
print("\nDiversityPhi done.")
print(f"\nAll done. Results in {OUTPUT_DIR}")

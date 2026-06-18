"""
notebooks/scorenorm_timing_probe.py
-------------------------------------
Quick sanity check: run a handful of reverse-process steps under
ScoreNormPhi on the A100 and time them, before committing to a full
500-step generation run. If per-step time is reasonable (sub-second to a
few seconds), proceed to the full experiment. If it's still minutes per
step, ScoreNormPhi is confirmed intractable here too and the thesis
sentence stands as written.

Run from repo root:
    PYTHONPATH=$(pwd) python notebooks/scorenorm_timing_probe.py
"""

import time
import torch
from diffusers import StableDiffusionPipeline
from sd.pipeline import SalienceGradSDPipeline
from sd.phi_sd import ScoreNormPhiSD

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

DEVICE   = "cuda"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
PROMPT   = "a green apple on a brown table"
SAL_SCALE = 200.0  # same starting scale as DiversityPhi's first ablation point
N_PROBE_STEPS = 5  # only run a handful of steps to measure timing
BATCH_SIZE = 4
SEED = 2024

print("Loading pipeline...")
pipe = SalienceGradSDPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
).to(DEVICE)

# Encode the null/empty prompt directly via tokenizer + text_encoder,
# bypassing encode_prompt/_encode_prompt version differences entirely.
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
    )[0]  # (1, seq_len, embed_dim)

phi_norm = ScoreNormPhiSD()
pipe.setup_phi(phi_norm, context={"null_embeds": null_embeds})
pipe.set_salience_scale(SAL_SCALE)
pipe.set_guidance_frequency(1)

generator = torch.Generator(device=DEVICE).manual_seed(SEED)

print(f"Timing {N_PROBE_STEPS} reverse steps under ScoreNormPhi (batch={BATCH_SIZE})...")

torch.cuda.synchronize()
start = time.time()

# Most pipeline implementations accept num_inference_steps directly;
# here we temporarily set it low purely to measure per-step cost.
_ = pipe(
    prompt=PROMPT,
    num_images_per_prompt=BATCH_SIZE,
    num_inference_steps=N_PROBE_STEPS,
    guidance_scale=5.0,
    generator=generator,
)

torch.cuda.synchronize()
elapsed = time.time() - start
per_step = elapsed / N_PROBE_STEPS

print(f"\nTotal time for {N_PROBE_STEPS} steps: {elapsed:.2f}s")
print(f"Per-step time: {per_step:.2f}s")
print(f"Estimated time for full 500-step run: {per_step * 500 / 60:.1f} minutes")

if per_step < 5:
    print("\nLooks tractable -- proceed to the full experiment script.")
elif per_step < 30:
    print("\nSlow but possibly workable for a reduced step count or smaller batch.")
else:
    print("\nStill intractable at this scale -- confirms the MPS finding on A100 too.")
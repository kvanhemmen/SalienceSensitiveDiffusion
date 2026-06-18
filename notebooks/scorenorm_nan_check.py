"""
notebooks/scorenorm_nan_check.py
------------------------------------
Diagnostic: run a single reverse step under ScoreNormPhi and check whether
grad_log_S contains NaN values, mirroring the failure mode previously
diagnosed for ScoreAlignmentPhi (float16 underflow in the second-order
backward pass once flash attention is disabled in favor of standard
attention).

Run from repo root:
    PYTHONPATH=$(pwd) python notebooks/scorenorm_nan_check.py
"""

import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from sd.pipeline import SalienceGradSDPipeline
from sd.phi_sd import ScoreNormPhiSD

DEVICE   = "cuda"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

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

# Build a fake latent + context to call _single_salience_grad directly,
# bypassing the full pipeline __call__ for a focused check.
latent_shape = (4, 64, 64)  # standard SD v1.5 latent shape (C, H, W)
z = torch.randn(latent_shape, device=DEVICE, dtype=torch.float16)
t = 500  # arbitrary mid-range timestep

context = {
    "unet": pipe.unet,
    "scheduler": pipe.scheduler,
    "null_embeds": null_embeds,
}
pipe.setup_phi(phi_norm, context=context)

grad = pipe._single_salience_grad(z, t, context)

print("\ngrad shape:", grad.shape)
print("grad dtype:", grad.dtype)
print("Contains NaN:", torch.isnan(grad).any().item())
print("Contains Inf:", torch.isinf(grad).any().item())
print("grad min/max:", grad.min().item(), grad.max().item())
print("grad norm:", grad.norm().item())
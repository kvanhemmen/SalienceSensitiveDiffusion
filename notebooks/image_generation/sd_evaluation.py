"""
notebooks/sd_evaluation.py
----------------------------
Evaluates DiversityPhi vs DDPM baseline on:
    1. LPIPS pairwise diversity  -- perceptual spread within batch
    2. T-CLIP                    -- prompt-image alignment
    3. Intra-batch CLIP variance -- semantic spread within batch

Run from the repo root:
    python notebooks/sd_evaluation.py
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from itertools import combinations
import os

import lpips
import clip as openai_clip

DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path("../outputs/sd_experiment")
METHODS    = ["baseline", "diversity_sal200", "diversity_sal2000", "diversity_sal20000"]
BATCH_SIZE = 16
PROMPTS    = {
    "a_green_apple_on_a_brown_table": "a green apple on a brown table",
    "a_wooden_chair_in_a_blue_room":  "a wooden chair in a blue room",
    "a_wolf_in_the_woods":            "a wolf in the woods",
    "a_piece_of_toast_on_a_plate":    "a piece of toast on a plate",
}

for method in METHODS:
    for prompt in PROMPTS:
        folder = OUTPUT_DIR / method / prompt
        for idx in [48, 49]:
            path = folder / f"{idx}.png"
            if path.exists():
                os.remove(path)
                print(f"Removed {path}")

print("Loading LPIPS...")
loss_fn = lpips.LPIPS(net="alex").to(DEVICE)

print("Loading CLIP...")
clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=DEVICE)

def load_images(method, prompt_dir):
    folder = OUTPUT_DIR / method / prompt_dir
    paths = sorted(folder.glob("*.png"))
    return [Image.open(p).convert("RGB") for p in paths]

def pil_to_lpips(imgs):
    tensors = []
    for img in imgs:
        t = torch.tensor(np.array(img.resize((256, 256)))).float() / 127.5 - 1.0
        tensors.append(t.permute(2, 0, 1))
    return torch.stack(tensors).to(DEVICE)

def compute_lpips_diversity_within_batch(imgs, batch_size=16):
    """Mean pairwise LPIPS within each batch, averaged across batches."""
    n = len(imgs)
    batch_scores = []
    for start in range(0, n, batch_size):
        batch = imgs[start:start + batch_size]
        if len(batch) < 2:
            continue
        t = pil_to_lpips(batch)
        scores = []
        for i, j in combinations(range(len(t)), 2):
            d = loss_fn(t[i].unsqueeze(0), t[j].unsqueeze(0)).item()
            scores.append(d)
        batch_scores.append(float(np.mean(scores)))
    return float(np.mean(batch_scores))

def compute_clip_embeddings(imgs):
    tensors = torch.stack([
        clip_preprocess(img) for img in imgs
    ]).to(DEVICE)
    with torch.no_grad():
        embeddings = clip_model.encode_image(tensors)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.float().cpu()

def compute_tclip(imgs, prompt):
    img_embeddings = compute_clip_embeddings(imgs)
    text_tokens = openai_clip.tokenize([prompt]).to(DEVICE)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding.float().cpu()
    similarities = (img_embeddings @ text_embedding.T).squeeze(-1)
    return float(similarities.mean())

def compute_clip_variance(imgs):
    embeddings = compute_clip_embeddings(imgs)
    return float(embeddings.var(dim=0).mean())

results = {method: {
    "lpips_diversity": [],
    "tclip": [],
    "clip_variance": [],
} for method in METHODS}

for prompt_dir, prompt_text in PROMPTS.items():
    print(f"\nPrompt: {prompt_text}")
    for method in METHODS:
        imgs = load_images(method, prompt_dir)
        if len(imgs) == 0:
            print(f"  {method}: 0 images — skipping")
            results[method]["lpips_diversity"].append(None)
            results[method]["tclip"].append(None)
            results[method]["clip_variance"].append(None)
            continue

        print(f"  {method}: {len(imgs)} images")
        lpips_div = compute_lpips_diversity_within_batch(imgs, batch_size=BATCH_SIZE)
        tclip     = compute_tclip(imgs, prompt_text)
        clip_var  = compute_clip_variance(imgs)

        results[method]["lpips_diversity"].append(lpips_div)
        results[method]["tclip"].append(tclip)
        results[method]["clip_variance"].append(clip_var)

        print(f"    LPIPS diversity: {lpips_div:.4f}")
        print(f"    T-CLIP:          {tclip:.4f}")
        print(f"    CLIP variance:   {clip_var:.6f}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

col_width = 16
header = f"{'Metric':<25}" + "".join(f"{m:>{col_width}}" for m in METHODS)
print("\n" + "=" * (25 + col_width * len(METHODS)))
print(header)
print("=" * (25 + col_width * len(METHODS)))

for metric, label in [
    ("lpips_diversity", "LPIPS Diversity"),
    ("tclip",           "T-CLIP"),
    ("clip_variance",   "CLIP Variance"),
]:
    row = f"{label:<25}"
    for method in METHODS:
        vals = [v for v in results[method][metric] if v is not None]
        mean = np.mean(vals) if vals else float("nan")
        row += f"{mean:>{col_width}.4f}"
    print(row)

print("=" * (25 + col_width * len(METHODS)))
print("\nAll values are means across prompts.")
print("LPIPS diversity: higher = more visually spread")
print("T-CLIP: higher = better prompt alignment")
print("CLIP variance: higher = more semantically spread")
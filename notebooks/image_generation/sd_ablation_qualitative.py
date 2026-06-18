"""
notebooks/sd_ablation_qualitative.py
--------------------------------------
Shows one seed across all 4 methods (baseline, sal200, sal2000, sal20000)
for each prompt. 4 rows x 4 cols: one row per prompt, one col per method.

Run from repo root:
    python notebooks/sd_ablation_qualitative.py
"""

import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from PIL import Image

matplotlib.rcParams.update({
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'black',
})

OUTPUT_DIR = Path("outputs/sd_experiment")
SAVE_PATH  = Path("outputs/sd_experiment/ablation_qualitative.png")
IMAGE_IDX  = 5  # which seed to show

METHODS = [
    ("baseline",          "Baseline"),
    ("diversity_sal200",  "s=200"),
    ("diversity_sal2000", "s=2000"),
    ("diversity_sal20000","s=20000"),
]

PROMPTS = [
    ("a_green_apple_on_a_brown_table", "A Green Apple on a Brown Table"),
    ("a_wooden_chair_in_a_blue_room",  "A Wooden Chair in a Blue Room"),
    ("a_wolf_in_the_woods",            "A Wolf in the Woods"),
    ("a_piece_of_toast_on_a_plate",    "A Piece of Toast on a Plate"),
]

def load_image(method, prompt_dir, idx):
    path = OUTPUT_DIR / method / prompt_dir / f"{idx}.png"
    return Image.open(path).convert("RGB")

n_rows = len(PROMPTS)
n_cols = len(METHODS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(n_cols * 3.5, n_rows * 3.5),
    facecolor="white",
)

for row, (prompt_dir, prompt_label) in enumerate(PROMPTS):
    for col, (method, method_label) in enumerate(METHODS):
        ax = axes[row, col]
        img = load_image(method, prompt_dir, IMAGE_IDX)
        ax.imshow(img)
        ax.axis("off")
        ax.set_facecolor("white")

        if row == 0:
            ax.set_title(method_label, fontsize=11, color="black")

        if col == 0:
            ax.text(
                -0.05, 0.5, prompt_label,
                transform=ax.transAxes,
                fontsize=9, color="black",
                va="center", ha="right",
                rotation=90,
            )

fig.suptitle(
    "DiversityPhi Guidance Scale Ablation (seed={})".format(IMAGE_IDX),
    fontsize=13, color="black"
)
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print(f"Saved to {SAVE_PATH}")
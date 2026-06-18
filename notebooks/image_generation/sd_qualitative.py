"""
notebooks/sd_qualitative.py
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

OUTPUT_DIR = Path("../outputs/sd_experiment")
SAVE_PATH  = Path("outputs/sd_experiment/qualitative_comparison.png")

PROMPTS = [
    ("a_green_apple_on_a_brown_table", "A Green Apple on a Brown Table"),
    ("a_wooden_chair_in_a_blue_room",  "A Wooden Chair in a Blue Room"),
    ("a_wolf_in_the_woods",            "A Wolf in the Woods"),
    ("a_piece_of_toast_on_a_plate",    "A Piece of Toast on a Plate"),
]

ROWS = [
    ("baseline",         PROMPTS[:2], "Baseline"),
    ("diversity_sal200", PROMPTS[:2], "Salience-Sensitive"),
    ("baseline",         PROMPTS[2:], "Baseline"),
    ("diversity_sal200", PROMPTS[2:], "Salience-Sensitive"),
]

IMAGE_INDICES = [1, 5]

def load_image(method, prompt_dir, idx):
    path = OUTPUT_DIR / method / prompt_dir / f"{idx}.png"
    return Image.open(path).convert("RGB")

fig, axes = plt.subplots(4, 4, figsize=(14, 14), facecolor="white")

for row, (method, prompt_pair, row_label) in enumerate(ROWS):
    for col_pair, (prompt_dir, prompt_label) in enumerate(prompt_pair):
        for img_pos, img_idx in enumerate(IMAGE_INDICES):
            col = col_pair * 2 + img_pos
            ax = axes[row, col]
            img = load_image(method, prompt_dir, img_idx)
            ax.imshow(img)
            ax.axis("off")
            ax.set_facecolor("white")

            # Row label on leftmost column
            if col == 0:
                ax.text(
                    -0.05, 0.5, row_label,
                    transform=ax.transAxes,
                    fontsize=11, color="black",
                    va="center", ha="right",
                    rotation=90,
                )

            # Prompt title on top rows only
            if row == 0 and img_pos == 0:
                ax.set_title(prompt_label, fontsize=11, color="black")
            elif row == 2 and img_pos == 0:
                ax.set_title(prompt_label, fontsize=11, color="black")

fig.suptitle(
    "Baseline vs Salience-Sensitive Guidance w/ DiversityPhi (guidance scale=200)",
    fontsize=13, color="black"
)
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print(f"Saved to {SAVE_PATH}")
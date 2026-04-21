"""
experiments/comparison.py
-------------------------
Saving utilities and multi-panel comparison figures for evaluating
salience-guided samplers against baseline DDPM.

Functions
---------
save_samples          -- save a sample tensor to disk (.pt and/or .npy)
save_figure           -- save a matplotlib figure to disk
make_comparison_figure -- four-panel comparison figure:
                            1. Side-by-side scatter plots
                            2. Distance-to-nearest-mode histograms
                            3. Pairwise diversity (mean inter-sample distance)
                            4. Mean GMM log-probability
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch import Tensor

from experiments.evaluation import distance_to_nearest_mode
from experiments.gmm import GaussianMixture, gmm_pdf_contour


# ---------------------------------------------------------------------------
# Saving utilities
# ---------------------------------------------------------------------------

def save_samples(
    samples: Tensor,
    name: str,
    root: Path,
    save_pt: bool = True,
    save_npy: bool = True,
) -> None:
    """
    Save a sample tensor to disk.

    Args:
        samples  : (N, D) Tensor
        name     : descriptive filename stem, e.g. "ddpm_baseline"
        root     : directory to save into (created if it does not exist)
        save_pt  : save as a PyTorch .pt file
        save_npy : save as a numpy .npy file
    """
    root.mkdir(parents=True, exist_ok=True)
    x = samples.detach().cpu()

    if save_pt:
        path = root / f"{name}.pt"
        torch.save(x, path)
        print(f"Saved samples -> {path}")

    if save_npy:
        path = root / f"{name}.npy"
        np.save(path, x.numpy())
        print(f"Saved samples -> {path}")


def save_figure(fig: plt.Figure, name: str, root: Path, dpi: int = 150) -> None:
    """
    Save a matplotlib figure to disk as a PNG.

    Args:
        fig  : matplotlib Figure
        name : descriptive filename stem, e.g. "comparison_ddpm_vs_diversity"
        root : directory to save into (created if it does not exist)
        dpi  : output resolution
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure  -> {path}")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _pairwise_diversity(samples: np.ndarray) -> float:
    """
    Mean Euclidean distance between all distinct pairs of samples.

    This is O(N^2) in memory — subsample if N is large.

    Args:
        samples : (N, D)

    Returns:
        mean pairwise distance as a float
    """
    # Use broadcasting: (N, 1, D) - (1, N, D) -> (N, N, D)
    diff = samples[:, None, :] - samples[None, :, :]       # (N, N, D)
    dists = np.linalg.norm(diff, axis=-1)                  # (N, N)
    # Upper triangle only (excluding diagonal) to avoid double-counting
    idx = np.triu_indices(len(samples), k=1)
    return float(dists[idx].mean())


def _mean_gmm_log_prob(samples: np.ndarray, gmm: GaussianMixture) -> float:
    """
    Mean log p(x) under the full GMM for a set of samples.

    Args:
        samples : (N, D) numpy array
        gmm     : GaussianMixture instance

    Returns:
        mean log probability as a float
    """
    x = torch.tensor(samples, dtype=torch.float32, device=gmm.device)
    log_probs = gmm.log_prob(x).detach().cpu().numpy()
    return float(log_probs.mean())


# ---------------------------------------------------------------------------
# Multi-panel comparison figure
# ---------------------------------------------------------------------------

def make_comparison_figure(
    sample_dict: Dict[str, Tensor],
    gmm: GaussianMixture,
    modes: np.ndarray,
    suptitle: str = "Sampler Comparison",
    max_pairs: int = 1000,
) -> plt.Figure:
    """
    Four-panel comparison figure for two or more samplers.

    Panels:
        Top row    : scatter plot for each sampler (one panel per sampler)
        Bottom-left  : distance-to-nearest-mode histogram
        Bottom-middle: mean pairwise diversity bar chart
        Bottom-right : mean GMM log-probability bar chart

    Args:
        sample_dict : {"sampler_name": samples_tensor (N, D), ...}
        gmm         : GaussianMixture — for contours and log-prob evaluation
        modes       : (K, D) numpy array of mode centres
        suptitle    : overall figure title
        max_pairs   : subsample to this many points for pairwise diversity
                      computation (O(N^2) otherwise)

    Returns:
        fig : matplotlib Figure
    """
    names = list(sample_dict.keys())
    n_samplers = len(names)

    # Convert all to clean numpy arrays (drop NaN/inf rows)
    arrays: Dict[str, np.ndarray] = {}
    for name, tensor in sample_dict.items():
        x = tensor.detach().cpu().numpy()
        mask = np.isfinite(x).all(axis=1)
        if not mask.all():
            print(f"  [{name}] dropping {(~mask).sum()} non-finite samples")
        arrays[name] = x[mask]

    # ------------------------------------------------------------------
    # Layout: top row has n_samplers scatter panels, bottom row has 3
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(6 * max(n_samplers, 3), 12))
    gs = gridspec.GridSpec(
        2, max(n_samplers, 3),
        figure=fig,
        hspace=0.35,
        wspace=0.3,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ------------------------------------------------------------------
    # Top row: scatter plots
    # ------------------------------------------------------------------
    scatter_axes = []
    for col, name in enumerate(names):
        ax = fig.add_subplot(gs[0, col])
        scatter_axes.append(ax)
        x = arrays[name]

        plt.sca(ax)
        gmm_pdf_contour(gmm, colors="gray", alpha=0.35, linewidths=0.8)

        ax.scatter(x[:, 0], x[:, 1], s=8, alpha=0.45, color=colors[col], label=name)

        for j, m in enumerate(modes):
            ax.scatter(m[0], m[1], s=180, marker="x", color="black",
                       linewidths=2, zorder=5,
                       label="mode" if (col == 0 and j == 0) else "_")

        ax.set_title(name, fontsize=12)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

    # Share axes across scatter panels so scales are directly comparable
    for ax in scatter_axes[1:]:
        ax.sharex(scatter_axes[0])
        ax.sharey(scatter_axes[0])

    # ------------------------------------------------------------------
    # Bottom-left: distance-to-nearest-mode histogram
    # ------------------------------------------------------------------
    ax_dist = fig.add_subplot(gs[1, 0])
    for col, name in enumerate(names):
        dists = distance_to_nearest_mode(arrays[name], modes)
        ax_dist.hist(dists, bins=35, alpha=0.55, color=colors[col], label=name)
        ax_dist.axvline(dists.mean(), color=colors[col], linestyle="--",
                        linewidth=1.5, label=f"{name} mean={dists.mean():.2f}")
    ax_dist.set_xlabel("Distance to nearest mode")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Distance to nearest mode")
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Bottom-middle: mean pairwise diversity bar chart
    # ------------------------------------------------------------------
    ax_div = fig.add_subplot(gs[1, 1])
    diversity_scores = {}
    for name in names:
        x = arrays[name]
        if len(x) > max_pairs:
            idx = np.random.choice(len(x), max_pairs, replace=False)
            x_sub = x[idx]
        else:
            x_sub = x
        diversity_scores[name] = _pairwise_diversity(x_sub)

    bars = ax_div.bar(
        names,
        [diversity_scores[n] for n in names],
        color=colors[:n_samplers],
        alpha=0.75,
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, name in zip(bars, names):
        ax_div.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{diversity_scores[name]:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax_div.set_title("Mean pairwise diversity\n(higher = more diverse)")
    ax_div.set_ylabel("Mean pairwise distance")
    ax_div.grid(True, axis="y", alpha=0.3)

    # ------------------------------------------------------------------
    # Bottom-right: mean GMM log-probability bar chart
    # ------------------------------------------------------------------
    ax_prob = fig.add_subplot(gs[1, 2])
    log_prob_scores = {}
    for name in names:
        log_prob_scores[name] = _mean_gmm_log_prob(arrays[name], gmm)

    bars = ax_prob.bar(
        names,
        [log_prob_scores[n] for n in names],
        color=colors[:n_samplers],
        alpha=0.75,
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, name in zip(bars, names):
        ax_prob.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{log_prob_scores[name]:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax_prob.set_title("Mean GMM log-probability\n(lower = more tail-seeking)")
    ax_prob.set_ylabel("Mean log p(x)")
    ax_prob.grid(True, axis="y", alpha=0.3)

    # ------------------------------------------------------------------
    # Print summary table to console
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"{'Metric':<30} " + "  ".join(f"{n:>10}" for n in names))
    print(f"{'-'*55}")
    for name in names:
        dists = distance_to_nearest_mode(arrays[name], modes)
        print(f"{'Mean dist to nearest mode':<30} {dists.mean():>10.4f}")
    for name in names:
        print(f"{'Mean pairwise diversity':<30} {diversity_scores[name]:>10.4f}")
    for name in names:
        print(f"{'Mean GMM log-prob':<30} {log_prob_scores[name]:>10.4f}")
    print(f"{'='*55}\n")

    fig.suptitle(suptitle, fontsize=14, y=1.01)
    return fig
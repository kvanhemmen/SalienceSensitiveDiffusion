"""
experiments/evaluation.py
-------------------------
Evaluation metrics and visualisation utilities for 2-D diffusion experiments.

Functions
---------
scatter_samples           -- scatter plot of generated samples vs. GMM contours
distance_to_nearest_mode  -- per-sample distance to the nearest GMM mode
summarise_samples         -- mean, covariance, and raw numpy array
kl_mvn                    -- KL divergence between two multivariate Gaussians
compare_samplers          -- side-by-side scatter plot for multiple samplers
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Distance and summary statistics
# ---------------------------------------------------------------------------

def distance_to_nearest_mode(
    samples: np.ndarray,
    modes: np.ndarray,
) -> np.ndarray:
    """
    Per-sample Euclidean distance to the nearest mode centre.

    Args:
        samples : (N, d) array of generated samples
        modes   : (K, d) array of mode centre coordinates

    Returns:
        distances : (N,) array of minimum distances
    """
    dists = np.linalg.norm(
        samples[:, None, :] - modes[None, :, :], axis=-1
    )
    return dists.min(axis=1)


def summarise_samples(samples: Tensor) -> Dict:
    """
    Compute mean, covariance, and raw numpy array from a sample tensor.

    Returns:
        dict with keys "mean", "cov", "samples"
    """
    x = samples.detach().cpu().numpy()
    return {
        "mean": x.mean(axis=0),
        "cov": np.cov(x.T),
        "samples": x,
    }


def kl_mvn(
    m0: np.ndarray,
    S0: np.ndarray,
    m1: np.ndarray,
    S1: np.ndarray,
) -> float:
    """
    KL divergence KL(N(m0,S0) || N(m1,S1)) in closed form.

    Args:
        m0, S0 : mean and covariance of the first Gaussian
        m1, S1 : mean and covariance of the second Gaussian

    Returns:
        KL divergence as a float
    """
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0
    tr_term = np.trace(iS1 @ S0)
    det_term = np.log(np.linalg.det(S1) / np.linalg.det(S0))
    quad_term = diff.T @ iS1 @ diff
    return float(0.5 * (tr_term + det_term + quad_term - N))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def scatter_samples(
    samples: Tensor,
    gmm=None,
    modes: Optional[np.ndarray] = None,
    title: str = "Samples",
    ax: Optional[plt.Axes] = None,
    **scatter_kwargs,
) -> plt.Axes:
    """
    Scatter plot of generated samples, optionally overlaid with GMM contours
    and mode markers.

    Args:
        samples        : (N, 2) Tensor
        gmm            : GaussianMixture instance (optional, for contours)
        modes          : (K, 2) numpy array of mode centres (optional)
        title          : plot title
        ax             : existing axes to draw on; creates a new figure if None
        **scatter_kwargs : forwarded to ax.scatter

    Returns:
        ax : the matplotlib Axes object
    """
    from experiments.gmm import gmm_pdf_contour

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    x = samples.detach().cpu().numpy()

    if gmm is not None:
        plt.sca(ax)
        gmm_pdf_contour(gmm, colors="gray", alpha=0.4)

    scatter_kwargs.setdefault("s", 10)
    scatter_kwargs.setdefault("alpha", 0.5)
    ax.scatter(x[:, 0], x[:, 1], **scatter_kwargs)

    if modes is not None:
        for i, m in enumerate(modes):
            ax.scatter(m[0], m[1], s=200, marker="x", label=f"mode {i+1}", zorder=5)

    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True)
    return ax


def compare_samplers(
    sample_dict: Dict[str, Tensor],
    gmm=None,
    modes: Optional[np.ndarray] = None,
    suptitle: str = "Sampler Comparison",
) -> plt.Figure:
    """
    Side-by-side scatter plots for multiple samplers.

    Args:
        sample_dict : {"sampler_name": samples_tensor, ...}
        gmm         : GaussianMixture for contours (optional)
        modes       : (K, 2) mode centres (optional)
        suptitle    : overall figure title

    Returns:
        fig : matplotlib Figure
    """
    n = len(sample_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, samples) in zip(axes, sample_dict.items()):
        scatter_samples(samples, gmm=gmm, modes=modes, title=name, ax=ax)

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    return fig


def plot_distance_histogram(
    sample_dict: Dict[str, np.ndarray],
    modes: np.ndarray,
    title: str = "Distance to nearest mode",
) -> plt.Figure:
    """
    Overlapping histograms of distance-to-nearest-mode for multiple samplers.

    Args:
        sample_dict : {"sampler_name": samples_numpy_array, ...}
        modes       : (K, 2) mode centres
        title       : plot title

    Returns:
        fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, samples in sample_dict.items():
        mask = np.isfinite(samples).all(axis=1)
        dists = distance_to_nearest_mode(samples[mask], modes)
        ax.hist(dists, bins=40, alpha=0.5, label=name)
        print(f"{name}: mean dist = {dists.mean():.4f}  ({mask.sum()}/{len(samples)} valid)")

    ax.set_xlabel("Distance to nearest mode")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

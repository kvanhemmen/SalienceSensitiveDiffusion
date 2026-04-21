"""
experiments/visualize_salience.py
----------------------------------
Visualise the salience landscape S(x) = ||grad_x phi(x)||^2 over a 2-D grid.

Because your data is 2-D, the salience can be plotted as a heatmap directly
over the sample space, overlaid with GMM contours and generated samples.
This makes it possible to visually verify that generated samples are landing
in high-salience regions.

Functions
---------
compute_salience_grid  -- evaluate log S(x) over a 2-D meshgrid
plot_salience_landscape -- multi-panel figure across multiple timesteps
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from salience.sampler import PhiBase, log_salience
from experiments.gmm import GaussianMixture, gmm_pdf_contour


# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------

def compute_salience_grid(
    phi: PhiBase,
    t: int,
    context: dict,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    resolution: int = 100,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate log S(x) = log ||grad_x phi(x)||^2 over a 2-D meshgrid.

    Each grid point is evaluated independently via log_salience, so this
    is exact rather than approximated.

    Args:
        phi        : PhiBase instance
        t          : diffusion timestep at which to evaluate
        context    : passed through to phi (e.g. {"model": ..., "scheduler": ...})
        xlim       : (min, max) for x axis
        ylim       : (min, max) for y axis
        resolution : number of grid points per axis (resolution^2 total evals)
        device     : torch device

    Returns:
        xx      : (resolution, resolution) meshgrid x coordinates
        yy      : (resolution, resolution) meshgrid y coordinates
        log_S   : (resolution, resolution) log salience values
    """
    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    xx, yy = np.meshgrid(xs, ys)

    log_S = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            x = torch.tensor(
                [xx[i, j], yy[i, j]], dtype=torch.float32, device=device
            )
            log_S[i, j] = log_salience(phi, x, t, context).item()

    return xx, yy, log_S


# ---------------------------------------------------------------------------
# Multi-panel figure
# ---------------------------------------------------------------------------

def plot_salience_landscape(
    phi: PhiBase,
    timesteps: List[int],
    context: dict,
    gmm: Optional[GaussianMixture] = None,
    samples: Optional[Tensor] = None,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    resolution: int = 100,
    device: torch.device = torch.device("cpu"),
    suptitle: str = "Salience landscape",
) -> plt.Figure:
    """
    Multi-panel heatmap of the salience landscape at multiple timesteps.

    Each panel shows:
        - log S(x) as a filled heatmap
        - GMM PDF contours overlaid in white (if gmm provided)
        - Generated samples scattered on top (if samples provided)

    Args:
        phi        : PhiBase instance
        timesteps  : list of diffusion timesteps to visualise
        context    : passed through to phi at every grid point
        gmm        : GaussianMixture for contour overlay (optional)
        samples    : (N, 2) Tensor of generated samples to scatter (optional)
        xlim       : x axis range
        ylim       : y axis range
        resolution : grid resolution per axis
        device     : torch device
        suptitle   : overall figure title

    Returns:
        fig : matplotlib Figure
    """
    n = len(timesteps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, t in zip(axes, timesteps):
        print(f"  Computing salience grid at t={t}...")

        xx, yy, log_S = compute_salience_grid(
            phi=phi,
            t=t,
            context=context,
            xlim=xlim,
            ylim=ylim,
            resolution=resolution,
            device=device,
        )

        # Heatmap of log salience
        im = ax.pcolormesh(
            xx, yy, log_S,
            cmap="plasma",
            shading="auto",
        )
        plt.colorbar(im, ax=ax, label="log S(x)")

        # GMM contours
        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(gmm, xlim=xlim, ylim=ylim, colors="white",
                            alpha=0.6, linewidths=1.0)

        # Generated samples
        if samples is not None:
            x_np = samples.detach().cpu().numpy()
            ax.scatter(
                x_np[:, 0], x_np[:, 1],
                s=8, alpha=0.5, color="cyan",
                label="samples", zorder=5,
            )

        ax.set_title(f"t = {t}", fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    return fig

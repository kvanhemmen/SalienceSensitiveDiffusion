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

            with torch.enable_grad():
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor="white")
    axes = axes.flatten()

    for ax, t in zip(axes, timesteps):
        ax.set_facecolor("white")
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

        im = ax.pcolormesh(
            xx, yy, log_S,
            cmap="plasma",
            shading="auto",
        )
        cbar = plt.colorbar(im, ax=ax, label="log S(x)")
        cbar.ax.yaxis.label.set_color("black")
        cbar.ax.tick_params(colors="black")

        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(gmm, xlim=xlim, ylim=ylim, colors="white",
                            alpha=0.6, linewidths=1.0)

        if samples is not None:
            x_np = samples.detach().cpu().numpy()
            ax.scatter(
                x_np[:, 0], x_np[:, 1],
                s=8, alpha=0.5, color="cyan",
                label="samples", zorder=5,
            )

        ax.set_title(f"t = {t}", fontsize=12, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$", color="black")
        ax.set_ylabel("$x_2$", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    fig.suptitle(suptitle, fontsize=14, color="black")
    plt.tight_layout()
    return fig


"""
Additional functions for visualize_salience.py
-----------------------------------------------
Add these functions to the bottom of experiments/visualize_salience.py.

These functions extend the existing salience visualisation infrastructure
to support batch-dependent phis (DiversityPhi, ScoreAlignmentPhi) by
capturing intermediate particle positions during sampling and using them
as the library context when computing the salience grid.
"""

# ---------------------------------------------------------------------------
# Trajectory capture
# ---------------------------------------------------------------------------

def sample_with_trajectory_capture(
    sampler,
    x_T: Tensor,
    capture_timesteps: List[int],
    guidance_scale: float = 1.0,
    context: Optional[dict] = None,
    verbose: bool = False,
) -> tuple[Tensor, dict[int, Tensor]]:
    """
    Run sample_particle_grad_guided and capture intermediate particle
    positions at specified timesteps.

    This is a standalone wrapper around the sampler's internal logic
    rather than a method on SalientSampler, keeping sampler.py clean.

    Args:
        sampler           : SalientSampler instance
        x_T               : initial noise, shape (N, d_in)
        capture_timesteps : list of timesteps at which to save x_t
        guidance_scale    : salience guidance scale
        context           : base context dict passed to phi
        verbose           : print progress

    Returns:
        x_0        : final denoised samples, shape (N, d_in)
        snapshots  : dict mapping timestep -> particle positions (N, d_in)
                     for each t in capture_timesteps
    """
    import torch.nn.functional as F

    if context is None:
        context = {}

    x = x_T.clone().to(sampler.device)
    N = x.shape[0]
    T = sampler.scheduler.num_timesteps
    snapshots = {}

    for t in reversed(range(T)):
        if verbose and t % 100 == 0:
            print(f"  t = {t}")

        # Capture snapshot before the step
        if t in capture_timesteps:
            snapshots[t] = x.detach().clone()

        use_guidance = (
            sampler.guidance_frequency > 0 and
            (t % sampler.guidance_frequency == 0)
        )

        t_vec = torch.full((N,), t, device=sampler.device, dtype=torch.long)

        with torch.no_grad():
            eps_hat = sampler.model(x, t_vec)
            x0_hat = sampler.scheduler.reconstruct_x0(x, t_vec, eps_hat)
            mu = sampler.scheduler.q_posterior(x0_hat, x, t_vec)

        if use_guidance:
            grads = sampler._batched_particle_grad(x, t, context)
            var_t = sampler.scheduler.get_variance(t)
            mu = mu + guidance_scale * var_t * grads

        noise = torch.zeros_like(x)
        if t > 0:
            noise = torch.randn_like(x)
        var_t = sampler.scheduler.get_variance(t)
        x = mu + (var_t ** 0.5) * noise

    return x.detach(), snapshots


# ---------------------------------------------------------------------------
# Batch-aware salience grid
# ---------------------------------------------------------------------------

def compute_salience_grid_with_library(
    phi: PhiBase,
    t: int,
    base_context: dict,
    library: Tensor,
    particle_index: int = 0,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    resolution: int = 80,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate log S(x) over a 2-D meshgrid for a batch-dependent phi,
    holding the library (all other particles) fixed.

    For DiversityPhi: context["library"] is set to the other N-1 particles.
    For ScoreAlignmentPhi: context["model"] and context["scheduler"] must
    be in base_context; the library is used to compute fixed reference scores.

    Args:
        phi            : PhiBase instance (DiversityPhi or ScoreAlignmentPhi)
        t              : diffusion timestep
        base_context   : base context dict (must include "model", "scheduler"
                         for ScoreAlignmentPhi)
        library        : (N, d_in) particle snapshot at timestep t.
                         Particle at particle_index is the query; all others
                         form the library.
        particle_index : which particle to treat as the query point
        xlim           : x axis range
        ylim           : y axis range
        resolution     : grid points per axis
        device         : torch device

    Returns:
        xx      : (resolution, resolution) meshgrid x coordinates
        yy      : (resolution, resolution) meshgrid y coordinates
        log_S   : (resolution, resolution) log salience values
    """
    # Build library excluding the query particle
    N = library.shape[0]
    lib = torch.cat([
        library[:particle_index],
        library[particle_index + 1:]
    ], dim=0).to(device)

    context = {
        **base_context,
        "library": lib,
        "self_index": None,   # self already excluded above
    }

    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    xx, yy = np.meshgrid(xs, ys)

    log_S = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            x_query = torch.tensor(
                [xx[i, j], yy[i, j]], dtype=torch.float32, device=device
            ).requires_grad_(True)

            # Concatenate query point with library to form a batch
            # Query is at index 0, library fills the rest
            X_batch = torch.cat([x_query.unsqueeze(0), lib], dim=0)  # (N, d_in)

            with torch.enable_grad():
                phi_vals = phi.forward_batched(X_batch, t, context)  # (N, 1)
                phi_query = phi_vals[0].squeeze()  # scalar for query point

                grad, = torch.autograd.grad(phi_query, x_query)
                grad = grad.clamp(-100.0, 100.0)
                log_S[i, j] = (2.0 * torch.log(grad.norm() + 1e-12)).item()

    return xx, yy, log_S


# ---------------------------------------------------------------------------
# Multi-panel batch-aware salience figure
# ---------------------------------------------------------------------------

def plot_salience_landscape_batch(
    phi: PhiBase,
    snapshots: dict[int, Tensor],
    base_context: dict,
    particle_index: int = 0,
    gmm=None,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    resolution: int = 80,
    device: torch.device = torch.device("cpu"),
    suptitle: str = "Salience landscape",
    facecolor: str = "white",
) -> plt.Figure:
    """
    Multi-panel salience heatmap for a batch-dependent phi, showing the
    salience landscape from the perspective of one particle with all
    other particles overlaid as scatter points.

    Each panel corresponds to one captured timestep. The salience landscape
    is computed by holding the other N-1 particles fixed at their snapshot
    positions and evaluating log S(x) for the query particle as it moves
    across the grid.

    Args:
        phi            : PhiBase instance (DiversityPhi or ScoreAlignmentPhi)
        snapshots      : dict from sample_with_trajectory_capture,
                         mapping timestep -> (N, d_in) particle positions
        base_context   : base context dict (model, scheduler, etc.)
        particle_index : which particle to treat as the query
        gmm            : GaussianMixture for contour overlay (optional)
        xlim           : x axis range
        ylim           : y axis range
        resolution     : grid resolution per axis
        device         : torch device
        suptitle       : figure title
        facecolor      : background colour

    Returns:
        fig : matplotlib Figure
    """
    timesteps = sorted(snapshots.keys(), reverse=True)
    n = len(timesteps)

    fig, axes = plt.subplots(
        1, n, figsize=(5 * n, 5), facecolor=facecolor
    )
    if n == 1:
        axes = [axes]

    for ax, t in zip(axes, timesteps):
        ax.set_facecolor(facecolor)
        print(f"  Computing batch salience grid at t={t}...")

        library = snapshots[t].to(device)

        xx, yy, log_S = compute_salience_grid_with_library(
            phi=phi,
            t=t,
            base_context=base_context,
            library=library,
            particle_index=particle_index,
            xlim=xlim,
            ylim=ylim,
            resolution=resolution,
            device=device,
        )

        # Salience heatmap
        im = ax.pcolormesh(
            xx, yy, log_S,
            cmap="plasma",
            shading="auto",
        )
        plt.colorbar(im, ax=ax, label="log S(x)")

        # GMM contours
        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(gmm, xlim=xlim, ylim=ylim,
                            colors="white", alpha=0.6, linewidths=1.0)

        # All particles at this timestep
        pts = library.detach().cpu().numpy()
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=8, alpha=0.4, color="cyan",
            label="other particles", zorder=5,
        )

        # Highlight the query particle
        query = snapshots[t][particle_index].detach().cpu().numpy()
        ax.scatter(
            query[0], query[1],
            s=60, color="white", edgecolors="black",
            linewidths=1.5, zorder=6, label=f"particle {particle_index}",
        )

        ax.set_title(f"t = {t}", fontsize=12, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$", color="black")
        ax.set_ylabel("$x_2$", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="black")
        ax.legend(fontsize=7, facecolor="white", edgecolor="black", labelcolor="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    fig.suptitle(suptitle, fontsize=14, color="black")
    plt.tight_layout()
    return fig

"""
Trajectory plotting additions for visualize_salience.py
--------------------------------------------------------
Add these functions to the bottom of experiments/visualize_salience.py,
after the batch salience functions.

These functions visualise the actual paths that particles take through
2D sample space during the reverse diffusion process, comparing guided
vs unguided trajectories to show how salience guidance bends paths.
"""

# ---------------------------------------------------------------------------
# DDPM baseline trajectory capture (no guidance)
# ---------------------------------------------------------------------------

def sample_baseline_with_trajectory_capture(
    model,
    scheduler,
    x_T: Tensor,
    capture_timesteps: List[int],
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> tuple[Tensor, dict[int, Tensor]]:
    """
    Run standard DDPM sampling (no guidance) and capture intermediate
    particle positions at specified timesteps.

    Args:
        model             : denoising MLP
        scheduler         : NoiseScheduler
        x_T               : initial noise, shape (N, d_in)
        capture_timesteps : timesteps at which to save x_t
        device            : torch device
        verbose           : print progress

    Returns:
        x_0       : final denoised samples, shape (N, d_in)
        snapshots : dict mapping timestep -> (N, d_in) particle positions
    """
    x = x_T.clone().to(device)
    N = x.shape[0]
    T = scheduler.num_timesteps
    snapshots = {}

    for t in reversed(range(T)):
        if verbose and t % 100 == 0:
            print(f"  t = {t}")

        if t in capture_timesteps:
            snapshots[t] = x.detach().clone()

        t_vec = torch.full((N,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            eps_hat = model(x, t_vec)
            x0_hat = scheduler.reconstruct_x0(x, t_vec, eps_hat)
            mu = scheduler.q_posterior(x0_hat, x, t_vec)

        noise = torch.zeros_like(x)
        if t > 0:
            noise = torch.randn_like(x)
        var_t = scheduler.get_variance(t)
        x = mu + (var_t ** 0.5) * noise

    return x.detach(), snapshots


# ---------------------------------------------------------------------------
# Trajectory comparison plot
# ---------------------------------------------------------------------------

def plot_trajectory_comparison(
    snapshots_baseline: dict[int, Tensor],
    snapshots_guided: dict[int, Tensor],
    particle_indices: List[int],
    gmm=None,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    suptitle: str = "Sampling trajectories: baseline vs guided",
    baseline_label: str = "DDPM baseline",
    guided_label: str = "Guided",
    facecolor: str = "white",
) -> plt.Figure:
    """
    Plot individual particle trajectories through 2D sample space,
    comparing DDPM baseline against salience-guided sampling.

    Each selected particle is shown as a line from x_T to x_0, with
    dots at each captured timestep. Baseline trajectories are shown
    in blue, guided in orange. GMM contours are overlaid.

    Args:
        snapshots_baseline : dict from sample_baseline_with_trajectory_capture
        snapshots_guided   : dict from sample_with_trajectory_capture
        particle_indices   : which particles to plot trajectories for
        gmm                : GaussianMixture for contour overlay (optional)
        xlim               : x axis range
        ylim               : y axis range
        suptitle           : figure title
        baseline_label     : legend label for baseline trajectories
        guided_label       : legend label for guided trajectories
        facecolor          : background colour

    Returns:
        fig : matplotlib Figure
    """
    # Sort timesteps high to low (T -> 0) so lines go noise -> data
    timesteps = sorted(snapshots_baseline.keys(), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=facecolor)

    titles = [baseline_label, guided_label]
    snapshot_pairs = [snapshots_baseline, snapshots_guided]
    colors = ["steelblue", "darkorange"]

    for ax, title, snapshots, color in zip(
        axes, titles, snapshot_pairs, colors
    ):
        ax.set_facecolor(facecolor)

        # GMM contours
        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(
                gmm, xlim=xlim, ylim=ylim,
                colors="gray", alpha=0.3, linewidths=0.8
            )

        # Plot trajectory for each selected particle
        for p_idx in particle_indices:
            xs = [snapshots[t][p_idx, 0].item() for t in timesteps]
            ys = [snapshots[t][p_idx, 1].item() for t in timesteps]

            # Line
            ax.plot(
                xs, ys,
                color=color, alpha=0.6, linewidth=1.0, zorder=3,
            )

            # Dots at each captured timestep, darker toward x_0
            n_steps = len(timesteps)
            for k, (x_pos, y_pos) in enumerate(zip(xs, ys)):
                alpha = 0.3 + 0.7 * (k / max(n_steps - 1, 1))
                ax.scatter(
                    x_pos, y_pos,
                    s=12, color=color, alpha=alpha, zorder=4,
                )

            # Mark start (x_T) and end (x_0)
            ax.scatter(
                xs[0], ys[0],
                s=50, color="white", edgecolors=color,
                linewidths=1.5, zorder=5, marker="o",
            )
            ax.scatter(
                xs[-1], ys[-1],
                s=50, color=color, edgecolors="black",
                linewidths=1.0, zorder=5, marker="*",
            )

        ax.set_title(title, fontsize=12, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$", color="black")
        ax.set_ylabel("$x_2$", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", linewidth=1.5,
               label=baseline_label),
        Line2D([0], [0], color="darkorange", linewidth=1.5,
               label=guided_label),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=6, label="$x_T$ (start)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
               markersize=8, label="$x_0$ (end)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=4,
        fontsize=9, facecolor="white", edgecolor="black",
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(suptitle, fontsize=14, color="black")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combined: salience landscape + trajectory overlay (single panel per t)
# ---------------------------------------------------------------------------

def plot_salience_with_trajectories(
    phi: PhiBase,
    snapshots_baseline: dict[int, Tensor],
    snapshots_guided: dict[int, Tensor],
    base_context: dict,
    particle_index: int = 0,
    gmm=None,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    resolution: int = 80,
    device: torch.device = torch.device("cpu"),
    suptitle: str = "Salience landscape with trajectories",
    facecolor: str = "white",
) -> plt.Figure:
    """
    Combined figure: salience heatmap at each captured timestep with
    baseline and guided trajectories overlaid.

    For each captured timestep, shows:
        - log S(x) heatmap (computed from guided snapshot library)
        - All guided particles as cyan dots
        - Trajectory of selected particle up to this timestep:
            baseline in blue, guided in orange

    Args:
        phi                : PhiBase instance
        snapshots_baseline : baseline trajectory snapshots
        snapshots_guided   : guided trajectory snapshots
        base_context       : context dict for phi (model, scheduler, etc.)
        particle_index     : which particle to highlight
        gmm                : GaussianMixture for contour overlay
        xlim               : x axis range
        ylim               : y axis range
        resolution         : salience grid resolution
        device             : torch device
        suptitle           : figure title
        facecolor          : background colour

    Returns:
        fig : matplotlib Figure
    """
    timesteps = sorted(snapshots_guided.keys(), reverse=True)
    n = len(timesteps)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), facecolor=facecolor)
    if n == 1:
        axes = [axes]

    for ax, t in zip(axes, timesteps):
        ax.set_facecolor(facecolor)
        print(f"  Computing combined figure at t={t}...")

        # Salience heatmap
        library = snapshots_guided[t].to(device)
        xx, yy, log_S = compute_salience_grid_with_library(
            phi=phi,
            t=t,
            base_context=base_context,
            library=library,
            particle_index=particle_index,
            xlim=xlim,
            ylim=ylim,
            resolution=resolution,
            device=device,
        )

        im = ax.pcolormesh(
            xx, yy, log_S,
            cmap="plasma", shading="auto", alpha=0.8,
        )
        plt.colorbar(im, ax=ax, label="log S(x)")

        # GMM contours
        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(
                gmm, xlim=xlim, ylim=ylim,
                colors="white", alpha=0.4, linewidths=0.8
            )

        # All guided particles at this timestep
        pts = library.detach().cpu().numpy()
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=6, alpha=0.3, color="cyan", zorder=4,
        )

        # Trajectory of query particle up to this timestep
        past_timesteps = [s for s in timesteps if s >= t]

        # Baseline path
        if snapshots_baseline:
            bxs = [snapshots_baseline[s][particle_index, 0].item()
                   for s in past_timesteps if s in snapshots_baseline]
            bys = [snapshots_baseline[s][particle_index, 1].item()
                   for s in past_timesteps if s in snapshots_baseline]
            ax.plot(bxs, bys, color="steelblue", alpha=0.8,
                    linewidth=1.5, zorder=5, label="baseline")
            if bxs:
                ax.scatter(bxs[-1], bys[-1], s=40, color="steelblue",
                           edgecolors="white", linewidths=1.0, zorder=6)

        # Guided path
        gxs = [snapshots_guided[s][particle_index, 0].item()
               for s in past_timesteps if s in snapshots_guided]
        gys = [snapshots_guided[s][particle_index, 1].item()
               for s in past_timesteps if s in snapshots_guided]
        ax.plot(gxs, gys, color="darkorange", alpha=0.8,
                linewidth=1.5, zorder=5, label="guided")
        if gxs:
            ax.scatter(gxs[-1], gys[-1], s=40, color="darkorange",
                       edgecolors="white", linewidths=1.0, zorder=6)

        ax.set_title(f"t = {t}", fontsize=12, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$", color="black")
        ax.set_ylabel("$x_2$", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="black")
        ax.legend(fontsize=7, facecolor="white", edgecolor="black", labelcolor="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    fig.suptitle(suptitle, fontsize=14, color="black")
    plt.tight_layout()
    return fig

"""
Vector field plotting additions for visualize_salience.py
----------------------------------------------------------
Add these functions to the bottom of experiments/visualize_salience.py.

These functions compute and plot the salience gradient vector field
grad_x log S(x) over a 2D grid, optionally overlaid with the salience
heatmap. This makes the guidance mechanism directly visible: arrows show
the direction the salience gradient pushes samples at each point in space.
"""

# ---------------------------------------------------------------------------
# Salience gradient grid computation
# ---------------------------------------------------------------------------

def compute_salience_gradient_grid(
    phi: PhiBase,
    t: int,
    context: dict,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    resolution: int = 20,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate log S(x) and grad_x log S(x) over a 2D meshgrid.

    For batch-independent phis (ScoreNormPhi): uses log_salience directly.
    For batch-dependent phis (DiversityPhi, ScoreAlignmentPhi): requires
    context["library"] to be set before calling.

    Args:
        phi        : PhiBase instance
        t          : diffusion timestep
        context    : passed through to phi
        xlim       : x axis range
        ylim       : y axis range
        resolution : grid points per axis (keep low, e.g. 20, for arrows)
        device     : torch device

    Returns:
        xx      : (resolution, resolution) meshgrid x coordinates
        yy      : (resolution, resolution) meshgrid y coordinates
        log_S   : (resolution, resolution) log salience values
        gx      : (resolution, resolution) x component of grad log S
        gy      : (resolution, resolution) y component of grad log S
    """
    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    xx, yy = np.meshgrid(xs, ys)

    log_S = np.zeros((resolution, resolution))
    gx = np.zeros((resolution, resolution))
    gy = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            x = torch.tensor(
                [xx[i, j], yy[i, j]], dtype=torch.float32, device=device
            ).requires_grad_(True)

            with torch.enable_grad():
                # For batch-dependent phis, context must already contain
                # the library. For ScoreAlignmentPhi, use forward_batched
                # path via the library concatenation approach.
                lib = context.get("library", None)

                if lib is not None:
                    # Batch-dependent phi: concatenate query with library
                    X_batch = torch.cat([x.unsqueeze(0), lib], dim=0)
                    phi_vals = phi.forward_batched(X_batch, t, context)
                    phi_query = phi_vals[0].squeeze()
                else:
                    # Batch-independent phi: call forward directly
                    phi_query = phi(x, t, context).squeeze()

                # First derivative: grad_x phi
                grad_phi, = torch.autograd.grad(
                    phi_query, x, create_graph=True
                )
                grad_phi = grad_phi.clamp(-100.0, 100.0)

                # log S = 2 * log ||grad_phi||
                log_S_val = 2.0 * torch.log(grad_phi.norm() + 1e-12)
                log_S[i, j] = log_S_val.item()

                # Second derivative: grad_x log S
                grad_log_S, = torch.autograd.grad(log_S_val, x)
                grad_log_S = grad_log_S.clamp(-100.0, 100.0)
                gx[i, j] = grad_log_S[0].item()
                gy[i, j] = grad_log_S[1].item()

    return xx, yy, log_S, gx, gy


def compute_salience_gradient_grid_batch(
    phi: PhiBase,
    t: int,
    base_context: dict,
    library: Tensor,
    particle_index: int = 0,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    resolution: int = 20,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate log S(x) and grad_x log S(x) for a batch-dependent phi,
    holding the library (all other particles) fixed.

    Args:
        phi            : PhiBase instance (DiversityPhi or ScoreAlignmentPhi)
        t              : diffusion timestep
        base_context   : base context dict (model, scheduler, etc.)
        library        : (N, d_in) particle snapshot at timestep t
        particle_index : which particle to treat as the query
        xlim           : x axis range
        ylim           : y axis range
        resolution     : grid points per axis
        device         : torch device

    Returns:
        xx, yy  : meshgrid coordinates
        log_S   : (resolution, resolution) log salience values
        gx, gy  : (resolution, resolution) gradient vector components
    """
    N = library.shape[0]
    lib = torch.cat([
        library[:particle_index],
        library[particle_index + 1:]
    ], dim=0).to(device)

    context = {
        **base_context,
        "library": lib,
    }

    return compute_salience_gradient_grid(
        phi=phi,
        t=t,
        context=context,
        xlim=xlim,
        ylim=ylim,
        resolution=resolution,
        device=device,
    )


# ---------------------------------------------------------------------------
# Combined heatmap + vector field figure
# ---------------------------------------------------------------------------

def plot_salience_vector_field(
    phi: PhiBase,
    timesteps: List[int],
    context: dict,
    library_snapshots: Optional[dict] = None,
    particle_index: int = 0,
    gmm=None,
    samples: Optional[Tensor] = None,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    heatmap_resolution: int = 80,
    arrow_resolution: int = 20,
    device: torch.device = torch.device("cpu"),
    suptitle: str = "Salience gradient vector field",
    facecolor: str = "white",
    arrow_scale: float = None,
) -> plt.Figure:
    """
    Multi-panel figure combining salience heatmap with gradient vector field.

    Each panel shows:
        - log S(x) as a filled heatmap (fine resolution)
        - grad_x log S(x) as arrows (coarse resolution)
        - GMM PDF contours overlaid (if gmm provided)
        - Generated samples scattered on top (if samples provided)

    For batch-dependent phis, pass library_snapshots (dict mapping
    timestep -> particle positions). The library at each timestep is
    used to compute the salience landscape for the query particle.

    For batch-independent phis (ScoreNormPhi), library_snapshots can
    be None.

    Args:
        phi               : PhiBase instance
        timesteps         : list of timesteps to visualise
        context           : base context dict (model, scheduler, etc.)
        library_snapshots : optional dict mapping timestep -> (N, d_in)
                            particle positions (for batch-dependent phis)
        particle_index    : query particle index (for batch-dependent phis)
        gmm               : GaussianMixture for contour overlay
        samples           : (N, 2) final samples to scatter (optional)
        xlim              : x axis range
        ylim              : y axis range
        heatmap_resolution: grid resolution for heatmap
        arrow_resolution  : grid resolution for arrows (keep low, ~15-25)
        device            : torch device
        suptitle          : figure title
        facecolor         : background colour
        arrow_scale       : quiver scale parameter (None = matplotlib default)

    Returns:
        fig : matplotlib Figure
    """


    n = len(timesteps)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor=facecolor)
    axes = axes.flatten()

    for ax, t in zip(axes, timesteps):
        ax.set_facecolor(facecolor)
        print(f"  Computing vector field at t={t}...")

        # Build library context for batch-dependent phis
        if library_snapshots is not None and t in library_snapshots:
            library = library_snapshots[t].to(device)
            lib = torch.cat([
                library[:particle_index],
                library[particle_index + 1:]
            ], dim=0)
            ctx = {**context, "library": lib}
        else:
            ctx = context

        # --- Heatmap (fine resolution) ---
        xs_fine = np.linspace(*xlim, heatmap_resolution)
        ys_fine = np.linspace(*ylim, heatmap_resolution)
        xx_fine, yy_fine = np.meshgrid(xs_fine, ys_fine)
        log_S_fine = np.zeros((heatmap_resolution, heatmap_resolution))

        for i in range(heatmap_resolution):
            for j in range(heatmap_resolution):
                x = torch.tensor(
                    [xx_fine[i, j], yy_fine[i, j]],
                    dtype=torch.float32, device=device
                ).requires_grad_(True)

                with torch.enable_grad():
                    lib_ctx = ctx.get("library", None)
                    if lib_ctx is not None:
                        X_batch = torch.cat([x.unsqueeze(0), lib_ctx], dim=0)
                        phi_vals = phi.forward_batched(X_batch, t, ctx)
                        phi_query = phi_vals[0].squeeze()
                    else:
                        phi_query = phi(x, t, ctx).squeeze()

                    grad_phi, = torch.autograd.grad(
                        phi_query, x, create_graph=False
                    )
                    grad_phi = grad_phi.clamp(-100.0, 100.0)
                    log_S_fine[i, j] = (
                        2.0 * torch.log(grad_phi.norm() + 1e-12)
                    ).item()

        im = ax.pcolormesh(
            xx_fine, yy_fine, log_S_fine,
            cmap="plasma", shading="auto", alpha=0.85,
        )
        cbar = plt.colorbar(im, ax=ax, label="log S(x)")
        cbar.ax.yaxis.label.set_color("black")
        cbar.ax.tick_params(colors="black")

        # --- Vector field (coarse resolution) ---
        xs_coarse = np.linspace(*xlim, arrow_resolution)
        ys_coarse = np.linspace(*ylim, arrow_resolution)
        xx_c, yy_c = np.meshgrid(xs_coarse, ys_coarse)
        gx = np.zeros((arrow_resolution, arrow_resolution))
        gy = np.zeros((arrow_resolution, arrow_resolution))

        for i in range(arrow_resolution):
            for j in range(arrow_resolution):
                x = torch.tensor(
                    [xx_c[i, j], yy_c[i, j]],
                    dtype=torch.float32, device=device
                ).requires_grad_(True)

                with torch.enable_grad():
                    lib_ctx = ctx.get("library", None)
                    if lib_ctx is not None:
                        X_batch = torch.cat([x.unsqueeze(0), lib_ctx], dim=0)
                        phi_vals = phi.forward_batched(X_batch, t, ctx)
                        phi_query = phi_vals[0].squeeze()
                    else:
                        phi_query = phi(x, t, ctx).squeeze()

                    grad_phi, = torch.autograd.grad(
                        phi_query, x, create_graph=True
                    )
                    grad_phi = grad_phi.clamp(-100.0, 100.0)
                    log_S_val = 2.0 * torch.log(grad_phi.norm() + 1e-12)

                    grad_log_S, = torch.autograd.grad(log_S_val, x)
                    grad_log_S = grad_log_S.clamp(-100.0, 100.0).detach()
                    gx[i, j] = grad_log_S[0].item()
                    gy[i, j] = grad_log_S[1].item()

        # Normalise arrows to unit length for clean visualisation
        magnitude = np.sqrt(gx ** 2 + gy ** 2) + 1e-12
        gx_norm = gx / magnitude
        gy_norm = gy / magnitude

        ax.quiver(
            xx_c, yy_c, gx_norm, gy_norm,
            color="white", alpha=0.7,
            scale=arrow_scale if arrow_scale is not None else arrow_resolution * 1.5,
            width=0.003,
            headwidth=4, headlength=5,
            zorder=5,
        )

        # GMM contours
        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(
                gmm, xlim=xlim, ylim=ylim,
                colors="white", alpha=0.4, linewidths=0.8,
            )

        # Particle scatter
        if samples is not None:
            pts = samples.detach().cpu().numpy()
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=8, alpha=0.4, color="cyan", zorder=6,
            )

        # Library particles for batch-dependent phis
        if library_snapshots is not None and t in library_snapshots:
            lib_pts = library_snapshots[t].detach().cpu().numpy()
            ax.scatter(
                lib_pts[:, 0], lib_pts[:, 1],
                s=6, alpha=0.3, color="cyan", zorder=5,
            )

        ax.set_title(f"t = {t}", fontsize=12, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$", color="black")
        ax.set_ylabel("$x_2$", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    fig.suptitle(suptitle, fontsize=14, color="black")
    plt.tight_layout()
    return fig


"""
Conditional vector field plotting for visualize_salience.py
------------------------------------------------------------
Add this function to the bottom of experiments/visualize_salience.py.

Shows salience gradient and classifier gradient fields side by side
across multiple timesteps, for a single target class.
"""

# ---------------------------------------------------------------------------
# Classifier gradient grid computation
# ---------------------------------------------------------------------------

def compute_classifier_gradient_grid(
    classifier: torch.nn.Module,
    target_class: int,
    t: int,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    resolution: int = 80,
    arrow_resolution: int = 20,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate log p(y | x, t) and grad_x log p(y | x, t) over a 2D meshgrid.

    Args:
        classifier     : time-conditioned noisy classifier
        target_class   : class label to compute gradient toward
        t              : diffusion timestep
        xlim           : x axis range
        ylim           : y axis range
        resolution     : grid resolution for heatmap
        arrow_resolution: grid resolution for arrows
        device         : torch device

    Returns:
        xx_fine, yy_fine : heatmap meshgrid coordinates
        log_p_fine       : (resolution, resolution) log prob values
        xx_c, yy_c       : arrow meshgrid coordinates
        gx, gy           : (arrow_resolution, arrow_resolution) gradient components
    """
    import torch.nn.functional as F

    # Heatmap
    xs_fine = np.linspace(*xlim, resolution)
    ys_fine = np.linspace(*ylim, resolution)
    xx_fine, yy_fine = np.meshgrid(xs_fine, ys_fine)
    log_p_fine = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            x = torch.tensor(
                [xx_fine[i, j], yy_fine[i, j]],
                dtype=torch.float32, device=device
            )
            t_vec = torch.full((1,), t, device=device, dtype=torch.long)
            with torch.no_grad():
                logits = classifier(x.unsqueeze(0), t_vec)
                log_p = F.log_softmax(logits, dim=-1)[0, target_class]
            log_p_fine[i, j] = log_p.item()

    # Arrow field
    xs_c = np.linspace(*xlim, arrow_resolution)
    ys_c = np.linspace(*ylim, arrow_resolution)
    xx_c, yy_c = np.meshgrid(xs_c, ys_c)
    gx = np.zeros((arrow_resolution, arrow_resolution))
    gy = np.zeros((arrow_resolution, arrow_resolution))

    for i in range(arrow_resolution):
        for j in range(arrow_resolution):
            x = torch.tensor(
                [xx_c[i, j], yy_c[i, j]],
                dtype=torch.float32, device=device
            ).requires_grad_(True)
            t_vec = torch.full((1,), t, device=device, dtype=torch.long)

            with torch.enable_grad():
                logits = classifier(x.unsqueeze(0), t_vec)
                log_p = F.log_softmax(logits, dim=-1)[0, target_class]
                grad, = torch.autograd.grad(log_p, x)
                grad = grad.clamp(-100.0, 100.0).detach()
                gx[i, j] = grad[0].item()
                gy[i, j] = grad[1].item()

    return xx_fine, yy_fine, log_p_fine, xx_c, yy_c, gx, gy


# ---------------------------------------------------------------------------
# Conditional vector field figure: salience + classifier, two rows
# ---------------------------------------------------------------------------

def plot_conditional_vector_fields(
    phi: PhiBase,
    classifier: torch.nn.Module,
    target_class: int,
    timesteps: List[int],
    context: dict,
    library_snapshots: Optional[dict] = None,
    particle_index: int = 0,
    gmm=None,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    heatmap_resolution: int = 60,
    arrow_resolution: int = 15,
    device: torch.device = torch.device("cpu"),
    suptitle: str = "Conditional guidance vector fields",
    facecolor: str = "white",
) -> plt.Figure:
    """
    Two-row figure showing salience and classifier gradient fields
    at multiple timesteps for a single target class.

    Row 1: salience gradient field (log S heatmap + grad log S arrows)
    Row 2: classifier gradient field (log p(y|x,t) heatmap + grad arrows)

    Each column is one timestep.

    Args:
        phi              : PhiBase instance
        classifier       : time-conditioned noisy classifier
        target_class     : class label to guide toward
        timesteps        : list of timesteps to visualise
        context          : base context dict (model, scheduler, etc.)
        library_snapshots: optional dict mapping timestep -> (N, d_in)
                           particle positions (for batch-dependent phis)
        particle_index   : query particle index (for batch-dependent phis)
        gmm              : GaussianMixture for contour overlay
        xlim             : x axis range
        ylim             : y axis range
        heatmap_resolution: grid resolution for heatmaps
        arrow_resolution : grid resolution for arrows
        device           : torch device
        suptitle         : figure title
        facecolor        : background colour

    Returns:
        fig : matplotlib Figure
    """
    import torch.nn.functional as F

    n = len(timesteps)
    fig, axes = plt.subplots(
        2, n,
        figsize=(5 * n, 10),
        facecolor=facecolor,
    )

    row_labels = ["Salience gradient", f"Classifier gradient (class {target_class})"]

    for col, t in enumerate(timesteps):
        print(f"  Computing fields at t={t}...")

        # Build library context for batch-dependent phis
        if library_snapshots is not None and t in library_snapshots:
            library = library_snapshots[t].to(device)
            lib = torch.cat([
                library[:particle_index],
                library[particle_index + 1:]
            ], dim=0)
            ctx = {**context, "library": lib}
        else:
            ctx = context

        # ----------------------------------------------------------------
        # Row 0: Salience gradient field
        # ----------------------------------------------------------------
        ax = axes[0, col]
        ax.set_facecolor(facecolor)

        # Heatmap
        xs_fine = np.linspace(*xlim, heatmap_resolution)
        ys_fine = np.linspace(*ylim, heatmap_resolution)
        xx_fine, yy_fine = np.meshgrid(xs_fine, ys_fine)
        log_S_fine = np.zeros((heatmap_resolution, heatmap_resolution))

        for i in range(heatmap_resolution):
            for j in range(heatmap_resolution):
                x = torch.tensor(
                    [xx_fine[i, j], yy_fine[i, j]],
                    dtype=torch.float32, device=device
                ).requires_grad_(True)

                with torch.enable_grad():
                    lib_ctx = ctx.get("library", None)
                    if lib_ctx is not None:
                        X_batch = torch.cat([x.unsqueeze(0), lib_ctx], dim=0)
                        phi_vals = phi.forward_batched(X_batch, t, ctx)
                        phi_query = phi_vals[0].squeeze()
                    else:
                        phi_query = phi(x, t, ctx).squeeze()

                    grad_phi, = torch.autograd.grad(
                        phi_query, x, create_graph=False
                    )
                    grad_phi = grad_phi.clamp(-100.0, 100.0)
                    log_S_fine[i, j] = (
                        2.0 * torch.log(grad_phi.norm() + 1e-12)
                    ).item()

        im0 = ax.pcolormesh(
            xx_fine, yy_fine, log_S_fine,
            cmap="plasma", shading="auto", alpha=0.85,
        )
        cbar0 = plt.colorbar(im0, ax=ax, label="log S(x)")
        cbar0.ax.yaxis.label.set_color("black")
        cbar0.ax.tick_params(colors="black")

        # Arrows
        xs_c = np.linspace(*xlim, arrow_resolution)
        ys_c = np.linspace(*ylim, arrow_resolution)
        xx_c, yy_c = np.meshgrid(xs_c, ys_c)
        gx_s = np.zeros((arrow_resolution, arrow_resolution))
        gy_s = np.zeros((arrow_resolution, arrow_resolution))

        for i in range(arrow_resolution):
            for j in range(arrow_resolution):
                x = torch.tensor(
                    [xx_c[i, j], yy_c[i, j]],
                    dtype=torch.float32, device=device
                ).requires_grad_(True)

                with torch.enable_grad():
                    lib_ctx = ctx.get("library", None)
                    if lib_ctx is not None:
                        X_batch = torch.cat([x.unsqueeze(0), lib_ctx], dim=0)
                        phi_vals = phi.forward_batched(X_batch, t, ctx)
                        phi_query = phi_vals[0].squeeze()
                    else:
                        phi_query = phi(x, t, ctx).squeeze()

                    grad_phi, = torch.autograd.grad(
                        phi_query, x, create_graph=True
                    )
                    grad_phi = grad_phi.clamp(-100.0, 100.0)
                    log_S_val = 2.0 * torch.log(grad_phi.norm() + 1e-12)
                    grad_log_S, = torch.autograd.grad(log_S_val, x)
                    grad_log_S = grad_log_S.clamp(-100.0, 100.0).detach()
                    gx_s[i, j] = grad_log_S[0].item()
                    gy_s[i, j] = grad_log_S[1].item()

        mag_s = np.sqrt(gx_s ** 2 + gy_s ** 2) + 1e-12
        ax.quiver(
            xx_c, yy_c, gx_s / mag_s, gy_s / mag_s,
            color="white", alpha=0.7,
            scale=arrow_resolution * 1.5,
            width=0.003, headwidth=4, headlength=5,
            zorder=5,
        )

        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(gmm, xlim=xlim, ylim=ylim,
                            colors="white", alpha=0.4, linewidths=0.8)

        ax.set_title(f"t = {t}", fontsize=12, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$", color="black")
        ax.set_ylabel("$x_2$", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

        if col == 0:
            ax.set_ylabel(row_labels[0], fontsize=11, color="black")

        # ----------------------------------------------------------------
        # Row 1: Classifier gradient field
        # ----------------------------------------------------------------
        ax = axes[1, col]
        ax.set_facecolor(facecolor)

        # Heatmap
        log_p_fine = np.zeros((heatmap_resolution, heatmap_resolution))
        for i in range(heatmap_resolution):
            for j in range(heatmap_resolution):
                x = torch.tensor(
                    [xx_fine[i, j], yy_fine[i, j]],
                    dtype=torch.float32, device=device
                )
                t_vec = torch.full((1,), t, device=device, dtype=torch.long)
                with torch.no_grad():
                    logits = classifier(x.unsqueeze(0), t_vec)
                    log_p = F.log_softmax(logits, dim=-1)[0, target_class]
                log_p_fine[i, j] = log_p.item()

        im1 = ax.pcolormesh(
            xx_fine, yy_fine, log_p_fine,
            cmap="viridis", shading="auto", alpha=0.85,
        )
        cbar1 = plt.colorbar(im1, ax=ax, label=f"log p(y={target_class} | x, t)")
        cbar1.ax.yaxis.label.set_color("black")
        cbar1.ax.tick_params(colors="black")

        # Arrows
        gx_c = np.zeros((arrow_resolution, arrow_resolution))
        gy_c = np.zeros((arrow_resolution, arrow_resolution))

        for i in range(arrow_resolution):
            for j in range(arrow_resolution):
                x = torch.tensor(
                    [xx_c[i, j], yy_c[i, j]],
                    dtype=torch.float32, device=device
                ).requires_grad_(True)
                t_vec = torch.full((1,), t, device=device, dtype=torch.long)

                with torch.enable_grad():
                    logits = classifier(x.unsqueeze(0), t_vec)
                    log_p = F.log_softmax(logits, dim=-1)[0, target_class]
                    grad, = torch.autograd.grad(log_p, x)
                    grad = grad.clamp(-100.0, 100.0).detach()
                    gx_c[i, j] = grad[0].item()
                    gy_c[i, j] = grad[1].item()

        mag_c = np.sqrt(gx_c ** 2 + gy_c ** 2) + 1e-12
        ax.quiver(
            xx_c, yy_c, gx_c / mag_c, gy_c / mag_c,
            color="white", alpha=0.7,
            scale=arrow_resolution * 1.5,
            width=0.003, headwidth=4, headlength=5,
            zorder=5,
        )

        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(gmm, xlim=xlim, ylim=ylim,
                            colors="white", alpha=0.4, linewidths=0.8)

        ax.set_title(f"t = {t}", fontsize=12, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$", color="black")
        ax.set_ylabel("$x_2$", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

        if col == 0:
            ax.set_ylabel(row_labels[1], fontsize=11, color="black")

    fig.suptitle(suptitle, fontsize=14, color="black")
    plt.tight_layout()
    return fig

"""
Simplified vector field plotting for visualize_salience.py
-----------------------------------------------------------
Add this function to the bottom of experiments/visualize_salience.py.

Shows salience and classifier gradient vector fields as clean arrow
plots on GMM contours, without the salience heatmap overlay.
"""

def plot_conditional_vector_fields_clean(
    phi: PhiBase,
    classifier: torch.nn.Module,
    target_class: int,
    timesteps: List[int],
    context: dict,
    library_snapshots: Optional[dict] = None,
    particle_index: int = 0,
    gmm=None,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    arrow_resolution: int = 15,
    device: torch.device = torch.device("cpu"),
    suptitle: str = "Conditional guidance vector fields",
    facecolor: str = "white",
) -> plt.Figure:
    """
    Two-row figure showing salience and classifier gradient vector fields
    as clean arrow plots at multiple timesteps for a single target class.

    Row 1: grad_x log S(x) arrows on GMM contours
    Row 2: grad_x log p(y|x,t) arrows on GMM contours

    Args:
        phi              : PhiBase instance
        classifier       : time-conditioned noisy classifier
        target_class     : class label to guide toward
        timesteps        : list of timesteps to visualise
        context          : base context dict (model, scheduler, etc.)
        library_snapshots: optional dict mapping timestep -> (N, d_in)
                           particle positions (for batch-dependent phis)
        particle_index   : query particle index (for batch-dependent phis)
        gmm              : GaussianMixture for contour overlay
        xlim             : x axis range
        ylim             : y axis range
        arrow_resolution : grid resolution for arrows
        device           : torch device
        suptitle         : figure title
        facecolor        : background colour

    Returns:
        fig : matplotlib Figure
    """
    import torch.nn.functional as F

    n = len(timesteps)
    fig, axes = plt.subplots(
        2, n,
        figsize=(5 * n, 10),
        facecolor=facecolor,
    )

    row_labels = [
        "Salience gradient $\\nabla_x \\log \\mathcal{S}(x)$",
        f"Classifier gradient $\\nabla_x \\log p(y={target_class} | x, t)$",
    ]

    xs_c = np.linspace(*xlim, arrow_resolution)
    ys_c = np.linspace(*ylim, arrow_resolution)
    xx_c, yy_c = np.meshgrid(xs_c, ys_c)

    for col, t in enumerate(timesteps):
        print(f"  Computing vector fields at t={t}...")

        # Build library context for batch-dependent phis
        if library_snapshots is not None and t in library_snapshots:
            library = library_snapshots[t].to(device)
            lib = torch.cat([
                library[:particle_index],
                library[particle_index + 1:]
            ], dim=0)
            ctx = {**context, "library": lib}
        else:
            ctx = context

        # ------------------------------------------------------------
        # Row 0: Salience gradient arrows
        # ------------------------------------------------------------
        ax = axes[0, col]
        ax.set_facecolor(facecolor)

        gx_s = np.zeros((arrow_resolution, arrow_resolution))
        gy_s = np.zeros((arrow_resolution, arrow_resolution))

        for i in range(arrow_resolution):
            for j in range(arrow_resolution):
                x = torch.tensor(
                    [xx_c[i, j], yy_c[i, j]],
                    dtype=torch.float32, device=device
                ).requires_grad_(True)

                with torch.enable_grad():
                    lib_val = ctx.get("library", None)
                    if lib_val is not None:
                        X_batch = torch.cat([x.unsqueeze(0), lib_val], dim=0)
                        phi_vals = phi.forward_batched(X_batch, t, ctx)
                        phi_query = phi_vals[0].squeeze()
                    else:
                        phi_query = phi(x, t, ctx).squeeze()

                    grad_phi, = torch.autograd.grad(
                        phi_query, x, create_graph=True
                    )
                    grad_phi = grad_phi.clamp(-100.0, 100.0)
                    log_S_val = 2.0 * torch.log(grad_phi.norm() + 1e-12)
                    grad_log_S, = torch.autograd.grad(log_S_val, x)
                    grad_log_S = grad_log_S.clamp(-100.0, 100.0).detach()
                    gx_s[i, j] = grad_log_S[0].item()
                    gy_s[i, j] = grad_log_S[1].item()

        mag_s = np.sqrt(gx_s ** 2 + gy_s ** 2) + 1e-12

        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(gmm, xlim=xlim, ylim=ylim,
                            colors="gray", alpha=0.4, linewidths=0.8)

        ax.quiver(
            xx_c, yy_c, gx_s / mag_s, gy_s / mag_s,
            color="darkorange", alpha=0.8,
            scale=arrow_resolution * 1.5,
            width=0.003, headwidth=4, headlength=5,
            zorder=5,
        )

        ax.set_title(f"t = {t}", fontsize=12, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

        if col == 0:
            ax.set_ylabel(row_labels[0], fontsize=10, color="black")
        else:
            ax.set_ylabel("")

        # ------------------------------------------------------------
        # Row 1: Classifier gradient arrows
        # ------------------------------------------------------------
        ax = axes[1, col]
        ax.set_facecolor(facecolor)

        gx_c = np.zeros((arrow_resolution, arrow_resolution))
        gy_c = np.zeros((arrow_resolution, arrow_resolution))

        for i in range(arrow_resolution):
            for j in range(arrow_resolution):
                x = torch.tensor(
                    [xx_c[i, j], yy_c[i, j]],
                    dtype=torch.float32, device=device
                ).requires_grad_(True)
                t_vec = torch.full((1,), t, device=device, dtype=torch.long)

                with torch.enable_grad():
                    logits = classifier(x.unsqueeze(0), t_vec)
                    log_p = F.log_softmax(logits, dim=-1)[0, target_class]
                    grad, = torch.autograd.grad(log_p, x)
                    grad = grad.clamp(-100.0, 100.0).detach()
                    gx_c[i, j] = grad[0].item()
                    gy_c[i, j] = grad[1].item()

        mag_c = np.sqrt(gx_c ** 2 + gy_c ** 2) + 1e-12

        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(gmm, xlim=xlim, ylim=ylim,
                            colors="gray", alpha=0.4, linewidths=0.8)

        ax.quiver(
            xx_c, yy_c, gx_c / mag_c, gy_c / mag_c,
            color="steelblue", alpha=0.8,
            scale=arrow_resolution * 1.5,
            width=0.003, headwidth=4, headlength=5,
            zorder=5,
        )

        ax.set_title(f"t = {t}", fontsize=12, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

        if col == 0:
            ax.set_ylabel(row_labels[1], fontsize=10, color="black")
        else:
            ax.set_ylabel("")

    fig.suptitle(suptitle, fontsize=14, color="black")
    plt.tight_layout()
    return fig

def plot_salience_vector_field_clean(
    phi: PhiBase,
    timesteps: List[int],
    context: dict,
    library_snapshots: Optional[dict] = None,
    particle_index: int = 0,
    gmm=None,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    arrow_resolution: int = 15,
    device: torch.device = torch.device("cpu"),
    suptitle: str = "Salience gradient vector field",
    facecolor: str = "white",
) -> plt.Figure:
    """
    Single-row figure showing salience gradient vector field as clean
    arrow plots on GMM contours at multiple timesteps.

    Args:
        phi              : PhiBase instance
        timesteps        : list of timesteps to visualise
        context          : base context dict (model, scheduler, etc.)
        library_snapshots: optional dict mapping timestep -> (N, d_in)
                           particle positions (for batch-dependent phis)
        particle_index   : query particle index (for batch-dependent phis)
        gmm              : GaussianMixture for contour overlay
        xlim             : x axis range
        ylim             : y axis range
        arrow_resolution : grid resolution for arrows
        device           : torch device
        suptitle         : figure title
        facecolor        : background colour

    Returns:
        fig : matplotlib Figure
    """
    n = len(timesteps)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor=facecolor)
    axes = axes.flatten()

    xs_c = np.linspace(*xlim, arrow_resolution)
    ys_c = np.linspace(*ylim, arrow_resolution)
    xx_c, yy_c = np.meshgrid(xs_c, ys_c)

    for ax, t in zip(axes, timesteps):
        ax.set_facecolor(facecolor)
        print(f"  Computing salience vector field at t={t}...")

        if library_snapshots is not None and t in library_snapshots:
            library = library_snapshots[t].to(device)
            lib = torch.cat([
                library[:particle_index],
                library[particle_index + 1:]
            ], dim=0)
            ctx = {**context, "library": lib}
        else:
            ctx = context

        gx_s = np.zeros((arrow_resolution, arrow_resolution))
        gy_s = np.zeros((arrow_resolution, arrow_resolution))

        for i in range(arrow_resolution):
            for j in range(arrow_resolution):
                x = torch.tensor(
                    [xx_c[i, j], yy_c[i, j]],
                    dtype=torch.float32, device=device
                ).requires_grad_(True)

                with torch.enable_grad():
                    lib_val = ctx.get("library", None)
                    if lib_val is not None:
                        X_batch = torch.cat([x.unsqueeze(0), lib_val], dim=0)
                        phi_vals = phi.forward_batched(X_batch, t, ctx)
                        phi_query = phi_vals[0].squeeze()
                    else:
                        phi_query = phi(x, t, ctx).squeeze()

                    grad_phi, = torch.autograd.grad(
                        phi_query, x, create_graph=True
                    )
                    grad_phi = grad_phi.clamp(-100.0, 100.0)
                    log_S_val = 2.0 * torch.log(grad_phi.norm() + 1e-12)
                    grad_log_S, = torch.autograd.grad(log_S_val, x)
                    grad_log_S = grad_log_S.clamp(-100.0, 100.0).detach()
                    gx_s[i, j] = grad_log_S[0].item()
                    gy_s[i, j] = grad_log_S[1].item()

        mag_s = np.sqrt(gx_s ** 2 + gy_s ** 2) + 1e-12

        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(gmm, xlim=xlim, ylim=ylim,
                            colors="gray", alpha=0.4, linewidths=0.8)

        ax.quiver(
            xx_c, yy_c, gx_s / mag_s, gy_s / mag_s,
            color="darkorange", alpha=0.8,
            scale=arrow_resolution * 1.5,
            width=0.003, headwidth=4, headlength=5,
            zorder=5,
        )

        ax.set_title(f"t = {t}", fontsize=12, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$", color="black")
        ax.set_ylabel("$x_2$", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    fig.suptitle(suptitle, fontsize=14, color="black")
    plt.tight_layout()
    return fig

def plot_salience_heatmap_clean(
    phi: PhiBase,
    timesteps: List[int],
    context: dict,
    library_snapshots: Optional[dict] = None,
    particle_index: int = 0,
    gmm=None,
    xlim: tuple = (-2, 12),
    ylim: tuple = (-2, 12),
    resolution: int = 80,
    device: torch.device = torch.device("cpu"),
    suptitle: str = "Salience landscape",
    facecolor: str = "white",
) -> plt.Figure:
    """
    Single-row figure showing log S(x) heatmap at multiple timesteps,
    with GMM contours overlaid. No arrows, no trajectories.

    For batch-dependent phis, pass library_snapshots to fix the library
    at each timestep. For batch-independent phis, library_snapshots can
    be None.

    Args:
        phi              : PhiBase instance
        timesteps        : list of timesteps to visualise
        context          : base context dict (model, scheduler, etc.)
        library_snapshots: optional dict mapping timestep -> (N, d_in)
        particle_index   : query particle index (for batch-dependent phis)
        gmm              : GaussianMixture for contour overlay
        xlim             : x axis range
        ylim             : y axis range
        resolution       : grid resolution per axis
        device           : torch device
        suptitle         : figure title
        facecolor        : background colour

    Returns:
        fig : matplotlib Figure
    """
    n = len(timesteps)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor=facecolor)
    axes = axes.flatten()

    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    xx, yy = np.meshgrid(xs, ys)

    for ax, t in zip(axes, timesteps):
        ax.set_facecolor(facecolor)
        print(f"  Computing salience heatmap at t={t}...")

        if library_snapshots is not None and t in library_snapshots:
            library = library_snapshots[t].to(device)
            lib = torch.cat([
                library[:particle_index],
                library[particle_index + 1:]
            ], dim=0)
            ctx = {**context, "library": lib}
        else:
            ctx = context

        log_S = np.zeros((resolution, resolution))

        for i in range(resolution):
            for j in range(resolution):
                x = torch.tensor(
                    [xx[i, j], yy[i, j]],
                    dtype=torch.float32, device=device
                ).requires_grad_(True)

                with torch.enable_grad():
                    lib_val = ctx.get("library", None)
                    if lib_val is not None:
                        X_batch = torch.cat([x.unsqueeze(0), lib_val], dim=0)
                        phi_vals = phi.forward_batched(X_batch, t, ctx)
                        phi_query = phi_vals[0].squeeze()
                    else:
                        phi_query = phi(x, t, ctx).squeeze()

                    grad_phi, = torch.autograd.grad(
                        phi_query, x, create_graph=False
                    )
                    grad_phi = grad_phi.clamp(-100.0, 100.0)
                    log_S[i, j] = (
                        2.0 * torch.log(grad_phi.norm() + 1e-12)
                    ).item()

        im = ax.pcolormesh(
            xx, yy, log_S,
            cmap="plasma", shading="auto",
        )
        cbar = plt.colorbar(im, ax=ax, label="log S(x)")
        cbar.ax.yaxis.label.set_color("black")
        cbar.ax.tick_params(colors="black")

        if gmm is not None:
            plt.sca(ax)
            gmm_pdf_contour(gmm, xlim=xlim, ylim=ylim,
                            colors="white", alpha=0.4, linewidths=0.8)

        ax.set_title(f"t = {t}", fontsize=12, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$", color="black")
        ax.set_ylabel("$x_2$", color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    fig.suptitle(suptitle, fontsize=14, color="black")
    plt.tight_layout()
    return fig
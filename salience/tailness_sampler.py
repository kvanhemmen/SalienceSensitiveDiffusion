"""
salience/tailness_sampler.py
----------------------------
Sampling loops purpose-built for the Tailness phi variants.

The core SalientSampler in sampler.py is phi-agnostic but requires the
caller to manage context. The Tailness phis (TailnessLooPhi,
TailnessGlobalPhi) need per-step context population that is intrinsic
to how they work — the reference pool or leave-one-out others must be
constructed from the current batch of K candidates at each step.

These samplers handle that context management, keeping the logic in one
place rather than scattered across notebook cells.

Functions
---------
sample_tailness_loo      -- full reverse loop using TailnessLooPhi
sample_tailness_global   -- full reverse loop using TailnessGlobalPhi
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from diffusion.scheduler import NoiseScheduler
from salience.sampler import log_salience
from salience.phi import TailnessLooPhi, TailnessGlobalPhi


# ---------------------------------------------------------------------------
# Shared candidate-sampling utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sample_k_candidates(
    model: nn.Module,
    scheduler: NoiseScheduler,
    x_t: Tensor,
    t_int: int,
    K: int,
) -> Tensor:
    """
    Draw K independent DDPM reverse-step candidates from x_t.

    Args:
        x_t   : (B, D)
        t_int : current timestep
        K     : number of candidates

    Returns:
        candidates : (B, K, D)
    """
    B, D = x_t.shape
    device = x_t.device

    x_rep = x_t.repeat_interleave(K, dim=0)                       # (B*K, D)
    t_vec = torch.full((B * K,), t_int, device=device, dtype=torch.long)
    eps_hat = model(x_rep, t_vec)
    _, x_prev = scheduler.step(eps_hat, t_int, x_rep)
    return x_prev.view(B, K, D)                                    # (B, K, D)


# ---------------------------------------------------------------------------
# Leave-one-out tailness sampler
# ---------------------------------------------------------------------------

def sample_tailness_loo(
    model: nn.Module,
    scheduler: NoiseScheduler,
    x_init: Tensor,
    *,
    K: int = 8,
    cov_reg: float = 1e-4,
    return_history: bool = False,
    verbose: bool = True,
):
    """
    Full reverse diffusion using TailnessLooPhi for candidate selection.

    At each step:
        1. Sample K candidates x_{t-1}^k from x_t.
        2. For each k, score phi(x^k) = NLL under a Gaussian fit
           to the Tweedie projections of all other K-1 candidates.
        3. Compute salience S_k = ||grad_x phi(x^k)||^2.
        4. Select the candidate with the highest salience.

    Args:
        model          : denoising network
        scheduler      : NoiseScheduler
        x_init         : (B, D) initial noise
        K              : number of candidates per step
        cov_reg        : covariance regularisation for the LOO Gaussian
        return_history : if True, also return list of x_t snapshots
        verbose        : print progress every 100 steps

    Returns:
        x_0            : (B, D) final samples
        history        : list of (B, D) CPU tensors if return_history else None
    """
    phi = TailnessLooPhi(cov_reg=cov_reg)
    device = x_init.device
    x_t = x_init.clone()
    B, D = x_t.shape
    history = [x_t.detach().cpu()] if return_history else None

    for t_int in reversed(range(len(scheduler))):
        if verbose and t_int % 100 == 0:
            print(f"  t = {t_int}")

        cands = _sample_k_candidates(model, scheduler, x_t, t_int, K)  # (B, K, D)

        salience_vals = torch.zeros(B, K, device=device)

        for k in range(K):
            x_k = cands[:, k, :]                                   # (B, D)
            others = torch.cat(
                [cands[:, :k, :], cands[:, k + 1:, :]], dim=1
            )                                                       # (B, K-1, D)

            for i in range(B):
                context = {
                    "others": others[i].unsqueeze(0),              # (1, K-1, D)
                    "model": model,
                    "scheduler": scheduler,
                }
                salience_vals[i, k] = log_salience(phi, x_k[i], t_int, context)

        best_idx = salience_vals.argmax(dim=1)                     # (B,)
        x_t = cands[torch.arange(B, device=device), best_idx]     # (B, D)

        if return_history:
            history.append(x_t.detach().cpu())

    return (x_t, history) if return_history else x_t


# ---------------------------------------------------------------------------
# Shared-Gaussian tailness sampler
# ---------------------------------------------------------------------------

def sample_tailness_global(
    model: nn.Module,
    scheduler: NoiseScheduler,
    x_init: Tensor,
    *,
    K: int = 8,
    cov_reg: float = 1e-4,
    return_history: bool = False,
    verbose: bool = True,
):
    """
    Full reverse diffusion using TailnessGlobalPhi for candidate selection.

    Same as sample_tailness_loo but the reference Gaussian is fit once
    from the full pool of B*K candidates rather than leave-one-out.
    This is cheaper and avoids per-candidate reference projections.

    Args:
        model          : denoising network
        scheduler      : NoiseScheduler
        x_init         : (B, D) initial noise
        K              : number of candidates per step
        cov_reg        : covariance regularisation for the shared Gaussian
        return_history : if True, also return list of x_t snapshots
        verbose        : print progress every 100 steps

    Returns:
        x_0            : (B, D) final samples
        history        : list of (B, D) CPU tensors if return_history else None
    """
    phi = TailnessGlobalPhi(cov_reg=cov_reg)
    device = x_init.device
    x_t = x_init.clone()
    B, D = x_t.shape
    history = [x_t.detach().cpu()] if return_history else None

    for t_int in reversed(range(len(scheduler))):
        if verbose and t_int % 100 == 0:
            print(f"  t = {t_int}")

        cands = _sample_k_candidates(model, scheduler, x_t, t_int, K)  # (B, K, D)
        ref_pool = cands.reshape(B * K, D)                         # shared reference

        salience_vals = torch.zeros(B, K, device=device)

        for k in range(K):
            x_k = cands[:, k, :]                                   # (B, D)

            for i in range(B):
                context = {
                    "ref_pool": ref_pool,
                    "model": model,
                    "scheduler": scheduler,
                }
                salience_vals[i, k] = log_salience(phi, x_k[i], t_int, context)

        best_idx = salience_vals.argmax(dim=1)                     # (B,)
        x_t = cands[torch.arange(B, device=device), best_idx]     # (B, D)

        if return_history:
            history.append(x_t.detach().cpu())

    return (x_t, history) if return_history else x_t

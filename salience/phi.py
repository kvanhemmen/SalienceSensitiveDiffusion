"""
salience/phi.py
---------------
Concrete feature map implementations for salience-guided sampling.

Each class inherits PhiBase and implements a specific notion of what
makes a diffusion sample "salient". Swapping the phi passed to
SalientSampler changes the sampling behaviour entirely.

Classes
-------
DiversityPhi
    phi(x) = mean([k(x, x_1), ..., k(x, x_N)])  in R^N

    Salience is high when x is dissimilar from a library of previously
    generated samples *in many linearly independent directions*. This
    promotes diversity across the generated set.

ScoreNormPhi
    phi(x) = || s_theta(x, t) ||  in R^1

    Salience is high where the norm of the score function is changing
    most rapidly — near the boundary between high-score and low-score
    regions rather than simply where the score is largest.

    Required context keys: "model", "scheduler".


TailnessLooPhi
    phi(x) = -log N(x0_hat(x); mu_{-k}, Sigma_{-k})

    Scalar phi. Salience is high when the Tweedie projection of x is an
    outlier under a Gaussian fit to all other K-1 projected candidates
    at this timestep (leave-one-out). This promotes tail-seeking /
    mode-diverse behaviour within a single generation batch.

TailnessGlobalPhi
    phi(x) = -log N(x0_hat(x); mu_t, Sigma_t)

    Same as TailnessLooPhi but the reference Gaussian is fit once from
    a shared pool of all K candidates in the batch, rather than
    leave-one-out. Cheaper and often sufficient.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from salience.sampler import PhiBase

from diffusion.scheduler import predict_x0
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_nll_diag(
    z: Tensor,    # (B, D)
    mu: Tensor,   # (D,) or (B, D)
    var: Tensor,  # (D,) or (B, D), strictly positive
) -> Tensor:
    """
    Per-sample diagonal Gaussian negative log-likelihood.

    Returns: (B,)
    """
    D = z.shape[-1]
    mahal = ((z - mu) ** 2 / var).sum(dim=-1)
    logdet = torch.log(var).sum(dim=-1) if var.ndim > 1 else torch.log(var).sum()
    return 0.5 * (mahal + logdet + D * math.log(2.0 * math.pi))


# ---------------------------------------------------------------------------
# DiversityPhi
# ---------------------------------------------------------------------------

class DiversityPhi(PhiBase):
    """
    Vector phi promoting diversity relative to a library of past samples.

        phi(x) = mean([k(x, x_1), ..., k(x, x_N)])  in R^N

    where k is a differentiable dissimilarity kernel (default: squared
    Euclidean distance) and x_1, ..., x_N are previously generated samples
    stored in context["library"].

    By default, compares the current intermediate noisy sample x to the mean of the library of previously
    generated final samples.

    Optionally, can compare the Tweedie estimate x̂0(x, t) instead, while
    keeping the library as previously generated x0 samples.

    context keys
    ------------
    "library"          : Tensor (N, d_in) — previously generated final x0 samples.
                         If absent or empty, log salience returns -inf
                         (no library yet means no diversity signal).
    "max_library_size" : int (optional) — subsample the library to this
                         many entries to bound the Jacobian size as N grows.

    Args:
        kernel : callable (x: (d_in,), y: (d_in,)) -> scalar Tensor.
                 Must be differentiable w.r.t. its first argument.
                 Defaults to squared Euclidean distance.
        use_tweedie : if True, compare using Tweedie estimate x̂0(x, t)
                      instead of raw xt.
    """

    def __init__(self, kernel=None, use_tweedie: bool = False):
        super().__init__()
        self.kernel = kernel if kernel is not None else (
            lambda x, y: ((x - y) ** 2).sum()
        )
        self.use_tweedie = use_tweedie

    def forward(self, x: Tensor, t: int, context: dict) -> Tensor:
        library = context.get("library", None)

        if library is None or library.shape[0] == 0:
            return x.sum().unsqueeze(0) * 0.0

        max_N = context.get("max_library_size", None)
        if max_N is not None and library.shape[0] > max_N:
            idx = torch.randperm(library.shape[0], device=x.device)[:max_N]
            library = library[idx]

        library = library.to(x.device)

        if self.use_tweedie:
            model = context["model"]
            scheduler = context["scheduler"]

            x_b = x.unsqueeze(0)
            t_vec = torch.full((1,), t, device=x.device, dtype=torch.long)
            eps_hat = model(x_b, t_vec)
            query = scheduler.reconstruct_x0(x_b, t_vec, eps_hat).squeeze(0)
        else:
            query = x

        vals = torch.stack([
            self.kernel(query, library[j])
            for j in range(library.shape[0])
        ])

        # Original vector-valued phi:
        # return vals

        # Fast scalar proxy:
        return vals.mean().unsqueeze(0)


# ---------------------------------------------------------------------------
# ScoreNormPhi
# ---------------------------------------------------------------------------

class ScoreNormPhi(PhiBase):
    """
    Scalar phi based on the norm of the diffusion score function.

        phi(x) = || s_theta(x, t) ||
               = || -eps_theta(x, t) / sqrt(1 - alpha_bar_t) ||

    where s_theta is the model's estimate of grad_x log p_t(x).

    Because phi is scalar, salience reduces to:

        S(x) = || grad_x phi(x) ||^2

    This is high where the *norm* of the score is changing most rapidly —
    i.e. near the boundary between high-score and low-score regions.
    This is a subtle but important distinction: the sampler does not
    simply seek regions of large score, but regions where score magnitude
    is most sensitive to position.

    Note: the score network forward pass must remain differentiable w.r.t.
    x, so this phi does NOT use torch.no_grad(). The model should be in
    eval() mode before sampling.

    Required context keys
    ---------------------
    "model"     : nn.Module  -- the denoising network
    "scheduler" : NoiseScheduler
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, t: int, context: dict) -> Tensor:
        """
        Args:
            x       : shape (d_in,) — single sample, gradient-enabled
            t       : diffusion timestep
            context : must contain "model" and "scheduler"

        Returns:
            phi(x)  : shape (1,) — norm of the score at x
        """
        model = context["model"]
        scheduler = context["scheduler"]

        x_b = x.unsqueeze(0)  # (1, d_in)
        t_vec = torch.full((1,), t, device=x.device, dtype=torch.long)

        eps_hat = model(x_b, t_vec)  # (1, d_in)

        sqrt_one_minus_alpha_bar = scheduler.sqrt_one_minus_alphas_cumprod[t].to(x.device)
        score = -eps_hat / sqrt_one_minus_alpha_bar  # (1, d_in)

        score_norm = score.norm()  # scalar
        return score_norm.unsqueeze(0)  # (1,)


# ---------------------------------------------------------------------------
# TailnessLooPhi
# ---------------------------------------------------------------------------

class TailnessLooPhi(PhiBase):
    """
    Leave-one-out tailness phi (scalar).

        phi(x^k) = -log N( x0_hat(x^k) ; mu_{-k}, Sigma_{-k} )

    where x0_hat is the Tweedie projection of x^k, and (mu_{-k}, Sigma_{-k})
    is a diagonal Gaussian fit to the Tweedie projections of all other
    K-1 candidates in the current batch.

    Salience S(x^k) = ||grad_x phi(x^k)||^2 is high when the Tweedie
    projection of x^k is an outlier from the rest of the batch, and
    when that outlierness changes rapidly as x^k moves. This promotes
    tail-seeking / mode-diverse behaviour within a single generation batch.

    context keys
    ------------
    "others"   : Tensor (B, K-1, d_in) — the other K-1 candidates for
                 each of the B samples in the batch. Must be set by the
                 caller (e.g. SalientSampler or a custom loop) before
                 each scoring call.
    "model"    : nn.Module — denoising network, used to project to x0_hat.
    "scheduler": NoiseScheduler — used to project to x0_hat.

    Args:
        cov_reg : diagonal covariance regularisation (added after clamping)
    """

    def __init__(self, cov_reg: float = 1e-4):
        super().__init__()
        self.cov_reg = cov_reg

    def forward(self, x: Tensor, t: int, context: dict) -> Tensor:
        """
        Args:
            x       : (d_in,) — single candidate
            t       : current diffusion timestep (used as t-1 for projection)
            context : must contain "others" (1, K-1, d_in), "model", "scheduler"

        Returns:
            phi(x)  : (1,) scalar
        """
        others: Tensor = context["others"]        # (1, K-1, d_in)
        model: nn.Module = context["model"]
        scheduler = context["scheduler"]

        x_b = x.unsqueeze(0)                      # (1, d_in)
        t_minus_1 = t - 1

        # Project candidate to x0_hat (keep gradient through x)
        if t_minus_1 >= 0:
            t_vec = torch.full((1,), t_minus_1, device=x.device, dtype=torch.long)
            eps_hat = model(x_b, t_vec)
            z = scheduler.reconstruct_x0(x_b, t_vec, eps_hat)    # (1, d_in)
        else:
            z = x_b

        # Project others to x0_hat (no gradient needed)
        with torch.no_grad():
            Km1 = others.shape[1]
            flat = others.reshape(Km1, -1).to(x.device)
            if t_minus_1 >= 0:
                t_oth = torch.full((Km1,), t_minus_1, device=x.device, dtype=torch.long)
                eps_oth = model(flat, t_oth)
                z_oth = scheduler.reconstruct_x0(flat, t_oth, eps_oth)   # (Km1, d_in)
            else:
                z_oth = flat

            mu = z_oth.mean(dim=0)                                # (d_in,)
            if Km1 == 1:
                var = torch.full_like(mu, self.cov_reg)
            else:
                centered = z_oth - mu.unsqueeze(0)
                var = (centered ** 2).mean(dim=0).clamp(min=1e-3) + self.cov_reg

        nll = _gaussian_nll_diag(z, mu, var).clamp(max=1e4)       # (1,)
        return nll                                                 # (1,)


# ---------------------------------------------------------------------------
# TailnessGlobalPhi
# ---------------------------------------------------------------------------

class TailnessGlobalPhi(PhiBase):
    """
    Shared-Gaussian tailness phi (scalar).

        phi(x) = -log N( x0_hat(x) ; mu_t, Sigma_t )

    where (mu_t, Sigma_t) is fit once from a shared reference pool of all
    K candidates in the batch for this timestep.

    Compared to TailnessLooPhi, the reference Gaussian is the same for all
    K candidates (no leave-one-out), which is cheaper and avoids per-candidate
    Tweedie projections of the reference set.

    context keys
    ------------
    "ref_pool" : Tensor (M, d_in) — shared reference pool (typically all
                 B*K candidates at this timestep). Must be set by the caller.
    "model"    : nn.Module — denoising network.
    "scheduler": NoiseScheduler.

    Args:
        cov_reg : diagonal covariance regularisation
    """

    def __init__(self, cov_reg: float = 1e-4):
        super().__init__()
        self.cov_reg = cov_reg

    def forward(self, x: Tensor, t: int, context: dict) -> Tensor:
        """
        Args:
            x       : (d_in,)
            t       : current diffusion timestep
            context : must contain "ref_pool" (M, d_in), "model", "scheduler"

        Returns:
            phi(x)  : (1,) scalar
        """
        ref_pool: Tensor = context["ref_pool"]    # (M, d_in)
        model: nn.Module = context["model"]
        scheduler = context["scheduler"]

        x_b = x.unsqueeze(0)                      # (1, d_in)
        t_minus_1 = t - 1

        # Project candidate to x0_hat (keep gradient through x)
        if t_minus_1 >= 0:
            t_vec = torch.full((1,), t_minus_1, device=x.device, dtype=torch.long)
            eps_hat = model(x_b, t_vec)
            z = scheduler.reconstruct_x0(x_b, t_vec, eps_hat)    # (1, d_in)
        else:
            z = x_b

        # Fit shared reference Gaussian (no gradient)
        with torch.no_grad():
            M = ref_pool.shape[0]
            ref = ref_pool.to(x.device)
            if t_minus_1 >= 0:
                t_ref = torch.full((M,), t_minus_1, device=x.device, dtype=torch.long)
                eps_ref = model(ref, t_ref)
                z_ref = scheduler.reconstruct_x0(ref, t_ref, eps_ref)    # (M, d_in)
            else:
                z_ref = ref

            mu = z_ref.mean(dim=0)                                # (d_in,)
            if M == 1:
                var = torch.full_like(mu, self.cov_reg)
            else:
                centered = z_ref - mu.unsqueeze(0)
                var = (centered ** 2).mean(dim=0).clamp(min=1e-3) + self.cov_reg

        nll = _gaussian_nll_diag(z, mu, var).clamp(max=1e4)       # (1,)
        return nll                                                 # (1,)

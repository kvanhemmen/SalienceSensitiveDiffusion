"""
experiments/gmm.py
------------------
Gaussian Mixture Model utilities for 2-D diffusion experiments.

Classes
-------
GaussianMixture  -- simple GMM with log_prob, score, pdf, and sampling

Functions
---------
gmm_pdf_contour  -- plot PDF contours of a GaussianMixture on the
                    current matplotlib axes
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.distributions as D
from torch import Tensor


class GaussianMixture:
    """
    Gaussian Mixture Model over R^d.

    Args:
        mus     : list of mean tensors, each shape (d,)
        covs    : list of diagonal covariance tensors, each shape (d,)
        weights : mixture weights (will be normalised to sum to 1)
        device  : torch device
    """

    def __init__(
        self,
        mus: List[Tensor],
        covs: List[Tensor],
        weights: List[float],
        device=torch.device("cpu"),
    ):
        self.device = torch.device(device)
        self.mus = [torch.as_tensor(mu, dtype=torch.float32, device=self.device) for mu in mus]
        self.covs = [torch.as_tensor(cov, dtype=torch.float32, device=self.device) for cov in covs]
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        self.weights = w / w.sum()
        self.dim = int(self.mus[0].numel())

    def _component_log_prob(self, x: Tensor, k: int) -> Tensor:
        dist = D.Independent(D.Normal(self.mus[k], self.covs[k].sqrt()), 1)
        return dist.log_prob(x)

    def log_prob(self, x: Tensor) -> Tensor:
        """Log probability density at x. Shape: (B,) for input (B, d)."""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        log_terms = [
            torch.log(self.weights[k]) + self._component_log_prob(x, k)
            for k in range(len(self.mus))
        ]
        return torch.logsumexp(torch.stack(log_terms, dim=0), dim=0)

    def pdf(self, x: Tensor) -> Tensor:
        """Probability density at x."""
        return self.log_prob(x).exp()

    def score(self, x: Tensor) -> Tensor:
        """Score function grad_x log p(x)."""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device).detach().requires_grad_(True)
        lp = self.log_prob(x).sum()
        return torch.autograd.grad(lp, x)[0].detach()

    def sample(self, n: int, seed: Optional[int] = None) -> Tensor:
        """Draw n samples from the mixture."""
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
        comp = torch.multinomial(self.weights, n, replacement=True, generator=g)
        out = [
            self.mus[idx] + self.covs[idx].sqrt() * torch.randn(self.dim, device=self.device, generator=g)
            for idx in comp.tolist()
        ]
        return torch.stack(out, dim=0)


def gmm_pdf_contour(
    gmm: GaussianMixture,
    xlim=(0, 10),
    ylim=(0, 10),
    ticks: int = 200,
    **kwargs,
) -> None:
    """
    Plot PDF contours of a 2-D GaussianMixture on the current axes.

    Args:
        gmm   : GaussianMixture instance (must be 2-D)
        xlim  : (min, max) for x axis
        ylim  : (min, max) for y axis
        ticks : grid resolution
        **kwargs : forwarded to matplotlib contour()
    """
    import matplotlib.pyplot as plt

    xx, yy = np.meshgrid(
        np.linspace(*xlim, ticks),
        np.linspace(*ylim, ticks),
    )
    grid = torch.tensor(
        np.stack([xx, yy], axis=-1), dtype=torch.float32, device=gmm.device
    )
    pdf = gmm.pdf(grid.reshape(-1, 2)).reshape(xx.shape).detach().cpu().numpy()
    plt.contour(xx, yy, pdf, **kwargs)

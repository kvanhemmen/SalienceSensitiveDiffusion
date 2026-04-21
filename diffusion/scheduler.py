"""
diffusion/scheduler.py
----------------------
DDPM noise schedule and reverse-process step utilities.

Classes
-------
NoiseScheduler  -- precomputes all beta/alpha schedule quantities and
                   implements forward noising, posterior mean, and
                   stochastic reverse steps.

Functions
---------
reverse_step    -- one full stochastic DDPM reverse step (no grad)
reverse_mean    -- posterior mean only, no noise added (no grad)
predict_x0      -- Tweedie / reconstruct_x0 estimate (no grad)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class NoiseScheduler:
    """
    Precomputes all quantities needed for DDPM forward and reverse processes.

    Args:
        num_timesteps : total number of diffusion steps T
        beta_start    : beta value at t=0
        beta_end      : beta value at t=T-1
        beta_schedule : "linear" or "quadratic"
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        self.num_timesteps = num_timesteps

        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "quadratic":
            self.betas = (
                torch.linspace(
                    beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32
                ) ** 2
            )
        else:
            raise ValueError(f"Unknown beta_schedule '{beta_schedule}'.")

        self.alphas = 1.0 - self.betas
        self.sqrt_alphas = self.alphas ** 0.5
        self.sqrt_betas = self.betas ** 0.5
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Forward noising coefficients
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod) ** 0.5

        # Tweedie / reconstruct_x0 coefficients
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # Posterior mean coefficients
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def reconstruct_x0(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Tweedie denoising: recover x0 estimate from x_t and predicted noise."""
        s1 = self.sqrt_inv_alphas_cumprod[t].reshape(-1, 1).to(x_t.device)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].reshape(-1, 1).to(x_t.device)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """Posterior mean mu_t(x_0, x_t)."""
        s1 = self.posterior_mean_coef1[t].reshape(-1, 1).to(x_t.device)
        s2 = self.posterior_mean_coef2[t].reshape(-1, 1).to(x_t.device)
        return s1 * x_0 + s2 * x_t

    def get_variance(self, t: int) -> Tensor:
        """Scalar posterior variance at timestep t."""
        if t == 0:
            return torch.tensor(0.0)
        variance = (
            self.betas[t]
            * (1.0 - self.alphas_cumprod_prev[t])
            / (1.0 - self.alphas_cumprod[t])
        )
        return variance.clamp(min=1e-20)

    def step(self, model_output: Tensor, timestep: int, sample: Tensor):
        """
        Full stochastic DDPM reverse step.

        Returns:
            pred_original_sample : x0 estimate, shape (B, D)
            pred_prev_sample     : x_{t-1} sample, shape (B, D)
        """
        t = timestep
        t_tensor = torch.full((sample.shape[0],), t, device=sample.device, dtype=torch.long)
        x0_hat = self.reconstruct_x0(sample, t_tensor, model_output)
        mu = self.q_posterior(x0_hat, sample, t_tensor)

        noise = torch.zeros_like(model_output)
        if t > 0:
            noise = torch.randn_like(model_output).to(model_output.device)

        x_prev = mu + (self.get_variance(t) ** 0.5) * noise
        return x0_hat, x_prev

    def add_noise(self, x_start: Tensor, x_noise: Tensor, timesteps: Tensor) -> Tensor:
        """Forward diffusion: q(x_t | x_0)."""
        s1 = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1).to(x_start.device)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1).to(x_start.device)
        return s1 * x_start + s2 * x_noise

    def to(self, device) -> "NoiseScheduler":
        for name, val in self.__dict__.items():
            if torch.is_tensor(val):
                setattr(self, name, val.to(device))
        return self

    def __len__(self) -> int:
        return self.num_timesteps


# ---------------------------------------------------------------------------
# Reverse-process utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def reverse_step(
    model: nn.Module,
    scheduler: NoiseScheduler,
    x_t: Tensor,
    t_int: int,
) -> Tensor:
    """
    One stochastic DDPM reverse step.

    Args:
        model     : denoising network
        scheduler : NoiseScheduler
        x_t       : (B, D) noisy samples at timestep t
        t_int     : integer timestep

    Returns:
        x_{t-1}   : (B, D)
    """
    t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
    eps_hat = model(x_t, t)
    _, x_tm1 = scheduler.step(eps_hat, t_int, x_t)
    return x_tm1


@torch.no_grad()
def reverse_mean(
    model: nn.Module,
    scheduler: NoiseScheduler,
    x_t: Tensor,
    t_int: int,
) -> Tensor:
    """
    Posterior mean mu_t only — no stochastic noise added.

    Useful when you want to inspect or modify the mean before sampling.

    Args:
        model     : denoising network
        scheduler : NoiseScheduler
        x_t       : (B, D)
        t_int     : integer timestep

    Returns:
        mu_{t-1}  : (B, D)
    """
    t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
    eps_hat = model(x_t, t)
    x0_hat = scheduler.reconstruct_x0(x_t, t, eps_hat)
    return scheduler.q_posterior(x0_hat, x_t, t)


@torch.no_grad()
def predict_x0(
    model: nn.Module,
    scheduler: NoiseScheduler,
    x_t: Tensor,
    t_int: int,
) -> Tensor:
    """
    Tweedie estimate of x_0 from x_t.

    Args:
        model     : denoising network
        scheduler : NoiseScheduler
        x_t       : (B, D)
        t_int     : integer timestep

    Returns:
        x0_hat    : (B, D)
    """
    t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
    eps_hat = model(x_t, t)
    return scheduler.reconstruct_x0(x_t, t, eps_hat)

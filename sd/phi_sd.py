"""
sd/phi_sd.py
------------
Stable Diffusion adapted feature map implementations for salience-guided
sampling in latent space.

These phis operate on SD latent vectors (B, C, H, W) rather than the 2D
point vectors used in the GMM experiments. The PhiBase interface is identical
— only the internals change to accommodate the UNet architecture and latent
space geometry.

Classes
-------
ScoreNormPhiSD
    phi(z) = || s_theta(z, t) ||
           = || -eps_theta(z, t) / sqrt(1 - alpha_bar_t) ||

    where z is a latent vector and s_theta is the score function estimate
    from the UNet. Can be toggled between unconditional and CFG-corrected
    (conditional) score via the `conditional` flag.

DiversityPhiSD
    Same forward_batched mechanism as DiversityPhi in the GMM setting.
    Computes pairwise squared Euclidean distances between all N latents
    in the current batch, with diagonal masked for self-exclusion.
    No external library — diversity is measured within the live batch.
"""

from __future__ import annotations

import torch
from torch import Tensor

from salience.sampler import PhiBase


# ---------------------------------------------------------------------------
# ScoreNormPhiSD
# ---------------------------------------------------------------------------

class ScoreNormPhiSD(PhiBase):
    """
    Scalar phi based on the norm of the UNet score function in latent space.

        phi(z) = || s_theta(z, t) ||

    where s_theta(z, t) = -eps_theta(z, t) / sqrt(1 - alpha_bar_t).

    Salience S(z) = || grad_z phi(z) ||^2 is high where the norm of the
    score is changing most rapidly — near the boundary between high-score
    and low-score regions rather than simply where the score is largest.

    The `conditional` flag controls which noise prediction is used:
        conditional=False : raw unconditional UNet prediction
        conditional=True  : CFG-corrected prediction using prompt embeddings

    Required context keys
    ---------------------
    "unet"         : UNet model
    "scheduler"    : DDPMScheduler (HuggingFace)
    "guidance_scale" : float — CFG scale (only used when conditional=True)
    "prompt_embeds": Tensor (2*B, seq, dim) — concatenated unconditional and
                     conditional embeddings (only used when conditional=True)

    Args:
        conditional : if True, use CFG-corrected score; if False (default),
                      use raw unconditional score. Can be toggled at any time
                      by setting phi.conditional = True/False.
    """

    def __init__(self, conditional: bool = False):
        super().__init__()
        self.conditional = conditional

    def forward(self, z: Tensor, t: int, context: dict) -> Tensor:
        """
        Args:
            z       : shape (C, H, W) — single latent, gradient-enabled
            t       : diffusion timestep (integer)
            context : must contain "unet", "scheduler";
                      additionally "guidance_scale" and "prompt_embeds"
                      when conditional=True

        Returns:
            phi(z)  : shape (1,) — norm of the score at z
        """
        unet = context["unet"]
        scheduler = context["scheduler"]

        z_b = z.unsqueeze(0)                                       # (1, C, H, W)
        t_tensor = torch.tensor([t], device=z.device, dtype=torch.long)

        # Scale the latent input as SD expects
        z_scaled = scheduler.scale_model_input(z_b, t_tensor[0])

        if self.conditional:
            guidance_scale = context["guidance_scale"]
            prompt_embeds = context["prompt_embeds"]               # (2, seq, dim)

            # Duplicate for CFG: [unconditional, conditional]
            z_input = torch.cat([z_scaled, z_scaled], dim=0)      # (2, C, H, W)
            t_input = t_tensor.repeat(2)                           # (2,)

            noise_pred = unet(
                z_input, t_input,
                encoder_hidden_states=prompt_embeds,
            ).sample                                               # (2, C, H, W)

            noise_uncond, noise_text = noise_pred.chunk(2)
            eps_hat = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        else:
            # Unconditional — pass null prompt embeddings from context
            null_embeds = context.get("null_embeds", None)
            if null_embeds is None:
                raise ValueError(
                    "ScoreNormPhiSD requires 'null_embeds' in context "
                    "for unconditional score computation."
                )
            eps_hat = unet(
                z_scaled, t_tensor[0],
                encoder_hidden_states=null_embeds,
            ).sample                                               # (1, C, H, W)

        # Score = -eps / sqrt(1 - alpha_bar_t)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(z.device)
        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t) ** 0.5
        score = -eps_hat / sqrt_one_minus_alpha_bar                # (1, C, H, W)

        return score.norm().unsqueeze(0)                           # (1,)

    def forward_batched(self, Z: Tensor, t: int, context: dict) -> Tensor:
        """
        Vectorised batched forward for ScoreNormPhiSD.

        Calls the UNet once for all N latents simultaneously rather than
        looping over samples.

        Args:
            Z : (N, C, H, W) -- batch of N latent vectors

        Returns:
            phi_vals : (N, 1)
        """
        unet = context["unet"]
        scheduler = context["scheduler"]

        N = Z.shape[0]
        t_tensor = torch.tensor([t], device=Z.device, dtype=torch.long).repeat(N)

        Z_scaled = scheduler.scale_model_input(Z, t_tensor[0])

        if self.conditional:
            guidance_scale = context["guidance_scale"]
            prompt_embeds = context["prompt_embeds"]  # (2, seq, dim)

            uncond_embeds = prompt_embeds[:1].expand(N, -1, -1)
            cond_embeds = prompt_embeds[1:].expand(N, -1, -1)
            all_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)  # (2N, seq, dim)

            Z_input = torch.cat([Z_scaled, Z_scaled], dim=0)  # (2N, C, H, W)
            t_input = t_tensor.repeat(2)

            noise_pred = unet(
                Z_input, t_input,
                encoder_hidden_states=all_embeds,
            ).sample  # (2N, C, H, W)

            noise_uncond, noise_text = noise_pred[:N], noise_pred[N:]
            eps_hat = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        else:
            null_embeds = context.get("null_embeds", None)
            if null_embeds is None:
                raise ValueError(
                    "ScoreNormPhiSD requires 'null_embeds' in context "
                    "for unconditional score computation."
                )
            null_embeds_expanded = null_embeds.expand(N, -1, -1)

            eps_hat = unet(
                Z_scaled, t_tensor,
                encoder_hidden_states=null_embeds_expanded,
            ).sample  # (N, C, H, W)

        alpha_bar_t = scheduler.alphas_cumprod[t].to(Z.device)
        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t) ** 0.5
        scores = -eps_hat / sqrt_one_minus_alpha_bar  # (N, C, H, W)

        score_norms = scores.reshape(N, -1).norm(dim=-1, keepdim=True)  # (N, 1)
        return score_norms


# ---------------------------------------------------------------------------
# DiversityPhiSD
# ---------------------------------------------------------------------------

class DiversityPhiSD(PhiBase):
    """
    Diversity phi operating in SD latent space.

    Uses the same forward_batched mechanism as DiversityPhi in the GMM
    setting: pairwise squared Euclidean distances between all N latents
    in the current batch, with the diagonal masked out for self-exclusion.

    There is no external library — diversity is measured within the live
    batch of N latents at the current timestep t. This makes the method
    fully parallel: all N repulsion gradients are computed in one pass
    via forward_batched.

    The single-sample forward() falls back to zero (no signal) since
    meaningful diversity requires at least two samples. It is only called
    during sequential (non-batched) salience computation, which is not
    the intended use case for this phi.

    No context keys required — diversity is computed purely from the
    batch structure passed to forward_batched.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: Tensor, t: int, context: dict) -> Tensor:
        """
        Single-sample forward — returns zero since diversity requires
        at least two samples. Use forward_batched for meaningful output.

        Args:
            z : shape (C, H, W)

        Returns:
            phi(z) : shape (1,) — zero
        """
        return torch.zeros(1, device=z.device)

    def forward_batched(self, Z: Tensor, t: int, context: dict) -> Tensor:
        """
        Vectorised batched forward for DiversityPhiSD.

        Computes all N phi values simultaneously using broadcasting.
        Each phi(z^i) is the mean squared Euclidean distance from z^i
        to all other latents in the batch (self excluded via diagonal mask).

        Args:
            Z : (N, C, H, W) — batch of N latent vectors

        Returns:
            phi_vals : (N, 1)
        """
        N = Z.shape[0]

        # Flatten latents to vectors for distance computation
        Z_flat = Z.reshape(N, -1)                                  # (N, C*H*W)

        # All pairwise squared distances: (N, N)
        diff = Z_flat.unsqueeze(1) - Z_flat.unsqueeze(0)          # (N, N, C*H*W)
        dists = (diff ** 2).sum(dim=-1)                            # (N, N)

        # Mask diagonal (self-distances)
        mask = 1.0 - torch.eye(N, device=Z.device)
        phi_vals = (dists * mask).sum(dim=-1, keepdim=True) / (N - 1)

        return phi_vals

class ScoreAlignmentPhiSD(PhiBase):
    """
    Score alignment phi operating in SD latent space.

    phi(z^i) = (1/N-1) sum_{j != i} cos_sim(s_theta(z^i, t), s_theta(z^j, t))

    where s_theta(z, t) = -eps_theta(z, t) / sqrt(1 - alpha_bar_t).

    Salience S(z^i) = ||grad_{z^i} phi(z^i)||^2 is high at the boundary
    between regions where z^i's denoising trajectory agrees with the batch
    consensus and regions where it diverges.

    Operates entirely in latent space — no VAE decode required.

    Required context keys
    ---------------------
    "unet"          : UNet model
    "scheduler"     : DDPMScheduler (HuggingFace)
    "prompt_embeds" : Tensor (2*B, seq, dim) — for CFG-corrected score
    "guidance_scale": float — CFG scale

    Args:
        conditional : if True, use CFG-corrected score; if False (default),
                      use raw unconditional score.
    """

    def __init__(self, conditional: bool = False, normalize_grad: bool = True):
        super().__init__()
        self.conditional = conditional
        self.normalize_grad = normalize_grad

    def forward(self, z: Tensor, t: int, context: dict) -> Tensor:
        """
        Single-sample forward — returns zero since meaningful score
        alignment requires at least two samples. Use forward_batched.
        """
        return torch.zeros(1, device=z.device)

    def forward_batched(self, Z: Tensor, t: int, context: dict) -> Tensor:
        """
        Vectorised batched forward for ScoreAlignmentPhiSD.

        Computes the mean cosine similarity between each latent's score
        and all other latents' scores in one pass.

        Args:
            Z : (N, C, H, W) — batch of N latent vectors

        Returns:
            phi_vals : (N, 1)
        """
        unet = context["unet"]
        scheduler = context["scheduler"]

        N = Z.shape[0]
        t_tensor = torch.tensor([t], device=Z.device, dtype=torch.long).repeat(N)

        Z_scaled = torch.stack([
            scheduler.scale_model_input(Z[i].unsqueeze(0), t_tensor[i])[0]
            for i in range(N)
        ])  # (N, C, H, W)

        if self.conditional:
            guidance_scale = context["guidance_scale"]
            prompt_embeds = context["prompt_embeds"]  # (2, seq, dim)

            # Expand prompt embeds to match batch
            uncond_embeds = prompt_embeds[:1].expand(N, -1, -1)   # (N, seq, dim)
            cond_embeds = prompt_embeds[1:].expand(N, -1, -1)     # (N, seq, dim)
            all_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)  # (2N, seq, dim)

            Z_input = torch.cat([Z_scaled, Z_scaled], dim=0)      # (2N, C, H, W)
            t_input = t_tensor.repeat(2)                           # (2N,)

            noise_pred = unet(
                Z_input, t_input,
                encoder_hidden_states=all_embeds,
            ).sample                                               # (2N, C, H, W)

            noise_uncond, noise_text = noise_pred[:N], noise_pred[N:]
            eps_hat = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        else:
            null_embeds = context.get("null_embeds", None)
            if null_embeds is None:
                raise ValueError(
                    "ScoreAlignmentPhiSD requires 'null_embeds' in context "
                    "for unconditional score computation."
                )
            null_embeds_expanded = null_embeds.expand(N, -1, -1)  # (N, seq, dim)

            eps_hat = unet(
                Z_scaled, t_tensor,
                encoder_hidden_states=null_embeds_expanded,
            ).sample                                               # (N, C, H, W)

        # Score = -eps / sqrt(1 - alpha_bar_t)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(Z.device)
        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t) ** 0.5
        scores = -eps_hat / sqrt_one_minus_alpha_bar               # (N, C, H, W)

        # Flatten scores for cosine similarity
        scores_flat = scores.reshape(N, -1)                        # (N, C*H*W)
        scores_norm = scores_flat / (
            scores_flat.norm(dim=-1, keepdim=True) + 1e-12
        )                                                          # (N, C*H*W)

        # Pairwise cosine similarity matrix: (N, N)
        cos_sim = scores_norm @ scores_norm.T

        # Mask diagonal and take mean over N-1 others
        mask = 1.0 - torch.eye(N, device=Z.device)
        phi_vals = (cos_sim * mask).sum(dim=-1, keepdim=True) / (N - 1)  # (N, 1)

        return phi_vals
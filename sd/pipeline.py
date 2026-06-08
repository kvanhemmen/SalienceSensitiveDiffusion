"""
sd/pipeline.py
--------------
Salience-guided Stable Diffusion pipeline.

Subclasses StableDiffusionPipeline directly — no dependency on Anuj's
scorers or GradSDPipeline. The denoising loop is implemented here in full,
with the salience gradient injected at each step exactly as in your 2D
sampler: the noise prediction is modified before the scheduler step,
matching Dhariwal & Nichol (2021) classifier guidance scaling.

The guidance update at each step is:

    noise_pred -= sqrt(1 - alpha_bar_t) * salience_scale * grad_z log S(z_t)

which is equivalent to nudging the posterior mean in the direction of
increasing salience.

Gradient computation is done entirely in latent space — no VAE decode,
no pixel-space scorer. This avoids MPS/CUDA issues and is faster.

Usage
-----
    from sd.pipeline import SalienceGradSDPipeline
    from sd.phi_sd import DiversityPhiSD

    pipe = SalienceGradSDPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to(device)

    phi = DiversityPhiSD()
    pipe.setup_phi(phi)
    pipe.set_salience_scale(1.0)

    images = pipe(
        prompt="a red apple on a white background",
        num_images_per_prompt=4,
        num_inference_steps=500,
    ).images
"""

from __future__ import annotations

import torch

from pathlib import Path
from tqdm.auto import tqdm
from torch import Tensor
from typing import Optional, List, Union, Callable, Dict, Any

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from salience.sampler import PhiBase


class SalienceGradSDPipeline(StableDiffusionPipeline):
    """
    Salience-guided Stable Diffusion pipeline.

    Implements a full DDPM denoising loop with salience gradient guidance
    injected at each step. Operates entirely in latent space.

    Methods to call before __call__:
        pipe.setup_phi(phi, context)     -- set the feature map and its context
        pipe.set_salience_scale(s)       -- set guidance strength (default 1.0)
        pipe.set_guidance_frequency(n)   -- apply guidance every n steps (default 1)
    """

    def setup_phi(self, phi: PhiBase, context: Optional[dict] = None):
        """
        Set the feature map phi and its context dict.

        Args:
            phi     : PhiBase instance (ScoreNormPhiSD or DiversityPhiSD)
            context : dict of external state passed to phi at every step.
                      For ScoreNormPhiSD: must contain "null_embeds".
                      For DiversityPhiSD: can be empty.
        """
        self.phi = phi
        self.phi_context = context if context is not None else {}

    def set_salience_scale(self, scale: float = 1.0):
        """Salience guidance strength."""
        self.salience_scale = scale

    def set_guidance_frequency(self, freq: int = 1):
        """Apply salience guidance every freq steps. 1 = every step."""
        self.guidance_frequency = freq

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 500,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        save_dir: Optional[Path] = None,
        offset: int = 0,
    ):
        """
        Run salience-guided text-to-image generation.

        Standard SD args are identical to StableDiffusionPipeline.__call__.
        Additional args:
            save_dir : if provided, save generated images here as PNGs
            offset   : starting index for saved image filenames
        """
        # --- Setup ---
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width  = width  or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt, height, width, callback_steps,
            negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_cfg = guidance_scale > 1.0

        salience_scale     = getattr(self, 'salience_scale', 1.0)
        guidance_frequency = getattr(self, 'guidance_frequency', 1)
        has_phi            = hasattr(self, 'phi')

        # --- Encode prompt ---
        prompt_embeds = self._encode_prompt(
            prompt, device, num_images_per_prompt,
            do_cfg, negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # --- Timesteps ---
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # --- Latents ---
        num_channels = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # --- Denoising loop ---
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with tqdm(total=num_inference_steps) as pbar:
            for i, t in enumerate(timesteps):

                # CFG: expand latents
                latent_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_input = self.scheduler.scale_model_input(latent_input, t)

                # UNet noise prediction
                noise_pred = self.unet(
                    latent_input, t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # CFG correction
                if do_cfg:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

                # Salience gradient guidance
                if has_phi and guidance_frequency > 0 and (i % guidance_frequency == 0):
                    t_int = int(t)
                    context = {
                        **self.phi_context,
                        "unet": self.unet,
                        "scheduler": self.scheduler,
                    }

                    grad = self._compute_salience_gradient(latents, t_int, context)

                    sqrt_1m_alpha = (
                        1.0 - self.scheduler.alphas_cumprod[t]
                    ).to(latents.device) ** 0.5

                    noise_pred = noise_pred - sqrt_1m_alpha * salience_scale * grad.to(noise_pred.dtype)

                # Scheduler step
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and
                    (i + 1) % self.scheduler.order == 0
                ):
                    pbar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # --- Decode ---
        image = self.decode_latents(latents)

        # Safety checker
        image, has_nsfw = self.run_safety_checker(
            image, device, prompt_embeds.dtype
        )

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        # Save if requested
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(image):
                img.save(save_dir / f"{offset + idx}.png")

        if not return_dict:
            return (image, has_nsfw)

        return StableDiffusionPipelineOutput(
            images=image,
            nsfw_content_detected=has_nsfw,
        )

    # ------------------------------------------------------------------
    # Salience gradient computation
    # ------------------------------------------------------------------

    def _compute_salience_gradient(
        self,
        latents: Tensor,
        t: int,
        context: dict,
    ) -> Tensor:
        """
        Compute grad_z log S(z_t) for the current batch of latents.

        Uses forward_batched when available (DiversityPhiSD) for a fully
        vectorised computation. Falls back to per-sample computation
        for phis without a batched implementation.

        Args:
            latents : (B, C, H, W)
            t       : timestep integer
            context : passed to phi

        Returns:
            grad : (B, C, H, W), same dtype as latents
        """
        B = latents.shape[0]

        if hasattr(self.phi, 'forward_batched') and B > 1:
            return self._batched_salience_grad(latents, t, context)
        else:
            grads = [
                self._single_salience_grad(latents[i], t, context)
                for i in range(B)
            ]
            return torch.stack(grads, dim=0)

    @torch.enable_grad()
    def _batched_salience_grad(
        self,
        Z: Tensor,
        t: int,
        context: dict,
    ) -> Tensor:
        """
        Compute grad_z log S(z) for all N latents in one pass via forward_batched.

        Args:
            Z       : (N, C, H, W)
            t       : timestep
            context : passed to phi

        Returns:
            grads : (N, C, H, W)
        """
        Z_req = Z.detach().to(Z.dtype).requires_grad_(True)

        phi_vals = self.phi.forward_batched(Z_req, t, context)    # (N, d_out)
        d_out = phi_vals.shape[1]

        if d_out == 1:
            grad_phi = torch.autograd.grad(
                phi_vals.squeeze(-1).sum(),
                Z_req,
                create_graph=True,
            )[0]                                                   # (N, C, H, W)
            grad_phi_flat = grad_phi.reshape(Z_req.shape[0], -1).clamp(-100.0, 100.0)

            log_S = 2.0 * torch.log(
                grad_phi_flat.norm(dim=-1) + 1e-12
            ).sum()

            grad_log_S = torch.autograd.grad(log_S, Z_req)[0]    # (N, C, H, W)

        else:
            rows = []
            for i in range(d_out):
                g = torch.autograd.grad(
                    phi_vals[:, i].sum(),
                    Z_req,
                    retain_graph=True,
                    create_graph=True,
                )[0]
                rows.append(g.reshape(Z_req.shape[0], -1).clamp(-100.0, 100.0))

            J = torch.stack(rows, dim=1)                          # (N, d_out, C*H*W)
            sv = torch.linalg.svdvals(J)
            log_S = 2.0 * torch.log(sv + 1e-12).sum()
            grad_log_S = torch.autograd.grad(log_S, Z_req)[0]

        return grad_log_S.clamp(-100.0, 100.0).detach().to(Z.dtype)

    @torch.enable_grad()
    def _single_salience_grad(
        self,
        z: Tensor,
        t: int,
        context: dict,
    ) -> Tensor:
        """
        Compute grad_z log S(z) for a single latent z: (C, H, W).

        Args:
            z       : (C, H, W)
            t       : timestep
            context : passed to phi

        Returns:
            grad : (C, H, W)
        """
        z_req = z.detach().float().requires_grad_(True)

        phi_x = self.phi(z_req, t, context)
        d_out = phi_x.shape[0]

        if d_out == 1:
            grad_phi, = torch.autograd.grad(
                phi_x.squeeze(), z_req, create_graph=True
            )
            grad_phi = grad_phi.clamp(-100.0, 100.0)
            log_S = 2.0 * torch.log(grad_phi.norm() + 1e-12)
        else:
            rows = []
            for i in range(d_out):
                g, = torch.autograd.grad(
                    phi_x[i], z_req,
                    retain_graph=True,
                    create_graph=True,
                )
                rows.append(g.reshape(-1).clamp(-100.0, 100.0))
            J = torch.stack(rows, dim=0)                          # (d_out, C*H*W)
            sv = torch.linalg.svdvals(J)
            log_S = 2.0 * torch.log(sv + 1e-12).sum()

        grad_log_S, = torch.autograd.grad(log_S, z_req)
        return grad_log_S.clamp(-100.0, 100.0).detach().to(z.dtype)

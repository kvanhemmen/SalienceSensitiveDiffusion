"""
salience/sampler.py
-------------------
Information-theoretic salience measure and salience-guided DDPM sampler.

Theory
------
Given a differentiable feature map phi: R^{d_in} -> R^{d_out}, the
preattentive saliency of a point x is defined by Loog (2005) as:

    S(x) := det( J_phi(x)^T  J_phi(x) )

where J_phi(x) is the (d_out x d_in) Jacobian of phi at x.
This is the Gram determinant of the Jacobian, equal to the squared
product of the singular values of J_phi.

For numerical stability we work with log S:

    log S(x) = 2 * sum_i  log sigma_i( J_phi(x) )

Special case d_out == 1 (scalar phi):
    J_phi reduces to the gradient vector grad_x phi(x),
    so  log S(x) = 2 * log || grad_x phi(x) ||.

Classes
-------
PhiBase        -- abstract base class for all feature maps phi
SalientSampler -- DDPM sampler that selects the most salient candidate
                  at each reverse step

Functions
---------
log_salience   -- compute log S(x) for a single sample given any PhiBase
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from diffusion.scheduler import NoiseScheduler, reverse_step as ddpm_reverse_step


# ---------------------------------------------------------------------------
# Abstract base class for phi
# ---------------------------------------------------------------------------

class PhiBase(ABC, nn.Module):
    """
    Abstract base for a feature map phi: R^{d_in} -> R^{d_out}.

    Subclasses implement forward(x, t, context) where:
        x       : Tensor of shape (d_in,)  -- single sample, NOT batched
        t       : int                      -- current diffusion timestep
        context : dict                     -- any external state phi needs

    phi must be differentiable w.r.t. x. The context dict is the mechanism
    for passing in stateful information (e.g. a library of previously
    generated samples, the denoising model, the scheduler, etc.).

    Example subclass skeleton::

        class MyPhi(PhiBase):
            def forward(self, x: Tensor, t: int, context: dict) -> Tensor:
                library = context["library"]         # (N, d_in)
                return torch.stack([...])            # shape (d_out,)
    """

    @abstractmethod
    def forward(self, x: Tensor, t: int, context: dict) -> Tensor:
        """
        Args:
            x       : shape (d_in,)  -- single sample
            t       : diffusion timestep
            context : dict of external state

        Returns:
            phi(x)  : shape (d_out,)
        """
        ...

    def forward_batched(self, X: Tensor, t: int, context: dict) -> Tensor:
        """
        Evaluate phi for a batch of samples X of shape (N, d_in).
        Returns (N, d_out).

        Default implementation calls forward() in a loop.
        Subclasses can override for vectorised computation.
        """
        return torch.stack([
            self.forward(X[i], t, context)
            for i in range(X.shape[0])
        ])


# ---------------------------------------------------------------------------
# Salience computation
# ---------------------------------------------------------------------------

@torch.enable_grad()
def log_salience(
    phi: PhiBase,
    x: Tensor,
    t: int,
    context: dict,
    grad_clip: float = 100.0,
) -> Tensor:
    """
    Compute log S(x) = log det(J_phi^T J_phi) for a single sample x.

    For d_out == 1 (scalar phi):
        log S = 2 * log || grad_x phi(x) ||

    For d_out > 1 (vector phi):
        log S = 2 * sum_i log sigma_i(J_phi(x))
        computed via full SVD of J_phi, shape (d_out, d_in).

    Args:
        phi       : a PhiBase instance
        x         : shape (d_in,); grad will be enabled internally
        t         : diffusion timestep
        context   : dict passed through to phi
        grad_clip : clip gradient entries to [-grad_clip, grad_clip]
                    before norm/SVD for numerical stability

    Returns:
        log_S     : scalar Tensor (detached, no grad)
    """
    x = x.detach().requires_grad_(True)
    phi_x = phi(x, t, context)          # (d_out,)
    d_out = phi_x.shape[0]

    if d_out == 1:
        grad, = torch.autograd.grad(phi_x.squeeze(), x)
        grad = grad.clamp(-grad_clip, grad_clip)
        log_S = 2.0 * torch.log(grad.norm() + 1e-12)

    else:
        rows = []
        for i in range(d_out):
            grad, = torch.autograd.grad(
                phi_x[i],
                x,
                retain_graph=(i < d_out - 1),
                create_graph=False,
            )
            rows.append(grad.clamp(-grad_clip, grad_clip))

        J = torch.stack(rows, dim=0)                       # (d_out, d_in)
        sv = torch.linalg.svdvals(J)                       # (min(d_out, d_in),)
        log_S = 2.0 * torch.log(sv + 1e-12).sum()

    return log_S.detach()


# ---------------------------------------------------------------------------
# Salience-guided DDPM sampler
# ---------------------------------------------------------------------------

class SalientSampler:
    """
    DDPM sampler with salience-guided candidate selection.

    At each reverse timestep t:
        1. Draw K candidate next states x_{t-1}^k independently from
           the DDPM posterior p_theta(x_{t-1} | x_t).
        2. Score each candidate: s_k = log S(x_{t-1}^k) under phi.
        3. Select x_{t-1} = argmax_k s_k and continue.

    The stochasticity of DDPM (vs. DDIM) is essential: without it all K
    candidates would be identical and selection would be vacuous.

    Args:
        model     : denoising network, called as model(x, t)
        scheduler : NoiseScheduler instance
        phi       : a PhiBase instance — the only thing that needs to
                    change when you want different sampling behaviour
        K         : number of candidates to draw per step
        device    : torch device
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: NoiseScheduler,
        phi: PhiBase,
        K: int = 16,
        device: torch.device = torch.device("cpu"),
        guidance_frequency: int = 1,
        batched: bool = False,
        normalize_salience_grad: bool = False,
    ):
        self.model = model
        self.scheduler = scheduler
        self.phi = phi
        self.K = K
        self.device = device
        self.guidance_frequency = guidance_frequency
        self.batched = batched
        self.normalize_salience_grad = normalize_salience_grad

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _single_reverse_step(self, x_i: Tensor, t_int: int) -> Tensor:
        """
        One DDPM reverse step for a single unbatched sample x_i: (d_in,).
        Returns x_{t-1}: (d_in,).
        """
        return ddpm_reverse_step(
            self.model,
            self.scheduler,
            x_i.unsqueeze(0),
            t_int,
        ).squeeze(0)

    def _select_most_salient(
        self,
        candidates: list[Tensor],
        t: int,
        context: dict,
    ) -> Tensor:
        """
        Score K candidates and return the one with the highest log salience.

        Args:
            candidates : list of K tensors, each shape (d_in,)
            t          : current timestep
            context    : passed through to phi

        Returns:
            best candidate : shape (d_in,)
        """
        with torch.enable_grad():
            scores = torch.stack([
                log_salience(self.phi, c, t, context)
                for c in candidates
            ])
        return candidates[scores.argmax().item()]

        # ------------------------------------------------------------------
        # Batched helpers
        # ------------------------------------------------------------------

    def _batched_reverse_step(self, x: Tensor, t_int: int) -> Tensor:
        """
        Draw K candidates for all N samples in one batched forward pass.

        Args:
            x     : (N, d_in)
            t_int : timestep

        Returns:
            candidates : (N, K, d_in)
        """
        N, d_in = x.shape
        x_rep = x.repeat_interleave(self.K, dim=0)  # (N*K, d_in)
        t_vec = torch.full((N * self.K,), t_int, device=self.device, dtype=torch.long)
        eps_hat = self.model(x_rep, t_vec)
        _, x_prev = self.scheduler.step(eps_hat, t_int, x_rep)
        return x_prev.view(N, self.K, d_in)

    def _batched_log_salience(
            self,
            candidates: Tensor,
            t: int,
            context: dict,
    ) -> Tensor:
        """
        Compute log salience for all N*K candidates in one pass.

        For scalar phi (d_out==1): one backward pass for all N*K candidates.
        For vector phi (d_out >1): one backward pass per output dimension,
            shared across the full N*K batch, then SVD per sample.

        Args:
            candidates : (N, K, d_in)
            t          : timestep
            context    : passed through to phi

        Returns:
            scores : (N, K)
        """
        N, K, d_in = candidates.shape
        flat = candidates.reshape(N * K, d_in).detach().requires_grad_(True)

        # Evaluate phi for all N*K candidates
        phi_vals = torch.stack([
            self.phi(flat[i], t, context)
            for i in range(N * K)
        ])  # (N*K, d_out)

        d_out = phi_vals.shape[1]

        if d_out == 1:
            # One backward pass covers all N*K scalars at once
            grads = torch.autograd.grad(
                phi_vals.squeeze(-1).sum(),
                flat,
                create_graph=False,
            )[0]  # (N*K, d_in)
            grads = grads.clamp(-100.0, 100.0)
            log_S = 2.0 * torch.log(grads.norm(dim=-1) + 1e-12)  # (N*K,)

        else:
            # One backward pass per output dimension, shared across batch
            rows = []
            for i in range(d_out):
                grad = torch.autograd.grad(
                    phi_vals[:, i].sum(),
                    flat,
                    retain_graph=(i < d_out - 1),
                    create_graph=False,
                )[0]  # (N*K, d_in)
                rows.append(grad.clamp(-100.0, 100.0))

            J = torch.stack(rows, dim=1)  # (N*K, d_out, d_in)
            sv = torch.linalg.svdvals(J)  # (N*K, min(d_out, d_in))
            log_S = 2.0 * torch.log(sv + 1e-12).sum(dim=-1)  # (N*K,)

        return log_S.detach().view(N, K)


    # ------------------------------------------------------------------
    # Public sampling interface
    # ------------------------------------------------------------------

    def sample_resampling_guided(
            self,
            x_T: Tensor,
            context: Optional[dict] = None,
            verbose: bool = True,
    ) -> Tensor:
        if context is None:
            context = {}

        x = x_T.clone().to(self.device)
        N = x.shape[0]
        T = self.scheduler.num_timesteps

        for t in reversed(range(T)):
            if verbose and t % 100 == 0:
                print(f"  t = {t}")

            use_salience = (
                    self.guidance_frequency > 0 and
                    (t % self.guidance_frequency == 0)
            )

            if self.batched:
                with torch.no_grad():
                    candidates = self._batched_reverse_step(x, t)  # (N, K, d_in)

                if use_salience:
                    scores = self._batched_log_salience(candidates, t, context)  # (N, K)
                    best_idx = scores.argmax(dim=1)  # (N,)
                else:
                    best_idx = torch.zeros(N, device=self.device, dtype=torch.long)

                x = candidates[torch.arange(N, device=self.device), best_idx]

            else:
                next_x = []
                for i in range(N):
                    if use_salience:
                        with torch.no_grad():
                            cands = [
                                self._single_reverse_step(x[i], t)
                                for _ in range(self.K)
                            ]
                        best = self._select_most_salient(cands, t, context)
                    else:
                        with torch.no_grad():
                            best = self._single_reverse_step(x[i], t)
                    next_x.append(best)
                x = torch.stack(next_x, dim=0)

        return x

    def sample_grad_guided(
            self,
            x_T: Tensor,
            guidance_scale: float = 1.0,
            context: Optional[dict] = None,
            verbose: bool = True,
    ) -> Tensor:
        """
        Classifier-guidance-style salience-guided reverse diffusion.

        At each reverse timestep t, instead of selecting among K candidates,
        the posterior mean is nudged in the direction of increasing salience:

            mu_t_guided = mu_t + guidance_scale * var_t * grad_{x_t} log S(x_t)

        where var_t is the posterior variance at timestep t, matching the
        scaling convention of Dhariwal & Nichol (2021).

        Respects self.resampling_frequency — guidance is only applied at
        timesteps where t % resampling_frequency == 0.

        If self.batched is True, all N gradient computations are performed
        in a single pass. Only valid for phi variants without sequential
        sample dependencies.

        Args:
            x_T            : initial noise, shape (N, d_in)
            guidance_scale : scalar s controlling guidance strength
            context        : dict passed through to phi
            verbose        : print progress every 100 steps

        Returns:
            x_0 : denoised samples, shape (N, d_in)
        """
        if context is None:
            context = {}

        x = x_T.clone().to(self.device)
        N = x.shape[0]
        T = self.scheduler.num_timesteps

        for t in reversed(range(T)):
            if verbose and t % 100 == 0:
                print(f"  t = {t}")

            use_guidance = (
                    self.guidance_frequency > 0 and
                    (t % self.guidance_frequency == 0)
            )

            t_vec = torch.full((N,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                eps_hat = self.model(x, t_vec)
                x0_hat = self.scheduler.reconstruct_x0(x, t_vec, eps_hat)
                mu = self.scheduler.q_posterior(x0_hat, x, t_vec)  # (N, d_in)

            if use_guidance:
                if self.batched:
                    grads = self._batched_salience_grad(x, t, context)  # (N, d_in)
                else:
                    grads = torch.stack([
                        self._salience_grad(x[i], t, context)
                        for i in range(N)
                    ])  # (N, d_in)

                var_t = self.scheduler.get_variance(t)
                mu = mu + guidance_scale * var_t * grads

            noise = torch.zeros_like(x)
            if t > 0:
                noise = torch.randn_like(x)
            var_t = self.scheduler.get_variance(t)
            x = mu + (var_t ** 0.5) * noise

        return x.detach()

    def _batched_salience_grad(
            self,
            x: Tensor,
            t: int,
            context: dict,
    ) -> Tensor:
        """
        Compute grad_{x_i} log S(x_i) for all N samples in one pass.

        For scalar phi (d_out==1):
            - One forward pass over all N samples to get phi(x_i)
            - One backward pass with create_graph=True to get grad_phi for all N
            - One backward pass through log ||grad_phi|| to get grad_log_S for all N

        For vector phi (d_out>1):
            - d_out forward-backward passes to build J for all N simultaneously
            - One backward pass through log det(J^T J) for all N

        Args:
            x       : (N, d_in), the current noisy states
            t       : timestep
            context : passed through to phi

        Returns:
            grads : (N, d_in), detached
        """
        N, d_in = x.shape
        x_req = x.detach().requires_grad_(True)

        # Evaluate phi for all N samples
        phi_vals = torch.stack([
            self.phi(x_req[i], t, context)
            for i in range(N)
        ])  # (N, d_out)

        d_out = phi_vals.shape[1]

        if d_out == 1:
            # First backward: grad_phi for all N in one pass
            grad_phi = torch.autograd.grad(
                phi_vals.squeeze(-1).sum(),
                x_req,
                create_graph=True,
            )[0]  # (N, d_in)
            grad_phi = grad_phi.clamp(-100.0, 100.0)

            # log S = 2 * log ||grad_phi|| per sample
            log_S = 2.0 * torch.log(
                grad_phi.norm(dim=-1) + 1e-12
            ).sum()  # scalar sum for backward

            # Second backward: grad_log_S for all N in one pass
            grad_log_S = torch.autograd.grad(log_S, x_req)[0]  # (N, d_in)

        else:
            # Build J for all N simultaneously: one backward per output dim
            rows = []
            for i in range(d_out):
                g = torch.autograd.grad(
                    phi_vals[:, i].sum(),
                    x_req,
                    retain_graph=True,
                    create_graph=True,
                )[0]  # (N, d_in)
                rows.append(g.clamp(-100.0, 100.0))

            J = torch.stack(rows, dim=1)  # (N, d_out, d_in)
            sv = torch.linalg.svdvals(J)  # (N, min(d_out, d_in))
            log_S = 2.0 * torch.log(sv + 1e-12).sum()  # scalar sum for backward

            grad_log_S = torch.autograd.grad(log_S, x_req)[0]  # (N, d_in)

        grad_log_S = grad_log_S.clamp(-100.0, 100.0).detach()
        if self.normalize_salience_grad:
            grad_norm = grad_log_S.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            grad_log_S = grad_log_S / grad_norm
        return grad_log_S


    def _salience_grad(
        self,
        x_i: Tensor,
        t: int,
        context: dict,
    ) -> Tensor:
        """
        Compute grad_{x_i} log S(x_i) for a single sample.

        This is the gradient of the salience score itself w.r.t. the
        input — a second-order quantity, since log S already involves
        a first derivative of phi.

        Args:
            x_i     : shape (d_in,)
            t       : timestep
            context : passed through to phi

        Returns:
            grad : shape (d_in,), detached
        """
        x = x_i.detach().requires_grad_(True)

        # Recompute log salience with create_graph=True so we can
        # differentiate through it
        phi_x = self.phi(x, t, context)
        d_out = phi_x.shape[0]

        if d_out == 1:
            grad_phi, = torch.autograd.grad(
                phi_x.squeeze(), x, create_graph=True
            )
            grad_phi = grad_phi.clamp(-100.0, 100.0)
            log_S = 2.0 * torch.log(grad_phi.norm() + 1e-12)
        else:
            rows = []
            for i in range(d_out):
                g, = torch.autograd.grad(
                    phi_x[i], x,
                    retain_graph=True,
                    create_graph=True,
                )
                rows.append(g.clamp(-100.0, 100.0))
            J = torch.stack(rows, dim=0)                       # (d_out, d_in)
            sv = torch.linalg.svdvals(J)
            log_S = 2.0 * torch.log(sv + 1e-12).sum()

        # Now differentiate log S w.r.t. x — this is the second derivative
        grad_log_S, = torch.autograd.grad(log_S, x)
        return grad_log_S.clamp(-100.0, 100.0).detach()

    def sample_particle_grad_guided(
            self,
            x_T: Tensor,
            guidance_scale: float = 1.0,
            context: Optional[dict] = None,
            verbose: bool = True,
    ) -> Tensor:
        """
        Particle guidance: batched diversity-promoting reverse diffusion.

        Maintains N particles {x_t^1, ..., x_t^N} in parallel. At each
        reverse step, each particle is repelled from all others by adding
        a salience gradient term to the posterior mean:

            mu_t^i_guided = mu_t^i + guidance_scale * var_t * grad_{x_t^i} log S(x_t^i)

        where S(x_t^i) is computed under phi with the library set to all
        other particles at the current timestep (self excluded via self_index).

        All N gradient computations are performed in a single batched pass,
        making this fully parallel unlike the original sequential DiversityPhi.

        Inspired by: "Particle Guidance: non-I.I.D. diverse sampling with
        diffusion models" — repulsion is applied to noisy intermediates x_t
        rather than finished samples.

        Respects self.resampling_frequency.

        Args:
            x_T            : initial noise, shape (N, d_in)
            guidance_scale : scalar controlling repulsion strength
            context        : base context dict merged with per-step particle
                             positions. If None, an empty dict is used.
            verbose        : print progress every 100 steps

        Returns:
            x_0 : denoised samples, shape (N, d_in)
        """
        if context is None:
            context = {}

        x = x_T.clone().to(self.device)
        N = x.shape[0]
        T = self.scheduler.num_timesteps

        for t in reversed(range(T)):
            if verbose and t % 1 == 0:
                print(f"  t = {t}")

            use_guidance = (
                    self.guidance_frequency > 0 and
                    (t % self.guidance_frequency == 0)
            )

            # Posterior mean for all N particles — no grad needed here
            t_vec = torch.full((N,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                eps_hat = self.model(x, t_vec)
                x0_hat = self.scheduler.reconstruct_x0(x, t_vec, eps_hat)
                mu = self.scheduler.q_posterior(x0_hat, x, t_vec)  # (N, d_in)

            if use_guidance:
                grads = self._batched_particle_grad(x, t, context)  # (N, d_in)
                var_t = self.scheduler.get_variance(t)
                mu = mu + guidance_scale * var_t * grads

            # Sample x_{t-1} from guided mean
            noise = torch.zeros_like(x)
            if t > 0:
                noise = torch.randn_like(x)
            var_t = self.scheduler.get_variance(t)
            x = mu + (var_t ** 0.5) * noise

        return x.detach()

    def _batched_particle_grad(
            self,
            x: Tensor,
            t: int,
            context: dict,
    ) -> Tensor:
        """
        Compute grad_{x^i} log S(x^i) for all N particles simultaneously,
        where each particle's phi excludes itself from the library.

        Builds phi_vals for all N particles in one pass, with each particle
        i using the library x with index i excluded (via self_index in context).

        Args:
            x       : (N, d_in) current particle positions
            t       : timestep
            context : base context; "library" will be set to x, "self_index"
                      set per particle

        Returns:
            grads : (N, d_in) repulsion gradients, detached
        """
        N, d_in = x.shape
        x_req = x.detach().requires_grad_(True)

        # Each particle i sees the full batch as library minus itself
        phi_vals = self.phi.forward_batched(x_req, t, context)  # (N, d_out)

        d_out = phi_vals.shape[1]
        # print("Called")
        #
        # print(phi_vals.requires_grad)
        # print(phi_vals.grad_fn)

        if d_out == 1:
            grad_phi = torch.autograd.grad(
                phi_vals.squeeze(-1).sum(),
                x_req,
                create_graph=True,
            )[0]  # (N, d_in)
            grad_phi = grad_phi.clamp(-100.0, 100.0)
            log_S = 2.0 * torch.log(
                grad_phi.norm(dim=-1) + 1e-12
            ).sum()
            grads = torch.autograd.grad(log_S, x_req)[0]  # (N, d_in)

        else:
            rows = []
            for i in range(d_out):
                g = torch.autograd.grad(
                    phi_vals[:, i].sum(),
                    x_req,
                    retain_graph=True,
                    create_graph=True,
                )[0]  # (N, d_in)
                rows.append(g.clamp(-100.0, 100.0))

            J = torch.stack(rows, dim=1)  # (N, d_out, d_in)
            sv = torch.linalg.svdvals(J)
            log_S = 2.0 * torch.log(sv + 1e-12).sum()
            grads = torch.autograd.grad(log_S, x_req)[0]  # (N, d_in)

        grads = grads.clamp(-100.0, 100.0).detach()
        if self.normalize_salience_grad:
            grad_norm = grads.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            grads = grads / grad_norm
        return grads

    def sample_particle_resampling_guided(
            self,
            x_T: Tensor,
            context: Optional[dict] = None,
            verbose: bool = True,
    ) -> Tensor:
        """
        Resampling-based particle guidance.

        At each reverse timestep t, for each particle i:
            1. Draw K candidates from the DDPM posterior
            2. Score each candidate using phi with the library set to all
               other particles' current positions at timestep t (self excluded)
            3. Select the most salient candidate

        All particles score against the same snapshot of x_t, making this
        fully parallelisable with no sequential dependency across particles.

        Respects self.resampling_frequency.

        Args:
            x_T     : initial noise, shape (N, d_in)
            context : base context dict; "library" and "self_index" are set
                      internally at each step and should not be passed in
            verbose : print progress every 100 steps

        Returns:
            x_0 : denoised samples, shape (N, d_in)
        """
        if context is None:
            context = {}

        x = x_T.clone().to(self.device)
        N = x.shape[0]
        T = self.scheduler.num_timesteps

        for t in reversed(range(T)):
            if verbose and t % 100 == 0:
                print(f"  t = {t}")

            use_salience = (
                    self.guidance_frequency > 0 and
                    (t % self.guidance_frequency == 0)
            )

            # Draw K candidates for all N particles in one batched pass
            with torch.no_grad():
                candidates = self._batched_reverse_step(x, t)  # (N, K, d_in)

            if use_salience:
                # Score all N*K candidates, each particle excluding itself
                # from the library. Library is the current snapshot x_t.
                scores = self._batched_particle_salience(
                    candidates, x, t, context
                )  # (N, K)
                best_idx = scores.argmax(dim=1)  # (N,)
            else:
                best_idx = torch.zeros(N, device=self.device, dtype=torch.long)

            x = candidates[torch.arange(N, device=self.device), best_idx]

        return x

    def _batched_particle_salience(
            self,
            candidates: Tensor,
            x_current: Tensor,
            t: int,
            context: dict,
    ) -> Tensor:
        """
        Score all N*K candidates where each particle i uses the current
        particle positions x_current as the library, excluding index i.

        Args:
            candidates : (N, K, d_in) — candidate next states
            x_current  : (N, d_in)   — current particle positions (the library)
            t          : timestep
            context    : base context

        Returns:
            scores : (N, K)
        """
        N, K, d_in = candidates.shape
        scores = torch.zeros(N, K, device=self.device)

        for i in range(N):
            # Build context for particle i: library is x_current, self excluded
            ctx_i = {
                **context,
                "library": x_current,
                "self_index": i,
            }
            for k in range(K):
                scores[i, k] = log_salience(self.phi, candidates[i, k], t, ctx_i)

        return scores

    def sample_particle_grad_classifier(
            self,
            x_T: Tensor,
            classifier: nn.Module,
            target_class: int,
            diversity_scale: float = 1.0,
            classifier_scale: float = 1.0,
            context: Optional[dict] = None,
            verbose: bool = True,
    ) -> Tensor:
        """
        Combined diversity + classifier gradient guidance.

        At each reverse timestep t:

            mu_guided = mu_t
                      + var_t * diversity_scale  * grad_{x_t} log S(x_t)
                      + var_t * classifier_scale * grad_{x_t} log p(y | x_t, t)

        The diversity gradient is computed via forward_batched for efficiency.
        The classifier gradient pulls samples toward target_class.
        Both are evaluated at the noisy intermediate x_t — the classifier
        is time-conditioned so this is principled at all noise levels.

        Respects self.guidance_frequency.

        Args:
            x_T              : initial noise, shape (N, d_in)
            classifier       : trained NoisyClassifier
            target_class     : integer class index to guide toward
            diversity_scale  : lambda_div — scale for diversity gradient
            classifier_scale : lambda_cls — scale for classifier gradient
            context          : base context dict (optional)
            verbose          : print progress every 100 steps

        Returns:
            x_0 : denoised samples, shape (N, d_in)
        """

        if context is None:
            context = {}

        x = x_T.clone().to(self.device)
        N = x.shape[0]
        T = self.scheduler.num_timesteps
        target = torch.full((N,), target_class, device=self.device, dtype=torch.long)

        for t in reversed(range(T)):
            if verbose and t % 100 == 0:
                print(f"  t = {t}")

            use_guidance = (
                    self.guidance_frequency > 0 and
                    t % self.guidance_frequency == 0
            )

            t_vec = torch.full((N,), t, device=self.device, dtype=torch.long)

            # Posterior mean — no grad needed for denoising step
            with torch.no_grad():
                eps_hat = self.model(x, t_vec)
                x0_hat = self.scheduler.reconstruct_x0(x, t_vec, eps_hat)
                mu = self.scheduler.q_posterior(x0_hat, x, t_vec)  # (N, d_in)

            if use_guidance:
                var_t = self.scheduler.get_variance(t)

                # --- Diversity gradient ---
                div_grads = self._batched_particle_grad(x, t, context)  # (N, d_in)

                # --- Classifier gradient ---
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t_vec)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[
                    torch.arange(N, device=self.device), target
                ]
                cls_grads = torch.autograd.grad(
                    selected.sum(), x_in
                )[0].clamp(-100.0, 100.0).detach()  # (N, d_in)

                # --- Combined update ---
                mu = (mu
                      + var_t * diversity_scale * div_grads
                      + var_t * classifier_scale * cls_grads)

            noise = torch.zeros_like(x)
            if t > 0:
                noise = torch.randn_like(x)
            var_t = self.scheduler.get_variance(t)
            x = mu + (var_t ** 0.5) * noise

        return x.detach()


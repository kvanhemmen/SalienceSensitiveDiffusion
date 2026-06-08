"""
experiments/conditional_gmm.py
-------------------------------
Two-class conditional GMM setup, noisy classifier training, and
classifier-guided DDPM sampling for the conditional salience experiment.

The GMM has 10 modes total — 5 per class — with classes spatially
separated. Mode positions are fixed by a frozen random seed so the
experiment is fully replicable.

Classes
-------
ConditionalGaussianMixture  -- GMM with per-sample class labels

Functions
---------
make_conditional_gmm        -- build the frozen 2-class 10-mode GMM
train_noisy_classifier      -- train a time-conditioned MLP classifier
sample_classifier_guided    -- standalone classifier-guided DDPM sampler
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from diffusion.model import MLP
from diffusion.scheduler import NoiseScheduler


# ---------------------------------------------------------------------------
# Frozen mode positions
# Mode layout: class 0 lives in x1 in [0, 4], class 1 in x1 in [6, 10]
# Both classes spread across x2 in [0, 10]
# Positions generated once with seed 42 and frozen.
# ---------------------------------------------------------------------------

_FROZEN_SEED = 42

def make_conditional_gmm(
    device: torch.device = torch.device("cpu"),
    cov_scale: float = 0.4,
    shape: str = "horseshoe",
) -> "ConditionalGaussianMixture":
    """
    Two-class 10-mode GMM.

    Args:
        device    : torch device
        cov_scale : isotropic variance for each mode
        shape     : "horseshoe" — two C-shapes opening rightward, well
                    separated (left and right halves of [0,10]^2)
                    "separated" — original random blobs (seed=42)
    """
    if shape == "horseshoe":
        cls0_mus = torch.tensor([
            [3, 6.5],  # top
            [1, 6.5],  # upper arm — curves left
            [-0.5, 5.0],  # middle — furthest left
            [1, 3.5],  # lower arm — curves left
            [3, 3.5],  # bottom
        ])

        cls1_mus = torch.tensor([
            [9, 6.5],  # top
            [7, 6.5],  # upper arm — curves left
            [5.5, 5.0],  # middle — furthest left
            [7, 3.5],  # lower arm — curves left
            [9, 3.5],  # bottom
        ])
        mus = torch.cat([cls0_mus, cls1_mus], dim=0)
        covs = torch.full((10, 2), cov_scale)
        weights = torch.ones(10) / 10.0
        labels = torch.tensor([0] * 5 + [1] * 5)

    elif shape == "separated":
        rng = torch.Generator()
        rng.manual_seed(_FROZEN_SEED)
        def rand(low, high, n):
            return low + (high - low) * torch.rand(n, generator=rng)
        cls0_mus = torch.stack([rand(0.5, 3.5, 5), rand(1.0, 9.0, 5)], dim=1)
        cls1_mus = torch.stack([rand(6.5, 9.5, 5), rand(1.0, 9.0, 5)], dim=1)
        mus = torch.cat([cls0_mus, cls1_mus], dim=0)
        covs = torch.full((10, 2), cov_scale)
        weights = torch.ones(10) / 10.0
        labels = torch.tensor([0] * 5 + [1] * 5)

    elif shape == "submodal":
        import math

        ring_radius = 1.5
        centre_weight = 0.65
        sub_weight = 0.07

        # Class 0 centred at (3, 5), Class 1 centred at (7, 5)
        class_centres = [(3.5, 5.0), (6.5, 5.0)]
        all_mus = []
        all_weights = []
        all_labels = []

        for cls_idx, (cx, cy) in enumerate(class_centres):
            # Central mode
            all_mus.append([cx, cy])
            all_weights.append(centre_weight / 2.0)  # divide by 2 for global weight
            all_labels.append(cls_idx)

            # 5 sub-modes evenly spaced on a ring
            for k in range(5):
                angle = 2 * math.pi * k / 5
                all_mus.append([
                    cx + ring_radius * math.cos(angle),
                    cy + ring_radius * math.sin(angle),
                ])
                all_weights.append(sub_weight / 2.0)
                all_labels.append(cls_idx)

        mus = torch.tensor(all_mus, dtype=torch.float32)
        covs = torch.cat([
            torch.full((1, 2), cov_scale),  # central mode
            torch.full((5, 2), cov_scale * 0.3),  # sub-modes class 0
            torch.full((1, 2), cov_scale),  # central mode
            torch.full((5, 2), cov_scale * 0.3),  # sub-modes class 1
        ], dim=0)
        weights = torch.tensor(all_weights)
        labels = torch.tensor(all_labels)
    else:
        raise ValueError(f"Unknown shape '{shape}'. Choose 'horseshoe', 'submodal' or 'separated'.")

    return ConditionalGaussianMixture(
        mus=mus, covs=covs, weights=weights, labels=labels, device=device,
    )


# ---------------------------------------------------------------------------
# Conditional GMM
# ---------------------------------------------------------------------------

class ConditionalGaussianMixture:
    """
    Gaussian Mixture Model with per-component class labels.

    Args:
        mus     : (K, d) mode centres
        covs    : (K, d) diagonal covariances
        weights : (K,) mixture weights (will be normalised)
        labels  : (K,) integer class label per component
        device  : torch device
    """

    def __init__(
        self,
        mus: Tensor,
        covs: Tensor,
        weights: Tensor,
        labels: Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.mus = mus.to(device)
        self.covs = covs.to(device)
        self.weights = (weights / weights.sum()).to(device)
        self.labels = labels.to(device)
        self.num_classes = int(labels.max().item()) + 1
        self.dim = int(mus.shape[1])
        self.K = int(mus.shape[0])

    def sample(
        self,
        n: int,
        target_class: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Draw n samples with their class labels.

        Args:
            n            : number of samples
            target_class : if provided, sample only from this class
            seed         : optional random seed

        Returns:
            x      : (n, d) samples
            labels : (n,) integer class labels
        """
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)

        if target_class is not None:
            mask = self.labels == target_class
            mus = self.mus[mask]
            covs = self.covs[mask]
            w = self.weights[mask]
            w = w / w.sum()
            labs = self.labels[mask]
        else:
            mus = self.mus
            covs = self.covs
            w = self.weights
            labs = self.labels

        comp = torch.multinomial(w, n, replacement=True, generator=g)
        noise = torch.randn(n, self.dim, device=self.device, generator=g)
        x = mus[comp] + covs[comp].sqrt() * noise
        y = labs[comp]
        return x, y

    def log_prob(self, x: Tensor) -> Tensor:
        """Full mixture log probability. Shape (B,) for input (B, d)."""
        x = x.to(self.device)
        log_terms = []
        for k in range(self.K):
            import torch.distributions as D
            dist = D.Independent(
                D.Normal(self.mus[k], self.covs[k].sqrt()), 1
            )
            log_terms.append(
                torch.log(self.weights[k]) + dist.log_prob(x)
            )
        return torch.logsumexp(torch.stack(log_terms, dim=0), dim=0)

    def pdf(self, x: Tensor) -> Tensor:
        return self.log_prob(x).exp()

    def mode_centres(self, target_class: Optional[int] = None) -> Tensor:
        """Return mode centres, optionally filtered by class."""
        if target_class is not None:
            return self.mus[self.labels == target_class]
        return self.mus

    def contour_plot(
        self,
        xlim=(-1, 11),
        ylim=(-1, 11),
        ticks: int = 200,
        **kwargs,
    ) -> None:
        """Plot PDF contours on the current matplotlib axes."""
        import numpy as np
        import matplotlib.pyplot as plt

        xx, yy = np.meshgrid(
            np.linspace(*xlim, ticks),
            np.linspace(*ylim, ticks),
        )
        grid = torch.tensor(
            np.stack([xx, yy], axis=-1), dtype=torch.float32, device=self.device
        )
        pdf = self.pdf(grid.reshape(-1, 2)).reshape(xx.shape).detach().cpu().numpy()
        plt.contour(xx, yy, pdf, **kwargs)


# ---------------------------------------------------------------------------
# Noisy classifier
# ---------------------------------------------------------------------------

class NoisyClassifier(nn.Module):
    """
    Time-conditioned MLP classifier for noisy intermediate samples x_t.

    Reuses the same MLP architecture as the denoiser but with a
    classification head (num_classes outputs) instead of a noise
    prediction head (d outputs).

    Args:
        hidden_size   : width of hidden layers (default 128)
        hidden_layers : number of residual blocks (default 3)
        emb_size      : embedding size (default 128)
        num_classes   : number of output classes (default 2)
    """

    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        num_classes: int = 2,
    ):
        super().__init__()
        # Reuse the same architecture as MLP denoiser
        self._mlp = MLP(
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            emb_size=emb_size,
            time_emb="sinusoidal",
            input_emb="identity",
        )
        # Replace the output layer: 2 -> num_classes
        in_features = self._mlp.joint_mlp[-1].in_features
        self._mlp.joint_mlp[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x : (B, 2) noisy samples
            t : (B,)   integer timestep indices

        Returns:
            logits : (B, num_classes)
        """
        return self._mlp(x, t)


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

def train_noisy_classifier(
    gmm: ConditionalGaussianMixture,
    scheduler: NoiseScheduler,
    device: torch.device,
    n_train: int = 100_000,
    batch_size: int = 512,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_size: int = 128,
    hidden_layers: int = 3,
    emb_size: int = 128,
    save_path: Optional[Path] = None,
) -> NoisyClassifier:
    """
    Train a time-conditioned classifier on noisy GMM samples.

    Training procedure:
        1. Sample clean (x_0, y) pairs from the GMM
        2. Sample random timesteps t ~ Uniform(0, T)
        3. Corrupt: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
        4. Train to predict y from (x_t, t) with cross-entropy loss

    Args:
        gmm          : ConditionalGaussianMixture
        scheduler    : NoiseScheduler (provides noise schedule)
        device       : torch device
        n_train      : number of training samples to generate
        batch_size   : training batch size
        epochs       : number of training epochs
        lr           : learning rate
        hidden_size  : classifier hidden size
        hidden_layers: classifier depth
        emb_size     : embedding size
        save_path    : if provided, save trained classifier here

    Returns:
        classifier : trained NoisyClassifier in eval mode
    """
    # Generate clean dataset
    x0, y = gmm.sample(n_train)
    x0 = x0.float()
    y = y.long()

    dataset = TensorDataset(x0, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    classifier = NoisyClassifier(
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        emb_size=emb_size,
        num_classes=gmm.num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
    T = scheduler.num_timesteps

    print("Training noisy classifier...")
    for epoch in tqdm(range(epochs)):
        classifier.train()
        total_loss = 0.0
        for x_clean, labels in loader:
            x_clean = x_clean.to(device)
            labels = labels.to(device)

            # Sample random timesteps and corrupt
            t = torch.randint(0, T, (x_clean.shape[0],), device=device).long()
            noise = torch.randn_like(x_clean)
            x_t = scheduler.add_noise(x_clean, noise, t)

            logits = classifier(x_t, t)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(loader)
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    classifier.eval()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), save_path)
        print(f"Saved classifier -> {save_path}")

    return classifier


# ---------------------------------------------------------------------------
# Standalone classifier-guided DDPM sampler
# ---------------------------------------------------------------------------

def sample_classifier_guided(
    model: nn.Module,
    classifier: NoisyClassifier,
    scheduler: NoiseScheduler,
    x_init: Tensor,
    target_class: int,
    guidance_scale: float = 1.0,
    resampling_frequency: int = 1,
    verbose: bool = True,
) -> Tensor:
    """
    Classifier-guided DDPM sampling (no salience).

    At each reverse timestep t:

        mu_guided = mu_t + var_t * guidance_scale * grad_{x_t} log p(y | x_t, t)

    The classifier gradient is computed with grad enabled;
    the denoising step uses no_grad.

    Args:
        model               : denoising network
        classifier          : trained NoisyClassifier
        scheduler           : NoiseScheduler
        x_init              : (N, d) initial noise
        target_class        : integer class to guide toward
        guidance_scale      : scale for classifier gradient
        resampling_frequency: apply guidance every n steps (1 = every step)
        verbose             : print progress every 100 steps

    Returns:
        x_0 : (N, d) guided samples
    """
    device = x_init.device
    x = x_init.clone()
    N = x.shape[0]
    T = scheduler.num_timesteps
    target = torch.full((N,), target_class, device=device, dtype=torch.long)

    for t in reversed(range(T)):
        if verbose and t % 100 == 0:
            print(f"  t = {t}")

        use_guidance = (
            resampling_frequency > 0 and
            t % resampling_frequency == 0
        )

        t_vec = torch.full((N,), t, device=device, dtype=torch.long)

        # Posterior mean — no grad needed for denoising step
        with torch.no_grad():
            eps_hat = model(x, t_vec)
            x0_hat = scheduler.reconstruct_x0(x, t_vec, eps_hat)
            mu = scheduler.q_posterior(x0_hat, x, t_vec)          # (N, d)

        if use_guidance:
            # Classifier gradient w.r.t. x_t
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t_vec)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[torch.arange(N, device=device), target]
            cls_grad = torch.autograd.grad(selected.sum(), x_in)[0]  # (N, d)
            cls_grad = cls_grad.clamp(-100.0, 100.0).detach()

            var_t = scheduler.get_variance(t)
            mu = mu + guidance_scale * var_t * cls_grad

        noise = torch.zeros_like(x)
        if t > 0:
            noise = torch.randn_like(x)
        var_t = scheduler.get_variance(t)
        x = mu + (var_t ** 0.5) * noise

    return x.detach()

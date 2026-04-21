# SalienceSensitiveDiffusion

Information-theoretic salience-guided DDPM sampling.

## Theory

Given a differentiable feature map φ: ℝ^{d_in} → ℝ^{d_out}, the preattentive
saliency of a point x is defined by Loog (2005) as:

    S(x) := det( J_φ(x)ᵀ J_φ(x) )

where J_φ(x) is the Jacobian of φ at x. This is the Gram determinant of the
Jacobian, measuring how much φ stretches the local volume around x.

The sampling procedure is: at each reverse diffusion step t, draw K candidate
next states from the DDPM posterior, score each by S(x), and keep the most
salient. 

## Repository structure

```
SalienceSensitiveDiffusion/
├── diffusion/
│   ├── model.py          # MLP denoiser and positional embeddings
│   └── scheduler.py      # NoiseScheduler, reverse_step, predict_x0
├── salience/
│   ├── sampler.py        # PhiBase, log_salience, SalientSampler
│   ├── phi.py            # DiversityPhi, TailnessLooPhi, TailnessGlobalPhi
│   └── tailness_sampler.py  # Sampling loops for Tailness phi variants
├── experiments/
│   ├── gmm.py            # GaussianMixture, gmm_pdf_contour
│   ├── evaluation.py     # Metrics and visualisation utilities
│   └── checkpoints.py    # Model loading and saving
└── notebooks/
    └── gmm_experiment.ipynb  # Experiments live here
```

## Design principles

**φ is the only thing you need to change** to alter sampling behaviour.
All φ implementations inherit `PhiBase` from `salience/sampler.py` and
implement a single method: `forward(x, t, context) -> Tensor`.

The `context` dict is the mechanism for passing external state into φ —
previously generated samples for diversity, the denoising model and scheduler
for tailness, etc.

## Implemented φ variants

### DiversityPhi (`salience/phi.py`)
Promotes diversity relative to a library of previously generated samples:

    φ(x) = mean([k(x, x_1), ..., k(x, x_N)])  ∈ ℝᴺ

High salience when x is dissimilar from the library.

**Required context keys:** `"library"` (Tensor, N × d_in),
optionally `"max_library_size"` (int).

### TailnessLooPhi (`salience/phi.py`)
Leave-one-out tailness: high salience when a sample's Tweedie projection
is an outlier from the rest of the batch.

    φ(x^k) = -log N( x̂_0(x^k) ; μ_{-k}, Σ_{-k} )

**Required context keys:** `"others"` (Tensor, 1 × K-1 × d_in),
`"model"`, `"scheduler"`.

### TailnessGlobalPhi (`salience/phi.py`)
Same idea but with a shared reference Gaussian fit from all K candidates.
Cheaper than leave-one-out, often sufficient.

**Required context keys:** `"ref_pool"` (Tensor, M × d_in),
`"model"`, `"scheduler"`.

## Usage

```python
from diffusion.model import MLP
from diffusion.scheduler import NoiseScheduler
from salience.sampler import SalientSampler
from salience.phi import DiversityPhi

model = ...  # trained MLP
scheduler = NoiseScheduler().to(device)
phi = DiversityPhi()

sampler = SalientSampler(model, scheduler, phi, K=16, device=device)

context = {"library": existing_samples}
x_0 = sampler.sample_resampling_guided(x_T, context=context)
```

For the tailness variants, use the dedicated samplers in
`salience/tailness_sampler.py` which manage context automatically:

```python
from salience.tailness_sampler import sample_tailness_global

x_0 = sample_tailness_global(model, scheduler, x_T, K=16)
```

## Adding a new φ

Subclass `PhiBase` and implement `forward`:

```python
from salience.sampler import PhiBase
import torch
from torch import Tensor

class MyPhi(PhiBase):
    def forward(self, x: Tensor, t: int, context: dict) -> Tensor:
        # x is shape (d_in,) — a single sample, not batched
        # return shape (d_out,) — must be differentiable w.r.t. x
        ...
```

Then pass an instance to `SalientSampler`.

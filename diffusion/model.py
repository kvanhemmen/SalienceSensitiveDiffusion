"""
diffusion/model.py
------------------
Denoising network architecture

Classes
-------
PositionalEmbedding  -- dispatcher for time/input embedding strategies
MLP                  -- unconditional denoising MLP
ConditionalMLP       -- class-conditioned denoising MLP
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional / time embeddings
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size)).to(x.device)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

    def __len__(self) -> int:
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x / self.size * self.scale).unsqueeze(-1)

    def __len__(self) -> int:
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self) -> int:
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)

    def __len__(self) -> int:
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * 0

    def __len__(self) -> int:
        return 1


class PositionalEmbedding(nn.Module):
    """Dispatcher that selects an embedding strategy by name."""

    _STRATEGIES = ("sinusoidal", "linear", "learnable", "identity", "zero")

    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()
        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif type == "identity":
            self.layer = IdentityEmbedding()
        elif type == "zero":
            self.layer = ZeroEmbedding()
        else:
            raise ValueError(
                f"Unknown embedding type '{type}'. "
                f"Choose from {self._STRATEGIES}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.ff(x))


# ---------------------------------------------------------------------------
# Denoising networks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Unconditional denoising MLP for 2-D data.

    Takes a batch of noisy 2-D samples x_t and timestep indices t,
    and predicts the noise epsilon added to produce x_t.

    Args:
        hidden_size  : width of all hidden layers
        hidden_layers: number of residual Block layers
        emb_size     : output dimension of each embedding module
        time_emb     : embedding strategy for the timestep
        input_emb    : embedding strategy for the 2-D input coordinates
    """

    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
    ):
        super().__init__()
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = (
            len(self.time_mlp.layer)
            + len(self.input_mlp1.layer)
            + len(self.input_mlp2.layer)
        )
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 2) noisy samples
            t : (B,)   integer timestep indices

        Returns:
            eps_hat : (B, 2) predicted noise
        """
        x1_emb = self.input_mlp1(x[:, 0]).to(x.device)
        x2_emb = self.input_mlp2(x[:, 1]).to(x.device)
        t_emb = self.time_mlp(t).to(x.device)
        h = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        return self.joint_mlp(h)


class ConditionalMLP(nn.Module):
    """
    Class-conditioned denoising MLP for 2-D data.

    Same architecture as MLP with an additional embedding for a
    scalar class label y.

    Args:
        hidden_size  : width of all hidden layers
        hidden_layers: number of residual Block layers
        emb_size     : output dimension of each embedding module
        time_emb     : embedding strategy for the timestep
        input_emb    : embedding strategy for the 2-D input / class coordinates
    """

    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
    ):
        super().__init__()
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.class_mlp = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = (
            len(self.time_mlp.layer)
            + len(self.input_mlp1.layer)
            + len(self.input_mlp2.layer)
            + len(self.class_mlp.layer)
        )
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x : (B, 2) noisy samples
            y : (B, 1) class labels
            t : (B,)   integer timestep indices

        Returns:
            eps_hat : (B, 2) predicted noise
        """
        x1_emb = self.input_mlp1(x[:, 0]).to(x.device)
        x2_emb = self.input_mlp2(x[:, 1]).to(x.device)
        t_emb = self.time_mlp(t).to(x.device)
        y_emb = self.class_mlp(y[:, 0]).to(x.device)
        h = torch.cat((x1_emb, x2_emb, t_emb, y_emb), dim=-1)
        return self.joint_mlp(h)

"""
experiments/checkpoints.py
--------------------------
Model loading and checkpoint management utilities.

Functions
---------
load_model      -- load an MLP denoiser from a checkpoint file,
                   handling both raw module and state-dict formats
load_base_noise -- load a fixed base noise tensor from disk
save_checkpoint -- save model state dict to disk
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from diffusion.model import MLP


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_module_prefix(state_dict: OrderedDict) -> OrderedDict:
    """Remove 'module.' prefix added by DataParallel / DistributedDataParallel."""
    out = OrderedDict()
    for k, v in state_dict.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(
    model_path: Path,
    device: torch.device = torch.device("cpu"),
    hidden_size: int = 128,
    hidden_layers: int = 3,
    emb_size: int = 128,
    time_emb: str = "sinusoidal",
    input_emb: str = "identity",
) -> MLP:
    """
    Load a trained MLP denoiser from a checkpoint file.

    Handles three checkpoint formats:
        1. A raw nn.Module saved with torch.save(model, path)
        2. A dict with a "state_dict" or "model_state_dict" key
        3. A raw state dict

    Args:
        model_path   : path to the checkpoint file
        device       : device to load the model onto
        hidden_size  : must match the saved model architecture
        hidden_layers: must match the saved model architecture
        emb_size     : must match the saved model architecture
        time_emb     : must match the saved model architecture
        input_emb    : must match the saved model architecture

    Returns:
        model : MLP in eval mode on the specified device
    """
    ckpt = torch.load(model_path, map_location=device)

    model = MLP(
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        emb_size=emb_size,
        time_emb=time_emb,
        input_emb=input_emb,
    ).to(device)

    if isinstance(ckpt, nn.Module):
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
    model.eval()
    return model


def load_base_noise(
    noise_path: Path,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Load a fixed base noise tensor from disk.

    Args:
        noise_path : path to a saved torch tensor of shape (N, D)
        device     : device to load onto

    Returns:
        noise tensor on the specified device
    """
    if not noise_path.exists():
        raise FileNotFoundError(
            f"Base noise tensor not found at {noise_path}. "
            "Generate one with torch.randn and torch.save, or update the path."
        )
    return torch.load(noise_path, map_location=device)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """
    Save model state dict to disk.

    Args:
        model : nn.Module to save
        path  : destination file path (will create parent dirs if needed)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")

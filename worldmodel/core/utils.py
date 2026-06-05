"""Small shared helpers: image <-> tensor, EMA, and video writing."""

from __future__ import annotations

import copy
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------- image conversion
def frames_to_model(frames: np.ndarray, device) -> torch.Tensor:
    """uint8 (..., H, W, C) in [0, 255] -> float tensor (..., C, H, W) in [-1, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(frames)).to(device).float()
    t = t.movedim(-1, -3)  # channels-last -> channels-first
    return t / 127.5 - 1.0


def model_to_uint8(t: torch.Tensor) -> np.ndarray:
    """float tensor (..., C, H, W) in [-1, 1] -> uint8 numpy (..., H, W, C)."""
    t = ((t.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return t.movedim(-3, -1).cpu().numpy()


# --------------------------------------------------------------------------- EMA
class Ema:
    """Exponential moving average of model parameters for stable sampling."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def module(self) -> nn.Module:
        return self.shadow


# --------------------------------------------------------------------------- video
def save_video(path: str, frames: list[np.ndarray] | np.ndarray, fps: int = 20) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def upscale(frame: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)


def comparison_frames(
    gt: np.ndarray, pred: np.ndarray, scale: int = 4, gap: int = 4
) -> list[np.ndarray]:
    """Build side-by-side (ground truth | prediction) frames for a rollout.

    gt, pred: (T, H, W, C) uint8.
    """
    out = []
    H = gt.shape[1]
    sep = np.full((H * scale, gap, 3), 255, np.uint8)
    for g, p in zip(gt, pred):
        out.append(np.concatenate([upscale(g, scale), sep, upscale(p, scale)], axis=1))
    return out

"""The Game contract and action-space types.

A game is any object implementing the Game protocol below. The world model is
built per run from the game's ``obs_shape`` and ``action_space``, so the core
never needs to know which game it is training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Discrete:
    """A discrete action space with ``n`` actions, labelled ``0 .. n-1``."""

    n: int

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.n))

    @property
    def shape(self) -> tuple[int, ...]:
        # The "raw" shape of a discrete action is a scalar.
        return ()


@dataclass(frozen=True)
class Box:
    """A continuous action vector with per-dimension bounds.

    ``low`` and ``high`` are broadcast to ``shape`` and stored as float32
    arrays so an actor can clamp against them directly.
    """

    shape: tuple[int, ...]
    low: np.ndarray
    high: np.ndarray

    def __init__(self, shape: tuple[int, ...], low, high):
        shape = tuple(int(s) for s in shape)
        low = np.broadcast_to(np.asarray(low, dtype=np.float32), shape).copy()
        high = np.broadcast_to(np.asarray(high, dtype=np.float32), shape).copy()
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def dim(self) -> int:
        return int(np.prod(self.shape))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        u = rng.random(self.shape, dtype=np.float32)
        return (self.low + u * (self.high - self.low)).astype(np.float32)

    def clip(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.low, self.high).astype(np.float32)


ActionSpace = Discrete | Box


@runtime_checkable
class Game(Protocol):
    """The contract every game must implement.

    Frames are always ``HxWx3`` uint8 arrays. ``done`` marks an episode
    boundary; frame-stacked conditioning must never cross it.
    """

    action_space: ActionSpace
    obs_shape: tuple[int, int, int]  # native (H, W, C) of the rendered frame

    def reset(self) -> np.ndarray:
        """Start a new episode, randomizing the initial state. Returns a frame."""
        ...

    def step(self, action) -> tuple[np.ndarray, bool]:
        """Apply ``action``; return ``(next_frame, done)``."""
        ...


def action_dim(space: ActionSpace) -> int:
    """Flat dimensionality of an action space (1 for Discrete)."""
    if isinstance(space, Discrete):
        return 1
    return space.dim

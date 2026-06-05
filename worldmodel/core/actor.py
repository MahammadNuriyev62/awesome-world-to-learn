"""Exploration actors.

Two distinct randomness roles in this project: the *actor* is temporally
correlated random (sticky / OU) so the agent commits to a direction and covers
the state space; the *batch sampler* (in buffer.py) is uniform random. No
trained policy is ever involved.

A game may override ``make_actor`` if it needs custom exploration.
"""

from __future__ import annotations

import numpy as np

from .contract import ActionSpace, Box, Discrete


class StickyDiscreteActor:
    """Hold a uniformly-random discrete action for ``repeat`` frames."""

    def __init__(self, n: int, repeat: int, rng: np.random.Generator):
        self.n = int(n)
        self.repeat = max(1, int(repeat))
        self.rng = rng
        self._action = 0
        self._left = 0

    def reset(self) -> None:
        self._left = 0

    def __call__(self) -> int:
        if self._left <= 0:
            self._action = int(self.rng.integers(self.n))
            self._left = self.repeat
        self._left -= 1
        return self._action


class OUContinuousActor:
    """Ornstein-Uhlenbeck process: temporally-correlated continuous actions.

    x <- x + theta * (mu - x) + sigma * N(0, 1), then clipped to the box.
    mu is the box midpoint.
    """

    def __init__(self, box: Box, theta: float, sigma: float, rng: np.random.Generator):
        self.box = box
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rng = rng
        self.mu = 0.5 * (box.low + box.high)
        self.scale = 0.5 * (box.high - box.low)
        self._x = self.mu.copy()

    def reset(self) -> None:
        self._x = self.mu.copy()

    def __call__(self) -> np.ndarray:
        noise = self.rng.standard_normal(self.box.shape).astype(np.float32)
        self._x = self._x + self.theta * (self.mu - self._x) + self.sigma * self.scale * noise
        return self.box.clip(self._x)


def make_actor(action_space: ActionSpace, config: dict, rng: np.random.Generator):
    """Return the exploration actor for an action space.

    The returned object is callable (``actor()`` -> action) and has ``reset()``
    to be called on every episode reset.
    """
    if isinstance(action_space, Discrete):
        return StickyDiscreteActor(action_space.n, config.get("action_repeat", 8), rng)
    if isinstance(action_space, Box):
        return OUContinuousActor(
            action_space,
            config.get("ou_theta", 0.15),
            config.get("ou_sigma", 0.3),
            rng,
        )
    raise TypeError(f"unsupported action space: {action_space!r}")

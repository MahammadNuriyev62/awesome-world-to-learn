"""GymGame: wrap any Gymnasium environment as a Game.

Uses the env's ``rgb_array`` render as the observation and maps its action space
to the contract. This exposes the whole Gymnasium ecosystem (classic control,
Atari via ALE, ...) and is the test that the core is not hardcoded to the ball.

    python -m worldmodel.core.train --game gym:CartPole-v1
    python -m worldmodel.core.train --game gym:Pendulum-v1      # continuous
    python -m worldmodel.core.train --game gym:ALE/Breakout-v5  # needs ale-py
"""

from __future__ import annotations

import numpy as np

from worldmodel.core.contract import Box, Discrete


def _finite(arr, fallback):
    arr = np.asarray(arr, dtype=np.float32)
    out = np.where(np.isfinite(arr), arr, fallback).astype(np.float32)
    return out


class GymGame:
    @staticmethod
    def default_config() -> dict:
        return {"resolution": 64, "episode_timeout": 300}

    def __init__(self, config: dict | None = None):
        import gymnasium as gym

        cfg = config or {}
        env_id = cfg.get("env_id")
        if env_id is None:
            raise ValueError("GymGame requires 'env_id' in config (use spec gym:ENV_ID)")
        self.env_id = env_id
        self.name = "gym_" + env_id.replace("/", "_").replace("-", "_")

        if env_id.startswith("ALE/"):  # register Atari envs if ale-py is present
            try:
                import ale_py
                gym.register_envs(ale_py)
            except ImportError:
                pass

        self.env = gym.make(env_id, render_mode="rgb_array")
        self._rng = np.random.default_rng(int(cfg.get("seed", 0)) + 4242)

        # map action space
        aspace = self.env.action_space
        if isinstance(aspace, gym.spaces.Discrete):
            self.action_space = Discrete(int(aspace.n))
            self._discrete = True
        elif isinstance(aspace, gym.spaces.Box):
            low = _finite(aspace.low, -1.0)
            high = _finite(aspace.high, 1.0)
            self.action_space = Box(tuple(aspace.shape), low, high)
            self._discrete = False
        else:
            raise TypeError(f"unsupported gym action space: {type(aspace).__name__}")

        frame = self.reset()
        self.obs_shape = frame.shape  # native (H, W, 3)

    def _render(self) -> np.ndarray:
        frame = self.env.render()
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        return frame

    def reset(self) -> np.ndarray:
        seed = int(self._rng.integers(0, 2**31 - 1))
        self.env.reset(seed=seed)
        return self._render()

    def step(self, action):
        if self._discrete:
            act = int(action)
        else:
            act = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        _, _, terminated, truncated, _ = self.env.step(act)
        done = bool(terminated or truncated)
        return self._render(), done

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

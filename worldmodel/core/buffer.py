"""Episode-aware uint8 ring buffer with uniform random minibatch sampling.

Stores raw frames as uint8 to keep memory small (100k @ 64x64x3 ~ 1.2 GB).

Conditioning is episode-aware: a sampled training tuple
``(cond_frames[k], action, target_frame)`` never crosses a reset. We guarantee
this with two per-slot fields:

  - ``ep_step``: the frame's index within its episode (0 at reset). A target
    needs ``ep_step >= k`` so its k predecessors are in the same episode.
  - ``wid``: a monotonic write id. We additionally require the k predecessors to
    have consecutive write ids, which rejects windows whose older frames were
    already overwritten by the ring (the only remaining hazard).

The action stored at a slot is the action that *led into* that frame, so for a
target at slot ``p`` the action is ``actions[p]`` and the conditioning frames are
the k slots before ``p``.
"""

from __future__ import annotations

import numpy as np

from .contract import ActionSpace, Discrete, action_dim


class FrameBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, int, int],
        frame_stack: int,
        action_space: ActionSpace,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.obs_shape = obs_shape
        self.k = int(frame_stack)
        self.action_space = action_space
        self.discrete = isinstance(action_space, Discrete)
        self.adim = action_dim(action_space)
        self.rng = np.random.default_rng(seed)

        H, W, C = obs_shape
        self.frames = np.zeros((self.capacity, H, W, C), dtype=np.uint8)
        self.actions = np.zeros((self.capacity, self.adim), dtype=np.float32)
        self.ep_step = np.full(self.capacity, -1, dtype=np.int32)
        self.wid = np.full(self.capacity, -1, dtype=np.int64)

        self.pos = 0
        self.size = 0
        self._counter = 0
        self._cur_ep_step = -1

    def __len__(self) -> int:
        return self.size

    def _encode_action(self, action) -> np.ndarray:
        if self.discrete:
            return np.array([float(int(action))], dtype=np.float32)
        return np.asarray(action, dtype=np.float32).reshape(self.adim)

    def _write(self, frame: np.ndarray, action: np.ndarray, ep_step: int) -> None:
        p = self.pos
        self.frames[p] = frame
        self.actions[p] = action
        self.ep_step[p] = ep_step
        self.wid[p] = self._counter
        self._counter += 1
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_reset(self, frame: np.ndarray) -> None:
        """Record the first frame of a new episode."""
        self._cur_ep_step = 0
        self._write(frame, np.zeros(self.adim, dtype=np.float32), 0)

    def add_step(self, action, frame: np.ndarray) -> None:
        """Record a frame produced by applying ``action`` to the previous frame."""
        self._cur_ep_step += 1
        self._write(frame, self._encode_action(action), self._cur_ep_step)

    # ------------------------------------------------------------------ sampling
    def can_sample(self) -> bool:
        return self.size > self.k + 1

    def _occupied_indices(self, n: int) -> np.ndarray:
        hi = self.capacity if self.size == self.capacity else self.pos
        return self.rng.integers(0, hi, size=n)

    def _valid_mask(self, idx: np.ndarray) -> np.ndarray:
        k, cap = self.k, self.capacity
        ok = self.ep_step[idx] >= k
        base_wid = self.wid[idx]
        for m in range(1, k + 1):
            pred = (idx - m) % cap
            ok &= self.wid[pred] == (base_wid - m)
        return ok

    def sample(self, batch_size: int):
        """Uniform random minibatch.

        Returns a dict with numpy arrays:
          cond:   uint8 (B, k, H, W, C)  oldest..newest
          action: int64 (B,) if discrete else float32 (B, adim)
          target: uint8 (B, H, W, C)
        """
        if not self.can_sample():
            raise RuntimeError("buffer does not have enough frames to sample yet")

        k, cap = self.k, self.capacity
        chosen = np.empty(batch_size, dtype=np.int64)
        filled = 0
        attempts = 0
        while filled < batch_size:
            cand = self._occupied_indices(batch_size * 2)
            valid = cand[self._valid_mask(cand)]
            take = min(len(valid), batch_size - filled)
            chosen[filled:filled + take] = valid[:take]
            filled += take
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("could not sample enough valid windows; buffer too small?")

        # cond frame indices: oldest (target-k) .. newest (target-1)
        offsets = np.arange(k, 0, -1, dtype=np.int64)  # [k, k-1, ..., 1]
        cond_idx = (chosen[:, None] - offsets[None, :]) % cap  # (B, k)
        cond = self.frames[cond_idx]  # (B, k, H, W, C)
        target = self.frames[chosen]  # (B, H, W, C)

        if self.discrete:
            action = self.actions[chosen, 0].astype(np.int64)
        else:
            action = self.actions[chosen].astype(np.float32)

        return {"cond": cond, "action": action, "target": target}

    def sample_sequence(self, batch_size: int, horizon: int):
        """Sample short in-episode trajectories for the rollout / drift loss.

        Returns a dict:
          cond:    uint8 (B, k, H, W, C)        seed stack (oldest..newest)
          actions: int64 (B, horizon) | float32 (B, horizon, adim)
          targets: uint8 (B, horizon, H, W, C)  ground-truth next frames
        Each trajectory lies within a single episode and is physically intact.
        """
        k, cap, hz = self.k, self.capacity, horizon
        window = k + hz  # frames from seed-start to final target inclusive
        chosen = np.empty(batch_size, dtype=np.int64)
        filled = 0
        attempts = 0
        while filled < batch_size:
            cand = self._occupied_indices(batch_size * 3)
            ok = self.ep_step[cand] >= (window - 1)
            base = self.wid[cand]
            for m in range(1, window):
                ok &= self.wid[(cand - m) % cap] == (base - m)
            valid = cand[ok]
            take = min(len(valid), batch_size - filled)
            chosen[filled:filled + take] = valid[:take]
            filled += take
            attempts += 1
            if attempts > 2000:
                raise RuntimeError("could not sample valid sequences; increase buffer / lower horizon")

        # final target = chosen (p). seed cond = k frames ending at p-hz.
        seed_off = np.arange(window - 1, hz - 1, -1, dtype=np.int64)  # p-(k+hz-1) .. p-hz
        cond_idx = (chosen[:, None] - seed_off[None, :]) % cap  # (B, k) oldest..newest
        step_off = np.arange(hz - 1, -1, -1, dtype=np.int64)  # p-(hz-1) .. p
        tgt_idx = (chosen[:, None] - step_off[None, :]) % cap  # (B, hz) first..last

        cond = self.frames[cond_idx]
        targets = self.frames[tgt_idx]
        if self.discrete:
            actions = self.actions[tgt_idx, 0].astype(np.int64)
        else:
            actions = self.actions[tgt_idx].astype(np.float32)
        return {"cond": cond, "actions": actions, "targets": targets}

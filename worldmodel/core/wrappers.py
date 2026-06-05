"""Observation wrappers. The only core wrapper is resize-to-canonical."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .contract import ActionSpace, Game


class ResizeToCanonical:
    """Wrap a game so its frames are square ``resolution x resolution x 3`` uint8.

    Conditioning and the model run at this canonical resolution regardless of
    the game's native render size. Action space and episode semantics pass
    through unchanged.
    """

    def __init__(self, game: Game, resolution: int, mode: str = "bilinear"):
        self.game = game
        self.resolution = int(resolution)
        self._resample = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "area": Image.BOX,
        }.get(mode, Image.BILINEAR)
        self.action_space: ActionSpace = game.action_space
        self.obs_shape = (self.resolution, self.resolution, 3)
        self.name = getattr(game, "name", "game")

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        if frame.shape[2] == 4:  # drop alpha
            frame = frame[:, :, :3]
        if frame.shape[:2] == (self.resolution, self.resolution):
            return np.ascontiguousarray(frame.astype(np.uint8))
        img = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        img = img.resize((self.resolution, self.resolution), self._resample)
        return np.asarray(img, dtype=np.uint8)

    def reset(self) -> np.ndarray:
        return self._resize(self.game.reset())

    def step(self, action):
        frame, done = self.game.step(action)
        return self._resize(frame), bool(done)

    # Let games expose extra helpers (e.g. make_actor) through the wrapper.
    def __getattr__(self, item):
        return getattr(self.game, item)

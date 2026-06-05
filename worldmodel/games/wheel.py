"""Wheel game: side view of a car's front wheel that rotates with speed.

You see the front-bottom of a car: a body/fender and a spoked front wheel sitting
on a dashed road. Accelerate forward or backward to spin the wheel up (the road
scrolls to convey motion), or brake to bring it to a stop. Momentum + rolling
friction; no goal, no reward.

Dynamics the world model has to learn from pixels: the wheel angle integrates
speed, the road dashes scroll at speed, and the three controls change speed.

Actions (Discrete 4): 0 coast, 1 forward, 2 backward, 3 brake.
Keys in the players: W/Up forward, S/Down backward, Space brake.

Pure numpy plus a simple rasterizer.
"""

from __future__ import annotations

import numpy as np

from worldmodel.core.contract import Discrete
from worldmodel.core.registry import register

COAST, FORWARD, BACKWARD, BRAKE = range(4)

PALETTES = {
    "auto": {
        "sky": [134, 178, 214],
        "ground": [54, 58, 66],
        "dash": [226, 202, 110],
        "body": [206, 72, 72],
        "tire": [24, 24, 28],
        "rim": [206, 210, 220],
        "hub": [70, 74, 84],
        "spoke": [188, 194, 206],
        "marker": [250, 214, 92],
    },
    "lime": {
        "sky": [30, 34, 44],
        "ground": [40, 44, 52],
        "dash": [120, 130, 150],
        "body": [120, 210, 90],
        "tire": [18, 18, 22],
        "rim": [210, 216, 226],
        "hub": [60, 64, 74],
        "spoke": [200, 206, 218],
        "marker": [250, 120, 90],
    },
}


@register("wheel")
class WheelGame:
    name = "wheel"
    space_action = BRAKE  # players map the space bar to this action

    @staticmethod
    def default_config() -> dict:
        return {
            "resolution": 64,
            "episode_timeout": 300,
            "accel": 0.35,
            "friction": 0.985,
            "brake_factor": 0.82,
            "max_speed": 6.0,
            "wheel_radius": 0.23,   # fraction of frame size
            "num_spokes": 5,
            "palette": "auto",
        }

    def __init__(self, config: dict | None = None):
        cfg = {**WheelGame.default_config(), **(config or {})}
        self.size = s = int(cfg["resolution"])
        self.scale = s / 64.0
        self.accel = float(cfg["accel"]) * self.scale
        self.friction = float(cfg["friction"])
        self.brake_factor = float(cfg["brake_factor"])
        self.max_speed = float(cfg["max_speed"]) * self.scale
        self.R = float(cfg["wheel_radius"]) * s
        self.num_spokes = int(cfg["num_spokes"])

        pal = PALETTES.get(cfg.get("palette", "auto"), PALETTES["auto"])
        self.col = {k: np.array(cfg.get(f"{k}_color", v), dtype=np.uint8) for k, v in pal.items()}

        # geometry
        self.ground_y = int(s * 0.74)
        self.cx = s * 0.5
        self.cy = self.ground_y - self.R * 0.78
        self.dash_spacing = max(6, int(s * 0.22))

        self.action_space = Discrete(4)
        self.obs_shape = (s, s, 3)

        self.rng = np.random.default_rng(int(cfg.get("seed", 0)) + 99)
        self.v = 0.0
        self.theta = 0.0
        self.offset = 0.0

    def reset(self) -> np.ndarray:
        self.v = float(self.rng.uniform(-3.0, 3.0)) * self.scale
        self.theta = float(self.rng.uniform(0, 2 * np.pi))
        self.offset = float(self.rng.uniform(0, self.dash_spacing))
        return self._render()

    def step(self, action) -> tuple[np.ndarray, bool]:
        a = int(action)
        if a == FORWARD:
            self.v += self.accel
        elif a == BACKWARD:
            self.v -= self.accel
        elif a == BRAKE:
            self.v *= self.brake_factor
            if abs(self.v) < 0.04 * self.scale:
                self.v = 0.0
        else:  # coast
            self.v *= self.friction

        self.v = float(np.clip(self.v, -self.max_speed, self.max_speed))
        self.theta += self.v / max(self.R, 1e-3)  # rolling: omega = v / R
        self.offset = (self.offset + self.v) % self.dash_spacing
        return self._render(), False

    # ------------------------------------------------------------------ render
    def _disk(self, img, cx, cy, r, color):
        s = self.size
        x0, x1 = max(0, int(cx - r - 1)), min(s, int(cx + r + 2))
        y0, y1 = max(0, int(cy - r - 1)), min(s, int(cy + r + 2))
        if x1 <= x0 or y1 <= y0:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1]
        mask = (xs + 0.5 - cx) ** 2 + (ys + 0.5 - cy) ** 2 <= r * r
        img[y0:y1, x0:x1][mask] = color

    def _spoke(self, img, ang, r0, r1, color, thick):
        n = int((r1 - r0) * 2) + 3
        for t in np.linspace(r0, r1, n):
            x = int(round(self.cx + t * np.cos(ang)))
            y = int(round(self.cy + t * np.sin(ang)))
            self._disk(img, x + 0.5, y + 0.5, thick, color)

    def _render(self) -> np.ndarray:
        s = self.size
        img = np.empty((s, s, 3), dtype=np.uint8)
        img[: self.ground_y] = self.col["sky"]
        img[self.ground_y:] = self.col["ground"]

        # scrolling road dashes (motion cue), drawn just below the ground line
        dy0 = self.ground_y + max(2, int(s * 0.05))
        dy1 = dy0 + max(2, int(s * 0.05))
        dash_w = max(3, int(self.dash_spacing * 0.5))
        start = -int(self.offset)
        for x in range(start, s, self.dash_spacing):
            x0, x1 = max(0, x), min(s, x + dash_w)
            if x1 > x0:
                img[dy0:dy1, x0:x1] = self.col["dash"]

        # car body / fender above the wheel (suggests the front of a car)
        bx0, bx1 = int(s * 0.12), int(s * 0.88)
        by0, by1 = int(s * 0.16), int(self.cy - self.R * 0.15)
        img[max(0, by0):max(0, by1), bx0:bx1] = self.col["body"]
        # a sloped hood toward the front (right side)
        for i, x in enumerate(range(int(s * 0.62), bx1)):
            top = by0 - int((x - s * 0.62) * 0.35)
            img[max(0, top):by0, x:x + 1] = self.col["body"]

        # wheel: tire, rim, spokes, hub
        self._disk(img, self.cx, self.cy, self.R, self.col["tire"])
        self._disk(img, self.cx, self.cy, self.R * 0.62, self.col["rim"])
        thick = max(0.6, 0.9 * self.scale)
        for k in range(self.num_spokes):
            ang = self.theta + k * 2 * np.pi / self.num_spokes
            color = self.col["marker"] if k == 0 else self.col["spoke"]
            self._spoke(img, ang, self.R * 0.16, self.R * 0.6, color, thick)
        self._disk(img, self.cx, self.cy, self.R * 0.17, self.col["hub"])
        return img

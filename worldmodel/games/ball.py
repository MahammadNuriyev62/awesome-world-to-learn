"""Ball game: a ball rolling with momentum in a top-down arena with random walls.

Open arena, no goal, no reward. A ball is pushed by discrete impulses, carries
momentum, damps with friction, and bounces off / stops against randomly placed
rectangular walls and the arena border. Every reset randomizes the ball state
and the wall layout.

Pure numpy plus a simple rasterizer (no pymunk required).
"""

from __future__ import annotations

import numpy as np

from worldmodel.core.contract import Discrete
from worldmodel.core.registry import register

# action ids
NOOP, UP, DOWN, LEFT, RIGHT = range(5)
_IMPULSE = {
    NOOP: (0.0, 0.0),
    UP: (0.0, -1.0),
    DOWN: (0.0, 1.0),
    LEFT: (-1.0, 0.0),
    RIGHT: (1.0, 0.0),
}

# named color palettes: (background, wall, ball) as RGB
PALETTES = {
    "dusk": ([18, 18, 24], [120, 124, 140], [235, 80, 60]),       # original
    "mint": ([20, 28, 30], [70, 150, 140], [245, 225, 120]),      # teal walls, gold ball
    "grape": ([22, 18, 34], [128, 96, 210], [120, 240, 170]),     # purple walls, mint ball
    "ocean": ([14, 24, 40], [52, 116, 168], [250, 196, 84]),      # blue walls, amber ball
    "paper": ([238, 236, 228], [150, 160, 180], [222, 86, 70]),   # light theme
    "candy": ([26, 22, 36], [236, 110, 168], [120, 220, 244]),    # pink walls, cyan ball
}


def _resolve_palette(cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bg, wall, ball = PALETTES.get(cfg.get("palette", "ocean"), PALETTES["ocean"])
    bg = cfg.get("bg_color", bg)
    wall = cfg.get("wall_color", wall)
    ball = cfg.get("ball_color", ball)
    to = lambda c: np.array(c, dtype=np.uint8)
    return to(bg), to(wall), to(ball)


@register("ball")
class BallGame:
    name = "ball"

    @staticmethod
    def default_config() -> dict:
        return {
            "resolution": 64,
            "episode_timeout": 300,
            # ball-specific
            "ball_radius": 4.0,
            "impulse": 0.9,
            "damping": 0.92,
            "edge_restitution": 0.75,
            "wall_restitution": 0.45,
            "min_walls": 1,
            "max_walls": 4,
            "max_speed": 6.0,
            # appearance: a named palette, or override bg_color/wall_color/ball_color
            "palette": "ocean",
        }

    def __init__(self, config: dict | None = None):
        cfg = {**BallGame.default_config(), **(config or {})}
        self.size = int(cfg["resolution"])
        self.r = float(cfg["ball_radius"]) * self.size / 64.0
        self.impulse = float(cfg["impulse"]) * self.size / 64.0
        self.damping = float(cfg["damping"])
        self.edge_rest = float(cfg["edge_restitution"])
        self.wall_rest = float(cfg["wall_restitution"])
        self.min_walls = int(cfg["min_walls"])
        self.max_walls = int(cfg["max_walls"])
        self.max_speed = float(cfg["max_speed"]) * self.size / 64.0

        self.bg, self.wall_color, self.ball_color = _resolve_palette(cfg)

        self.action_space = Discrete(5)
        self.obs_shape = (self.size, self.size, 3)

        seed = int(cfg.get("seed", 0))
        self.rng = np.random.default_rng(seed + 12345)

        self.pos = np.zeros(2, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.walls: list[tuple[float, float, float, float]] = []

    # ------------------------------------------------------------------ layout
    def _random_walls(self) -> list[tuple[float, float, float, float]]:
        n = int(self.rng.integers(self.min_walls, self.max_walls + 1))
        walls = []
        s = self.size
        for _ in range(n):
            w = self.rng.uniform(s * 0.12, s * 0.35)
            h = self.rng.uniform(s * 0.12, s * 0.35)
            x0 = self.rng.uniform(s * 0.1, s * 0.9 - w)
            y0 = self.rng.uniform(s * 0.1, s * 0.9 - h)
            walls.append((x0, y0, x0 + w, y0 + h))
        return walls

    def _inside_any_wall(self, p: np.ndarray, pad: float) -> bool:
        for (x0, y0, x1, y1) in self.walls:
            if (x0 - pad) <= p[0] <= (x1 + pad) and (y0 - pad) <= p[1] <= (y1 + pad):
                return True
        return False

    def _random_free_pos(self) -> np.ndarray:
        lo, hi = self.r + 1.0, self.size - self.r - 1.0
        for _ in range(200):
            p = self.rng.uniform(lo, hi, size=2)
            if not self._inside_any_wall(p, pad=self.r + 1.0):
                return p
        return np.array([self.size * 0.5, self.size * 0.5])

    def reset(self) -> np.ndarray:
        self.walls = self._random_walls()
        self.pos = self._random_free_pos()
        # small random initial velocity for momentum diversity
        self.vel = self.rng.uniform(-2.0, 2.0, size=2) * self.size / 64.0
        return self._render()

    # ------------------------------------------------------------------ physics
    def _resolve_walls(self) -> None:
        r = self.r
        for _ in range(2):  # a couple of passes for corners / multiple walls
            for (x0, y0, x1, y1) in self.walls:
                ex0, ey0, ex1, ey1 = x0 - r, y0 - r, x1 + r, y1 + r
                px, py = self.pos
                if not (ex0 < px < ex1 and ey0 < py < ey1):
                    continue
                # penetration depth to each expanded side; push out the least one
                pen_left = px - ex0
                pen_right = ex1 - px
                pen_top = py - ey0
                pen_bottom = ey1 - py
                m = min(pen_left, pen_right, pen_top, pen_bottom)
                if m == pen_left:
                    self.pos[0] = ex0
                    self.vel[0] = -abs(self.vel[0]) * self.wall_rest
                elif m == pen_right:
                    self.pos[0] = ex1
                    self.vel[0] = abs(self.vel[0]) * self.wall_rest
                elif m == pen_top:
                    self.pos[1] = ey0
                    self.vel[1] = -abs(self.vel[1]) * self.wall_rest
                else:
                    self.pos[1] = ey1
                    self.vel[1] = abs(self.vel[1]) * self.wall_rest

    def _resolve_edges(self) -> None:
        r = self.r
        lo, hi = r, self.size - r
        if self.pos[0] < lo:
            self.pos[0] = lo
            self.vel[0] = abs(self.vel[0]) * self.edge_rest
        elif self.pos[0] > hi:
            self.pos[0] = hi
            self.vel[0] = -abs(self.vel[0]) * self.edge_rest
        if self.pos[1] < lo:
            self.pos[1] = lo
            self.vel[1] = abs(self.vel[1]) * self.edge_rest
        elif self.pos[1] > hi:
            self.pos[1] = hi
            self.vel[1] = -abs(self.vel[1]) * self.edge_rest

    def step(self, action) -> tuple[np.ndarray, bool]:
        a = int(action)
        ix, iy = _IMPULSE.get(a, (0.0, 0.0))
        self.vel[0] += ix * self.impulse
        self.vel[1] += iy * self.impulse

        speed = float(np.hypot(*self.vel))
        if speed > self.max_speed:
            self.vel *= self.max_speed / speed

        self.pos = self.pos + self.vel
        self._resolve_walls()
        self._resolve_edges()
        self.vel *= self.damping

        # endless game: episode boundary is the framework timeout
        return self._render(), False

    # ------------------------------------------------------------------ render
    def _render(self) -> np.ndarray:
        s = self.size
        img = np.empty((s, s, 3), dtype=np.uint8)
        img[:] = self.bg

        for (x0, y0, x1, y1) in self.walls:
            xi0, yi0 = int(round(x0)), int(round(y0))
            xi1, yi1 = int(round(x1)), int(round(y1))
            xi0, yi0 = max(0, xi0), max(0, yi0)
            xi1, yi1 = min(s, xi1), min(s, yi1)
            if xi1 > xi0 and yi1 > yi0:
                img[yi0:yi1, xi0:xi1] = self.wall_color

        # ball as an anti-aliased-ish filled circle
        cx, cy = self.pos
        r = self.r
        x_lo, x_hi = max(0, int(cx - r - 1)), min(s, int(cx + r + 2))
        y_lo, y_hi = max(0, int(cy - r - 1)), min(s, int(cy + r + 2))
        if x_hi > x_lo and y_hi > y_lo:
            ys, xs = np.mgrid[y_lo:y_hi, x_lo:x_hi]
            dist = np.hypot(xs + 0.5 - cx, ys + 0.5 - cy)
            mask = dist <= r
            img[y_lo:y_hi, x_lo:x_hi][mask] = self.ball_color
        return img

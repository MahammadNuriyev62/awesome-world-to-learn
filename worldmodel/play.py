"""Play a game live in the terminal (no GUI needed).

Renders frames with truecolor half-block characters and reads the keyboard in
raw mode, so it works over SSH / in the VSCode integrated terminal.

    python -m worldmodel.play --game ball
    python -m worldmodel.play --game gym:ALE/Breakout-v5 --resize 96

Controls:
    WASD or arrow keys : move (ball: up/left/down/right)
    0-9                : pick a discrete action by index (any game)
    space              : no-op / action 0
    r                  : reset the episode
    q or ctrl-c        : quit

Continuous (Box) games map arrow keys to nudges on the first action dim.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import time

import numpy as np

from worldmodel.core.contract import Box, Discrete
from worldmodel.core.registry import load_game
from worldmodel.core.config import build_config
from worldmodel.core.wrappers import ResizeToCanonical

# ball's known semantics, for nice WASD play
_BALL_KEYS = {"w": 1, "s": 2, "a": 3, "d": 4, " ": 0}
_ARROW = {"A": "up", "B": "down", "C": "right", "D": "left"}
_BALL_ARROW = {"up": 1, "down": 2, "left": 3, "right": 4}


def frame_to_ansi(img: np.ndarray, scale: int = 1) -> str:
    """Render an (H, W, 3) uint8 image as half-block truecolor text."""
    if scale > 1:
        img = np.repeat(np.repeat(img, scale, 0), scale, 1)
    H, W, _ = img.shape
    if H % 2:  # pad to even height
        img = np.concatenate([img, np.zeros((1, W, 3), np.uint8)], 0)
        H += 1
    out = []
    for y in range(0, H, 2):
        top, bot = img[y], img[y + 1]
        row = []
        for x in range(W):
            tr, tg, tb = top[x]
            br, bg, bb = bot[x]
            row.append(f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀")
        out.append("".join(row) + "\x1b[0m")
    return "\n".join(out)


def read_keys(fd: int, timeout: float) -> bytes:
    keys = b""
    r, _, _ = select.select([fd], [], [], timeout)
    while r:
        chunk = os.read(fd, 1024)
        if not chunk:
            break
        keys += chunk
        r, _, _ = select.select([fd], [], [], 0)
    return keys


def parse_keys(keys: bytes):
    """Return (tokens, quit, reset). tokens is a list of 'w'/'a'/.../'up'/'0'..."""
    tokens, quit_, reset = [], False, False
    i = 0
    while i < len(keys):
        b = keys[i]
        if b == 0x1B and keys[i + 1 : i + 2] == b"[":  # escape sequence (arrow)
            c = chr(keys[i + 2]) if i + 2 < len(keys) else ""
            if c in _ARROW:
                tokens.append(_ARROW[c])
            i += 3
            continue
        ch = chr(b).lower()
        if ch in ("q", "\x03"):
            quit_ = True
        elif ch == "r":
            reset = True
        elif ch in "wasd " or ch.isdigit():
            tokens.append(ch)
        i += 1
    return tokens, quit_, reset


class DiscreteController:
    def __init__(self, n: int, ball_like: bool, space_action: int = 0):
        self.n = n
        self.ball_like = ball_like
        self.space_action = int(space_action)
        self.last = 0

    def action(self, tokens):
        act = 0  # default no-op each frame (direct control)
        for tk in tokens:
            if tk in _BALL_ARROW and self.ball_like:
                act = _BALL_ARROW[tk]
            elif tk in ("up", "down", "left", "right"):
                # generic: map arrows to first few action ids
                act = {"up": 1, "down": 2, "left": 3, "right": 4}.get(tk, 0) % self.n
            elif tk == " ":
                act = self.space_action  # e.g. brake in the wheel game
            elif self.ball_like and tk in _BALL_KEYS:
                act = _BALL_KEYS[tk]
            elif tk.isdigit():
                act = min(int(tk), self.n - 1)
        self.last = act
        return act

    def label(self, a):
        names = {0: "noop", 1: "up", 2: "down", 3: "left", 4: "right"}
        return names.get(a, str(a)) if self.ball_like else str(a)


class BoxController:
    def __init__(self, box: Box):
        self.box = box
        self.vec = (0.5 * (box.low + box.high)).astype(np.float32)
        self.step = (0.34 * (box.high - box.low)).astype(np.float32)

    def action(self, tokens):
        for tk in tokens:
            if tk in ("up", "right") or tk in ("w", "d"):
                self.vec[0] = min(self.vec[0] + self.step[0], self.box.high[0])
            elif tk in ("down", "left") or tk in ("a", "s"):
                self.vec[0] = max(self.vec[0] - self.step[0], self.box.low[0])
            elif tk == " ":
                self.vec[:] = 0.5 * (self.box.low + self.box.high)
        return self.box.clip(self.vec)

    def label(self, a):
        return "[" + " ".join(f"{v:+.2f}" for v in np.atleast_1d(a)) + "]"


def _seed_action(action_space):
    if isinstance(action_space, Discrete):
        return 0
    return (0.5 * (action_space.low + action_space.high)).astype(np.float32)


class DreamEnv:
    """Play the trained world model itself: the U-Net hallucinates each frame.

    Seeded with k real frames, then driven by actions alone — the real game is
    discarded after seeding, so you are playing the model's learned dynamics.
    """

    def __init__(self, ckpt, device, sampler_steps=None, sampler=None):
        import torch
        from .core.eval import load_checkpoint

        self.torch = torch
        self.game, self.model, self.cfg, self.full = load_checkpoint(ckpt, device)
        if sampler_steps:
            self.model.sampler_steps = sampler_steps
        if sampler:
            self.model.sampler = sampler
        self.device = device
        self.k = self.cfg.frame_stack
        self.action_space = self.game.action_space
        self.discrete = isinstance(self.action_space, Discrete)
        self.space_action = getattr(self.game, "space_action", 0)
        self.name = f"DREAM:{getattr(self.game, 'name', 'game')}"

    def _seed_cond(self):
        from .core.utils import frames_to_model

        f = self.game.reset()
        seed = [f]
        sa = _seed_action(self.action_space)
        for _ in range(self.k - 1):
            f, _ = self.game.step(sa)
            seed.append(f)
        return frames_to_model(np.stack(seed[-self.k:]), self.device)[None]

    def _act_tensor(self, action):
        if self.discrete:
            return self.torch.tensor([int(action)], device=self.device, dtype=self.torch.long)
        return self.torch.tensor(np.asarray(action, np.float32)[None], device=self.device)

    def reset(self):
        from .core.utils import model_to_uint8

        self.cond = self._seed_cond()
        return model_to_uint8(self.cond[0, -1])

    def step(self, action):
        from .core.utils import model_to_uint8

        with self.torch.no_grad():
            nxt = self.model.sample_frame(self.cond, self._act_tensor(action))
        self.cond = self.torch.cat([self.cond[:, 1:], nxt[:, None]], dim=1)
        return model_to_uint8(nxt[0]), False


class CompareEnv(DreamEnv):
    """Step the real game and the model in lockstep on the same actions.

    Renders ``real | dream`` so you can watch the model track or drift.
    """

    def reset(self):
        from .core.utils import frames_to_model, model_to_uint8

        f = self.game.reset()
        seed = [f]
        sa = _seed_action(self.action_space)
        for _ in range(self.k - 1):
            f, _ = self.game.step(sa)
            seed.append(f)
        self.cond = frames_to_model(np.stack(seed[-self.k:]), self.device)[None]
        self.real_frame = seed[-1]
        return self._compose(self.real_frame, model_to_uint8(self.cond[0, -1]))

    def step(self, action):
        from .core.utils import model_to_uint8

        self.real_frame, _ = self.game.step(action)
        with self.torch.no_grad():
            nxt = self.model.sample_frame(self.cond, self._act_tensor(action))
        self.cond = self.torch.cat([self.cond[:, 1:], nxt[:, None]], dim=1)
        return self._compose(self.real_frame, model_to_uint8(nxt[0])), False

    @staticmethod
    def _compose(real, dream):
        H = real.shape[0]
        sep = np.full((H, 1, 3), 255, np.uint8)
        return np.concatenate([real, sep, dream], axis=1)


def build_env(args):
    """Construct the thing to play: the real game, the model dream, or both.

    ``args`` needs: game, model, compare, sampler_steps, sampler, device, resize.
    Returns (env, ball_like, cfg, full). The env exposes reset()/step(action)/
    action_space/name, so callers (terminal or web) treat all three uniformly.
    """
    if getattr(args, "model", None):
        Env = CompareEnv if getattr(args, "compare", False) else DreamEnv
        env = Env(args.model, args.device, args.sampler_steps, args.sampler)
        cfg, full = env.cfg, env.full
    else:
        spec = load_game(args.game)
        overrides = {}
        if getattr(args, "resolution", None):  # render the real game at higher res
            overrides["resolution"] = args.resolution
        cfg, full = build_config(spec.default_config, overrides)
        env = spec.make(full)
        if getattr(args, "resize", None):
            env = ResizeToCanonical(env, args.resize, mode="nearest")
    ball_like = "ball" in getattr(env, "name", "")
    return env, ball_like, cfg, full


def play(args):
    game, ball_like, cfg, full = build_env(args)
    if getattr(args, "model", None) and getattr(args, "compare", False):
        print("left = real game, right = model dream (same actions)")
    if isinstance(game.action_space, Discrete):
        ctrl = DiscreteController(game.action_space.n, ball_like, getattr(game, "space_action", 0))
    elif isinstance(game.action_space, Box):
        ctrl = BoxController(game.action_space)
    else:
        raise SystemExit("unsupported action space")

    if not sys.stdin.isatty():
        raise SystemExit("play.py needs an interactive terminal (a tty).")

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    frame_time = 1.0 / args.fps
    frame = game.reset()
    steps, t_last, fps_est = 0, time.time(), args.fps
    try:
        tty.setcbreak(fd)
        sys.stdout.write("\x1b[2J\x1b[?25l")  # clear + hide cursor
        while True:
            keys = read_keys(fd, frame_time)
            tokens, quit_, reset = parse_keys(keys)
            if quit_:
                break
            if reset:
                frame = game.reset()
                steps = 0
                continue
            a = ctrl.action(tokens)
            frame, done = game.step(a)
            steps += 1

            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 / max(now - t_last, 1e-6)
            t_last = now

            img = frame_to_ansi(frame, scale=args.scale)
            hud = (
                f"\x1b[0m{game.name if hasattr(game,'name') else args.game}  "
                f"step={steps}  action={ctrl.label(a)}  ~{fps_est:4.1f}fps   "
                f"[WASD/arrows move | 0-9 action | r reset | q quit]"
            )
            sys.stdout.write("\x1b[H" + img + "\n" + hud + "\x1b[K\n")
            sys.stdout.flush()
            if done:
                frame = game.reset()
                steps = 0
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\x1b[?25h\x1b[0m\n")  # show cursor
        sys.stdout.flush()
        if hasattr(game, "close"):
            game.close()


def main():
    p = argparse.ArgumentParser(description="Play a game (or a trained world model) in the terminal")
    p.add_argument("--game", default="ball")
    p.add_argument("--model", default=None, help="checkpoint to play the world model instead of the real game")
    p.add_argument("--compare", action="store_true", help="with --model: show real | dream side by side")
    p.add_argument("--sampler-steps", dest="sampler_steps", type=int, default=None,
                   help="override diffusion sampler steps (fewer = faster, for interactive play)")
    p.add_argument("--sampler", choices=["heun", "ddim"], default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--scale", type=int, default=1, help="integer upscale of the rendered frame")
    p.add_argument("--resize", type=int, default=None, help="resize frames to NxN (for big gym frames)")
    args = p.parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    play(args)


if __name__ == "__main__":
    main()

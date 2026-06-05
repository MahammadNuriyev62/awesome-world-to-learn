"""Autoregressive rollout evaluation vs ground truth. Game-agnostic.

Used both inside training (periodic videos + metrics) and standalone from a
checkpoint:

    python -m worldmodel.core.eval --ckpt runs/latest/model.pt --rollout 128
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .actor import make_actor
from .config import build_config
from .model import DiffusionWorldModel
from .registry import load_game
from .utils import comparison_frames, frames_to_model, model_to_uint8, save_video
from .wrappers import ResizeToCanonical


@torch.no_grad()
def rollout_eval(game, model: DiffusionWorldModel, cfg, full_cfg, device, length: int, rng=None):
    """Roll the model out autoregressively against the real game.

    Seeds with k real frames, then drives the model with the same action
    sequence the real game receives, so predicted and ground-truth frames are
    directly comparable.

    Returns (gt_frames, pred_frames, metrics) where frames are (T, H, W, C)
    uint8 over the predicted horizon.
    """
    model.eval()
    rng = rng or np.random.default_rng(cfg.seed + 777)
    actor = make_actor(game.action_space, full_cfg, rng)
    k = cfg.frame_stack
    timeout = full_cfg.get("episode_timeout", cfg.episode_timeout)

    # seed: k real frames from a fresh episode
    f = game.reset()
    actor.reset()
    seed = [f]
    steps = 1
    for _ in range(k - 1):
        a = actor()
        f, done = game.step(a)
        seed.append(f)
        steps += 1
        if done:  # very short episode; just restart
            return rollout_eval(game, model, cfg, full_cfg, device, length, rng)

    cond = frames_to_model(np.stack(seed[-k:]), device)[None]  # (1, k, C, H, W)

    gt, actions = [], []
    for _ in range(length):
        a = actor()
        f, done = game.step(a)
        actions.append(a)
        gt.append(f)
        steps += 1
        if done or steps >= timeout:
            break

    if model.cond_embed.discrete:
        act_t = torch.tensor(actions, device=device, dtype=torch.long)[None]  # (1, T)
    else:
        act_t = torch.tensor(np.stack(actions), device=device, dtype=torch.float32)[None]

    pred = model.imagine(cond, act_t)[0]  # (T, C, H, W)
    pred_u8 = model_to_uint8(pred)
    gt_u8 = np.stack(gt)

    # metrics in [-1, 1] space
    gt_t = frames_to_model(gt_u8, device)
    err = (pred[: len(gt_t)] - gt_t).abs()
    mse_per_step = ((pred[: len(gt_t)] - gt_t) ** 2).mean(dim=(1, 2, 3))
    # identity-shortcut probe: how much does prediction differ from the last seed frame?
    last_seed = cond[0, -1]
    motion = (pred[0] - last_seed).abs().mean().item()
    metrics = {
        "rollout_mae": err.mean().item(),
        "rollout_mse_first": mse_per_step[0].item(),
        "rollout_mse_last": mse_per_step[-1].item(),
        "pred_motion_first_step": motion,
    }
    return gt_u8, pred_u8[: len(gt_u8)], metrics


def run_eval_video(game, model, cfg, full_cfg, device, length, out_path, rng=None):
    gt, pred, metrics = rollout_eval(game, model, cfg, full_cfg, device, length, rng)
    frames = comparison_frames(gt, pred, scale=max(1, 256 // cfg.resolution))
    save_video(out_path, frames, fps=20)
    return metrics


def load_checkpoint(ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    spec = load_game(ckpt["game_spec"])
    cfg, full = build_config(spec.default_config, ckpt.get("cli_overrides", {}))
    game = ResizeToCanonical(spec.make(full), cfg.resolution)
    model = DiffusionWorldModel(game.obs_shape, game.action_space, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return game, model, cfg, full


def main():
    p = argparse.ArgumentParser(description="Autoregressive rollout from a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    game, model, cfg, full = load_checkpoint(args.ckpt, args.device)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "rollout.mp4")
    metrics = run_eval_video(game, model, cfg, full, args.device, args.rollout, out)
    print("rollout written to", out)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

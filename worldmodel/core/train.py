"""Game-agnostic online training loop.

Each env step pushes a real (prev frames, action, next frame) tuple into the
replay buffer; after warmup we take ``train_ratio`` gradient steps per env step
on uniform random minibatches from the buffer. The diffusion model is sampled
only during periodic eval rollouts, never inside the collection loop.

    python -m worldmodel.core.train --game ball
    python -m worldmodel.core.train --game gym:ALE/Breakout-v5
    python -m worldmodel.core.train --game ./my_game.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .actor import make_actor
from .buffer import FrameBuffer
from .config import build_config
from .eval import run_eval_video
from .model import DiffusionWorldModel
from .registry import load_game, registered_names
from .utils import Ema, frames_to_model
from .wrappers import ResizeToCanonical


# ---------------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(description="Online diffusion world-model trainer")
    p.add_argument("--game", required=True, help="name | gym:ENV | ./file.py | mod:Class")
    p.add_argument("--steps", dest="total_steps", type=int, default=None, help="total gradient steps")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--frame-stack", dest="frame_stack", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--train-ratio", dest="train_ratio", type=int, default=None)
    p.add_argument("--warmup", dest="warmup_frames", type=int, default=None)
    p.add_argument("--buffer-capacity", dest="buffer_capacity", type=int, default=None)
    p.add_argument("--episode-timeout", dest="episode_timeout", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-every", dest="eval_every", type=int, default=None)
    p.add_argument("--rollout-length", dest="rollout_length", type=int, default=None)
    p.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=None)
    p.add_argument("--rollout-loss-weight", dest="rollout_loss_weight", type=float, default=None)
    p.add_argument("--amp", dest="amp_dtype", choices=["fp16", "bf16", "fp32"], default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-dir", dest="run_dir", default=None)
    p.add_argument("--wandb", action="store_true", default=None)
    p.add_argument("--set", action="append", default=[], metavar="key=value",
                   help="override any config field, repeatable")
    return p.parse_args()


def cli_overrides_from_args(args) -> dict:
    overrides = {}
    for k, v in vars(args).items():
        if k in ("game", "set") or v is None:
            continue
        overrides[k] = v
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        overrides[key] = val
    return overrides


# ---------------------------------------------------------------------- helpers
def amp_context(amp_dtype: str, device: str):
    if device != "cuda" or amp_dtype == "fp32":
        return torch.autocast("cuda", enabled=False)
    dt = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dt)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------- training
def train(args):
    cli = cli_overrides_from_args(args)
    spec = load_game(args.game)
    cfg, full = build_config(spec.default_config, cli)

    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    set_seed(cfg.seed)
    os.makedirs(cfg.run_dir, exist_ok=True)

    game = ResizeToCanonical(spec.make(full), cfg.resolution)
    print(f"game={spec.display_name}  obs={game.obs_shape}  action={game.action_space}")

    rng = np.random.default_rng(cfg.seed)
    actor = make_actor(game.action_space, full, rng)
    buffer = FrameBuffer(cfg.buffer_capacity, game.obs_shape, cfg.frame_stack, game.action_space, cfg.seed)
    model = DiffusionWorldModel(game.obs_shape, game.action_space, cfg).to(device)
    ema = Ema(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    use_fp16_scaler = cfg.amp_dtype == "fp16" and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"diffusion=EDM  params={nparams:.1f}M  device={device}  amp={cfg.amp_dtype}")
    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump(full, f, indent=2, default=str)

    run = None
    if cfg.wandb:
        try:
            import wandb
            run = wandb.init(project=cfg.wandb_project, config=full, dir=cfg.run_dir)
        except Exception as e:  # pragma: no cover
            print("wandb disabled:", e)

    # -------------------------------------------------------- collection helpers
    timeout = full.get("episode_timeout", cfg.episode_timeout)
    ep_len = 0

    def reset_env():
        nonlocal ep_len
        f = game.reset()
        actor.reset()
        buffer.add_reset(f)
        ep_len = 0

    def env_step():
        nonlocal ep_len
        a = actor()
        f, done = game.step(a)
        buffer.add_step(a, f)
        ep_len += 1
        if done or ep_len >= timeout:
            reset_env()

    reset_env()

    # ----------------------------------------------------------------- main loop
    def train_step() -> dict:
        batch = buffer.sample(cfg.batch_size)
        cond = frames_to_model(batch["cond"], device)
        target = frames_to_model(batch["target"], device)
        if buffer.discrete:
            action = torch.from_numpy(batch["action"]).to(device).long()
        else:
            action = torch.from_numpy(batch["action"]).to(device).float()

        opt.zero_grad(set_to_none=True)
        with amp_context(cfg.amp_dtype, device):
            loss = model.denoise_loss(cond, action, target)
            logs = {"loss": loss.item()}

            use_roll = (
                cfg.rollout_loss_weight > 0
                and gstep >= cfg.rollout_loss_after
                and buffer.size > cfg.frame_stack + cfg.rollout_loss_horizon + 2
            )
            if use_roll:
                rl = rollout_loss_term()
                loss = loss + cfg.rollout_loss_weight * rl
                logs["rollout_loss"] = rl.item()

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        ema.update(model)
        return logs

    def rollout_loss_term():
        seq = buffer.sample_sequence(cfg.batch_size, cfg.rollout_loss_horizon)
        cond = frames_to_model(seq["cond"], device)
        targets = frames_to_model(seq["targets"], device)  # (B, H, C, H, W)
        if buffer.discrete:
            actions = torch.from_numpy(seq["actions"]).to(device).long()
        else:
            actions = torch.from_numpy(seq["actions"]).to(device).float()
        c = cond
        total = 0.0
        for t in range(cfg.rollout_loss_horizon):
            pred = model.rollout_predict(c, actions[:, t])
            total = total + F.mse_loss(pred, targets[:, t])
            c = torch.cat([c[:, 1:], pred.detach()[:, None]], dim=1)  # scheduled sampling
        return total / cfg.rollout_loss_horizon

    gstep = 0
    env_steps = 0
    running = {}
    t0 = time.time()
    train_start = None  # wall-clock when the first grad step runs (excludes warmup)
    append_jsonl(os.path.join(cfg.run_dir, "metrics.jsonl"),
                 {"phase": "start", "t": round(t0, 1), "total_steps": cfg.total_steps,
                  "resolution": cfg.resolution, "frame_stack": cfg.frame_stack})
    pbar = tqdm(total=cfg.total_steps, desc="train", dynamic_ncols=True)
    while gstep < cfg.total_steps:
        env_step()
        env_steps += 1

        if buffer.size < cfg.warmup_frames or not buffer.can_sample():
            if env_steps % 500 == 0:
                pbar.set_postfix_str(f"warmup {buffer.size}/{cfg.warmup_frames}")
                append_jsonl(os.path.join(cfg.run_dir, "metrics.jsonl"),
                             {"step": 0, "phase": "warmup", "frames": buffer.size, "target": cfg.warmup_frames})
            continue

        for _ in range(cfg.train_ratio):
            if gstep >= cfg.total_steps:
                break
            if train_start is None:
                train_start = time.time()
            logs = train_step()
            for k, v in logs.items():
                running[k] = 0.98 * running.get(k, v) + 0.02 * v
            gstep += 1
            pbar.update(1)

            if gstep % cfg.log_every == 0:
                now = time.time()
                sps = gstep / max(now - train_start, 1e-6)
                eta_min = (cfg.total_steps - gstep) / max(sps, 1e-6) / 60.0
                msg = " ".join(f"{k}={running[k]:.4f}" for k in running)
                pbar.set_postfix_str(f"{msg} buf={buffer.size} {sps:.1f}it/s eta~{eta_min:.0f}m")
                append_jsonl(
                    os.path.join(cfg.run_dir, "metrics.jsonl"),
                    {"step": gstep, "phase": "train", "t": round(now, 1),
                     "elapsed_s": round(now - train_start, 1), "it_s": round(sps, 2),
                     "eta_min": round(eta_min, 1), "buffer": buffer.size,
                     **{k: round(float(v), 6) for k, v in running.items()}},
                )
                if run:
                    run.log({**running, "buffer": buffer.size, "env_steps": env_steps}, step=gstep)

            if gstep % cfg.eval_every == 0 or gstep == cfg.total_steps:
                evaluate(game, ema.module(), cfg, full, device, gstep, run)
            if gstep % cfg.ckpt_every == 0 or gstep == cfg.total_steps:
                save_checkpoint(cfg, full, args.game, ema.module(), cli, gstep)
    pbar.close()
    if run:
        run.finish()
    print("done. run dir:", cfg.run_dir)


def append_jsonl(path: str, obj: dict) -> None:
    """Append one metrics record as a JSON line. Tailable live progress log."""
    try:
        with open(path, "a") as f:
            f.write(json.dumps(obj) + "\n")
    except OSError:
        pass


def evaluate(game, model, cfg, full, device, gstep, run=None):
    out = os.path.join(cfg.run_dir, f"rollout_{gstep:07d}.mp4")
    try:
        metrics = run_eval_video(game, model, cfg, full, device, cfg.rollout_length, out)
        print(f"\n[eval @ {gstep}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()) + f"  -> {out}")
        append_jsonl(os.path.join(cfg.run_dir, "metrics.jsonl"), {"step": gstep, "phase": "eval", **metrics})
        if run:
            run.log({f"eval/{k}": v for k, v in metrics.items()}, step=gstep)
    except Exception as e:  # pragma: no cover
        print(f"\n[eval @ {gstep}] failed: {e}")


def save_checkpoint(cfg, full, game_spec, model, cli, gstep):
    path = os.path.join(cfg.run_dir, "model.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "config": {k: getattr(cfg, k) for k in vars(cfg)},
            "full_config": full,
            "cli_overrides": cli,
            "game_spec": game_spec,
            "step": gstep,
        },
        path,
    )


def main():
    args = parse_args()
    try:
        train(args)
    except KeyError as e:
        print(e)
        print("registered games:", registered_names())
        raise SystemExit(1)


if __name__ == "__main__":
    main()

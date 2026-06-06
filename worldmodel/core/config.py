"""Configuration dataclass and the base < game < CLI merge.

``Config`` holds the core training/model defaults that are identical for every
game. A game's ``default_config()`` may override any of these (e.g. resolution)
and may also add game-specific keys; those extra keys are preserved in the full
merged dict and handed to the game factory, so the core stays game-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class Config:
    # --- canonical observation / conditioning ---
    resolution: int = 64  # canonical square resolution; use 32 for fast iteration
    frame_stack: int = 4

    # --- replay buffer ---
    buffer_capacity: int = 100_000
    warmup_frames: int = 5_000

    # --- optimisation ---
    batch_size: int = 64
    train_ratio: int = 2  # gradient steps per env step, after warmup
    lr: float = 2e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    grad_clip: float = 1.0

    # --- exploration actor ---
    action_repeat: int = 8  # discrete: hold each random action this many frames
    ou_theta: float = 0.15  # continuous: OU mean-reversion
    ou_sigma: float = 0.3
    episode_timeout: int = 300

    # --- U-Net ---
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (16,)
    dropout: float = 0.0
    cond_embed_dim: int = 256

    # --- EDM diffusion ---
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    p_mean: float = -1.2
    p_std: float = 1.2
    rho: float = 7.0
    sampler: str = "heun"  # "heun" or "ddim"
    sampler_steps: int = 18

    # --- multi-step rollout loss (drift control, milestone 4) ---
    rollout_loss_weight: float = 0.0  # 0 disables; ~0.1-0.5 to enable
    rollout_loss_horizon: int = 4
    rollout_loss_after: int = 20_000  # only enable after this many steps

    # --- schedule ---
    total_steps: int = 100_000
    eval_every: int = 5_000
    rollout_length: int = 64
    ckpt_every: int = 5_000
    log_every: int = 100

    # --- runtime ---
    seed: int = 0
    amp_dtype: str = "fp16"  # "fp16" (native on T4), "bf16", or "fp32"
    device: str = "cuda"
    run_dir: str = "runs/latest"
    wandb: bool = False
    wandb_project: str = "worldmodel"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=_jsonable)


def _jsonable(o: Any):
    if isinstance(o, tuple):
        return list(o)
    return str(o)


_TUPLE_FIELDS = {"channel_mults", "attn_resolutions"}


def _coerce(name: str, value: Any, current: Any) -> Any:
    """Best-effort coercion of CLI strings to the field's type."""
    if name in _TUPLE_FIELDS and isinstance(value, str):
        return tuple(int(x) for x in value.replace(" ", "").split(",") if x)
    if isinstance(current, bool) and isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "y")
    if isinstance(current, int) and not isinstance(current, bool) and isinstance(value, str):
        return int(value)
    if isinstance(current, float) and isinstance(value, str):
        return float(value)
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def build_config(game_defaults: dict | None, cli_overrides: dict | None) -> tuple[Config, dict]:
    """Merge base < game < CLI. Returns (Config, full_merged_dict).

    The full dict keeps game-specific keys (unknown to Config) so the game
    factory can read them.
    """
    base = asdict(Config())
    game_defaults = dict(game_defaults or {})
    cli_overrides = {k: v for k, v in (cli_overrides or {}).items() if v is not None}

    merged: dict = {**base, **game_defaults, **cli_overrides}

    known = {f.name for f in fields(Config)}
    kwargs = {}
    defaults = Config()
    for k in known:
        if k in merged:
            kwargs[k] = _coerce(k, merged[k], getattr(defaults, k))
    cfg = Config(**kwargs)

    # Reflect coerced known values back into the full dict for the record/game.
    for k in known:
        merged[k] = getattr(cfg, k)
    return cfg, merged

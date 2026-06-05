# Lightweight Online Diffusion World Model (multi-game)

A small, game-agnostic trainer that learns a diffusion world model **on the fly**.
A random actor plays a real game live, frames stream into a replay buffer, and a
conditional U-Net trains concurrently to predict the next frame given the recent
frames and an action. There is no task and no reward. One world model per game.

Target hardware is a single NVIDIA T4 (16 GB). Everything is small and uses mixed
precision.

## Install

```bash
pip install torch torchvision numpy einops imageio imageio-ffmpeg tqdm gymnasium pillow
# optional extras
pip install wandb                       # logging
pip install "gymnasium[classic-control]"  # CartPole / Pendulum rendering (pygame)
pip install ale-py                      # Atari via the ALE
```

## Quick start

```bash
# train the built-in ball game
python -m worldmodel.core.train --game ball

# fast iteration at 32x32
python -m worldmodel.core.train --game ball --resolution 32 --steps 20000

# the deterministic MSE baseline (milestone 1) instead of diffusion
python -m worldmodel.core.train --game ball --objective regression

# any gymnasium env through the adapter (proves the core is not ball-specific)
python -m worldmodel.core.train --game gym:Pendulum-v1        # continuous action
python -m worldmodel.core.train --game gym:CartPole-v1        # discrete action
python -m worldmodel.core.train --game gym:ALE/Breakout-v5    # needs ale-py

# a custom single-file game
python -m worldmodel.core.train --game ./my_game.py
python -m worldmodel.core.train --game worldmodel.games.ball:BallGame

# standalone rollout video from a checkpoint
python -m worldmodel.core.eval --ckpt runs/latest/model.pt --rollout 128
```

During training, periodic side-by-side rollout videos (`runs/<name>/rollout_*.mp4`,
ground truth | prediction) and a checkpoint (`model.pt`) are written to the run dir.

## How it works

The training loop is identical for every game:

1. Step the real game with a temporally-correlated random actor (sticky-random
   for discrete actions, OU noise for continuous).
2. Push each `(prev frames, action, next frame)` tuple into a uint8 ring buffer
   that respects episode boundaries.
3. After warmup, take `train_ratio` gradient steps per env step on **uniform
   random** minibatches from the buffer.
4. Periodically run an autoregressive rollout vs ground truth and save a video.

The diffusion model is **only sampled during eval and rollouts, never inside the
data-collection loop** — all training targets come from the real game.

### The Game contract

A game is any object with:

- `action_space`: `Discrete(n)` or `Box(shape, low, high)`
- `obs_shape`: native `(H, W, C)`
- `reset() -> frame` (randomizes the initial state)
- `step(action) -> (frame, done)`
- optional `name` and `default_config()`

The model is built per run from `obs_shape` and `action_space`. Only the
action-conditioning head adapts (an embedding table for `Discrete`, a small MLP
for `Box`); the U-Net backbone is fixed. Add a game by implementing the contract
in one file and registering it with `@register("name")`, or wrap any gymnasium
env with `GymGame`.

## Layout

```
worldmodel/
  core/
    contract.py   # Game protocol, Discrete/Box action spaces
    registry.py   # @register + load by name / gym: / path / module:Class
    config.py     # base config + merge (base < game < CLI)
    wrappers.py   # resize-to-canonical wrapper
    buffer.py     # episode-aware uint8 ring buffer
    actor.py      # make_actor: sticky (discrete) / OU (continuous)
    model.py      # DiffusionWorldModel: U-Net + adaptive action head (EDM + MSE)
    utils.py      # image<->tensor, EMA, video helpers
    train.py      # game-agnostic online loop
    eval.py       # autoregressive rollout vs ground truth
  games/
    ball.py       # the first game (registers "ball")
    gym_adapter.py# GymGame(env_id) wrapping any gymnasium env
```

## Key defaults (override per game via `default_config`, or on the CLI)

| setting | default | flag |
| --- | --- | --- |
| canonical resolution | 64 (use 32 for speed) | `--resolution` |
| frame stack | 4 | `--frame-stack` |
| buffer capacity | 100k frames | `--buffer-capacity` |
| warmup frames | 5000 | `--warmup` |
| batch size | 64 | `--batch-size` |
| train ratio | 2 grad steps / env step | `--train-ratio` |
| objective | `edm` (or `regression`) | `--objective` |
| total steps | 100k | `--steps` |
| eval every | 5000 | `--eval-every` |

Any config field can also be set with `--set key=value` (repeatable).

## Milestones

0. Contract, registry, ball game, actor, episode-aware buffer + random-play video.
1. Deterministic regression U-Net (MSE) baseline — fast one-step signal.
2. EDM-style diffusion objective; one-step next-frame generation.
3. Generality: a gymnasium env through `GymGame`, `train.py` unchanged.
4. Autoregressive rollout eval + an optional short rollout / scheduled-sampling
   loss for drift control (`--set rollout_loss_weight=0.3`).

## Notes / design decisions

- Collect from the real game, never from the world model.
- Train from the buffer with uniform random minibatches (decorrelates gradients);
  the actor is correlated random, the sampler is uniform random.
- Randomize the initial state on every reset; use short episode timeouts.
- Frame stacking is episode-aware: a conditioning stack never crosses a reset.
- Collision / blocked frames are kept — "pushed into a wall, did not move" is a
  rule to learn, not noise.
- Watch the identity shortcut: rollout videos and the `pred_motion_first_step`
  metric exist to confirm the model predicts motion, not a copy of the input.

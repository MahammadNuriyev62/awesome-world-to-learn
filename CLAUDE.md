# CLAUDE.md

## Project: Lightweight Online Diffusion World Model (multi-game)

A game-agnostic trainer. Given any game that implements the Game contract, it trains a
small diffusion-based world model for that game **on the fly**: a random actor plays the
real game live, frames stream into a replay buffer, and the model trains concurrently.
There is no task and no reward. The only objective is to predict the next frame given the
recent frames and an action. Target hardware is a single NVIDIA T4 (16 GB).

One world model per game (separate runs and checkpoints). Games are pluggable, ideally
single-file, modules. The ball game is the first of many.

### Scope decision

- Default and v1: **one model per game.** The core is parameterized by the game's
  observation shape and action space at construction; you point the trainer at a game and
  it trains a model for that game.
- Out of scope for v1: a single shared model that learns all games at once. That is a
  research-tier extension (game-id conditioning, balanced cross-game replay, a larger
  model, much more compute). Keep the door open but do not build it yet.

## The Game contract

A game is any object implementing:

- `action_space`: one of
  - `Discrete(n)`: n actions, or
  - `Box(shape, low, high)`: a continuous action vector.
- `obs_shape`: native `(H, W, C)` of the rendered frame.
- `reset() -> frame`: start a new episode and **randomize the initial state** (positions,
  velocities, layout). Returns an HxWx3 uint8 frame.
- `step(action) -> (frame, done)`: apply the action, return the next uint8 frame and a
  `done` flag marking an episode boundary. Endless games return `done=False` and rely on
  the framework's episode timeout.
- optional `name: str` and `default_config() -> dict` for per-game overrides.

Notes:

- The model is built per run from `obs_shape` and `action_space`, so games may differ in
  resolution and action space with no change to the core. A game (or its config) may
  resize its frames to a smaller canonical resolution to stay T4-friendly.
- `done` matters for correctness: frame-stacked conditioning must never span a reset, or
  the model will treat a teleport as dynamics. The buffer must respect episode boundaries.

## Built-in games and adapters

- `games/ball.py`: open top-down arena, a ball rolling with momentum, randomly placed
  rectangular walls it bounces off or stops against, no goal. Default 64x64, 5 discrete
  actions (impulse up / down / left / right / no-op). Pure numpy plus a simple rasterizer;
  pymunk optional.
- `games/gym_adapter.py`: `GymGame(env_id)` wraps any Gymnasium environment, using its
  `rgb_array` render as the observation and mapping its action space to the contract. This
  gives the whole Gymnasium ecosystem (classic control, Atari via ALE, and so on) for
  free. Use it to validate that the core is not hardcoded to the ball.

Adding a game means implementing the contract in one file (or wrapping a gym env) and
registering it.

## Registry and loading

- A registry maps names to game classes via a `@register("ball")` decorator.
- The trainer loads a game by registered name (`--game ball`), by file path
  (`--game ./mygame.py`), or by dotted path (`--game worldmodel.games.ball:BallGame`),
  preserving the "define a game in one file" property.

## Core architecture (game-agnostic)

1. `Game` contract (above), plus a resize wrapper to the canonical resolution.
2. `FrameBuffer`: a uint8 ring buffer with **episode-aware frame stacking** (stacks never
   cross a reset) and uniform random minibatch sampling of
   `(cond_frames, action, target_frame)`.
3. `make_actor(action_space)`: returns the exploration actor.
   - Discrete: sticky-random, hold a random action for `repeat` frames.
   - Continuous: temporally-correlated (OU) noise, smoothed across frames.
   A game may override `make_actor` if it needs custom exploration.
4. `DiffusionWorldModel(obs_shape, action_space)`: a small conditional U-Net at the
   canonical resolution. The U-Net core is fixed; only the **action-conditioning head**
   adapts, an embedding table for `Discrete` and a small MLP for `Box`. Exposes
   `denoise_loss(cond, action, target)` (one step, training) and `imagine(cond, actions)`
   (multi-step sampling, eval and rollouts only).
5. `train.py`: the online loop, identical for every game. Step the real game with the
   actor, push to the buffer, and after warmup do N gradient steps per env step sampled
   from the buffer. Periodically run an autoregressive rollout and save a video.
6. `eval.py`: autoregressive rollout vs ground truth. Game-agnostic, runs on any game.

## Critical design decisions (do not skip these)

These are deliberate and prevent known failure modes. They are all game-agnostic.

- **Collect from the real game, never from the world model.** Each
  `(prev frames, action, next frame)` tuple from the real env is a training target. The
  diffusion model is only sampled during eval and rollouts, never inside the
  data-collection loop. Training is cheap; sampling is the only expensive operation.
- **Train from the replay buffer with uniform random minibatches**, not from the live
  frame stream. Live frames are correlated and non-stationary; uniform sampling from the
  buffer decorrelates gradients. Two distinct randomness roles: the actor is temporally
  correlated random, the batch sampler is uniform random.
- **Exploration uses correlated (sticky or OU) actions, not i.i.d. per-frame actions.**
  Hold or smooth each random action across several frames so the agent commits to a
  direction and covers the state space instead of vibrating in place. No trained agent is
  involved.
- **Randomize the initial state on every reset.** Spawn at a random valid state and
  regenerate the layout. State-space coverage comes from resets, not from the policy. Use
  short episode timeouts so resets happen often.
- **Respect episode boundaries in frame stacking.** Conditioning frames must all come from
  one episode; never stack across a reset.
- **Keep collision / blocked frames.** A frame where force is applied and the object does
  not move (pushed into a wall) is a core rule to learn, not noise. Do not dedup these.
  The only thing to guard against is a flood of truly inert frames (object at rest with no
  input); with an active actor this is rare, so at most lightly downsample long identical
  idle runs.
- **Condition on a stack of the last k frames (default 4)** so velocity is observable and
  the prediction problem is Markov. A single frame does not reveal motion direction.
- **Watch for the identity shortcut.** The model can cheat by predicting "same frame
  again." Monitor that rollouts show correct motion and collisions, not just copies of the
  input.

## Build order (milestones)

0. **Contract, registry, ball game, actor, buffer.** Define the `Game` contract, the
   registry and loader, `games/ball.py`, `make_actor`, and the episode-aware
   `FrameBuffer`. Sanity-check with a random-play video: ball rolls, damps, bounces, and
   resets randomize state.
1. **Deterministic baseline.** A plain regression U-Net with MSE loss, same conditioning.
   Validates the full pipeline and gives a fast one-step-prediction signal.
2. **Diffusion model.** Swap MSE for an EDM-style denoising objective; confirm one-step
   next-frame generation matches the ground truth.
3. **Prove generality.** Wrap one Gymnasium env via `GymGame`, ideally one with a
   different action space, and confirm `train.py` runs unchanged. This is the test that
   nothing is hardcoded to the ball.
4. **Autoregressive rollout and drift control.** Add the multi-step rollout eval, then a
   short multi-step rollout loss (or scheduled sampling) to reduce compounding error over
   long horizons.
5. **Tune.** Train ratio, buffer size, exploration, resolution, sampler steps.

## Suggested defaults (per game; override via default_config)

- canonical resolution: 64x64 (use 32x32 for fast iteration)
- frame_stack: 4
- buffer_capacity: 100_000 frames (uint8)
- warmup_frames: 5_000
- batch_size: 64
- train_ratio: 2 (gradient steps per env step, after warmup)
- action_repeat (discrete) / OU theta (continuous): 8 / 0.15
- episode_timeout: 300 steps
- unet: base channels 64, about 3 downsampling stages (~5 to 20M params)
- optimizer: AdamW, lr 2e-4, mixed precision (bf16 or fp16)
- diffusion: EDM preconditioning; eval sampler Heun or DDIM, about 10 to 30 steps
- eval_every: 5_000 steps; rollout_length: 64

## Layout

```
.
├── CLAUDE.md
├── README.md
└── worldmodel/
    ├── __init__.py
    ├── core/
    │   ├── contract.py     # Game protocol, Discrete/Box action spaces
    │   ├── registry.py     # @register + load by name / path
    │   ├── wrappers.py     # resize-to-canonical wrapper
    │   ├── buffer.py       # episode-aware uint8 ring buffer
    │   ├── actor.py        # make_actor: sticky (discrete) / OU (continuous)
    │   ├── model.py        # DiffusionWorldModel: U-Net + adaptive action head
    │   ├── train.py        # game-agnostic online loop
    │   ├── eval.py         # autoregressive rollout vs ground truth
    │   └── config.py       # base config + merge (base < game < CLI)
    └── games/
        ├── __init__.py
        ├── ball.py         # the first game (registers "ball")
        └── gym_adapter.py  # GymGame(env_id) wrapping any gymnasium env
```

## Commands

```bash
# setup
pip install torch torchvision numpy einops imageio imageio-ffmpeg tqdm gymnasium
# optional: pip install wandb pymunk "gymnasium[atari]" ale-py

# train a built-in game
python -m worldmodel.core.train --game ball

# train any gymnasium env through the adapter
python -m worldmodel.core.train --game gym:ALE/Breakout-v5

# train a custom single-file game
python -m worldmodel.core.train --game ./my_game.py

# standalone rollout video from a checkpoint
python -m worldmodel.core.eval --ckpt runs/latest/model.pt --rollout 128
```

## Hardware and constraints

- Single NVIDIA T4 (16 GB). Mixed precision. Keep the U-Net small.
- Store frames as uint8 (100k at 64x64x3 is about 1.2 GB RAM).
- The physics sim is cheap; GPU training is the throughput limiter, so the env will not
  bottleneck a synchronous loop. Gym envs vary; heavy ones may want a faster collection
  path later.

## Conventions

- Python and PyTorch. Prefer clear, small modules over a framework.
- Type hints and dataclass configs.
- Keep the core game-agnostic: no per-game branches in `core/`. Per-game behavior lives in
  the game module or its config.
- Avoid em dashes in generated docs and comments.

## Implementation notes (as built)

- A small `core/utils.py` holds shared image<->tensor conversion, the EMA helper, and
  video writing.
- `worldmodel/play.py` lets you play any game live in the terminal (truecolor half-block
  rendering, WASD/arrow keys), and with `--model CKPT` you can play the trained world
  model itself (its autoregressive "dream"), optionally `--compare` against the real game.
- AMP defaults to fp16, which is native on the T4 (Turing); bf16 is emulated there.

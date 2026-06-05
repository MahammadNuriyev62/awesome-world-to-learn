"""DiffusionWorldModel: a small conditional U-Net with an adaptive action head.

The U-Net backbone is fixed across games. Only the action-conditioning head
adapts to the action space: an embedding table for ``Discrete`` and a small MLP
for ``Box``. The model is built per run from ``obs_shape`` and ``action_space``.

Two objectives share one backbone:
  - ``regression``: predict the next frame directly (MSE). Milestone 1 baseline.
  - ``edm``: Karras et al. EDM preconditioning + denoising loss. Milestone 2.

Tensor conventions (inside the model): images are float in [-1, 1].
  cond:   (B, k, C, H, W)
  target: (B, C, H, W)
  action: (B,) int64 for Discrete, or (B, adim) float for Box.

Public methods:
  denoise_loss(cond, action, target) -> scalar   (one training step)
  imagine(cond, actions)            -> (B, T, C, H, W)   (sampling; eval only)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contract import ActionSpace, Box, Discrete, action_dim


# --------------------------------------------------------------------------- utils
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal embedding of a 1-D tensor of values."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels, eps=1e-6)


# --------------------------------------------------------------------------- blocks
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)  # FiLM scale + shift
        self.norm2 = _gn(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = _gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))  # (B, heads, HW, d)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


# --------------------------------------------------------------------------- unet
class UNet(nn.Module):
    """Conditional U-Net. Input channels = cond frames (k*C) + target (C)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        base: int,
        mults: tuple[int, ...],
        num_res_blocks: int,
        attn_resolutions: tuple[int, ...],
        resolution: int,
        emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.downs = nn.ModuleList()
        chs = [base]
        ch = base
        res = resolution
        for i, m in enumerate(mults):
            out = base * m
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out, emb_dim, dropout)])
                ch = out
                if res in attn_resolutions:
                    block.append(AttnBlock(ch))
                self.downs.append(block)
                chs.append(ch)
            if i != len(mults) - 1:
                self.downs.append(nn.ModuleList([Downsample(ch)]))
                chs.append(ch)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout), AttnBlock(ch), ResBlock(ch, ch, emb_dim, dropout)])

        self.ups = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            out = base * m
            for _ in range(num_res_blocks + 1):
                block = nn.ModuleList([ResBlock(ch + chs.pop(), out, emb_dim, dropout)])
                ch = out
                if res in attn_resolutions:
                    block.append(AttnBlock(ch))
                self.ups.append(block)
            if i != 0:
                self.ups.append(nn.ModuleList([Upsample(ch)]))
                res *= 2

        self.out_norm = _gn(ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        hs = [h]
        for block in self.downs:
            if isinstance(block[0], Downsample):
                h = block[0](h)
                hs.append(h)
            else:
                h = block[0](h, emb)
                for extra in block[1:]:
                    h = extra(h)
                hs.append(h)
        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        for block in self.ups:
            if isinstance(block[0], Upsample):
                h = block[0](h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                h = block[0](h, emb)
                for extra in block[1:]:
                    h = extra(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# --------------------------------------------------------- action / noise conditioning
class ConditionEmbed(nn.Module):
    """Combine noise-level and action into one conditioning vector."""

    def __init__(self, action_space: ActionSpace, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.noise_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        if isinstance(action_space, Discrete):
            self.discrete = True
            self.action_embed = nn.Embedding(action_space.n, emb_dim)
        elif isinstance(action_space, Box):
            self.discrete = False
            self.action_embed = nn.Sequential(
                nn.Linear(action_dim(action_space), emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
            )
        else:
            raise TypeError(action_space)

    def forward(self, c_noise: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        noise_emb = self.noise_mlp(sinusoidal_embedding(c_noise, self.emb_dim))
        if self.discrete:
            act_emb = self.action_embed(action.long())
        else:
            act_emb = self.action_embed(action.float())
        return noise_emb + act_emb


# --------------------------------------------------------------------------- model
class DiffusionWorldModel(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], action_space: ActionSpace, config):
        super().__init__()
        H, W, C = obs_shape
        assert H == W, "model assumes square canonical frames"
        self.C = C
        self.k = config.frame_stack
        self.objective = config.objective
        emb_dim = config.cond_embed_dim

        self.cond_embed = ConditionEmbed(action_space, emb_dim)
        self.unet = UNet(
            in_ch=(self.k + 1) * C,
            out_ch=C,
            base=config.base_channels,
            mults=tuple(config.channel_mults),
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=tuple(config.attn_resolutions),
            resolution=H,
            emb_dim=emb_dim,
            dropout=config.dropout,
        )

        # EDM hyper-parameters
        self.sigma_data = config.sigma_data
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.p_mean = config.p_mean
        self.p_std = config.p_std
        self.rho = config.rho
        self.sampler = config.sampler
        self.sampler_steps = config.sampler_steps

    # ---------------------------------------------------------------- core forward
    def _net(self, x: torch.Tensor, cond: torch.Tensor, action: torch.Tensor, c_noise: torch.Tensor):
        """Raw network call. ``cond`` is (B, k, C, H, W); flattened to channels."""
        B = cond.shape[0]
        cond_flat = cond.reshape(B, self.k * self.C, *cond.shape[-2:])
        emb = self.cond_embed(c_noise, action)
        return self.unet(torch.cat([x, cond_flat], dim=1), emb)

    # ---------------------------------------------------------------- EDM precond
    def edm_denoise(self, cond, action, x, sigma):
        """EDM-preconditioned denoiser D(x; sigma) -> estimate of clean target."""
        sigma = sigma.reshape(-1, 1, 1, 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1.0 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = 0.25 * torch.log(sigma.reshape(-1) + 1e-20)
        F_x = self._net(c_in * x, cond, action, c_noise)
        return c_skip * x + c_out * F_x

    # ---------------------------------------------------------------- training loss
    def denoise_loss(self, cond, action, target) -> torch.Tensor:
        if self.objective == "regression":
            c_noise = torch.zeros(cond.shape[0], device=cond.device)
            zeros = torch.zeros_like(target)
            pred = self._net(zeros, cond, action, c_noise)
            return F.mse_loss(pred, target)

        # EDM denoising loss
        B = target.shape[0]
        rnd = torch.randn(B, device=target.device)
        sigma = (rnd * self.p_std + self.p_mean).exp()
        n = torch.randn_like(target) * sigma.reshape(-1, 1, 1, 1)
        D = self.edm_denoise(cond, action, target + n, sigma)
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        loss = (weight.reshape(-1, 1, 1, 1) * (D - target) ** 2).mean()
        return loss

    # ----------------------------------------------- differentiable one-step pred
    def rollout_predict(self, cond, action) -> torch.Tensor:
        """A cheap, differentiable one-step next-frame prediction.

        Used only by the optional short-rollout / scheduled-sampling drift loss
        (milestone 4). For ``regression`` this is the exact predictor; for
        ``edm`` it is a single denoise step from a moderate-noise sample, an
        approximation of the full sampler that stays cheap and differentiable.
        """
        B = cond.shape[0]
        if self.objective == "regression":
            c_noise = torch.zeros(B, device=cond.device)
            zeros = torch.zeros(B, self.C, *cond.shape[-2:], device=cond.device)
            return self._net(zeros, cond, action, c_noise)
        sigma = torch.full((B,), self.sigma_data, device=cond.device)
        x = torch.randn(B, self.C, *cond.shape[-2:], device=cond.device) * self.sigma_data
        return self.edm_denoise(cond, action, x, sigma)

    # ---------------------------------------------------------------- sampling
    def _sigma_schedule(self, n: int, device) -> torch.Tensor:
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv = self.sigma_min ** (1 / self.rho)
        max_inv = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** self.rho
        return torch.cat([sigmas, torch.zeros(1, device=device)])  # append sigma=0

    @torch.no_grad()
    def sample_frame(self, cond, action) -> torch.Tensor:
        """Sample one next frame given a conditioning stack and action."""
        if self.objective == "regression":
            c_noise = torch.zeros(cond.shape[0], device=cond.device)
            zeros = torch.zeros(cond.shape[0], self.C, *cond.shape[-2:], device=cond.device)
            return self._net(zeros, cond, action, c_noise).clamp(-1, 1)

        device = cond.device
        B = cond.shape[0]
        sigmas = self._sigma_schedule(self.sampler_steps, device)
        x = torch.randn(B, self.C, *cond.shape[-2:], device=device) * sigmas[0]
        for i in range(len(sigmas) - 1):
            s, s_next = sigmas[i], sigmas[i + 1]
            sigma_b = torch.full((B,), float(s), device=device)
            D = self.edm_denoise(cond, action, x, sigma_b)
            d = (x - D) / s
            x_next = x + (s_next - s) * d
            if self.sampler == "heun" and s_next > 0:
                sigma_n = torch.full((B,), float(s_next), device=device)
                D2 = self.edm_denoise(cond, action, x_next, sigma_n)
                d2 = (x_next - D2) / s_next
                x_next = x + (s_next - s) * 0.5 * (d + d2)
            x = x_next
        return x.clamp(-1, 1)

    @torch.no_grad()
    def imagine(self, cond, actions) -> torch.Tensor:
        """Autoregressive rollout. ``actions`` is (B, T) Discrete or (B, T, adim).

        Returns predicted frames (B, T, C, H, W) in [-1, 1].
        """
        cond = cond.clone()
        T = actions.shape[1]
        preds = []
        for t in range(T):
            a_t = actions[:, t]
            nxt = self.sample_frame(cond, a_t)  # (B, C, H, W)
            preds.append(nxt)
            cond = torch.cat([cond[:, 1:], nxt[:, None]], dim=1)  # drop oldest, append
        return torch.stack(preds, dim=1)

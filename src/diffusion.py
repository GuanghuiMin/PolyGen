

# -*- coding: utf-8 -*-
"""
Core diffusion components for conditional histogram generation.

This module provides:
  - Histogram <-> logits helpers (stable)
  - Cosine noise schedule (betas/alphas/alpha_bars)
  - v-parameterization utilities (q_sample, targets, conversions)
  - A lightweight DiT-1D denoiser with FiLM conditioning
  - DDIM sampler (supports classifier-free guidance)

You can import these from your diffusion_trainer.py and wire them with your
ConditionEncoder (use the semantic embedding "cond_emb" as `cond`).

Author: guanghui + assistant
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Histogram <-> logits helpers
# ----------------------------

def hist_to_logits(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Map a probability histogram p (sum=1, p>=0) to unconstrained logits.
    Args:
        p: [B, M] tensor with probabilities.
    Returns:
        z: [B, M] logits such that softmax(z) ~ p (up to a constant shift).
    """
    return torch.log(torch.clamp(p, min=eps))


def logits_to_hist(z: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Map logits back to a valid histogram via softmax.
    Args:
        z: [B, M] logits
        tau: temperature; >1 smooths, <1 sharpens
    """
    return F.softmax(z / tau, dim=-1)


# ----------------------------
# Cosine noise schedule
# ----------------------------

@dataclass
class NoiseSchedule:
    """
    Precomputes cosine schedule with T steps.
    Stores:
      - betas:        [T]
      - alphas:       [T]
      - alpha_bars:   [T]
    """
    T: int
    s: float = 0.008   # Nichol & Dhariwal cosine offset

    def __post_init__(self):
        t = torch.linspace(0, self.T, self.T + 1, dtype=torch.float32)  # 0..T
        f = torch.cos(((t / self.T) + self.s) / (1 + self.s) * math.pi / 2) ** 2
        alpha_bar = (f / f[0]).clamp(min=1e-7)  # [T+1], alpha_bar(0)=1
        # Convert alpha_bar to betas between steps
        betas = []
        for i in range(1, self.T + 1):
            prev = alpha_bar[i - 1]
            curr = alpha_bar[i]
            beta = (1 - curr / prev).clamp(1e-8, 0.999)
            betas.append(beta)
        self.betas = torch.stack(betas)               # [T]
        self.alphas = (1.0 - self.betas)             # [T]
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # [T]

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def gather_alpha(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            t: int64 tensor of shape [B] with values in [0, T-1]
        Returns:
            alpha_t, alpha_bar_t, one_minus_alpha_bar_t: each [B, 1]
        """
        assert t.dtype == torch.long
        a = self.alphas.index_select(0, t).unsqueeze(-1)                 # [B,1]
        ab = self.alpha_bars.index_select(0, t).unsqueeze(-1)            # [B,1]
        one_minus_ab = (1.0 - ab).clamp(min=1e-8)
        return a, ab, one_minus_ab


# --------------------------------------
# v-parameterization (Imagen/DiT style)
# --------------------------------------

def q_sample_vparam(z0: torch.Tensor, t: torch.Tensor, schedule: NoiseSchedule,
                    eps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward process (q): produce x_t and the v-target for loss.
      x_t = sqrt(alpha_bar_t) * z0 + sqrt(1-alpha_bar_t) * eps
      v   = sqrt(alpha_bar_t) * eps - sqrt(1-alpha_bar_t) * z0
    Args:
        z0: [B, M] clean logits
        t : [B] long timesteps in [0, T-1]
        schedule: NoiseSchedule
        eps: optional Gaussian noise, if None sampled N(0, I)
    Returns:
        x_t : [B, M]
        v   : [B, M] target
    """
    _, ab, one_minus_ab = schedule.gather_alpha(t)     # [B,1]
    if eps is None:
        eps = torch.randn_like(z0)
    x_t = ab.sqrt() * z0 + one_minus_ab.sqrt() * eps
    v   = (ab.sqrt()) * eps - (one_minus_ab.sqrt()) * z0
    return x_t, v


def v_to_eps_x0(x_t: torch.Tensor, v_hat: torch.Tensor, t: torch.Tensor, schedule: NoiseSchedule):
    """
    Convert predicted v to (eps_hat, x0_hat) at step t.
    """
    _, ab, one_minus_ab = schedule.gather_alpha(t)  # [B,1]
    sqrt_ab = ab.sqrt()
    sqrt_one_minus_ab = one_minus_ab.sqrt()

    # From v = sqrt(ab)*eps - sqrt(1-ab)*x0  =>
    # eps = (v + sqrt(1-ab)*x0) / sqrt(ab)
    # x0  = (sqrt(ab)*x_t - sqrt(1-ab)*eps)
    # But we need x0 in terms of (x_t, v). Solve:
    #   v = sqrt(ab)*eps - sqrt(1-ab)*x0
    #   x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps
    # => x0 = sqrt(ab)*x_t - sqrt(1-ab)*v
    x0_hat = (x_t * sqrt_ab - v_hat * sqrt_one_minus_ab) / (ab + 1e-8).sqrt()
    # And eps_hat = (x_t - sqrt(ab)*x0) / sqrt(1-ab)
    eps_hat = (x_t - sqrt_ab * x0_hat) / (sqrt_one_minus_ab + 1e-8)
    return eps_hat, x0_hat


# ----------------------------
# Time / condition embeddings
# ----------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] int64 timesteps
        Returns: [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32)
                          * (-math.log(10000.0) / max(1, half - 1)))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        pe = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            pe = F.pad(pe, (0, 1))
        return self.proj(pe)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation from a conditioning vector.
    """
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, d_model)
        self.to_shift = nn.Linear(cond_dim, d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x: [B, T, d]
        cond: [B, cond_dim]
        """
        s = self.to_scale(cond).unsqueeze(1)
        b = self.to_shift(cond).unsqueeze(1)
        return x * (1 + s) + b


# ----------------------------
# DiT-1D denoiser (FiLM)
# ----------------------------

class DiT1D(nn.Module):
    """
    A compact Transformer denoiser for 1D histogram logits.

    Inputs:
      - x_t:  [B, M]   noisy logits at step t
      - t:    [B]      integer timesteps
      - cond: [B, C]   condition embedding (e.g., ConditionEncoder.cond_emb)

    Outputs:
      - v_hat: [B, M]  predicted v (Imagen/DiT parameterization)
    """
    def __init__(
        self,
        bins: int,
        cond_dim: int,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        film_each_layer: bool = True,
    ):
        super().__init__()
        self.bins = bins
        self.d_model = d_model
        self.token_proj = nn.Linear(1, d_model)  # per-bin scalar -> d
        self.pos_emb = nn.Parameter(torch.randn(1, bins, d_model) * 0.02)

        self.t_embed = SinusoidalTimeEmbedding(d_model)
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.film = FiLM(d_model, d_model)
        self.film_each_layer = film_each_layer

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, 1)  # d -> scalar per bin

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x_t:   [B, M]    noisy logits
        t:     [B]       integer steps
        cond:  [B, C]    condition embedding
        return v_hat: [B, M]
        """
        B, M = x_t.shape
        assert M == self.bins, f"bins mismatch: expected {self.bins}, got {M}"

        h = self.token_proj(x_t.unsqueeze(-1))         # [B, M, d]
        h = h + self.pos_emb                           # learned positional encoding

        t_emb = self.t_embed(t)                        # [B, d]
        c_emb = self.cond_proj(cond)                   # [B, d]
        # add time embedding
        h = h + t_emb.unsqueeze(1)

        # one FiLM pre-backbone for quick conditioning
        h = self.film(h, c_emb)

        # optional FiLM at each layer (simple way: chunk apply)
        if self.film_each_layer:
            # We can't easily insert per-layer FiLM without rewriting TransformerEncoder.
            # Approximation: after backbone, apply another FiLM.
            h = self.backbone(h)
            h = self.film(h, c_emb)
        else:
            h = self.backbone(h)

        v_hat = self.out_proj(h).squeeze(-1)           # [B, M]
        return v_hat


# ----------------------------
# Loss (v-parameterization)
# ----------------------------

def diffusion_loss_vpred(model: nn.Module,
                         z0: torch.Tensor,
                         t: torch.Tensor,
                         cond: torch.Tensor,
                         schedule: NoiseSchedule) -> torch.Tensor:
    """
    Sample a noised input and compute MSE between v_hat and v_target.
    Args:
        model: DiT1D
        z0: [B, M] clean logits
        t:  [B] long timesteps
        cond: [B, C] condition embedding
    """
    x_t, v_target = q_sample_vparam(z0, t, schedule)             # [B,M]
    v_hat = model(x_t=x_t, t=t, cond=cond)                       # [B,M]
    return F.mse_loss(v_hat, v_target)


# ----------------------------
# DDIM Sampler (v-parameterization)
# ----------------------------

@torch.no_grad()
def ddim_sample(model: nn.Module,
                cond: torch.Tensor,
                schedule: NoiseSchedule,
                steps: int = 50,
                eta: float = 0.0,
                guidance: Optional[Tuple[torch.Tensor, float]] = None,
                bins: Optional[int] = None,
                tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DDIM (eta=0 by default) sampling in v-parameterization.
    Optionally supports CFG: guidance=(uncond_cond, scale).

    Args:
        model: DiT1D
        cond: [B, C] condition embedding
        schedule: NoiseSchedule (with T training steps)
        steps: number of sampling steps (<= T). We'll use uniform stride.
        eta: ddim eta; 0 -> deterministic
        guidance: (cond_uncond, scale) if classifier-free guidance is used
        bins: if provided, allocate x_t of shape [B, bins]; otherwise infer from model
        tau: temperature for logits_to_hist (used by caller)
    Returns:
        z0_hat: [B, M] predicted clean logits
        p_hat:  [B, M] histogram via softmax(z0_hat/tau)
    """
    device = cond.device
    T = schedule.T
    assert steps >= 1 and steps <= T
    # uniform time grid
    ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    # infer bins from model pos_emb
    if bins is None:
        try:
            bins = model.pos_emb.size(1)
        except Exception:
            raise ValueError("Please provide `bins` explicitly to ddim_sample().")

    # start from Gaussian noise in logits space
    x_t = torch.randn(cond.size(0), bins, device=device)

    # helper to run model with optional CFG
    def model_v(x, t_step, cond_in):
        if guidance is None:
            return model(x_t=x, t=t_step, cond=cond_in)
        cond_uncond, scale = guidance
        v_c = model(x_t=x, t=t_step, cond=cond_in)
        v_u = model(x_t=x, t=t_step, cond=cond_uncond)
        return v_u + (v_c - v_u) * scale

    for i in range(steps):
        t_i = ts[i].expand(cond.size(0))  # [B]
        v_hat = model_v(x_t, t_i, cond)   # [B, M]
        eps_hat, x0_hat = v_to_eps_x0(x_t, v_hat, t_i, schedule)

        # DDIM update
        # x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat + sqrt(1 - alpha_bar_{t-1}) * eps_hat'
        # with eps_hat' = eta * eps + (1-eta) * eps_hat (here we use deterministic when eta=0)
        if i == steps - 1:
            x_t = x0_hat
            break

        t_next = ts[i + 1].expand(cond.size(0))
        _, ab_next, one_minus_ab_next = schedule.gather_alpha(t_next)
        if eta > 0:
            eps = torch.randn_like(eps_hat)
            eps_prime = eta * eps + (1 - eta) * eps_hat
        else:
            eps_prime = eps_hat

        x_t = ab_next.sqrt() * x0_hat + one_minus_ab_next.sqrt() * eps_prime

    z0_hat = x_t
    p_hat = logits_to_hist(z0_hat, tau=tau)
    return z0_hat, p_hat


def ddpm_sample(model: nn.Module,
                cond: torch.Tensor,
                schedule: NoiseSchedule,
                steps: int = 50,
                guidance: Optional[Tuple[torch.Tensor, float]] = None,
                bins: Optional[int] = None,
                tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DDPM sampling with full stochasticity for better long-tail coverage.
    Each step adds noise according to the learned posterior variance.

    Args:
        model: DiT1D
        cond: [B, C] condition embedding  
        schedule: NoiseSchedule (with T training steps)
        steps: number of sampling steps (<= T)
        guidance: (cond_uncond, scale) if classifier-free guidance is used
        bins: if provided, allocate x_t of shape [B, bins]; otherwise infer from model
        tau: temperature for logits_to_hist
    Returns:
        z0_hat: [B, M] predicted clean logits
        p_hat:  [B, M] histogram via softmax(z0_hat/tau)
    """
    device = cond.device
    T = schedule.T
    assert steps >= 1 and steps <= T
    
    # uniform time grid
    ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)
    
    # infer bins from model
    if bins is None:
        try:
            bins = model.pos_emb.size(1)
        except Exception:
            raise ValueError("Please provide `bins` explicitly to ddpm_sample().")
    
    # start from Gaussian noise
    x_t = torch.randn(cond.size(0), bins, device=device)
    
    # helper to run model with optional CFG
    def model_v(x, t_step, cond_in):
        if guidance is None:
            return model(x_t=x, t=t_step, cond=cond_in)
        cond_uncond, scale = guidance
        v_c = model(x_t=x, t=t_step, cond=cond_in)
        v_u = model(x_t=x, t=t_step, cond=cond_uncond)
        return v_u + (v_c - v_u) * scale
    
    for i in range(steps):
        t_i = ts[i].expand(cond.size(0))  # [B]
        v_hat = model_v(x_t, t_i, cond)   # [B, M]
        eps_hat, x0_hat = v_to_eps_x0(x_t, v_hat, t_i, schedule)
        
        if i == steps - 1:
            # Final step: return clean prediction
            x_t = x0_hat
            break
        
        # DDPM update with learned posterior variance
        t_next = ts[i + 1].expand(cond.size(0))
        alpha_t, alpha_bar_t, one_minus_alpha_bar_t = schedule.gather_alpha(t_i)
        _, alpha_bar_next, _ = schedule.gather_alpha(t_next)
        
        # Posterior variance: β_t * (1 - α̅_{t-1}) / (1 - α̅_t)
        beta_t = 1 - alpha_t
        posterior_var = beta_t * (1 - alpha_bar_next) / (1 - alpha_bar_t + 1e-8)
        posterior_log_var = torch.log(posterior_var.clamp(min=1e-20))
        
        # Sample from posterior: μ + σ * ε
        # μ = (√α̅_{t-1} * β_t * x0 + √α_t * (1-α̅_{t-1}) * x_t) / (1-α̅_t)
        coeff_x0 = alpha_bar_next.sqrt() * beta_t / (1 - alpha_bar_t + 1e-8)
        coeff_xt = alpha_t.sqrt() * (1 - alpha_bar_next) / (1 - alpha_bar_t + 1e-8)
        mean = coeff_x0 * x0_hat + coeff_xt * x_t
        
        if posterior_log_var.max() > -20:  # Add noise if variance is not too small
            noise = torch.randn_like(x_t)
            x_t = mean + torch.exp(0.5 * posterior_log_var) * noise
        else:
            x_t = mean
    
    z0_hat = x_t
    p_hat = logits_to_hist(z0_hat, tau=tau)
    return z0_hat, p_hat
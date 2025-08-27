#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python diffusion_trainer.py
--csv ./data/copolymer.csv
--cond_ckpt ./outputs/cond_enc_runs_sched40/cond_encoder_final.pt
--out_dir outputs/diffusion_runs_long_tail
--epochs 30 --batch_size 8 --accum_steps 2
--bins_new 50
--lambda_kl 0.2 --lambda_emd 0.03 --emd_warmup_epochs 3
--sample_steps 50 --eval_sample_steps 200 --logit_temp 0.8
--exp_tail --exp_tail_beta 1 --exp_tail_start 21
--fig_format pdf
"""

import os
import math
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Perf knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---------- Project imports ----------
# Adjust these imports if your module names differ.
from data.dataset import ChainSetDataset, collate_fn_set_transformer
from src.encoder import ConditionEncoder
from src.diffusion import (
    DiT1D, NoiseSchedule, hist_to_logits, logits_to_hist,
    q_sample_vparam, diffusion_loss_vpred, ddim_sample
)

# -------- skew-mitigation extra imports & helpers --------

@torch.no_grad()
def compute_bin_frequency(loader, device, bins: int, max_batches: int = 200):
    """Estimate dataset-level average probability per bin (freq[M])."""
    bin_sum = torch.zeros(bins, device=device)
    count = 0
    for i, batch in enumerate(loader):
        p = batch["block_dists"].to(device)
        bin_sum += p.sum(dim=0)
        count += p.size(0)
        if i + 1 >= max_batches:
            break
    freq = (bin_sum / max(1, count)).clamp_min(1e-8)
    return freq

@torch.no_grad()
def compute_global_prior(loader, device, bins: int, max_batches: int = 200):
    """Global mean histogram as a prior (probabilities)."""
    freq = compute_bin_frequency(loader, device, bins, max_batches)
    prior_p = (freq / freq.sum()).clamp_min(1e-8)
    return prior_p


def emd1d_torch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """1D EMD via CDF L1; p,q: [B,M] -> [B]."""
    cdf_p = p.cumsum(dim=-1)
    cdf_q = q.cumsum(dim=-1)
    return (cdf_p - cdf_q).abs().sum(dim=-1)

# ---------- Rebinning helpers ----------

def make_rebin_matrix(old_bins: int, new_bins: int, scheme: str = "uniform") -> torch.Tensor:
    """
    Build a linear map W s.t. p_new = W @ p_old (column vector), with W:[new,old].
    For batch row-vectors, use: p_new_batch = p_old_batch @ W.T
    """
    assert 1 < new_bins <= old_bins
    import numpy as np
    W = torch.zeros(new_bins, old_bins, dtype=torch.float32)
    if scheme == "uniform":
        edges = np.linspace(0, old_bins, new_bins + 1, dtype=int)
        for i in range(new_bins):
            s, e = edges[i], edges[i+1]
            if e > s:
                W[i, s:e] = 1.0
    else:
        # fallback uniform
        edges = np.linspace(0, old_bins, new_bins + 1, dtype=int)
        for i in range(new_bins):
            s, e = edges[i], edges[i+1]
            if e > s:
                W[i, s:e] = 1.0
    # Each old bin contributes fully to exactly one new bin; no renorm needed (prob sums preserved)
    return W


# ----------------- Utils -----------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def get_cond_emb(cond_encoder: ConditionEncoder, cond_feat: torch.Tensor) -> torch.Tensor:
    """Encode condition_features -> cond_emb (freeze encoder)."""
    cond_encoder.eval()
    out = cond_encoder(cond=cond_feat)
    return out["cond_emb"]  # [B, d_model]


def build_loaders(csv_path: str, batch_size: int, num_workers: int = 0, max_samples=None):
    dataset = ChainSetDataset(csv_path, max_samples=max_samples, contrastive=True)
    # simple random split (90/10)
    n = len(dataset)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(1234))

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn_set_transformer, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn_set_transformer, pin_memory=True, drop_last=False
    )
    return train_set, val_set, train_loader, val_loader


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(p||q) on last dim, batch-reduced mean."""
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()

def emd_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Earth Mover's Distance (Wasserstein-1) for 1D histograms on last dim; batch mean."""
    cdf_p = p.cumsum(dim=-1)
    cdf_q = q.cumsum(dim=-1)
    return (cdf_p - cdf_q).abs().sum(dim=-1).mean()

# ---------- Tail mass & soft quantile (differentiable) ----------
def tail_mass(p: torch.Tensor, tail_start_bin: int) -> torch.Tensor:
    """
    Compute per-sample tail mass S = sum_{i >= s} p_i (1-based s=tail_start_bin).
    p: [B,M] probabilities, rows sum to 1.
    Returns: [B]
    """
    s = max(1, int(tail_start_bin)) - 1
    return p[:, s:].sum(dim=-1)

def tail_mass_loss(p_true: torch.Tensor, p_pred: torch.Tensor, tail_start_bin: int) -> torch.Tensor:
    """MSE between tail masses of GT and prediction; returns batch mean."""
    S_t = tail_mass(p_true, tail_start_bin)
    S_p = tail_mass(p_pred, tail_start_bin)
    return torch.mean((S_p - S_t) ** 2)

def soft_quantile_index(p: torch.Tensor, levels: torch.Tensor, kappa: float = 25.0) -> torch.Tensor:
    """
    Differentiable estimate of (0,1]-quantiles (in bin index, 1-based).
    For each u in `levels`, returns expected index where CDF crosses u using a smooth
    count of bins with CDF >= u: q(u) ≈ ∑_k σ(kappa*(cdf_k - u)).
    Args:
        p: [B,M] probabilities
        levels: [L] tensor of quantile levels (e.g., [0.90,0.95,0.99])
        kappa: sharpness; larger = sharper step
    Returns:
        q_idx: [B,L] soft quantile indices (1-based scale approximately)
    """
    B, M = p.shape
    cdf = p.cumsum(dim=-1)                              # [B,M]
    # [1,M,L] broadcast compare
    u = levels.view(1, 1, -1)                           # [1,1,L]
    # sigmoid over bins, sum over bins -> soft index
    sig = torch.sigmoid(kappa * (cdf.unsqueeze(-1) - u))  # [B,M,L]
    q_idx = sig.sum(dim=1)                              # [B,L]
    return q_idx

def soft_quantile_loss(p_true: torch.Tensor, p_pred: torch.Tensor, levels: torch.Tensor, kappa: float = 25.0) -> torch.Tensor:
    """
    MSE between soft quantile indices of GT and prediction across given levels.
    Returns batch mean.
    """
    q_t = soft_quantile_index(p_true, levels, kappa)    # [B,L]
    q_p = soft_quantile_index(p_pred, levels, kappa)    # [B,L]
    return torch.mean((q_p - q_t) ** 2)

#
# ---------- Tail-aware helpers (EXP weights, Focal-KL, Reverse-KL, Weighted EMD) ----------
# (Optional and default-off; Power-law tail alignment below is an alternative without reweighting.)

def build_exp_tail_weights(bins: int, beta: float = 1.0, tail_start: int | None = None, normalize: bool = True) -> torch.Tensor:
    """Exponential bin weights that emphasize the tail."""
    idx = torch.arange(bins, dtype=torch.float32)
    if tail_start is None or tail_start <= 1:
        s = 0
    else:
        s = min(bins - 1, int(tail_start) - 1)
    w = torch.ones(bins, dtype=torch.float32)
    if s < bins - 1:
        span = max(1, (bins - 1 - s))
        t = (idx - float(s)).clamp(min=0.0) / float(span)  # in [0,1]
        w = torch.where(idx >= s, torch.exp(beta * t), torch.ones_like(idx))
    if normalize:
        w = w / w.mean()
    return w

def weighted_kl_divergence(p: torch.Tensor, q: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Weighted KL(p||q) on last dim; w broadcast to [B,M]. """
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return (w * p * (p.log() - q.log())).sum(dim=-1).mean()

def reverse_kl_divergence(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batch-mean KL(q||p) on last dim."""
    q = torch.clamp(q, eps, 1.0)
    p = torch.clamp(p, eps, 1.0)
    return (q * (q.log() - p.log())).sum(dim=-1).mean()

def weighted_reverse_kl_divergence(q: torch.Tensor, p: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Weighted KL(q||p) on last dim; w broadcastable to [B,M]."""
    q = torch.clamp(q, eps, 1.0)
    p = torch.clamp(p, eps, 1.0)
    return (w * q * (q.log() - p.log())).sum(dim=-1).mean()

def weighted_emd_1d(p: torch.Tensor, q: torch.Tensor, w_cdf: torch.Tensor | None = None) -> torch.Tensor:
    """
    Weighted 1D EMD as L1 between CDFs; if w_cdf is provided (shape [M] or [1,M]),
    weight each CDF bin. Returns batch mean.
    """
    cdf_p = p.cumsum(dim=-1)
    cdf_q = q.cumsum(dim=-1)
    diff = (cdf_p - cdf_q).abs()
    if w_cdf is not None:
        diff = diff * (w_cdf if w_cdf.dim() > 1 else w_cdf.view(1, -1))
    return diff.sum(dim=-1).mean()


# ---------- Long-tail property alignment (power-law) ----------
@torch.no_grad()
def _prepare_tail_x(bins: int, tail_start_bin: int) -> torch.Tensor:
    """
    Precompute X for tail regression: x_k = log(1 + bin_index), where bin_index is 1-based.
    Returns shape [M_tail].
    """
    s = max(1, int(tail_start_bin))
    idx = torch.arange(s, bins + 1, dtype=torch.float32)  # 1-based inclusive
    x = torch.log1p(idx)  # log(1+bin)
    return x  # [M_tail]

def powerlaw_tail_slope(p: torch.Tensor, tail_start_bin: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Estimate per-sample power-law slope on the tail via least squares:
    y = log p_k, x = log(1 + bin_index), for bins >= tail_start_bin (1-based).
    p: [B, M] probabilities.
    Returns slopes: [B]
    """
    B, M = p.shape
    s = max(1, int(tail_start_bin)) - 1  # to 0-based slice
    p_tail = p[:, s:].clamp_min(eps)          # [B, M_tail]
    x = torch.log1p(torch.arange(s + 1, M + 1, device=p.device, dtype=torch.float32))  # [M_tail]
    y = torch.log(p_tail)                     # [B, M_tail]
    # Closed-form slope for simple linear regression
    x_mean = x.mean()
    y_mean = y.mean(dim=1, keepdim=True)
    num = ((x - x_mean) * (y - y_mean)).sum(dim=1)            # [B]
    den = ((x - x_mean) ** 2).sum() + 1e-12                   # scalar
    slope = num / den                                         # [B]
    return slope

def powerlaw_shape_loss(p_true: torch.Tensor, p_pred: torch.Tensor, tail_start_bin: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Shape matching on the tail in log-space (scale-invariant):
    Center both by per-sample mean across tail bins and compute L2.
    Returns batch mean loss.
    """
    B, M = p_true.shape
    s = max(1, int(tail_start_bin)) - 1
    x_true = torch.log(p_true[:, s:].clamp_min(eps))  # [B, M_tail]
    x_pred = torch.log(p_pred[:, s:].clamp_min(eps))  # [B, M_tail]
    x_true = x_true - x_true.mean(dim=1, keepdim=True)
    x_pred = x_pred - x_pred.mean(dim=1, keepdim=True)
    return ((x_true - x_pred) ** 2).mean()


def save_hist_overlay(p_true: np.ndarray, p_pred: np.ndarray, out_path: Path, title: str = ""):
    """Overlay plot of two histograms."""
    x = np.arange(1, p_true.shape[-1] + 1)
    plt.figure(figsize=(6, 4), dpi=160)
    plt.plot(x, p_true, label="target", linewidth=2)
    plt.plot(x, p_pred, label="pred", linestyle="--", linewidth=2)
    plt.xlabel("block length bin")
    plt.ylabel("probability")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ----------------- Training & Eval -----------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    out_dir = Path(args.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    # 1) Data
    train_set, val_set, train_loader, val_loader = build_loaders(args.csv, args.batch_size, args.num_workers, args.max_samples)

    # 2) Load & freeze ConditionEncoder
    ckpt = torch.load(args.cond_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    cond_in_dim = int(ckpt_args.get("cond_in_dim", 17))
    cond_d_model = int(ckpt_args.get("d_model", 128))
    cond_proj_dim = int(ckpt_args.get("proj_dim", 256))
    cond_temp = float(ckpt_args.get("temperature", 0.10))
    cond_layers = int(ckpt_args.get("num_layers", 3))

    cond_encoder = ConditionEncoder(
        in_dim=cond_in_dim,
        d_model=cond_d_model,
        proj_dim=cond_proj_dim,
        num_layers=cond_layers,
        dropout=ckpt_args.get("dropout", 0.1),  # Use the same dropout as training
        temperature=cond_temp,
    ).to(device)
    
    # Load with compatibility for old checkpoint structure
    try:
        cond_encoder.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        print(f"Warning: Direct loading failed ({e}). Attempting compatibility loading...")
        ik = cond_encoder.load_state_dict(ckpt["model"], strict=False)
        try:
            if hasattr(ik, "missing_keys") and ik.missing_keys:
                print(f"Missing keys: {ik.missing_keys}")
            if hasattr(ik, "unexpected_keys") and ik.unexpected_keys:
                print(f"Unexpected keys: {ik.unexpected_keys}")
        except Exception:
            pass
        print("Loaded with compatibility mode.")
    for p in cond_encoder.parameters():
        p.requires_grad = False
    cond_encoder.eval()
    print(f"Loaded ConditionEncoder from {args.cond_ckpt} (cond_emb dim = {cond_d_model})")

    # 3) Discover bins from one batch
    with torch.no_grad():
        tmp_batch = next(iter(train_loader))
        bins = tmp_batch["block_dists"].shape[-1]
    print(f"Detected bins for block_dist: {bins}")

    # Build exponential tail weights if requested (mean-normalized for stability)
    w_exp = None
    if getattr(args, 'exp_tail', False):
        w_exp = build_exp_tail_weights(
            bins=bins,
            beta=float(getattr(args, 'exp_tail_beta', 1.0)),
            tail_start=getattr(args, 'exp_tail_start', None),
            normalize=True,
        )  # [M], cpu for now

    # Optional down-binning
    use_rebin = (args.bins_new > 0) and (args.bins_new < bins)
    if use_rebin:
        W_rebin = make_rebin_matrix(bins, args.bins_new, scheme="uniform").to(device)  # [Bnew,Bold]
        bins_old = bins
        bins = args.bins_new
        print(f"Rebin enabled (TRAIN=uniform): {bins_old} -> {bins} bins (cli asked: {args.rebin_scheme})")
    else:
        W_rebin = None

    # 4) DiT model & schedule
    dit = DiT1D(
        bins=bins, cond_dim=cond_d_model,
        d_model=args.dit_d_model, n_layers=args.dit_layers, n_heads=args.dit_heads,
        ff_mult=args.dit_ff_mult, dropout=args.dit_dropout, film_each_layer=True
    ).to(device)
    schedule = NoiseSchedule(T=args.T).to(device)

    # ---- Skew mitigation stats (optional) ----
    w_bin = None
    z_prior = None
    if args.reweight_bins or args.residual_prior:
        freq_old = compute_bin_frequency(train_loader, device, (W_rebin.shape[1] if W_rebin is not None else bins))
        if W_rebin is not None:
            freq = (W_rebin @ freq_old.view(-1, 1)).view(-1)
        else:
            freq = freq_old
        w_bin = ((1.0 / freq) ** float(args.reweight_alpha))
        w_bin = w_bin / w_bin.mean()
    if args.residual_prior:
        prior_old = compute_global_prior(train_loader, device, (W_rebin.shape[1] if W_rebin is not None else bins))
        if W_rebin is not None:
            prior_p = (W_rebin @ prior_old.view(-1,1)).view(-1)
        else:
            prior_p = prior_old
        z_prior = hist_to_logits(prior_p.unsqueeze(0)).detach()  # [1, M]
    if w_exp is not None:
        w_exp = w_exp.to(device)

    # 5) Optimizer
    optimizer = optim.AdamW(dit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 6) Training
    global_step = 0
    steps_per_epoch = len(train_loader)
    print(f"Per-step batch={args.batch_size}, accum_steps={args.accum_steps} -> effective batch={args.batch_size * args.accum_steps}")

    for epoch in range(1, args.epochs + 1):
        dit.train()
        running = 0.0
        for step_i, batch in enumerate(train_loader, start=1):
            cond_feat = batch["condition_features"].to(device, non_blocking=True)    # [B, C=17]
            p = batch["block_dists"].to(device, non_blocking=True)                   # [B, M]
            if W_rebin is not None:
                p = p @ W_rebin.t()
            z0 = hist_to_logits(p)                                                   # [B, M]

            with torch.no_grad():
                cond = get_cond_emb(cond_encoder, cond_feat)                         # [B, cond_dim]

            # Classifier-free guidance training trick: random drop condition
            if args.cfg_prob > 0:
                drop_mask = (torch.rand(cond.size(0), device=cond.device) < args.cfg_prob).float().unsqueeze(-1)
                cond = cond * (1.0 - drop_mask)  # drop -> zero

            # sample timesteps
            t = torch.randint(low=0, high=schedule.T, size=(p.size(0),), device=device, dtype=torch.long)

            # autocast
            try:
                autocast_context = torch.amp.autocast('cuda', enabled=(device.type == "cuda"))
            except AttributeError:
                autocast_context = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))

            with autocast_context:
                # Targets in logits space (optionally residualized by global prior)
                if args.residual_prior and (z_prior is not None):
                    z0_target = z0 - z_prior
                else:
                    z0_target = z0

                # Diffusion step in v-parameterization
                x_t, v_target = q_sample_vparam(z0_target, t, schedule)  # [B,M]
                v_hat = dit(x_t=x_t, t=t, cond=cond)                     # [B,M]

                # Per-bin weighted MSE in v-space (optionally with exponential tail emphasis)
                w = None
                if args.reweight_bins and (w_bin is not None):
                    w = w_bin.view(1, -1)
                if w_exp is not None:
                    w = w_exp.view(1, -1) if w is None else (w * w_exp.view(1, -1))
                if w is not None:
                    v_mse = ((v_hat - v_target) ** 2 * w).mean()
                else:
                    v_mse = torch.mean((v_hat - v_target) ** 2)

                # Optional auxiliary KL on histogram reconstruction (x0) to stabilize fitting
                kl_aux = None
                if getattr(args, 'lambda_kl', 0.0) > 0.0:
                    # Reconstruct x0 from v-parameterization
                    _, alpha_bar_t, one_minus_alpha_bar_t = schedule.gather_alpha(t)  # [B,1]
                    sqrt_alpha_bar = alpha_bar_t.sqrt()
                    sqrt_one_minus_alpha_bar = one_minus_alpha_bar_t.sqrt()
                    x0_hat_aux = sqrt_alpha_bar * x_t + sqrt_one_minus_alpha_bar * v_hat
                    if args.residual_prior and (z_prior is not None):
                        x0_hat_aux = x0_hat_aux + z_prior
                    # temperature 1.0 during training; eval uses args.logit_temp
                    p_hat_aux = torch.softmax(x0_hat_aux, dim=-1)

                    # ----- Long-tail property alignment (no reweighting) -----
                    tail_loss = None
                    if getattr(args, 'lambda_powerlaw', 0.0) and args.lambda_powerlaw > 0.0:
                        # slope alignment
                        slope_true = powerlaw_tail_slope(p, args.tail_start_bin)          # [B]
                        slope_pred = powerlaw_tail_slope(p_hat_aux, args.tail_start_bin)  # [B]
                        slope_mse = torch.mean((slope_true - slope_pred) ** 2)
                        # optional shape alignment in log-space
                        if getattr(args, 'powerlaw_shape_w', 0.0) and args.powerlaw_shape_w > 0.0:
                            shape_mse = powerlaw_shape_loss(p, p_hat_aux, args.tail_start_bin)
                            tail_loss = slope_mse + float(args.powerlaw_shape_w) * shape_mse
                        else:
                            tail_loss = slope_mse

                    # focal underestimation weights: only emphasize bins where p > p_hat
                    if getattr(args, 'focal_tail_gamma', 0.0) and args.focal_tail_gamma > 0.0:
                        w_focal = (p - p_hat_aux).clamp(min=0.0).pow(float(args.focal_tail_gamma))  # [B,M]
                    else:
                        w_focal = 1.0

                    if w_exp is not None:
                        w_total = w_focal * w_exp.view(1, -1)
                        kl_aux = weighted_kl_divergence(p, p_hat_aux, w_total)
                    else:
                        if isinstance(w_focal, torch.Tensor):
                            kl_aux = weighted_kl_divergence(p, p_hat_aux, w_focal)
                        else:
                            kl_aux = kl_divergence(p, p_hat_aux)

                # ----- Tail mass / soft-quantile alignment (data-driven, differentiable) -----
                tail_mass_reg = None
                if getattr(args, "lambda_tail_mass", 0.0) and args.lambda_tail_mass > 0.0:
                    tail_mass_reg = tail_mass_loss(p, p_hat_aux, args.tail_start_bin)

                quantile_reg = None
                if getattr(args, "lambda_quantile", 0.0) and args.lambda_quantile > 0.0:
                    # parse levels once (cache on args)
                    if not hasattr(args, "_quantile_levels_tensor"):
                        try:
                            lv = [float(x) for x in str(args.quantile_levels).split(",") if x.strip()]
                        except Exception:
                            lv = [0.90, 0.95]
                        lv = [min(max(1e-6, v), 1.0 - 1e-6) for v in lv]
                        args._quantile_levels_tensor = torch.tensor(lv, dtype=torch.float32, device=p.device)
                    quantile_reg = soft_quantile_loss(p, p_hat_aux, args._quantile_levels_tensor, args.quantile_kappa)

                # Optional reverse-KL for coverage (mitigate under-coverage on tails)
                rkl_aux = None
                if getattr(args, 'lambda_rkl', 0.0) and args.lambda_rkl > 0.0:
                    if w_exp is not None:
                        rkl_aux = weighted_reverse_kl_divergence(p_hat_aux, p, w_exp.view(1, -1))
                    else:
                        rkl_aux = reverse_kl_divergence(p_hat_aux, p)

                loss = v_mse
                if kl_aux is not None:
                    loss = loss + args.lambda_kl * kl_aux
                if rkl_aux is not None:
                    loss = loss + args.lambda_rkl * rkl_aux

                if 'tail_loss' in locals() and tail_loss is not None:
                    loss = loss + args.lambda_powerlaw * tail_loss

                # accumulate
                if tail_mass_reg is not None:
                    loss = loss + args.lambda_tail_mass * tail_mass_reg
                if quantile_reg is not None:
                    loss = loss + args.lambda_quantile * quantile_reg

                # Optional EMD(CDF) regularizer on histogram shape
                if args.lambda_emd > 0.0 and epoch > args.emd_warmup_epochs:
                    # Recover x0 from v-parameterization
                    _, alpha_bar_t, one_minus_alpha_bar_t = schedule.gather_alpha(t)  # [B,1]
                    sqrt_alpha_bar = alpha_bar_t.sqrt()
                    sqrt_one_minus_alpha_bar = one_minus_alpha_bar_t.sqrt()
                    x0_hat = sqrt_alpha_bar * x_t + sqrt_one_minus_alpha_bar * v_hat
                    if args.residual_prior and (z_prior is not None):
                        x0_hat = x0_hat + z_prior
                    p_hat = torch.softmax(x0_hat, dim=-1)
                    if getattr(args, 'tail_emd', False) and (w_exp is not None):
                        emd_loss = weighted_emd_1d(p, p_hat, w_cdf=w_exp.view(1, -1))
                    else:
                        emd_loss = emd1d_torch(p, p_hat).mean()
                    loss = loss + args.lambda_emd * emd_loss

                loss = loss / args.accum_steps

            scaler.scale(loss).backward()
            running += loss.item()
            global_step += 1

            if (global_step % args.accum_steps) == 0:
                # gradient clipping (with AMP): unscale then clip
                if args.grad_clip and args.grad_clip > 0:
                    try:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(dit.parameters(), max_norm=args.grad_clip)
                    except Exception:
                        pass
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (global_step % args.log_every) == 0:
                avg = running * args.accum_steps / args.log_every
                msg = f"[epoch {epoch:02d} | step {global_step}] loss={avg:.5f}"
                if 'v_mse' in locals():
                    try:
                        msg += f" | v_mse={float(v_mse):.4f}"
                    except Exception:
                        pass
                if args.lambda_emd > 0.0 and 'emd_loss' in locals():
                    try:
                        msg += f" emd_reg={float(emd_loss):.4f}"
                    except Exception:
                        pass
                if kl_aux is not None:
                    try:
                        msg += f" kl_aux={float(kl_aux):.4f}"
                    except Exception:
                        pass
                if rkl_aux is not None:
                    try:
                        msg += f" rkl_aux={float(rkl_aux):.4f}"
                    except Exception:
                        pass
                if 'tail_mass_reg' in locals() and tail_mass_reg is not None:
                    try:
                        msg += f" tailM={float(tail_mass_reg):.4f}"
                    except Exception:
                        pass
                if 'quantile_reg' in locals() and quantile_reg is not None:
                    try:
                        msg += f" qReg={float(quantile_reg):.4f}"
                    except Exception:
                        pass
                print(msg)
                running = 0.0

        # ---- Eval after each epoch ----
        kl_mean, emd_mean = evaluate(dit, cond_encoder, val_loader, schedule, args, device, bins, epoch, out_dir, W_rebin)
        print(f"[epoch {epoch:02d}] VAL: KL={kl_mean:.5f} | EMD={emd_mean:.5f}")

        # Save ckpt
        ckpt_path = out_dir / "ckpts" / f"dit_epoch{epoch:02d}.pt"
        torch.save({
            "model": dit.state_dict(),
            "epoch": epoch,
            "bins": bins,
            "args": vars(args),
        }, ckpt_path)
        print(f"Saved: {ckpt_path}")

    # save final
    final_path = out_dir / "ckpts" / "dit_final.pt"
    torch.save({
        "model": dit.state_dict(),
        "epoch": args.epochs,
        "bins": bins,
        "args": vars(args),
    }, final_path)
    print(f"Saved final: {final_path}")


@torch.no_grad()
def evaluate(dit, cond_encoder, val_loader, schedule, args, device, bins, epoch, out_dir: Path, W_rebin=None) -> Tuple[float, float]:
    dit.eval()
    cond_encoder.eval()

    # Prepare prior for residual mode
    z_eval_prior = None
    if args.residual_prior:
        # infer bins from first batch in val_loader
        try:
            first = next(iter(val_loader))
            _bins = first["block_dists"].shape[-1]
        except Exception:
            _bins = bins
        prior_old = compute_global_prior(val_loader, device, bins=_bins)
        if W_rebin is not None and prior_old.numel() != bins:
            W_tmp = W_rebin if W_rebin.shape[1] == prior_old.numel() else make_rebin_matrix(prior_old.numel(), bins).to(device)
            prior_p = (W_tmp @ prior_old.view(-1,1)).view(-1)
        else:
            prior_p = prior_old
        z_eval_prior = hist_to_logits(prior_p.unsqueeze(0)).to(device)

    # Holder for W_rebin for use inside batch loop
    class _W: pass
    W_rebin_holder = _W(); W_rebin_holder.W = W_rebin

    kls, emds = [], []
    first_batch = True
    for batch in val_loader:
        cond_feat = batch["condition_features"].to(device)
        p_true = batch["block_dists"].to(device)
        if hasattr(W_rebin_holder, 'W') and (W_rebin_holder.W is not None):
            p_true = p_true @ W_rebin_holder.W.t()

        cond = get_cond_emb(cond_encoder, cond_feat)

        # uncond cond for CFG (zeros)
        uncond = torch.zeros_like(cond)

        eval_steps = args.eval_sample_steps if getattr(args, 'eval_sample_steps', None) else args.sample_steps
        z0_hat, _p_ignored = ddim_sample(
            model=dit,
            cond=cond,
            schedule=schedule,
            steps=eval_steps,
            eta=0.0,
            guidance=(uncond, args.guidance_scale) if args.guidance_scale > 1.0 else None,
            bins=bins,
            tau=1.0,
        )
        if args.residual_prior and (z_eval_prior is not None):
            z0_hat = z0_hat + z_eval_prior
        # Apply evaluation-time logit temperature for calibration
        temp_eval = getattr(args, 'logit_temp', 1.0)
        p_hat = torch.softmax(z0_hat / max(1e-6, temp_eval), dim=-1)

        kls.append(kl_divergence(p_true, p_hat).item())
        emds.append(emd_1d(p_true, p_hat).item())

        # Save some overlays for the first validation batch
        if first_batch:
            first_batch = False
            B = min(p_true.size(0), args.sample_plots)
            for i in range(B):
                pt = p_true[i].detach().cpu().numpy()
                pp = p_hat[i].detach().cpu().numpy()
                save_hist_overlay(pt, pp, out_dir / "samples" / f"epoch{epoch:02d}_sample{i:02d}.{args.fig_format}",
                                  title=f"epoch {epoch} sample {i}")

    return float(np.mean(kls)), float(np.mean(emds))


# ----------------- CLI -----------------
def build_argparser():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--csv", type=str, required=True, help="Path to copolymer.csv")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)

    # condition encoder checkpoint
    p.add_argument("--cond_ckpt", type=str, required=True, help=".pt checkpoint file of ConditionEncoder")
    p.add_argument("--cond_dropout", type=float, default=0.0, help="Extra dropout inside ConditionEncoder projection (kept frozen).")

    # training
    p.add_argument("--out_dir", type=str, default="outputs/diffusion_runs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    # DiT config
    p.add_argument("--dit_d_model", type=int, default=256)
    p.add_argument("--dit_layers", type=int, default=8)
    p.add_argument("--dit_heads", type=int, default=8)
    p.add_argument("--dit_ff_mult", type=int, default=4)
    p.add_argument("--dit_dropout", type=float, default=0.0)

    # diffusion schedule
    p.add_argument("--T", type=int, default=1000)

    # sampling / CFG
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--eval_sample_steps", type=int, default=None,
                   help="If set, number of DDIM steps used at evaluation; otherwise falls back to --sample_steps.")
    p.add_argument("--guidance_scale", type=float, default=1.0, help=">1 to enable CFG at eval")
    p.add_argument("--cfg_prob", type=float, default=0.1, help="Probability to drop condition during training for CFG learning.")

    # plots
    p.add_argument("--fig_format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--sample_plots", type=int, default=6, help="Number of overlay plots to save per epoch.")

    # rebinning (downsample number of histogram bins)
    p.add_argument("--bins_new", type=int, default=0, help="If >0 and <detected bins, rebin targets to this many bins for training/eval.")
    p.add_argument("--rebin_scheme", type=str, default="uniform", choices=["uniform"], help="Rebinning scheme (uniform for now).")

    # ---------------- Skew mitigation ----------------
    p.add_argument("--reweight_bins", action="store_true",
                   help="Use inverse-frequency per-bin reweighting in v-loss.")
    p.add_argument("--reweight_alpha", type=float, default=0.6,
                   help="Exponent for inverse-frequency reweighting (0.5~0.8 recommended).")
    p.add_argument("--lambda_emd", type=float, default=0.0,
                   help="Weight of EMD(CDF) regularizer on p_hat vs p_true (0.05~0.1).")
    p.add_argument("--residual_prior", action="store_true",
                   help="Train diffusion on residual logits: z0 - z_prior (global mean prior).")

    # training stability
    p.add_argument("--grad_clip", type=float, default=1.0, help="Clip global grad-norm (0 = disable).")
    p.add_argument("--emd_warmup_epochs", type=int, default=3, help="Enable EMD regularizer only after this many epochs.")
    # calibration & auxiliary losses
    p.add_argument("--logit_temp", type=float, default=1.0,
                   help="Temperature applied to logits at evaluation to calibrate sharpness (softmax(z/τ)).")
    p.add_argument("--lambda_kl", type=float, default=0.0,
                   help="Weight for auxiliary KL(p_true || p_hat(x0)) term during training.")

    # Exponential tail emphasis (loss-space), aligned with log-scale visualization
    p.add_argument("--exp_tail", action="store_true",
                   help="Enable exponential per-bin loss weighting toward the tail (mean-normalized).")
    p.add_argument("--exp_tail_beta", type=float, default=1.0,
                   help="Strength of exponential growth toward the tail; larger → stronger emphasis.")
    p.add_argument("--exp_tail_start", type=int, default=None,
                   help="1-based bin index where tail emphasis starts; bins before this keep weight 1.")

    # -------- New: Tail-aware extras --------
    p.add_argument("--focal_tail_gamma", type=float, default=0.0,
                   help="If >0, apply focal underestimation ((p - p_hat)_+^gamma) inside KL to emphasize bins the model underestimates.")
    p.add_argument("--lambda_rkl", type=float, default=0.0,
                   help="Weight for reverse KL term KL(p_hat || p) to improve coverage.")
    p.add_argument("--tail_emd", action="store_true",
                   help="If set, apply exponential tail weights to EMD(CDF) regularizer as well.")

    # -------- Long-tail property alignment (data-driven, no reweighting) --------
    p.add_argument("--lambda_powerlaw", type=float, default=0.0,
                   help="Weight for power-law tail alignment loss (slope on log p vs log(1+bin)).")
    p.add_argument("--powerlaw_shape_w", type=float, default=0.0,
                   help="Extra weight for tail shape alignment in log-space; 0 disables.")
    # Reuse --tail_start_bin from existing flags if present; otherwise define default here
    if "--tail_start_bin" not in {a.option_strings[0] for a in p._actions if a.option_strings}:
        p.add_argument("--tail_start_bin", type=int, default=21,
                       help="1-based bin index where the tail starts for tail alignment losses.")

    # -------- Tail mass & quantile objectives (differentiable) --------
    p.add_argument("--lambda_tail_mass", type=float, default=0.0,
                   help="Weight for tail mass alignment loss: MSE between sum_{i>=tail_start} p_i of GT and prediction.")
    p.add_argument("--lambda_quantile", type=float, default=0.0,
                   help="Weight for soft-quantile alignment loss across specified levels (e.g., 0.9,0.95,0.99).")
    p.add_argument("--quantile_levels", type=str, default="0.90,0.95",
                   help="Comma-separated quantile levels in (0,1); used when --lambda_quantile>0.")
    p.add_argument("--quantile_kappa", type=float, default=25.0,
                   help="Sharpness for soft quantile (sigmoid) approximation; larger gives sharper step.")

    return p


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
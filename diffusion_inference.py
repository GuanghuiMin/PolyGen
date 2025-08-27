"""
# DDIM (deterministic)
python diffusion_inference.py \
--csv ./data/copolymer.csv \
--cond_ckpt outputs/cond_enc_runs/cond_encoder_final.pt \
--dit_ckpt outputs/diffusion_runs/ckpts/dit_final.pt \
--out_dir outputs/vis_dit_final \
--sample_steps 200 --guidance_scale 1.2 --eval_bins 50 --logit_temp 1.4 --fig_format pdf

# DDPM (ancestral, eta=1.0)
python diffusion_inference.py \
--csv ./data/copolymer.csv \
--cond_ckpt outputs/cond_enc_runs/cond_encoder_final.pt \
--dit_ckpt outputs/diffusion_runs/ckpts/dit_final.pt \
--out_dir outputs/vis_dit_ddpm \
--sample_steps 200 --use_ddpm --guidance_scale 1.2 --eval_bins 50 --logit_temp 1.4
"""

import os
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Perf knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Project imports
from data.dataset import ChainSetDataset, collate_fn_set_transformer
from data.block_dist import mayo_lewis_from_sequence
from src.encoder import ConditionEncoder

from src.diffusion import (
    DiT1D, NoiseSchedule, hist_to_logits, logits_to_hist, q_sample_vparam,
    diffusion_loss_vpred, ddim_sample
)

# ---- Extras: dataset prior & temperature softmax ----
@torch.no_grad()
def compute_bin_frequency(loader, device, bins: int, max_batches: int = 200):
    """Estimate dataset-level average probability per bin (freq[M])."""
    bin_sum = torch.zeros(bins, device=device)
    n = 0
    for i, batch in enumerate(loader):
        p = batch["block_dists"].to(device)  # [B,M]
        if p.size(-1) != bins:
            if p.size(-1) > bins:
                p = p[..., :bins]
            else:
                pad = bins - p.size(-1)
                p = F.pad(p, (0, pad))
        bin_sum += p.sum(dim=0)
        n += p.size(0)
        if i + 1 >= max_batches:
            break
    freq = (bin_sum / max(1, n)).clamp_min(1e-8)
    return freq

@torch.no_grad()
def compute_global_prior(loader, device, bins: int, max_batches: int = 200):
    freq = compute_bin_frequency(loader, device, bins, max_batches)
    prior_p = (freq / freq.sum()).clamp_min(1e-8)
    return prior_p


# ----------------- Utils -----------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return (p * (p.log() - q.log())).sum(dim=-1)


def emd_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    cdf_p = p.cumsum(dim=-1)
    cdf_q = q.cumsum(dim=-1)
    return (cdf_p - cdf_q).abs().sum(dim=-1)


def adjust_tensor_bins(tensor: torch.Tensor, target_bins: int) -> torch.Tensor:
    """Adjust tensor to match target number of bins by truncating or padding."""
    current_bins = tensor.size(-1)
    if current_bins == target_bins:
        return tensor
    elif current_bins > target_bins:
        return tensor[..., :target_bins]
    else:
        pad_size = target_bins - current_bins
        return F.pad(tensor, (0, pad_size), mode='constant', value=0.0)


def adjust_bins_and_norm(tensor: torch.Tensor, target_bins: int) -> torch.Tensor:
    """Truncate/pad to target_bins and renormalize to sum=1 along the last dim."""
    t = adjust_tensor_bins(tensor, target_bins)
    s = t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return t / s


def save_hist_overlay(p_true: np.ndarray, p_pred: np.ndarray, out_path: Path, title: str = "", eval_bins: int = None, sequences: List[str] = None):
    """Save overlay histogram with log scale on y-axis and Mayo-Lewis theoretical curve."""
    if eval_bins is None:
        max_bins = min(50, p_true.shape[-1])
    else:
        max_bins = eval_bins
    x = np.arange(1, max_bins + 1)
    plt.figure(figsize=(6, 4), dpi=160)
    width = 0.4
    p_true_plot = p_true[:max_bins] if len(p_true) >= max_bins else np.pad(p_true, (0, max_bins - len(p_true)))
    p_pred_plot = p_pred[:max_bins] if len(p_pred) >= max_bins else np.pad(p_pred, (0, max_bins - len(p_pred)))
    plt.bar(x - width/2, p_true_plot, width=width, label="ground truth", alpha=0.8, color='#2E8B57')
    plt.bar(x + width/2, p_pred_plot, width=width, label="prediction", alpha=0.8, color='#FF6B35')
    
    # Add Mayo-Lewis theoretical curve if sequences are provided
    if sequences is not None:
        try:
            mayo_lewis_dist = mayo_lewis_from_sequence(sequences, max_length=max_bins)
            plt.plot(x, mayo_lewis_dist, 'o--', linewidth=2, markersize=4, color='#8B0000', label='Mayo-Lewis theory', alpha=0.8)
        except Exception as e:
            print(f"Warning: Could not compute Mayo-Lewis curve: {e}")
    
    plt.xlabel("block length bin", fontsize=14, fontweight='bold')
    plt.ylabel("probability", fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.ylim(bottom=1e-3)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    legend = plt.legend(fontsize=12)
    for text in legend.get_texts():
        text.set_weight('bold')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_topk_panel(trues: np.ndarray, preds: np.ndarray, ids: List[int], out_path: Path, title: str = "", eval_bins: int = 50, sequences: List[List[str]] = None):
    """Draw a panel (grid) of target vs prediction overlays for given indices with Mayo-Lewis curves."""
    import math as _math
    K = len(ids)
    cols = min(4, K)
    rows = _math.ceil(K / cols)
    max_bins = eval_bins
    x = np.arange(1, max_bins + 1)
    plt.figure(figsize=(4*cols, 3*rows), dpi=160)
    for i, idx in enumerate(ids):
        ax = plt.subplot(rows, cols, i+1)
        width = 0.4
        true_data = trues[idx]
        pred_data = preds[idx]
        
        if len(true_data) < max_bins:
            true_padded = np.pad(true_data, (0, max_bins - len(true_data)), mode='constant', constant_values=0)
        else:
            true_padded = true_data[:max_bins]
        if len(pred_data) < max_bins:
            pred_padded = np.pad(pred_data, (0, max_bins - len(pred_data)), mode='constant', constant_values=0)
        else:
            pred_padded = pred_data[:max_bins]
        ax.bar(x - width/2, true_padded, width=width, label="ground truth", alpha=0.8, color='#2E8B57')
        ax.bar(x + width/2, pred_padded, width=width, label="prediction", alpha=0.8, color='#FF6B35')
        
        # Add Mayo-Lewis theoretical curve if sequences are provided
        if sequences is not None and i < len(sequences) and sequences[i] is not None:
            try:
                mayo_lewis_dist = mayo_lewis_from_sequence(sequences[i], max_length=max_bins)
                ax.plot(x, mayo_lewis_dist, 'o--', linewidth=2, markersize=3, color='#8B0000', label='Mayo-Lewis theory', alpha=0.8)
            except Exception as e:
                print(f"Warning: Could not compute Mayo-Lewis curve for idx {idx}: {e}")
        
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-3, top=1)
        ax.set_title(f"idx={idx}", fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=10, labelcolor='black', width=1.5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')
        if i % cols == 0:
            ax.set_ylabel("probability", fontsize=12, fontweight='bold')
        if i // cols == rows - 1:
            ax.set_xlabel("bin", fontsize=12, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    figlegend = plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12)
    for text in figlegend.get_texts():
        text.set_weight('bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend at top
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_mean_overlay(trues: np.ndarray, preds: np.ndarray, out_path: Path, title: str = "", eval_bins: int = 50, sequences: List[List[str]] = None):
    """Plot side-by-side bars for mean(gt) vs mean(pred) with log probabilities and Mayo-Lewis theoretical curve."""
    mean_t = trues.mean(axis=0)
    mean_p = preds.mean(axis=0)
    max_bins = eval_bins
    x = np.arange(1, max_bins + 1)
    if len(mean_t) < max_bins:
        mean_t_padded = np.pad(mean_t, (0, max_bins - len(mean_t)), mode='constant', constant_values=0)
    else:
        mean_t_padded = mean_t[:max_bins]
    if len(mean_p) < max_bins:
        mean_p_padded = np.pad(mean_p, (0, max_bins - len(mean_p)), mode='constant', constant_values=0)
    else:
        mean_p_padded = mean_p[:max_bins]
    plt.figure(figsize=(6, 4), dpi=160)
    width = 0.4
    plt.bar(x - width/2, mean_t_padded, width=width, label="mean ground truth", alpha=0.7)
    plt.bar(x + width/2, mean_p_padded, width=width, label="mean prediction", alpha=0.7)
    
    # Add Mayo-Lewis theoretical curve if sequences are provided
    if sequences is not None:
        try:
            # Flatten sequences list and compute theoretical distribution
            flat_sequences = []
            for seq_list in sequences:
                flat_sequences.extend(seq_list)
            mayo_lewis_dist = mayo_lewis_from_sequence(flat_sequences, max_length=max_bins)
            plt.plot(x, mayo_lewis_dist, 'o--', linewidth=2, markersize=4, color='#8B0000', label='Mayo-Lewis theory', alpha=0.8)
        except Exception as e:
            print(f"Warning: Could not compute Mayo-Lewis curve: {e}")
    
    plt.xlabel("block length bin",fontsize=14, fontweight='bold')
    plt.ylabel("probability",fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.ylim(bottom=1e-3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def compute_cond_norm_stats(dataset, cond_dim: int, max_samples: Optional[int] = None):
    """Compute z-score stats for condition_features over dataset."""
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    mean = torch.zeros(cond_dim, dtype=torch.float64)
    m2 = torch.zeros(cond_dim, dtype=torch.float64)
    count = 0
    for i in range(n):
        x = dataset[i]['condition_features'].to(dtype=torch.float64)
        count += 1
        delta = x - mean
        mean += delta / count
        m2 += delta * (x - mean)
    var = m2 / max(1, (count - 1))
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean.to(torch.float32), std.to(torch.float32)


@torch.no_grad()
def get_cond_emb(cond_encoder: ConditionEncoder, cond_feat: torch.Tensor, normalize: str,
                 mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    if normalize == "zscore" and mean is not None and std is not None:
        cond_feat = (cond_feat - mean.to(cond_feat.device)) / (std.to(cond_feat.device) + 1e-6)
    elif normalize == "layernorm":
        ln = nn.LayerNorm(cond_feat.size(-1)).to(cond_feat.device)
        cond_feat = ln(cond_feat)
    cond_encoder.eval()
    out = cond_encoder(cond=cond_feat)
    return out["cond_emb"]


# ----------------- DDPM(-like) sampler in v-param -----------------
@torch.no_grad()
def ddpm_sample_logits(
    model: DiT1D,
    cond: torch.Tensor,
    schedule: NoiseSchedule,
    steps: int,
    bins: int,
    guidance: Optional[Tuple[torch.Tensor, float]] = None,
    eta: float = 1.0,
):
    """
    Ancestral sampler with v-parameterization.
    When eta=1.0 → DDPM; eta=0.0 → DDIM; 0<eta<1.0 → stochastic DDIM.

    Model is assumed to predict v (same as training with v-parameterization).
    Returns:
        z0_hat: logits at t=0 (before softmax)
        x0_traj (optional None here to reduce memory)
    """
    device = cond.device
    B = cond.size(0)

    # Create inference time-step schedule (t_T-1 ... 0) with approximately uniform spacing
    ts = torch.linspace(schedule.T - 1, 0, steps, device=device).long()

    # Initialize x_T ~ N(0, I) in logits space
    x_t = torch.randn(B, bins, device=device)

    # Helper to fetch cumulative alphas for a batch of t
    def gather_alpha_bar(t_long: torch.Tensor):
        # schedule.gather_alpha(t) assumed to return (alpha_t, alpha_bar_t, one_minus_alpha_bar_t)
        # but alpha_t not used; we just need alpha_bar and (1 - alpha_bar)
        _, alpha_bar_t, one_minus_alpha_bar_t = schedule.gather_alpha(t_long)
        return alpha_bar_t, one_minus_alpha_bar_t  # shapes: [B,1]

    for i in range(len(ts)):
        t = ts[i].expand(B)  # [B]
        t_prev = ts[i + 1].expand(B) if i + 1 < len(ts) else (torch.zeros_like(t))  # [B]

        # predict v with (optional) CFG
        if guidance is not None and guidance[1] > 1.0:
            uncond, w = guidance
            v_cond = model(x_t=x_t, t=t, cond=cond)          # [B,M]
            v_uncond = model(x_t=x_t, t=t, cond=uncond)      # [B,M]
            v_hat = (1 + w) * v_cond - w * v_uncond
        else:
            v_hat = model(x_t=x_t, t=t, cond=cond)

        # Convert v_hat to x0_hat and eps_hat at current t
        alpha_bar_t, one_minus_alpha_bar_t = gather_alpha_bar(t)              # [B,1], [B,1]
        sqrt_ab_t = alpha_bar_t.sqrt()
        sqrt_omb_t = one_minus_alpha_bar_t.sqrt()
        # x0 = sqrt(ab) * x_t - sqrt(1-ab) * v
        x0_hat = sqrt_ab_t * x_t - sqrt_omb_t * v_hat
        # eps = sqrt(1-ab) * x_t + sqrt(ab) * v
        eps_hat = sqrt_omb_t * x_t + sqrt_ab_t * v_hat

        # If last step: directly output x0 (logits)
        if i == len(ts) - 1:
            x_t = x0_hat
            break

        # Compute stochasticity sigma_t for (t -> t_prev)
        alpha_bar_prev, one_minus_alpha_bar_prev = gather_alpha_bar(t_prev)   # [B,1], [B,1]

        # DDIM sigma formula (Cao & Song), eta=1 gives DDPM-like ancestral noise
        # sigma_t^2 = eta^2 * (1 - ab_{t-1})/(1 - ab_t) * (1 - ab_t/ab_{t-1})
        sigma_t = eta * torch.sqrt(
            (one_minus_alpha_bar_prev / one_minus_alpha_bar_t).clamp_min(1e-12)
            * (1.0 - (alpha_bar_t / alpha_bar_prev).clamp_max(1.0))
        )  # [B,1]

        # Deterministic coefficient on eps_hat
        # c = sqrt(1 - ab_{t-1} - sigma_t^2)
        c = torch.sqrt((one_minus_alpha_bar_prev - sigma_t**2).clamp_min(0.0))

        # x_{t-1} reconstruction
        noise = torch.randn_like(x_t) if (eta > 0.0) else 0.0
        x_t = alpha_bar_prev.sqrt() * x0_hat + c * eps_hat + sigma_t * noise

    # Return final logits at t=0
    return x_t, None


# ----------------- Main eval & viz -----------------

def visualize(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    out_dir = Path(args.out_dir)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    # Data (use full dataset by default)
    dataset = ChainSetDataset(args.csv, max_samples=args.max_samples, contrastive=True)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn_set_transformer,
                        pin_memory=(device.type == "cuda"))

    # Condition encoder
    ckpte = torch.load(args.cond_ckpt, map_location=device, weights_only=False)
    ck_args = ckpte.get("args", {})
    cond_in_dim = int(ck_args.get("cond_in_dim", 17))
    cond_d_model = int(ck_args.get("d_model", 128))
    cond_proj_dim = int(ck_args.get("proj_dim", 256))
    cond_temp = float(ck_args.get("temperature", 0.10))
    cond_layers = int(ck_args.get("num_layers", 3))

    cond_encoder = ConditionEncoder(
        in_dim=cond_in_dim, d_model=cond_d_model, proj_dim=cond_proj_dim,
        num_layers=cond_layers, dropout=ck_args.get("dropout", 0.1), temperature=cond_temp
    ).to(device)
    cond_encoder.load_state_dict(ckpte["model"], strict=True)
    for p in cond_encoder.parameters():
        p.requires_grad = False
    cond_encoder.eval()

    # normalization stats if needed
    cond_mean = cond_std = None
    if args.normalize_cond == "zscore":
        cond_mean, cond_std = compute_cond_norm_stats(dataset, cond_in_dim, max_samples=args.norm_max_samples)

    # Diffusion model
    ckptd = torch.load(args.dit_ckpt, map_location=device, weights_only=False)
    model_bins = int(ckptd.get("bins", 50))
    dit_args = ckptd.get("args", {})
    dit = DiT1D(
        bins=model_bins, cond_dim=cond_d_model,
        d_model=int(dit_args.get("dit_d_model", 256)),
        n_layers=int(dit_args.get("dit_layers", 8)),
        n_heads=int(dit_args.get("dit_heads", 8)),
        ff_mult=int(dit_args.get("dit_ff_mult", 4)),
        dropout=float(dit_args.get("dit_dropout", 0.0)),
        film_each_layer=True,
    ).to(device)
    dit.load_state_dict(ckptd["model"], strict=True)
    dit.eval()

    schedule = NoiseSchedule(T=int(dit_args.get("T", args.sample_T))).to(device)
    
    # Check data bins vs model bins
    with torch.no_grad():
        tmp_batch = next(iter(loader))
        data_bins = tmp_batch["block_dists"].shape[-1]
    
    print(f"Data bins: {data_bins}, Model bins: {model_bins}")
    if data_bins != model_bins:
        print(f"Warning: Data has {data_bins} bins but model was trained with {model_bins} bins")
        print("Will truncate or pad data to match model dimensions")
    
    bins = model_bins  # Use model bins for generation

    eval_bins = int(getattr(args, "eval_bins",50))
    print(f"Evaluation bins (truncate/pad+renorm): {eval_bins}")

    # Prepare residual prior in MODEL bins (so it can be added to z0_hat before any rebin/renorm)
    z_eval_prior = None
    if getattr(args, "residual_prior", False):
        prior_p = compute_global_prior(loader, device, bins=model_bins, max_batches=getattr(args, "prior_max_batches", 200))
        z_eval_prior = hist_to_logits(prior_p.unsqueeze(0)).to(device)

    # Evaluation loops
    all_KL: List[float] = []
    all_EMD: List[float] = []
    all_idx: List[int] = []

    max_batches = math.inf if args.max_eval is None else max(1, args.max_eval)
    processed = 0
    row_idx = 0

    preds_store = []
    trues_store = []
    seqs_store = []  # Store sequences for Mayo-Lewis calculation

    # Decide sampler
    eta_eff = 1.0 if args.use_ddpm else float(getattr(args, 'eta', 0.0))
    mode = 'DDPM (eta=1.0)' if args.use_ddpm or eta_eff >= 1.0 - 1e-6 else ('stochastic DDIM (eta>0)' if eta_eff > 0 else 'DDIM (eta=0)')
    print(f"Sampling mode: {mode}; steps={args.sample_steps}; guidance_scale={args.guidance_scale}")

    for batch in loader:
        cond_feat = batch['condition_features'].to(device)
        p_true = batch['block_dists'].to(device)
        # Extract original sequences from the collated chain_sets
        # batch['chain_sets'] is [B, max_set_size, max_chain_length] encoded
        # We need to decode back to original string sequences
        batch_sequences = []
        for i in range(len(batch['chain_sets'])):
            sample_sequences = []
            for j in range(batch['chain_sets'].size(1)):
                if batch['set_masks'][i, j]:  # Valid sequence
                    encoded_seq = batch['chain_sets'][i, j]
                    # Decode back to string (1='A', 2='B', 0='PAD')
                    decoded = ''.join(['A' if x == 1 else 'B' if x == 2 else '' for x in encoded_seq.tolist()])
                    decoded = decoded.rstrip()  # Remove trailing empty chars
                    if decoded:  # Only add non-empty sequences
                        sample_sequences.append(decoded)
            batch_sequences.append(sample_sequences)
        
        cond = get_cond_emb(cond_encoder, cond_feat, args.normalize_cond, cond_mean, cond_std)

        uncond = torch.zeros_like(cond)

        if args.use_ddpm or eta_eff > 0.0:
            # DDPM / stochastic DDIM via ancestral update in v-param
            z0_hat, _ = ddpm_sample_logits(
                model=dit,
                cond=cond,
                schedule=schedule,
                steps=args.sample_steps,
                bins=bins,
                guidance=(uncond, args.guidance_scale) if args.guidance_scale > 1.0 else None,
                eta=eta_eff if not args.use_ddpm else 1.0,
            )
        else:
            # Plain DDIM (deterministic)
            z0_hat, _ = ddim_sample(
                model=dit,
                cond=cond,
                schedule=schedule,
                steps=args.sample_steps,
                eta=0.0,
                guidance=(uncond, args.guidance_scale) if args.guidance_scale > 1.0 else None,
                bins=bins,
                tau=1.0,
            )

        # add dataset prior back if residual mode is requested
        if z_eval_prior is not None:
            z0_hat = z0_hat + z_eval_prior

        # temperature softmax
        tau = max(1e-3, float(getattr(args, "logit_temp", 1.0)))
        p_pred = torch.softmax(z0_hat / tau, dim=-1)

        # Map both GT and prediction to eval_bins and renormalize
        p_true = adjust_bins_and_norm(p_true, eval_bins)
        p_pred = adjust_bins_and_norm(p_pred, eval_bins)

        kl = kl_divergence(p_true, p_pred).detach().cpu().numpy()
        emd = emd_1d(p_true, p_pred).detach().cpu().numpy()

        B = p_true.size(0)
        all_KL.extend(kl.tolist())
        all_EMD.extend(emd.tolist())
        all_idx.extend(list(range(row_idx, row_idx + B)))
        row_idx += B

        preds_store.append(p_pred.detach().cpu())
        trues_store.append(p_true.detach().cpu())
        seqs_store.extend(batch_sequences)  # Add sequences for this batch

        processed += 1
        if processed >= max_batches:
            break

    # concat tensors for overlays
    PRED = torch.cat(preds_store, dim=0).numpy()
    TRUE = torch.cat(trues_store, dim=0).numpy()

    # Mean distribution comparison (first eval_bins bins)
    out_dir = Path(args.out_dir)
    save_mean_overlay(TRUE, PRED, out_dir / f"mean_distribution_overlay.{args.fig_format}",
                      title=f"Mean ground truth vs Mean prediction (first {args.eval_bins} bins)", 
                      eval_bins=args.eval_bins, sequences=seqs_store)

    # Save metrics CSV
    df = pd.DataFrame({
        'index': all_idx,
        'KL': all_KL,
        'EMD': all_EMD,
    }).sort_values('index').reset_index(drop=True)
    csv_path = out_dir / 'metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics: {csv_path}")

    # Aggregate plots
    def hist_plot(values, title, fname):
        plt.figure(figsize=(5,4), dpi=160)
        plt.hist(values, bins=40, alpha=0.8, color='#4682B4', edgecolor='black', linewidth=0.8)
        plt.xlabel(title, fontsize=14, fontweight='bold')
        plt.ylabel('count', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_dir / f"{fname}.{args.fig_format}", bbox_inches='tight')
        plt.close()

    hist_plot(all_KL,  'KL divergence (val)', 'hist_KL')
    hist_plot(all_EMD, 'EMD-1D (val)',       'hist_EMD')

    # Scatter KL vs EMD
    plt.figure(figsize=(6, 4), dpi=160)
    plt.scatter(all_EMD, all_KL, s=15, alpha=0.7, color='#DC143C', edgecolors='black', linewidths=0.3)
    plt.xlabel('EMD-1D', fontsize=14, fontweight='bold')
    plt.ylabel('KL', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_KL_vs_EMD.{args.fig_format}", bbox_inches='tight')
    plt.close()

    # Choose best/median/worst by EMD
    emd_np = np.array(all_EMD)
    order = np.argsort(emd_np)
    picks = []
    if len(order) > 0:
        picks.append(order[0])                            # best
        picks.append(order[len(order)//2])                # median
        picks.append(order[-1])                           # worst

    for rank, idx in zip(["best","median","worst"], picks):
        pt = TRUE[idx]
        pp = PRED[idx]
        seq_for_idx = seqs_store[idx] if idx < len(seqs_store) else None
        save_hist_overlay(pt, pp, out_dir / "overlays" / f"{rank}_emd.{args.fig_format}", sequences=seq_for_idx)

    # ---- Top-K smallest KL samples ----
    kl_np = np.array(all_KL)
    order_kl = np.argsort(kl_np)
    K = min(args.topk_small_kl, len(order_kl))
    top_ids = order_kl[:K]
    (out_dir / "overlays_kl_small").mkdir(parents=True, exist_ok=True)
    for j, idx in enumerate(top_ids):
        pt = TRUE[idx]
        pp = PRED[idx]
        seq_for_idx = seqs_store[idx] if idx < len(seqs_store) else None
        save_hist_overlay(pt, pp, out_dir / "overlays_kl_small" / f"kl_small_{j:02d}_idx{idx}.{args.fig_format}", sequences=seq_for_idx)

    if K > 0:
        # Get sequences for the top K indices
        seqs_for_topk = [seqs_store[idx] if idx < len(seqs_store) else None for idx in top_ids]
        save_topk_panel(TRUE, PRED, top_ids.tolist(), out_dir / f"top{K}_small_KL_panel.{args.fig_format}", sequences=seqs_for_topk)

    print("Visualization done.")


# ----------------- CLI -----------------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default='./data/copolymer.csv')
    p.add_argument('--cond_ckpt', type=str, default='./outputs/cond_enc_runs/cond_encoder_final.pt')
    p.add_argument('--dit_ckpt', type=str, default='./outputs/dit_runs/dit_final.pt')
    p.add_argument('--out_dir', type=str, default='outputs/vis_dit_final')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--max_eval', type=int, default=None, help='Max number of evaluation batches')

    # normalization consistent with ConditionEncoder training
    p.add_argument('--normalize_cond', type=str, default='layernorm', choices=['none','zscore','layernorm'])
    p.add_argument('--norm_max_samples', type=int, default=None)

    # sampling
    p.add_argument('--sample_steps', type=int, default=50)
    p.add_argument('--sample_T', type=int, default=1000)
    p.add_argument('--guidance_scale', type=float, default=2.0)
    p.add_argument('--use_ddpm', action='store_true', help='Use DDPM (ancestral) sampling; equivalent to eta=1.0')
    p.add_argument('--eta', type=float, default=0.0, help='DDIM/DDPM interpolation: 0=DDIM, 1≈DDPM')

    # evaluation controls
    p.add_argument('--logit_temp', type=float, default=1.0,
                   help='Softmax temperature at eval (z/τ). τ<1 sharpens, τ>1 smooths.')
    p.add_argument('--residual_prior', action='store_true',
                   help='If model was trained on residual logits, add dataset prior back at eval.')
    p.add_argument('--prior_max_batches', type=int, default=200,
                   help='Batches to estimate dataset prior when --residual_prior is on.')

    # misc
    p.add_argument('--fig_format', type=str, default='pdf', choices=['pdf','png','svg'])
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--topk_small_kl', type=int, default=12, help='Number of smallest-KL samples to visualize (individual + panel).')

    p.add_argument('--eval_bins', type=int, default=50,
                   help='Truncate/pad both ground truth and predictions to this number of bins and renormalize (for fair comparison/plotting).')

    return p


def main():
    args = build_argparser().parse_args()
    visualize(args)


if __name__ == '__main__':
    main()
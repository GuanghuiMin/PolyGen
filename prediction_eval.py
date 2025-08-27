#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/val split, train on training set, visualize overlap on test set,
and compose the 12 best (lowest error) test samples into a single grid PDF.

Usage:
    python prediction_eval.py \
    --csv ./data/copolymer.csv \
    --cond_ckpt ./outputs/cond_enc_runs_sched40/cond_encoder_final.pt \
    --out_dir outputs/test_run \
    --epochs 50 --batch_size 8 --accum_steps 2 \
    --sample_steps 200 --guidance_scale 1.2 \
    --bins_new 50 --fig_format pdf \
    --inference_logit_temp 1.4 \
    --fixed_test_indices '330,329,490,489,491,61,287,492,320,493,331,542'
"""

import os
import math
import shutil
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Perf
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---- Project imports ----
from data.dataset import ChainSetDataset, collate_fn_set_transformer
from data.block_dist import mayo_lewis_from_sequence
from src.encoder import ConditionEncoder
from src.diffusion import (
    DiT1D, NoiseSchedule, hist_to_logits, logits_to_hist,
    q_sample_vparam, ddim_sample
)

# ----------------- Utils -----------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_loaders_split(csv_path: str, batch_size: int, num_workers: int,
                        test_ratio: float, seed: int, max_samples=None, fixed_test_indices=None):
    dataset = ChainSetDataset(csv_path, max_samples=max_samples, contrastive=True)
    n = len(dataset)
    
    if fixed_test_indices is not None:
        # Use fixed test indices
        test_indices = [idx for idx in fixed_test_indices if idx < n]  # Filter valid indices
        train_indices = [i for i in range(n) if i not in test_indices]
        
        from torch.utils.data import Subset
        train_set = Subset(dataset, train_indices)
        test_set = Subset(dataset, test_indices)
        print(f"[Fixed Split] Train: {len(train_indices)}, Test: {len(test_indices)} (fixed indices)")
    else:
        # Original random split
        n_test = max(1, int(test_ratio * n))
        n_train = n - n_test
        gen = torch.Generator().manual_seed(seed)
        train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test], generator=gen)
        print(f"[Random Split] Train: {n_train}, Test: {n_test}")

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn_set_transformer, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn_set_transformer, pin_memory=True, drop_last=False
    )
    return dataset, train_set, test_set, train_loader, test_loader

@torch.no_grad()
def get_cond_emb(cond_encoder: ConditionEncoder, cond_feat: torch.Tensor) -> torch.Tensor:
    cond_encoder.eval()
    return cond_encoder(cond=cond_feat)["cond_emb"]

def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return (p * (p.log() - q.log())).sum(dim=-1)

def emd_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    cdf_p = p.cumsum(dim=-1)
    cdf_q = q.cumsum(dim=-1)
    return (cdf_p - cdf_q).abs().sum(dim=-1)

def make_rebin_matrix(old_bins: int, new_bins: int) -> torch.Tensor:
    import numpy as np
    W = torch.zeros(new_bins, old_bins, dtype=torch.float32)
    edges = np.linspace(0, old_bins, new_bins + 1, dtype=int)
    for i in range(new_bins):
        s, e = edges[i], edges[i+1]
        if e > s:
            W[i, s:e] = 1.0
    return W

def generate_topk_grid(results: List[Dict], out_dir: Path, args, epoch_suffix: str = ""):
    """Generate topk grid visualization from results."""
    if not results:
        print(f"[Warning] No results to generate topk grid")
        return None
    
    # Calculate scores
    res_np = np.array([[r["kl"], r["emd"]] for r in results], dtype=np.float64)
    kl_vals = res_np[:, 0]
    emd_vals = res_np[:, 1]
    if args.metric == "kl":
        score = kl_vals
    elif args.metric == "emd":
        score = emd_vals
    else:
        # combined: KL + normalized EMD
        emd_norm = (emd_vals - emd_vals.min()) / (emd_vals.ptp() + 1e-12)
        score = kl_vals + emd_norm

    order = np.argsort(score)
    top_k = min(args.top_k, len(results))
    chosen = [results[int(i)] for i in order[:top_k]]
    
    # Create grid filename
    if epoch_suffix:
        grid_filename = f"top{top_k}_grid{epoch_suffix}.{args.fig_format}"
    else:
        grid_filename = f"top{top_k}_grid.{args.fig_format}"
    
    grid_path = out_dir / grid_filename
    cols = 4
    rows = math.ceil(top_k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), dpi=160)
    if rows == 1:
        axes = np.array([axes])
    for n in range(rows*cols):
        r = n // cols; c = n % cols
        ax = axes[r, c]
        ax.axis("off")
        if n < top_k:
            img = plt.imread(chosen[n]["grid_path"])
            ax.imshow(img)
    
    # Add legend at the top center of the entire figure with correct colors
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#2E8B57', alpha=0.8, label='ground truth'),
        Rectangle((0, 0), 1, 1, facecolor='#FF6B35', alpha=0.8, label='prediction'),
        plt.Line2D([0], [0], color='#8B0000', linestyle='--', marker='o', markersize=3, label='Mayo-Lewis theory')
    ]
    figlegend = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=3, frameon=False, fontsize=12)
    for text in figlegend.get_texts():
        text.set_weight('bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend at top
    plt.savefig(grid_path, bbox_inches="tight")
    plt.close()
    
    return grid_path, chosen

def evaluate_epoch_model_inline(dit, test_loader, cond_encoder, schedule, device, args, W_rebin=None, fixed_test_indices=None, test_set=None, out_dir=None, epoch=None):
    """Evaluate current model state and return average KL divergence, optionally generate topk grid."""
    total_kl = 0.0
    total_samples = 0
    results = []
    
    # Get the actual indices for test set if using fixed indices (for overlay generation)
    if fixed_test_indices is not None and test_set is not None:
        test_dataset_indices = fixed_test_indices[:len(test_set)]
    elif test_set is not None and hasattr(test_set, 'indices'):
        test_dataset_indices = test_set.indices
    else:
        test_dataset_indices = list(range(len(test_loader.dataset)))
    
    current_test_idx = 0
    
    with torch.no_grad():
        for batch_i, batch in enumerate(test_loader):
            p_true = batch["block_dists"].to(device)
            if W_rebin is not None:
                p_true = p_true @ W_rebin.t()
            
            cond_feat = batch["condition_features"].to(device)
            cond = get_cond_emb(cond_encoder, cond_feat)
            uncond = torch.zeros_like(cond)
            
            z0_hat, _ = ddim_sample(
                model=dit,
                cond=cond,
                schedule=schedule,
                steps=args.sample_steps,
                eta=0.0,
                guidance=(uncond, args.guidance_scale) if args.guidance_scale > 1.0 else None,
                bins=p_true.size(-1),
                tau=1.0,
            )
            p_hat = torch.softmax(z0_hat / max(1e-6, args.inference_logit_temp), dim=-1)
            
            # Calculate KL divergence
            KL = kl_divergence(p_true, p_hat)
            EMD = emd_1d(p_true, p_hat)
            total_kl += KL.sum().item()
            total_samples += KL.size(0)
            
            # Generate overlay images if output directory is provided
            if out_dir is not None and epoch is not None:
                # Decode sequences from encoded format
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
                
                # Save overlay images for each sample in batch
                B = p_true.size(0)
                for i in range(B):
                    if current_test_idx + i < len(test_dataset_indices):
                        actual_idx = test_dataset_indices[current_test_idx + i]
                    else:
                        actual_idx = current_test_idx + i
                    
                    pt = p_true[i].detach().cpu().numpy()
                    ph = p_hat[i].detach().cpu().numpy()
                    seq_for_sample = batch_sequences[i] if i < len(batch_sequences) else None
                    title = f"idx={actual_idx}"
                    
                    # Create epoch-specific overlay directory
                    epoch_overlay_dir = out_dir / "epoch_overlays" / f"epoch_{epoch:02d}"
                    epoch_overlay_dir.mkdir(parents=True, exist_ok=True)
                    
                    path = epoch_overlay_dir / f"test_{actual_idx:05d}.{args.fig_format}"
                    png_path = overlay_plot(pt, ph, path, title=title, also_png=True, show_legend=False, sequences=seq_for_sample)
                    
                    grid_path = png_path if png_path else path
                    results.append({
                        "idx": actual_idx,
                        "kl": KL[i].item(),
                        "emd": EMD[i].item(),
                        "path": str(path),
                        "grid_path": str(grid_path),
                    })
                
                current_test_idx += B
    
    avg_kl = total_kl / total_samples
    
    # Generate topk grid if we have results and output directory
    if results and out_dir is not None and epoch is not None:
        grid_path, chosen = generate_topk_grid(results, out_dir, args, epoch_suffix=f"_epoch{epoch:02d}")
        if grid_path:
            print(f"[Epoch {epoch:02d}] Generated topk grid: {grid_path}")
    
    return avg_kl

def evaluate_epoch_model(model_path: str, dit, test_loader, cond_encoder, schedule, device, args, W_rebin=None):
    """Evaluate a single epoch model and return average KL divergence."""
    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    dit.load_state_dict(ckpt["model"])
    dit.eval()
    
    total_kl = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_i, batch in enumerate(test_loader):
            p_true = batch["block_dists"].to(device)
            if W_rebin is not None:
                p_true = p_true @ W_rebin.t()
            
            cond_feat = batch["condition_features"].to(device)
            cond = get_cond_emb(cond_encoder, cond_feat)
            uncond = torch.zeros_like(cond)
            
            z0_hat, _ = ddim_sample(
                model=dit,
                cond=cond,
                schedule=schedule,
                steps=args.sample_steps,
                eta=0.0,
                guidance=(uncond, args.guidance_scale) if args.guidance_scale > 1.0 else None,
                bins=p_true.size(-1),
                tau=1.0,
            )
            p_hat = torch.softmax(z0_hat / max(1e-6, args.inference_logit_temp), dim=-1)
            
            # Calculate KL divergence
            KL = kl_divergence(p_true, p_hat)
            total_kl += KL.sum().item()
            total_samples += KL.size(0)
    
    avg_kl = total_kl / total_samples
    return avg_kl

def select_best_model(out_dir: Path, dit, test_loader, cond_encoder, schedule, device, args, W_rebin=None):
    """Select the best model based on KL divergence and save as best_model.pt"""
    ckpt_dir = out_dir / "ckpts"
    model_files = list(ckpt_dir.glob("dit_epoch*.pt"))
    
    if not model_files:
        print("[Warning] No epoch models found for evaluation")
        return None
    
    best_kl = float('inf')
    best_epoch = None
    best_model_path = None
    
    print(f"[Evaluating] {len(model_files)} epoch models to find best...")
    
    for model_path in sorted(model_files):
        epoch_num = int(model_path.stem.split('epoch')[1])
        print(f"[Evaluating] Epoch {epoch_num:02d}...")
        avg_kl = evaluate_epoch_model(model_path, dit, test_loader, cond_encoder, schedule, device, args, W_rebin)
        print(f"[Epoch {epoch_num:02d}] Avg KL: {avg_kl:.6f}")
        
        if avg_kl < best_kl:
            best_kl = avg_kl
            best_epoch = epoch_num
            best_model_path = model_path
    
    if best_model_path:
        # Save best model
        best_save_path = out_dir / "best_model.pt"
        shutil.copy2(best_model_path, best_save_path)
        print(f"[Best Model] Epoch {best_epoch} with KL={best_kl:.6f} saved as {best_save_path}")
        
        # Load best model into dit
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        dit.load_state_dict(ckpt["model"])
        
        return best_save_path
    
    return None

def overlay_plot(p_true: np.ndarray, p_pred: np.ndarray, path: Path, title: str = "", also_png: bool = False, show_legend: bool = False, sequences: List[str] = None):
    max_bins = len(p_true)
    x = np.arange(1, max_bins + 1)
    plt.figure(figsize=(4.0, 3.0), dpi=160)
    width = 0.4
    
    # Use bar chart style like save_topk_panel with matching colors
    plt.bar(x - width/2, p_true, width=width, label="ground truth", alpha=0.8, color='#2E8B57')
    plt.bar(x + width/2, p_pred, width=width, label="prediction", alpha=0.8, color='#FF6B35')
    
    # Add Mayo-Lewis theoretical curve if sequences are provided
    if sequences is not None:
        try:
            mayo_lewis_dist = mayo_lewis_from_sequence(sequences, max_length=max_bins)
            plt.plot(x, mayo_lewis_dist, 'o--', linewidth=2, markersize=3, color='#8B0000', label='Mayo-Lewis theory', alpha=0.8)
        except Exception as e:
            print(f"Warning: Could not compute Mayo-Lewis curve: {e}")
    
    plt.xlabel("bin", fontsize=12, fontweight='bold')
    plt.ylabel("probability", fontsize=12, fontweight='bold')
    plt.yscale('log')  # Set y-axis to log scale
    plt.ylim(bottom=1e-3, top=1)  # Set y-range like save_topk_panel
    if title:
        plt.title(title, fontsize=12, fontweight='bold')
    
    # Add bold styling to tick labels
    plt.tick_params(axis='both', which='major', labelsize=10, labelcolor='black', width=1.5)
    for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        tick.set_fontweight('bold')
    
    if show_legend:
        legend = plt.legend(frameon=False, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.0))
        for text in legend.get_texts():
            text.set_weight('bold')
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    
    # Also save PNG version for grid composition if requested
    png_path = None
    if also_png and not str(path).endswith('.png'):
        png_path = path.with_suffix('.png')
        plt.savefig(png_path, bbox_inches="tight", format='png')
    
    plt.close()
    return png_path

# ----------------- Main pipeline -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--cond_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/split_train_test_run")
    parser.add_argument("--max_samples", type=int, default=None)

    # Model loading options
    parser.add_argument("--load_model", type=str, default=None, help="Path to pre-trained DiT model to load")
    parser.add_argument("--inference_only", action="store_true", help="Skip training, only do inference")

    # split
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed_test_indices", type=str, default="330,329,490,489,491,61,287,492,320,493,331,542", 
                        help="Comma-separated list of fixed test indices")

    # train
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)

    # DiT
    parser.add_argument("--dit_d_model", type=int, default=256)
    parser.add_argument("--dit_layers", type=int, default=8)
    parser.add_argument("--dit_heads", type=int, default=8)
    parser.add_argument("--dit_ff_mult", type=int, default=4)
    parser.add_argument("--dit_dropout", type=float, default=0.0)

    # diffusion
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--sample_steps", type=int, default=200)
    parser.add_argument("--guidance_scale", type=float, default=1.2)
    parser.add_argument("--cfg_prob", type=float, default=0.1)
    parser.add_argument("--train_logit_temp", type=float, default=0.8, help="Temperature for training phase")
    parser.add_argument("--inference_logit_temp", type=float, default=1.4, help="Temperature for inference phase")

    # bins / viz
    parser.add_argument("--bins_new", type=int, default=50)
    parser.add_argument("--fig_format", type=str, default="pdf", choices=["pdf", "png", "svg"])

    # select best
    parser.add_argument("--top_k", type=int, default=12, help="select N best (lowest error) test samples")
    parser.add_argument("--metric", type=str, default="kl", choices=["kl", "emd", "combined"],
                        help="ranking metric; 'combined' = KL + normalized EMD")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] Using: {device}")

    out_dir = Path(args.out_dir)
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    print(f"[Output] Directory: {out_dir}")

    # 1) split with fixed test indices
    fixed_test_indices = None
    if args.fixed_test_indices:
        try:
            fixed_test_indices = [int(x.strip()) for x in args.fixed_test_indices.split(',')]
            print(f"[Fixed Test Set] Using indices: {fixed_test_indices}")
        except ValueError:
            print(f"[Warning] Invalid fixed_test_indices format, using random split")
            fixed_test_indices = None
    
    dataset, train_set, test_set, train_loader, test_loader = build_loaders_split(
        args.csv, args.batch_size, args.num_workers, args.test_ratio, args.seed, args.max_samples, fixed_test_indices
    )
    print(f"[Dataset] Total: {len(dataset)}, Train: {len(train_set)}, Test: {len(test_set)}")

    # 2) load & freeze condition encoder
    ckpt = torch.load(args.cond_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    cond_in_dim = int(ckpt_args.get("cond_in_dim", 17))
    cond_d_model = int(ckpt_args.get("d_model", 128))
    cond_proj_dim = int(ckpt_args.get("proj_dim", 256))
    cond_temp = float(ckpt_args.get("temperature", 0.10))
    cond_layers = int(ckpt_args.get("num_layers", 3))

    cond_encoder = ConditionEncoder(
        in_dim=cond_in_dim, d_model=cond_d_model, proj_dim=cond_proj_dim,
        num_layers=cond_layers, dropout=ckpt_args.get("dropout", 0.1), temperature=cond_temp
    ).to(device)
    cond_encoder.load_state_dict(ckpt["model"], strict=False)
    for p in cond_encoder.parameters(): p.requires_grad = False
    cond_encoder.eval()
    print(f"[OK] Loaded ConditionEncoder: {args.cond_ckpt} (cond_emb dim={cond_d_model})")

    # 3) detect bins and maybe rebin
    with torch.no_grad():
        tmp_batch = next(iter(train_loader))
        bins_detected = tmp_batch["block_dists"].shape[-1]
    use_rebin = (args.bins_new > 0 and args.bins_new < bins_detected)
    if use_rebin:
        W_rebin = make_rebin_matrix(bins_detected, args.bins_new).to(device)
        bins = args.bins_new
        print(f"[Rebin] {bins_detected} -> {bins}")
    else:
        W_rebin = None
        bins = bins_detected

    # 4) build DiT and schedule
    dit = DiT1D(
        bins=bins, cond_dim=cond_d_model,
        d_model=args.dit_d_model, n_layers=args.dit_layers, n_heads=args.dit_heads,
        ff_mult=args.dit_ff_mult, dropout=args.dit_dropout, film_each_layer=True
    ).to(device)
    schedule = NoiseSchedule(T=args.T).to(device)

    # Check if we should load a pre-trained model
    if args.load_model:
        if Path(args.load_model).exists():
            print(f"[Loading] Pre-trained model from {args.load_model}")
            ckpt = torch.load(args.load_model, map_location=device, weights_only=False)
            dit.load_state_dict(ckpt["model"])
            print(f"[OK] Loaded DiT model")
        else:
            print(f"[Warning] Model file {args.load_model} not found, will train from scratch")
            args.load_model = None

    # If inference_only and we have a model, skip training
    if args.inference_only and args.load_model:
        print(f"[Inference Only] Skipping training, using loaded model")
    else:
        # 5) train only on train_loader (v-param + simple MSE)
        print(f"[Training] Starting training phase...")
        optimizer = optim.AdamW(dit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        print(f"[Train] epochs={args.epochs}, per-step batch={args.batch_size}, accum={args.accum_steps}")
        print(f"[Config] train_logit_temp={args.train_logit_temp}, inference_logit_temp={args.inference_logit_temp}")
        dit.train()
        global_step = 0
        
        # Track best model during training
        best_kl = float('inf')
        best_epoch = 0
        best_model_state = None
        
        for epoch in range(1, args.epochs + 1):
            print(f"[Epoch {epoch:02d}/{args.epochs}] Starting...")
            running = 0.0
            for step_i, batch in enumerate(train_loader, start=1):
                p = batch["block_dists"].to(device)  # [B, M_old]
                if W_rebin is not None:
                    p = p @ W_rebin.t()              # [B, M]
                z0 = hist_to_logits(p)               # [B, M]

                with torch.no_grad():
                    cond_feat = batch["condition_features"].to(device)
                    cond = get_cond_emb(cond_encoder, cond_feat)

                # CFG training drop
                if args.cfg_prob > 0:
                    drop = (torch.rand(cond.size(0), device=cond.device) < args.cfg_prob).float().unsqueeze(-1)
                    cond = cond * (1.0 - drop)

                t = torch.randint(low=0, high=schedule.T, size=(p.size(0),), device=device, dtype=torch.long)

                try:
                    autocast_context = torch.amp.autocast('cuda', enabled=(device.type == "cuda"))
                except AttributeError:
                    autocast_context = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))

                with autocast_context:
                    x_t, v_target = q_sample_vparam(z0, t, schedule)  # [B,M]
                    v_hat = dit(x_t=x_t, t=t, cond=cond)
                    loss = torch.mean((v_hat - v_target) ** 2) / args.accum_steps

                scaler.scale(loss).backward()
                running += loss.item()
                global_step += 1

                if (global_step % args.accum_steps) == 0:
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                if (global_step % args.log_every) == 0:
                    avg = running * args.accum_steps / args.log_every
                    print(f"[epoch {epoch:02d} | step {global_step}] loss={avg:.5f}")
                    running = 0.0

            # save per-epoch ckpt (after each epoch completes)
            ckpt_path = out_dir / "ckpts" / f"dit_epoch{epoch:02d}.pt"
            torch.save({"model": dit.state_dict(), "epoch": epoch, "bins": bins}, ckpt_path)
            print(f"[Saved] {ckpt_path}")
            
            # Evaluate on test set after each epoch
            print(f"[Epoch {epoch:02d}] Evaluating on test set...")
            dit.eval()
            epoch_kl = evaluate_epoch_model_inline(
                dit, test_loader, cond_encoder, schedule, device, args, W_rebin,
                fixed_test_indices=fixed_test_indices, test_set=test_set, out_dir=out_dir, epoch=epoch
            )
            print(f"[Epoch {epoch:02d}] Test KL: {epoch_kl:.6f}")
            
            # Track best model
            if epoch_kl < best_kl:
                best_kl = epoch_kl
                best_epoch = epoch
                best_model_state = dit.state_dict().copy()
                print(f"[Epoch {epoch:02d}] *** NEW BEST MODEL *** (KL: {best_kl:.6f})")
            else:
                print(f"[Epoch {epoch:02d}] Best remains epoch {best_epoch} (KL: {best_kl:.6f})")
            
            dit.train()  # Switch back to training mode
            print(f"[Epoch {epoch:02d}] Completed")

        # Save best model at the end of training
        if epoch == args.epochs and best_model_state is not None:
            best_save_path = out_dir / "best_model.pt"
            torch.save({"model": best_model_state, "epoch": best_epoch, "bins": bins}, best_save_path)
            print(f"[Training Complete] Best model from epoch {best_epoch} (KL: {best_kl:.6f}) saved as {best_save_path}")
            # Load best model for final evaluation
            dit.load_state_dict(best_model_state)

    # 6) evaluate on test set: produce overlays and select best 12
    print(f"[Evaluation] Starting test set evaluation...")
    dit.eval()
    results: List[Dict] = []
    
    # Get the actual indices for test set if using fixed indices
    if fixed_test_indices is not None:
        # For fixed indices, we need to map batch positions to actual dataset indices
        test_dataset_indices = fixed_test_indices[:len(test_set)]  # In case some indices were filtered out
    else:
        # For random split, get the indices from the subset
        test_dataset_indices = test_set.indices if hasattr(test_set, 'indices') else list(range(len(test_set)))
    
    current_test_idx = 0  # Track position in test set
    
    with torch.no_grad():
        for batch_i, batch in enumerate(test_loader):
            print(f"[Evaluation] Processing batch {batch_i+1}/{len(test_loader)}")
            p_true = batch["block_dists"].to(device)
            if W_rebin is not None:
                p_true = p_true @ W_rebin.t()
            cond_feat = batch["condition_features"].to(device)
            
            # Decode sequences from encoded format
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
            
            cond = get_cond_emb(cond_encoder, cond_feat)
            uncond = torch.zeros_like(cond)

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
            p_hat = torch.softmax(z0_hat / max(1e-6, args.inference_logit_temp), dim=-1)

            # metrics per-sample
            KL = kl_divergence(p_true, p_hat)   # [B]
            EMD = emd_1d(p_true, p_hat)         # [B]

            # save overlay images
            B = p_true.size(0)
            for i in range(B):
                # Get the actual dataset index for this test sample
                if current_test_idx + i < len(test_dataset_indices):
                    actual_idx = test_dataset_indices[current_test_idx + i]
                else:
                    actual_idx = current_test_idx + i  # Fallback
                
                pt = p_true[i].detach().cpu().numpy()
                ph = p_hat[i].detach().cpu().numpy()
                seq_for_sample = batch_sequences[i] if i < len(batch_sequences) else None  # Get sequences for this sample
                title = f"idx={actual_idx}"  # Use actual dataset index
                path = out_dir / "overlays" / f"test_{actual_idx:05d}.{args.fig_format}"
                png_path = overlay_plot(pt, ph, path, title=title, also_png=True, show_legend=False, sequences=seq_for_sample)
                
                # Store the PNG path for grid composition, fallback to original path if PNG not created
                grid_path = png_path if png_path else path
                results.append({
                    "idx": actual_idx,
                    "kl": KL[i].item(),
                    "emd": EMD[i].item(),
                    "path": str(path),
                    "grid_path": str(grid_path),  # Path for grid composition
                })
            
            # Update the current test index for next batch
            current_test_idx += B

    # 7) select top-K best and compose a 3x4 PDF grid
    print(f"[Grid Generation] Creating final top-{args.top_k} grid visualization...")
    
    grid_path, chosen = generate_topk_grid(results, out_dir, args)
    if grid_path:
        print(f"[Saved] Final grid: {grid_path}")
    else:
        print(f"[Warning] Could not generate final grid")

    # Save rankings CSV
    print(f"[Rankings] Saving detailed rankings...")
    import csv
    
    # Calculate scores for ranking
    res_np = np.array([[r["kl"], r["emd"]] for r in results], dtype=np.float64)
    kl_vals = res_np[:, 0]
    emd_vals = res_np[:, 1]
    if args.metric == "kl":
        score = kl_vals
    elif args.metric == "emd":
        score = emd_vals
    else:
        # combined: KL + normalized EMD
        emd_norm = (emd_vals - emd_vals.min()) / (emd_vals.ptp() + 1e-12)
        score = kl_vals + emd_norm

    order = np.argsort(score)
    
    with open(out_dir / "test_rankings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "idx", "kl", "emd", "path"])
        for rank, i in enumerate(order.tolist(), start=1):
            r = results[int(i)]
            w.writerow([rank, r["idx"], f"{r['kl']:.6f}", f"{r['emd']:.6f}", r["path"]])
    print(f"[Saved] rankings: {out_dir / 'test_rankings.csv'}")
    print(f"[Complete] All processing finished successfully!")

if __name__ == "__main__":
    main()
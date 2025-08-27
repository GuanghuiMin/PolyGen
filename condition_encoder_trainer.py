"""
python condition_encoder_trainer.py \
--csv ./data/copolymer.csv \
--out_dir outputs/cond_enc_runs_emdsoft \
--train_epochs 20 \
--batch_size 4 --accum_steps 4 \
--queue_size 8192 --stopgrad_pos \
--emd_softcon --emd_target_tau 0.5 --emd_self_weight 0.0 \
--temperature 0.07 --scale_s 20.0 --pos_margin 0.15 \
--vicreg_w 1.0 --vicreg_var_gamma 1.0 --vicreg_cov_w 0.01 \
--cond_jitter_std 0.02 \
--fig_format pdf \
--normalize_cond zscore --supcon_k 3 --hard_neg_k 256
"""

import os
import math
import argparse
from pathlib import Path
from typing import Optional
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Perf knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---- Project imports ----
from data.dataset import ChainSetDataset, collate_fn_set_transformer
from src.encoder import ConditionEncoder  # adjust path if your encoder.py is elsewhere


#
# -------- schedules --------
def lin_anneal(start: float, end: float, t: float) -> float:
    t = max(0.0, min(1.0, t))
    return start + (end - start) * t

# ----------------- Stats / EMD helpers -----------------
def compute_cond_norm_stats(dataset, cond_dim: int, max_samples: Optional[int] = None):
    """One-pass mean/std over dataset['condition_features']."""
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

def pairwise_emd_1d(p: torch.Tensor) -> torch.Tensor:
    """
    Fast 1D EMD via cumulative sums.
    p: [B, M] (probabilities)
    return: [B, B] EMD distances
    """
    cdf = p.cumsum(dim=-1)  # [B,M]
    emd = torch.cdist(cdf, cdf, p=1)  # [B,B]  (L1 of CDF diff)
    return emd

def build_positive_mask_from_block(p: torch.Tensor, k: int = 3, emd_thresh: Optional[float] = None) -> torch.Tensor:
    """
    Build positive mask using kNN or threshold on EMD over block_dists.
    Returns mask [B,B], with zeros on diagonal.
    """
    with torch.no_grad():
        D = pairwise_emd_1d(p)  # [B,B]
        B = D.size(0)
        D = D + torch.eye(B, device=p.device) * 1e6  # ignore self
        mask = torch.zeros(B, B, device=p.device, dtype=torch.bool)
        if emd_thresh is not None:
            mask |= (D < emd_thresh)
        if k > 0:
            idx = torch.topk(-D, k=k, dim=1).indices  # k smallest distances
            mask.scatter_(1, idx, True)
        mask = mask & (~torch.eye(B, device=p.device, dtype=torch.bool))
        return mask

# --------- Soft EMD-driven contrastive targets ---------
@torch.no_grad()
def emd_soft_targets(p_block: torch.Tensor, tau_tgt: float, self_weight: float = 0.0) -> torch.Tensor:
    """Return row-stochastic soft targets T from EMD:
    D_ij = EMD(p_i, p_j);  A_ij = exp(-D_ij / tau_tgt);  then normalize rows.
    Diagonal is set to `self_weight` before renorm.
    Args:
        p_block: [B,M] batch of histograms (sum 1)
        tau_tgt: target temperature
        self_weight: value for diagonal before renorm; 0.0 disables self-targets
    Returns:
        T: [B,B] with T_i· = soft distribution over j
    """
    D = pairwise_emd_1d(p_block)  # [B,B]
    A = torch.exp(- D / max(1e-6, tau_tgt))
    B = A.size(0)
    eye = torch.eye(B, device=A.device, dtype=A.dtype)
    if self_weight == 0.0:
        A = A * (1.0 - eye)
    else:
        A = A * (1.0 - eye) + eye * float(self_weight)
    Z = A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    T = A / Z
    return T


# ----------------- Utils -----------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def build_batch_from_indices(dataset, indices: torch.Tensor):
    """Given dataset indices, build a batch using the same collate_fn."""
    items = [dataset[int(i)] for i in indices.tolist()]
    return collate_fn_set_transformer(items)

def enqueue(queue: torch.Tensor, queue_ptr: int, queue_len: int, keys: torch.Tensor):
    """Enqueue keys into a ring buffer queue."""
    K = queue.size(0)
    B = keys.size(0)
    if B >= K:
        queue.copy_(keys[-K:].detach())
        return queue, 0, K
    end = queue_ptr + B
    if end <= K:
        queue[queue_ptr:end] = keys.detach()
    else:
        first = K - queue_ptr
        queue[queue_ptr:] = keys[:first].detach()
        queue[:end - K] = keys[first:].detach()
    new_ptr = (queue_ptr + B) % K
    new_len = min(K, queue_len + B)
    return queue, new_ptr, new_len


# ----------------- Training & Viz -----------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Dataset / Loader
    dataset = ChainSetDataset(args.csv, max_samples=args.max_samples, contrastive=True)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_set_transformer,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ---- condition normalization stats ----
    cond_mean = cond_std = None
    if args.normalize_cond == "zscore":
        cond_mean, cond_std = compute_cond_norm_stats(dataset, args.cond_in_dim, max_samples=args.norm_max_samples)
        print(f"[norm] computed z-score stats over {min(len(dataset), args.norm_max_samples or len(dataset))} samples")

    # Model
    model = ConditionEncoder(
        in_dim=args.cond_in_dim,
        d_model=args.d_model,
        proj_dim=args.proj_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduler (epoch-wise)
    scheduler = None
    if args.lr_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=args.lr_min)

    # GradScaler
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Queue
    use_queue = args.queue_size > 0
    if use_queue:
        queue = torch.zeros(args.queue_size, args.proj_dim, device=device, dtype=torch.float32)
        queue_ptr = 0
        queue_len = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    print(f"Per-step batch_size={args.batch_size}, accum_steps={args.accum_steps} -> effective batch={args.batch_size * args.accum_steps}")

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        running = 0.0

        # ---- epoch-wise schedules ----
        # LR warmup
        if args.lr_warmup_epochs > 0 and epoch <= args.lr_warmup_epochs:
            warm_ratio = epoch / float(max(1, args.lr_warmup_epochs))
            for g in optimizer.param_groups:
                g["lr"] = args.lr * warm_ratio
        elif not args.lr_cosine:
            for g in optimizer.param_groups:
                g["lr"] = args.lr

        # anneal contrastive hyper-params across epochs
        t_epoch = 0.0 if args.train_epochs <= 1 else (epoch - 1) / float(args.train_epochs - 1)
        temperature_cur = lin_anneal(args.temp_start, args.temp_end, t_epoch)
        scale_s_cur = lin_anneal(args.scale_start, args.scale_end, t_epoch)
        pos_margin_cur = lin_anneal(args.margin_start, args.margin_end, t_epoch)
        tau_cur = lin_anneal(args.tau_start, args.tau_end, t_epoch) if (args.emd_softcon and args.anneal_tau) else args.emd_target_tau

        # queue warmup: disable queue for first N epochs
        use_queue_now = (args.queue_size > 0) and (epoch > args.queue_warmup_epochs)

        for step_i, batch in enumerate(loader, start=1):
            # Build anchor & positive condition batches
            cond_a = batch['condition_features'].to(device, non_blocking=True)  # [B, C]
            pos_idx = batch['positive_indices']                                  # [B]
            pos_batch = build_batch_from_indices(dataset, pos_idx)
            cond_p = pos_batch['condition_features'].to(device, non_blocking=True)

            # Normalize conditions if requested
            if args.normalize_cond == "zscore" and cond_mean is not None:
                cond_a = (cond_a - cond_mean.to(device)) / (cond_std.to(device) + 1e-6)
                cond_p = (cond_p - cond_mean.to(device)) / (cond_std.to(device) + 1e-6)
            elif args.normalize_cond == "layernorm":
                ln = nn.LayerNorm(cond_a.size(-1)).to(device)
                cond_a = ln(cond_a)
                cond_p = ln(cond_p)

            # light augmentation on conditions
            if args.cond_jitter_std > 0:
                noise_a = torch.randn_like(cond_a) * args.cond_jitter_std
                noise_p = torch.randn_like(cond_p) * args.cond_jitter_std
                cond_a = cond_a + noise_a
                cond_p = cond_p + noise_p

            # autocast
            try:
                autocast_context = torch.amp.autocast('cuda', enabled=(device.type == "cuda"))
            except AttributeError:
                autocast_context = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))

            with autocast_context:
                out_a = model(cond=cond_a)
                if args.stopgrad_pos:
                    with torch.no_grad():
                        out_p = model(cond=cond_p)
                else:
                    out_p = model(cond=cond_p)

                z_a, z_p = out_a['z'], out_p['z']  # [B, D]
                B = z_a.size(0)

                # ---------- Build in-batch positives by EMD over block_dists ----------
                p_block = batch['block_dists'].to(device)  # [B, M]
                pos_mask_bb = build_positive_mask_from_block(
                    p_block, k=args.supcon_k, emd_thresh=args.supcon_emd_thresh
                )  # [B,B]

                # ---------- Build keys = [in-batch anchors; hard negatives from queue] ----------
                # L2-normalize embeddings first (cosine space)
                z_a = F.normalize(z_a, dim=-1)
                z_p = F.normalize(z_p, dim=-1)

                if use_queue_now and queue_len > 0:
                    neg_keys_full = queue[:queue_len]  # [K', D]
                    if args.hard_neg_k > 0:
                        sims = z_a @ neg_keys_full.t()       # [B, K']
                        topk_idx = torch.topk(sims, k=min(args.hard_neg_k, neg_keys_full.size(0)), dim=1).indices
                        unique_idx = torch.unique(topk_idx.reshape(-1))
                        neg_keys = neg_keys_full.index_select(0, unique_idx)
                    else:
                        neg_keys = neg_keys_full
                    keys = torch.cat([z_a, neg_keys.detach()], dim=0)   # [B+Ksel, D]
                else:
                    keys = z_a

                # cosine logits (no temp here)
                logits = z_a @ keys.t()                                # [B, B+K]
                # mask out self in the first B columns
                self_mask = torch.zeros_like(logits, device=device, dtype=torch.bool)
                self_mask[:, :B] = torch.eye(B, device=device, dtype=torch.bool)
                mask_value = -65500.0 if logits.dtype == torch.float16 else -1e9
                logits = logits.masked_fill(self_mask, mask_value)

                # positive mask of shape [B, B+K] (only in the first B columns)
                pos_mask = torch.zeros(B, keys.size(0), device=device, dtype=torch.bool)
                pos_mask[:, :B] = pos_mask_bb

                # additive margin on positives
                if pos_margin_cur > 0:
                    logits = logits - (pos_mask.float() * pos_margin_cur)

                # final scale & temperature
                logits = (scale_s_cur * logits) / max(1e-6, temperature_cur)

                # ---------- Contrastive objective ----------
                if args.emd_softcon:
                    # Soft EMD targets over in-batch pairs
                    T_emd = emd_soft_targets(p_block, tau_tgt=tau_cur,
                                             self_weight=args.emd_self_weight)  # [B,B]
                    logits_in = logits[:, :B]
                    if args.emd_self_weight == 0.0:
                        eye = torch.eye(B, device=device, dtype=torch.bool)
                        logits_in = logits_in.masked_fill(eye, mask_value)
                    P = torch.softmax(logits_in, dim=-1)  # [B,B]
                    # Row-wise KL(T || P)
                    cl_loss = (T_emd * (T_emd.add(1e-12).log() - P.add(1e-12).log())).sum(dim=-1).mean()
                else:
                    # SupCon with multi-positives (hard mask)
                    exp_logits = torch.exp(logits)
                    numer = (exp_logits * pos_mask.float()).sum(dim=-1) + 1e-8
                    denom = exp_logits.sum(dim=-1) + 1e-8
                    supcon_loss = -torch.log(numer / denom)
                    has_pos = (pos_mask.sum(dim=-1) > 0).float()
                    cl_loss = (supcon_loss * has_pos).sum() / (has_pos.sum() + 1e-8)

                # ---------- VICReg regularizer on z_a (anti-collapse) ----------
                vic_loss = z_a.sum() * 0.0
                if args.vicreg_w > 0:
                    eps = 1e-4
                    std_z = torch.sqrt(z_a.var(dim=0) + eps)
                    var_loss = torch.relu(args.vicreg_var_gamma - std_z).mean()
                    N = z_a.size(0)
                    zc = z_a - z_a.mean(dim=0)
                    cov = (zc.t() @ zc) / max(1, N - 1)
                    off_diag = cov - torch.diag(torch.diag(cov))
                    cov_loss = (off_diag ** 2).mean()
                    vic_loss = var_loss + args.vicreg_cov_w * cov_loss

                loss = (cl_loss + args.vicreg_w * vic_loss) / args.accum_steps

                # ------ diagnostics ------
                with torch.no_grad():
                    pos_pairs = (z_a @ z_p.t()).diag().mean().item()
                    pos_sim = pos_pairs
                    if use_queue and queue_len > 0:
                        top_neg_sim = (z_a @ queue[:queue_len].t()).amax(dim=1).mean().item()
                    else:
                        mask_nonpos = (~pos_mask_bb) & (~self_mask)
                        if mask_nonpos.any():
                            sims = (z_a @ z_a.t())[mask_nonpos]
                            top_neg_sim = sims.max().item()
                        else:
                            top_neg_sim = float('nan')

            scaler.scale(loss).backward()
            running += loss.item()
            global_step += 1

            if (global_step % args.accum_steps) == 0:
                # optional grad clipping (unscale first for amp)
                if args.grad_clip and args.grad_clip > 0:
                    try:
                        scaler.unscale_(optimizer)
                    except Exception:
                        pass
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # update queue with positive keys
                if use_queue_now:
                    queue, queue_ptr, queue_len = enqueue(queue, queue_ptr, queue_len, z_p.detach())

            if (global_step % args.log_every) == 0:
                avg = running * args.accum_steps / args.log_every
                extra = f" | pos_sim={pos_sim:.3f} top_neg_sim={top_neg_sim:.3f}" if 'pos_sim' in locals() else ""
                print(f"[epoch {epoch:02d} | step {global_step}] loss={avg:.4f}{extra} (queue_len={queue_len if use_queue_now else 0}) "
                      f"| temp={temperature_cur:.3f} scale={scale_s_cur:.1f} margin={pos_margin_cur:.2f} tau={tau_cur:.2f} lr={optimizer.param_groups[0]['lr']:.2e}")
                running = 0.0

        # step cosine scheduler at epoch end
        if scheduler is not None and epoch > args.lr_warmup_epochs:
            scheduler.step()

        # Save checkpoint each epoch
        ckpt = {"model": model.state_dict(), "epoch": epoch, "args": vars(args)}
        ckpt_path = out_dir / f"cond_encoder_epoch{epoch:02d}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"Saved: {ckpt_path}")

    # Final
    final_path = out_dir / "cond_encoder_final.pt"
    torch.save({"model": model.state_dict(), "epoch": args.train_epochs, "args": vars(args)}, final_path)
    print(f"Saved final: {final_path}")

    # Visualization
    visualize_tsne(model, dataset, device, out_dir, args, cond_mean, cond_std)


@torch.no_grad()
def visualize_tsne(model, dataset, device, out_dir: Path, args, cond_mean=None, cond_std=None):
    model.eval()
    n_vis = min(args.tsne_samples, len(dataset))
    idxs = np.linspace(0, len(dataset) - 1, num=n_vis, dtype=int)

    B = args.batch_size
    zs, hs = [], []
    probAAs, probBBs, meanBlocks, alternations = [], [], [], []

    for start in range(0, n_vis, B):
        end = min(start + B, n_vis)
        items = [dataset[i] for i in idxs[start:end]]
        batch = collate_fn_set_transformer(items)

        cond = batch['condition_features'].to(device)
        if hasattr(args, 'normalize_cond') and args.normalize_cond == "zscore" and cond_mean is not None:
            cond = (cond - cond_mean.to(device)) / (cond_std.to(device) + 1e-6)
        elif hasattr(args, 'normalize_cond') and args.normalize_cond == "layernorm":
            ln = nn.LayerNorm(cond.size(-1)).to(device)
            cond = ln(cond)

        out = model(cond=cond)
        z = out['z'].detach().cpu().numpy()
        h = out['cond_emb'].detach().cpu().numpy()

        zs.append(z)
        hs.append(h)

        pa = batch['probAAs'].detach().cpu().numpy()
        pb = batch['probBBs'].detach().cpu().numpy()
        mb = batch['target_stats']['mean_block'].detach().cpu().numpy()
        alt = batch['target_stats']['alternation_idx'].detach().cpu().numpy()
        probAAs.append(pa); probBBs.append(pb); meanBlocks.append(mb); alternations.append(alt)

    Z = np.concatenate(zs, axis=0)   # [n_vis, D]
    H = np.concatenate(hs, axis=0)   # [n_vis, d_model]
    y_aa  = np.concatenate(probAAs, axis=0).reshape(-1)
    y_bb  = np.concatenate(probBBs, axis=0).reshape(-1)
    y_mb  = np.concatenate(meanBlocks, axis=0).reshape(-1)
    y_alt = np.concatenate(alternations, axis=0).reshape(-1)

    def run_tsne(X, perplexity=30, seed=42):
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, max(5, (X.shape[0]-1)//3)),
            init="pca",
            learning_rate="auto",
            random_state=seed,
            n_iter=1000,
        )
        return tsne.fit_transform(X)

    Z2 = run_tsne(Z, perplexity=args.tsne_perplexity)
    H2 = run_tsne(H, perplexity=args.tsne_perplexity)

    def scatter_color(X2, color, title, base_name, fig_format="pdf"):
        plt.figure(figsize=(6, 5), dpi=160)
        sc = plt.scatter(X2[:, 0], X2[:, 1], c=color, s=8, alpha=0.85)
        plt.tick_params(axis="both", which="major", labelsize=18)
        plt.tick_params(axis="both", which="minor", labelsize=18)
        plt.tight_layout()
        path = out_dir / f"{base_name}.{fig_format}"
        plt.savefig(path, format=fig_format, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {path}")

    # four colorings for both z and h
    scatter_color(Z2, y_aa,  "t-SNE of z (colored by probAA)",           "tsne_z_probAA", args.fig_format)
    scatter_color(Z2, y_bb,  "t-SNE of z (colored by probBB)",           "tsne_z_probBB", args.fig_format)
    scatter_color(Z2, y_mb,  "t-SNE of z (colored by mean_block)",       "tsne_z_meanBlock", args.fig_format)
    scatter_color(Z2, y_alt, "t-SNE of z (colored by alternation_idx)",  "tsne_z_alternation", args.fig_format)

    scatter_color(H2, y_aa,  "t-SNE of cond_emb (colored by probAA)",          "tsne_h_probAA", args.fig_format)
    scatter_color(H2, y_bb,  "t-SNE of cond_emb (colored by probBB)",          "tsne_h_probBB", args.fig_format)
    scatter_color(H2, y_mb,  "t-SNE of cond_emb (colored by mean_block)",      "tsne_h_meanBlock", args.fig_format)
    scatter_color(H2, y_alt, "t-SNE of cond_emb (colored by alternation_idx)", "tsne_h_alternation", args.fig_format)

    Z3 = run_tsne(Z, perplexity=max(40, args.tsne_perplexity))
    scatter_color(Z3, y_aa, "t-SNE of z (perplexity=40) colored by probAA", "tsne_z_probAA_p40", args.fig_format)


# ----------------- CLI -----------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="./data/copolymer.csv")
    p.add_argument("--out_dir", type=str, default="outputs/cond_enc_runs")
    p.add_argument("--max_samples", type=int, default=None)

    # training
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # encoder config
    p.add_argument("--cond_in_dim", type=int, default=17)   # 10 multi + 7 single
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.07,
                   help="Softmax temperature for contrastive logits (smaller → stronger).")

    # queue
    p.add_argument("--queue_size", type=int, default=8192,
                   help="MoCo queue size (0 disables queue → SimCLR fallback).")
    p.add_argument("--stopgrad_pos", action="store_true",
                   help="Stop gradient on positive branch (MoCo-like).")

    # tsne
    p.add_argument("--tsne_samples", type=int, default=2000)
    p.add_argument("--tsne_perplexity", type=int, default=30)
    p.add_argument("--fig_format", type=str, default="pdf",
                   choices=["pdf", "png", "svg"])

    # normalization
    p.add_argument("--normalize_cond", type=str, default="zscore", choices=["none", "zscore", "layernorm"])
    p.add_argument("--norm_max_samples", type=int, default=None, help="Limit samples when computing z-score stats.")

    # SupCon & hard negatives
    p.add_argument("--supcon_k", type=int, default=3, help="kNN positives per anchor based on EMD over block_dists.")
    p.add_argument("--supcon_emd_thresh", type=float, default=None, help="Optional EMD threshold for positives.")
    p.add_argument("--hard_neg_k", type=int, default=256, help="Select top-k hard negatives from queue each step (0=use all).")

    # contrastive strength knobs
    p.add_argument("--pos_margin", type=float, default=0.15,
                   help="Additive margin on positive logits (ArcFace-style; larger → stronger).")
    p.add_argument("--scale_s", type=float, default=20.0,
                   help="Logit scale factor after cosine (common 10~64).")

    # regularizers (VICReg)
    p.add_argument("--vicreg_w", type=float, default=1.0,
                   help="Weight of VICReg variance/covariance regularizer on z (0 disables).")
    p.add_argument("--vicreg_var_gamma", type=float, default=1.0,
                   help="Target minimum std per-dimension for VICReg variance term.")
    p.add_argument("--vicreg_cov_w", type=float, default=0.01,
                   help="Weight of off-diagonal covariance penalty in VICReg.")

    # light augmentation on conditions
    p.add_argument("--cond_jitter_std", type=float, default=0.02,
                   help="Gaussian noise std added to normalized conditions (0 disables).")

    # EMD-driven soft contrastive (optional)
    p.add_argument("--emd_softcon", action="store_true",
                   help="Use soft targets from exp(-EMD/τ_tgt) instead of hard positive masks.")
    p.add_argument("--emd_target_tau", type=float, default=0.5,
                   help="Temperature for target kernel: A_ij ∝ exp(-EMD_ij/τ_tgt). Smaller → sharper targets.")
    p.add_argument("--emd_self_weight", type=float, default=0.0,
                   help="Weight on diagonal targets; usually keep 0 to avoid trivial self-alignment.")

    # ---- training schedules / annealing ----
    p.add_argument("--lr_min", type=float, default=1e-5, help="Minimum LR for cosine schedule.")
    p.add_argument("--lr_warmup_epochs", type=int, default=2, help="Warmup LR linearly for first N epochs.")
    p.add_argument("--lr_cosine", action="store_true", help="Use CosineAnnealingLR across epochs.")

    p.add_argument("--queue_warmup_epochs", type=int, default=2,
                   help="Disable memory queue for first N epochs (in-batch negatives only).")

    p.add_argument("--anneal_tau", action="store_true", help="Linearly anneal EMD target tau across epochs.")
    p.add_argument("--tau_start", type=float, default=1.0)
    p.add_argument("--tau_end", type=float, default=0.4)

    p.add_argument("--temp_start", type=float, default=0.15)
    p.add_argument("--temp_end", type=float, default=0.07)

    p.add_argument("--scale_start", type=float, default=10.0)
    p.add_argument("--scale_end", type=float, default=20.0)

    p.add_argument("--margin_start", type=float, default=0.0)
    p.add_argument("--margin_end", type=float, default=0.15)

    p.add_argument("--grad_clip", type=float, default=1.0, help="Clip global grad-norm (0 disables).")

    return p


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
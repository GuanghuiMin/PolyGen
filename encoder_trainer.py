# scripts/train_contrastive_encoder.py
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import numpy as np


from data.dataset import ChainSetDataset, collate_fn_set_transformer
from src.encoder import ContrastiveSetTransformerEncoder

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def build_batch_from_indices(dataset, indices: torch.Tensor):
    items = [dataset[int(i)] for i in indices.tolist()]
    return collate_fn_set_transformer(items)

def simclr_info_nce_from_pairs(z: torch.Tensor, temperature: float = 0.07):
    z = nn.functional.normalize(z, dim=-1)
    N = z.size(0)
    assert N % 2 == 0, "z must have even batch size for SimCLR pairing."
    B = N // 2

    logits = (z @ z.t()) / temperature  # [2B, 2B]
    mask = torch.eye(N, device=z.device, dtype=torch.bool)
    mask_value = -65500.0 if logits.dtype == torch.float16 else -1e9
    logits = logits.masked_fill(mask, mask_value)

    positives = torch.arange(N, device=z.device)
    positives = positives.roll(shifts=B)  # [0..B-1] <-> [B..2B-1]

    labels = positives
    loss = nn.functional.cross_entropy(logits, labels)
    return loss

def moco_info_nce(z_anchor: torch.Tensor,
                  z_pos: torch.Tensor,
                  queue: torch.Tensor,
                  queue_len: int,
                  temperature: float = 0.07,
                  detach_pos: bool = True):

    B, D = z_anchor.shape
    pos_keys = z_pos.detach() if detach_pos else z_pos
    neg_keys = queue[:queue_len].detach() if queue_len > 0 else z_anchor.new_zeros(0, z_anchor.size(1))

    keys = torch.cat([pos_keys, neg_keys], dim=0)  # [B + queue_len, D]
    logits = (z_anchor @ keys.t()) / temperature   # [B, B + queue_len]

    labels = torch.arange(B, device=z_anchor.device, dtype=torch.long)

    loss = nn.functional.cross_entropy(logits, labels)
    return loss

def enqueue(queue: torch.Tensor, queue_ptr: int, queue_len: int, keys: torch.Tensor):
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

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset = ChainSetDataset(args.csv, max_samples=args.max_samples, contrastive=True)

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_set_transformer,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True, 
    )

    model = ContrastiveSetTransformerEncoder(
        d_model=args.d_model,
        n_heads=args.n_heads,
        token_n_layers=args.token_layers,
        token_n_inducing=args.token_inducing,
        token_n_seeds=args.token_seeds,
        set_n_layers=args.set_layers,
        set_n_seeds=args.set_seeds,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        
    model.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_queue = args.queue_size > 0
    if use_queue:
        queue = torch.zeros(args.queue_size, args.proj_dim, device=device, dtype=torch.float32)
        queue_ptr = 0
        queue_len = 0

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for batch in loader:
            # anchors batch
            chain_sets = batch['chain_sets'].to(device, non_blocking=True)
            set_masks  = batch['set_masks'].to(device, non_blocking=True)

            pos_indices = batch['positive_indices']  # [B]
            pos_batch = build_batch_from_indices(dataset, pos_indices)
            pos_chain_sets = pos_batch['chain_sets'].to(device, non_blocking=True)
            pos_set_masks  = pos_batch['set_masks'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            try:
                autocast_context = torch.amp.autocast('cuda', enabled=(device.type == "cuda"))
            except AttributeError:
                autocast_context = torch.cuda.amp.autocast(enabled=(device.type == "cuda"))
                
            with autocast_context:
                out_anchor = model(chain_sets=chain_sets, set_masks=set_masks)

                if args.stopgrad_pos:
                    with torch.no_grad():
                        out_pos = model(chain_sets=pos_chain_sets, set_masks=pos_set_masks)
                else:
                    out_pos = model(chain_sets=pos_chain_sets, set_masks=pos_set_masks)

                z_anchor, z_pos = out_anchor['z'], out_pos['z']  # [B, D], [B, D]

                if use_queue:
                    loss = moco_info_nce(
                        z_anchor=z_anchor,
                        z_pos=z_pos,
                        queue=queue,
                        queue_len=queue_len,
                        temperature=args.temperature,
                        detach_pos=args.stopgrad_pos
                    )
                else:
                    z = torch.cat([z_anchor, z_pos], dim=0)  # [2B, D]
                    loss = simclr_info_nce_from_pairs(z, temperature=args.temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if use_queue:
                queue, queue_ptr, queue_len = enqueue(queue, queue_ptr, queue_len, z_pos.detach())

            running += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                avg = running / args.log_every
                print(f"[epoch {epoch:02d} | step {global_step}] loss={avg:.4f}")
                running = 0.0

        ckpt_path = out_dir / f"encoder_epoch{epoch:02d}.pt"
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)}, ckpt_path)
        print(f"Saved: {ckpt_path}")

    final_path = out_dir / "encoder_final.pt"
    torch.save({'model': model.state_dict(), 'epoch': args.epochs, 'args': vars(args)}, final_path)
    print(f"Saved final: {final_path}")

    visualize_tsne(model, dataset, device, out_dir, args)

@torch.no_grad()
def visualize_tsne(model, dataset, device, out_dir: Path, args):
    model.eval()
    n_vis = min(args.tsne_samples, len(dataset))
    idxs = np.linspace(0, len(dataset) - 1, num=n_vis, dtype=int)

    B = args.batch_size
    zs = []
    hs = []
    probAAs = []
    probBBs = []

    meanBlocks = []
    alternations = []

    for start in range(0, n_vis, B):
        end = min(start + B, n_vis)
        items = [dataset[i] for i in idxs[start:end]]
        batch = collate_fn_set_transformer(items)

        chain_sets = batch['chain_sets'].to(device)
        set_masks  = batch['set_masks'].to(device)
        out = model(chain_sets=chain_sets, set_masks=set_masks)
        z = out['z'].detach().cpu().numpy()
        h = out['set_emb'].detach().cpu().numpy()

        zs.append(z)
        hs.append(h)

        pa = batch['probAAs'].detach().cpu().numpy()
        pb = batch['probBBs'].detach().cpu().numpy()
        probAAs.append(pa)
        probBBs.append(pb)

        mb = batch['target_stats']['mean_block'].detach().cpu().numpy()
        alt = batch['target_stats']['alternation_idx'].detach().cpu().numpy()
        meanBlocks.append(mb)
        alternations.append(alt)

    Z = np.concatenate(zs, axis=0)  # [n_vis, D]
    H = np.concatenate(hs, axis=0)  # [n_vis, D’]
    y_aa = np.concatenate(probAAs, axis=0).reshape(-1)
    y_bb = np.concatenate(probBBs, axis=0).reshape(-1)

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
        plt.colorbar(sc, fraction=0.046, pad=0.04)
        plt.title(title)
        plt.tight_layout()
        path = out_dir / f"{base_name}.{fig_format}"
        plt.savefig(path, format=fig_format, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {path}")

    scatter_color(Z2, y_aa, "t-SNE of z colored by probAA", "tsne_z_probAA", args.fig_format)
    scatter_color(Z2, y_bb, "t-SNE of z colored by probBB", "tsne_z_probBB", args.fig_format)
    scatter_color(H2, y_aa, "t-SNE of set_emb colored by probAA", "tsne_h_probAA", args.fig_format)
    scatter_color(H2, y_bb, "t-SNE of set_emb colored by probBB", "tsne_h_probBB", args.fig_format)

    scatter_color(Z2, y_mb,  "t-SNE of z colored by mean_block",        "tsne_z_meanBlock", args.fig_format)
    scatter_color(Z2, y_alt, "t-SNE of z colored by alternation_index", "tsne_z_alternation", args.fig_format)
    scatter_color(H2, y_mb,  "t-SNE of set_emb colored by mean_block",        "tsne_h_meanBlock", args.fig_format)
    scatter_color(H2, y_alt, "t-SNE of set_emb colored by alternation_index", "tsne_h_alternation", args.fig_format)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./data/copolymer.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/encoder_runs")
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Encoder config
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--token_layers", type=int, default=2)
    parser.add_argument("--token_inducing", type=int, default=16)
    parser.add_argument("--token_seeds", type=int, default=1)
    parser.add_argument("--set_layers", type=int, default=2)
    parser.add_argument("--set_seeds", type=int, default=1)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--queue_size", type=int, default=8192,
                        help="MoCo queue size (0 disables queue, uses SimCLR fallback).")
    parser.add_argument("--stopgrad_pos", action="store_true",
                        help="Stop gradient on positive branch (MoCo-like).")

    # t-SNE
    parser.add_argument("--tsne_samples", type=int, default=1500,
                        help="Number of samples for t-SNE visualization.")
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    parser.add_argument("--fig_format", type=str, default="pdf", choices=["pdf", "png", "svg"],
                        help="File format for saved t-SNE figures.")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
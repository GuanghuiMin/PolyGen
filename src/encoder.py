import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Tuple


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Transform to multi-head
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask_value = -65500.0 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, mask_value)
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out(context)


class SetAttentionBlock(nn.Module):
    """Set Attention Block (SAB)"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class InducedSetAttentionBlock(nn.Module):
    """Induced Set Attention Block (ISAB)"""
    def __init__(self, d_model, n_heads, n_inducing, dropout=0.1):
        super().__init__()
        self.n_inducing = n_inducing
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing, d_model))
        
        self.attention1 = MultiheadAttention(d_model, n_heads, dropout)
        self.attention2 = MultiheadAttention(d_model, n_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Expand inducing points for batch
        I = self.inducing_points.expand(batch_size, -1, -1)
        
        # I = I + Attention(I, X, X)
        attn_out1 = self.attention1(I, x, x, mask)
        I = self.norm1(I + self.dropout(attn_out1))
        I = self.norm2(I + self.ffn1(I))
        
        # H = X + Attention(X, I, I)
        attn_out2 = self.attention2(x, I, I)
        x = self.norm3(x + self.dropout(attn_out2))
        x = self.norm4(x + self.ffn2(x))
        
        return x


class PoolingMultiheadAttention(nn.Module):
    """Pooling by Multihead Attention (PMA)"""
    def __init__(self, d_model, n_heads, n_seeds, dropout=0.1):
        super().__init__()
        self.n_seeds = n_seeds
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model))
        self.attention = MultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Expand seeds for batch
        seeds = self.seeds.expand(batch_size, -1, -1)
        
        # Seeds attend to input set
        attn_out = self.attention(seeds, x, x, mask)
        seeds = self.norm1(seeds + self.dropout(attn_out))
        seeds = self.norm2(seeds + self.ffn(seeds))
        
        return seeds


class SetTransformer(nn.Module):
    """Set Transformer for sequence embedding"""
    def __init__(self, d_model=128, n_heads=8, n_layers=4, n_inducing=16, n_seeds=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding for A/B tokens
        self.token_embedding = nn.Embedding(3, d_model)  # 0: pad, 1: A, 2: B
        
        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # Max seq length
        
        # Encoder layers (ISAB for efficiency with long sequences)
        self.encoder_layers = nn.ModuleList([
            InducedSetAttentionBlock(d_model, n_heads, n_inducing, dropout)
            for _ in range(n_layers)
        ])
        
        # Pooling layer
        self.pooling = PoolingMultiheadAttention(d_model, n_heads, n_seeds, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sequences, mask=None):
        """
        Args:
            sequences: (batch_size, seq_len) - tokenized sequences
            mask: (batch_size, seq_len) - padding mask
        Returns:
            embeddings: (batch_size, n_seeds, d_model) - set embeddings
        """
        batch_size, seq_len = sequences.shape
        
        # Token embedding
        x = self.token_embedding(sequences)  # (B, L, D)
        
        # Add position encoding
        if seq_len <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            # Handle sequences longer than expected
            pos_enc = self.pos_encoding.repeat(1, (seq_len // self.pos_encoding.size(1)) + 1, 1)
            x = x + pos_enc[:, :seq_len, :]
        
        x = self.dropout(x)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Pool to fixed size representation
        x = self.pooling(x, mask)  # (B, n_seeds, D)
        
        return x

class ContrastiveSetTransformerEncoder(nn.Module):
    """
    Contrastive Set Transformer Encoder

    This module adapts the token-level SetTransformer (for single chain sequences)
    to handle a *set of chains* per sample and produce:
      - set-level embedding suitable for downstream prediction
      - an L2-normalized projection ('z') for contrastive losses (InfoNCE, SupCon, etc.)

    Expected inputs (from your collate_fn_set_transformer):
        chain_sets: LongTensor [B, S, L]
            Tokenized polymer chains (A=1, B=2, PAD=0) per set.
        set_masks: BoolTensor [B, S]
            True for real chains, False for padded chains.

    Notes on masks:
      - Token mask for the inner SetTransformer uses PAD=0 to mask tokens.
      - Set mask is converted to attention masks for set-wise SAB and PMA.

    Usage:
        enc = ContrastiveSetTransformerEncoder(d_model=128, proj_dim=128, temperature=0.07)
        out = enc(chain_sets=batch['chain_sets'], set_masks=batch['set_masks'])
        z = out['z']          # [B, proj_dim], L2-normalized for InfoNCE
        h = out['set_emb']    # [B, D], pooled set embedding
    """
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        # token-level (per-chain) encoder config
        token_n_layers: int = 2,
        token_n_inducing: int = 16,
        token_n_seeds: int = 1,
        # set-level (per-set) encoder config
        set_n_layers: int = 2,
        set_n_seeds: int = 1,
        # projection head for contrastive learning
        proj_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

        # Reuse your token-level SetTransformer to embed each single chain
        self.token_encoder = SetTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=token_n_layers,
            n_inducing=token_n_inducing,
            n_seeds=token_n_seeds,
            dropout=dropout,
        )

        # Set-level encoder over the set of chain embeddings
        self.set_layers = nn.ModuleList([
            SetAttentionBlock(d_model, n_heads, dropout) for _ in range(set_n_layers)
        ])
        self.set_pool = PoolingMultiheadAttention(d_model, n_heads, set_n_seeds, dropout)


        # Projection head for contrastive learning (normalized output)
        self.norm = nn.LayerNorm(d_model * set_n_seeds)
        self.projector = nn.Sequential(
            nn.Linear(d_model * set_n_seeds, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, proj_dim),
        )

    @staticmethod
    def _make_token_mask(tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: LongTensor [BS, L] with PAD=0
        returns: Float mask broadcastable to attention scores [BS, 1, Q, K] (we use [BS, 1, 1, L])
                 with 1.0 for valid positions and 0.0 for pads.
        """
        mask_l = (tokens != 0).float()  # [BS, L]
        return mask_l.unsqueeze(1).unsqueeze(2)  # [BS, 1, 1, L]

    @staticmethod
    def _make_set_mask(set_mask_bool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        set_mask_bool: BoolTensor [B, S]
        returns:
          sab_mask: FloatTensor [B, 1, S, S]  (for self-attention among S chains)
          pma_mask: FloatTensor [B, 1, 1, S]  (for pooling seeds attending S chains)
        """
        # SAB: allow attend only among real chains
        sab = (set_mask_bool.unsqueeze(1) & set_mask_bool.unsqueeze(2)).float()  # [B, S, S]
        sab = sab.unsqueeze(1)  # [B, 1, S, S]

        # PMA: keys are the set elements
        pma = set_mask_bool.float().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        return sab, pma

    def encode_chains(self, chain_sets: torch.Tensor) -> torch.Tensor:
        """
        Encode each chain independently with the token-level SetTransformer.

        chain_sets: LongTensor [B, S, L]
        returns: FloatTensor [B, S, D] chain embeddings
        """
        B, S, L = chain_sets.shape
        flat = chain_sets.reshape(B * S, L)                     # [BS, L]
        token_mask = self._make_token_mask(flat)                # [BS, 1, 1, L]
        chain_emb = self.token_encoder(flat, mask=token_mask)   # [BS, token_n_seeds, D]
        chain_emb = chain_emb.mean(dim=1)                       # [BS, D] (if token_n_seeds>1)
        chain_emb = chain_emb.view(B, S, -1)                    # [B, S, D]
        return chain_emb

    def encode_set(
        self,
        chain_sets: torch.Tensor,
        set_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce set embedding and projected contrastive vector.

        returns:
          set_emb: [B, D]  (if set_n_seeds==1) else [B, set_n_seeds*D] before projection
          z:       [B, proj_dim] (L2-normalized)
        """
        B, S, L = chain_sets.shape
        assert set_masks is not None, "set_masks [B, S] is required."

        # Step 1: encode each chain to a D-dim vector
        chain_emb = self.encode_chains(chain_sets)  # [B, S, D]

        # Step 2: set-level encoder with SAB
        sab_mask, pma_mask = self._make_set_mask(set_masks)  # masks for SAB / PMA
        x = chain_emb
        for layer in self.set_layers:
            x = layer(x, sab_mask)  # [B, S, D]

        # Step 3: pool the set to fixed-size representation
        pooled = self.set_pool(x, pma_mask)  # [B, set_n_seeds, D]
        pooled = pooled.reshape(B, -1)       # [B, set_n_seeds*D]


        # Step 5: projection head for contrastive learning
        pooled = self.norm(pooled)                 # stabilize
        z = self.projector(pooled)                 # [B, proj_dim]
        z = F.normalize(z, dim=-1)                # L2 normalize

        # If set_n_seeds==1, expose a convenient [B, D] set embedding; else keep flattened
        set_emb = pooled if pooled.dim() == 2 else pooled.view(B, -1)
        return set_emb, z

    def forward(
        self,
        chain_sets: torch.Tensor,
        set_masks: torch.Tensor,
        return_seq_emb: bool = False,
    ):
        """
        Forward wrapper producing:
          - 'set_emb': pooled set embedding (pre-projection)
          - 'z':       normalized projection for contrastive loss
          - optionally 'chain_embs': per-chain embeddings [B, S, D]
        """
        B, S, L = chain_sets.shape
        # Per-chain embeddings (to return if needed)
        chain_embs = self.encode_chains(chain_sets)  # [B, S, D]

        # Set-level aggregation + projection
        sab_mask, pma_mask = self._make_set_mask(set_masks)
        x = chain_embs
        for layer in self.set_layers:
            x = layer(x, sab_mask)
        pooled = self.set_pool(x, pma_mask).reshape(B, -1)
        pooled = self.norm(pooled)
        z = F.normalize(self.projector(pooled), dim=-1)

        out = {
            "set_emb": pooled,   # [B, set_n_seeds*D]
            "z": z,              # [B, proj_dim], normalized
        }
        if return_seq_emb:
            out["chain_embs"] = chain_embs  # [B, S, D]
        return out

    # ---------- Optional helpers for contrastive training ----------

    def pairwise_logits(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ z.t()) / self.temperature

    def info_nce(
        self,
        z: torch.Tensor,
        positive_mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        B = z.size(0)
        logits = self.pairwise_logits(z)                      # [B, B]
        logits = logits - torch.eye(B, device=z.device) * 1e9  # remove self-comparisons

        exp_logits = torch.exp(logits)                        # [B, B]
        pos = exp_logits * positive_mask.float()              # [B, B]
        denom = exp_logits.sum(dim=-1) + eps                  # [B]
        numer = pos.sum(dim=-1) + eps                         # [B]

        loss = -torch.log(numer / denom)                      # [B]
        # Only average over anchors that have at least one positive
        has_pos = (positive_mask.sum(dim=-1) > 0).float()     # [B]
        loss = (loss * has_pos).sum() / (has_pos.sum() + eps)
        return loss


# -------------------- ConditionEncoder --------------------
class ConditionEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,            
        d_model: int = 128, 
        proj_dim: int = 256, 
        num_layers: int = 3, 
        dropout: float = 0.1,
        temperature: float = 0.10,
    ):
        super().__init__()
        self.temperature = temperature
        
        layers = []
        dims = [in_dim] + [d_model] * num_layers
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
            ]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(d_model)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, cond: torch.Tensor):
        h = self.backbone(cond)   # [B, d_model]
        h = self.norm(h)
        z = self.projector(h)     # [B, proj_dim]
        z = F.normalize(z, dim=-1)
        return {"cond_emb": h, "z": z}

    def pairwise_logits(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ z.t()) / self.temperature

    def info_nce(
        self,
        z: torch.Tensor,
        positive_mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:

        B = z.size(0)
        logits = self.pairwise_logits(z)                        # [B, B]
        logits = logits - torch.eye(B, device=z.device) * 1e9   

        exp_logits = torch.exp(logits)                          # [B, B]
        pos = exp_logits * positive_mask.float()                # [B, B]
        denom = exp_logits.sum(dim=-1) + eps                    # [B]
        numer = pos.sum(dim=-1) + eps                           # [B]

        loss = -torch.log(numer / denom)                        # [B]
        has_pos = (positive_mask.sum(dim=-1) > 0).float()       # [B]
        loss = (loss * has_pos).sum() / (has_pos.sum() + eps)
        return loss
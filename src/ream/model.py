from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .config import ModelConfig


class TextBranch(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(cfg.text_model)
        hidden = int(self.encoder.config.hidden_size)
        self.proj = nn.Sequential(nn.Linear(hidden, cfg.representation_dim), nn.GELU(), nn.Dropout(cfg.dropout))
        self.logit = nn.Linear(cfg.representation_dim, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            # Only forwarded when the tokenizer produced the tensor; some encoders reject the argument.
            kwargs["token_type_ids"] = token_type_ids
        cls = self.encoder(**kwargs).last_hidden_state[:, 0]
        h = self.proj(cls)
        return h, self.logit(h).squeeze(-1)


class SpatialBranch(nn.Module):
    def __init__(self, cfg: ModelConfig, input_dim: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.representation_dim), nn.GELU(), nn.Dropout(cfg.dropout),
        )
        self.logit = nn.Linear(cfg.representation_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        return h, self.logit(h).squeeze(-1)


class CharCNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(257, cfg.char_embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(cfg.char_embedding_dim, cfg.char_cnn_channels, k) for k in cfg.char_kernel_sizes
        ])

    def forward(self, chars):
        # chars: [B, N, L]
        b, n, l = chars.shape
        x = self.embedding(chars.reshape(b * n, l)).transpose(1, 2)
        pooled = [torch.amax(torch.relu(conv(x)), dim=-1) for conv in self.convs]
        return torch.cat(pooled, dim=-1).reshape(b, n, -1)


class NeighborhoodBranch(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.char_cnn = CharCNN(cfg)
        char_dim = cfg.char_cnn_channels * len(cfg.char_kernel_sizes)
        self.category_embedding = nn.Embedding(4096, cfg.category_embedding_dim, padding_idx=0)
        self.neighbor_proj = nn.Sequential(
            nn.Linear(char_dim + cfg.category_embedding_dim + cfg.rbf_bins, cfg.representation_dim),
            nn.GELU(),
        )
        self.pair_proj = nn.Sequential(nn.Linear(cfg.representation_dim * 4, cfg.representation_dim), nn.GELU())
        self.summary_proj = nn.Sequential(nn.Linear(cfg.representation_dim * 2, cfg.representation_dim), nn.GELU(), nn.Dropout(cfg.dropout))
        self.logit = nn.Linear(cfg.representation_dim, 1)
        # Neighbor distances reach the model normalized to [0, 1], so the RBF centres span that range.
        centers = torch.linspace(0.0, 1.0, cfg.rbf_bins)
        self.register_buffer("rbf_centers", centers)

    def _encode_neighbors(self, chars, categories, distances):
        name = self.char_cnn(chars)
        cat = self.category_embedding(categories)
        rbf = torch.exp(-0.5 * ((distances.unsqueeze(-1) - self.rbf_centers) / self.cfg.rbf_sigma) ** 2)
        return self.neighbor_proj(torch.cat([name, cat, rbf], dim=-1))

    def _direction(self, va, da, ma, vb, db, mb):
        norm_a = va / va.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        norm_b = vb / vb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cosine = torch.einsum("bnd,bmd->bnm", norm_a, norm_b)
        distance_penalty = (da.unsqueeze(-1) - db.unsqueeze(-2)).abs()
        scores = cosine - self.cfg.beta * distance_penalty
        valid = ma.unsqueeze(-1) & mb.unsqueeze(-2)
        scores = scores.masked_fill(~valid, -1e4)
        weights = torch.softmax(scores / self.cfg.temperature, dim=-1)
        # Masking before the softmax alone leaves a uniform row where a side has no neighbors, so
        # invalid entries are zeroed after it and the row is renormalized.
        weights = weights * valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        matched = torch.einsum("bnm,bmd->bnd", weights, vb)
        pair = torch.cat([va, matched, (va - matched).abs(), va * matched], dim=-1)
        pair = self.pair_proj(pair) * ma.unsqueeze(-1).float()
        summary = pair.sum(dim=1) / ma.sum(dim=1, keepdim=True).clamp_min(1).float()
        effective = weights.max(dim=-1).values
        effective = (effective * ma.float()).sum(dim=1) / ma.sum(dim=1).clamp_min(1).float()
        return summary, effective

    def forward(self, batch):
        va = self._encode_neighbors(batch["left_neighbor_chars"], batch["left_neighbor_categories"], batch["left_neighbor_distances"])
        vb = self._encode_neighbors(batch["right_neighbor_chars"], batch["right_neighbor_categories"], batch["right_neighbor_distances"])
        # Averaged over both directions so an imbalance in neighborhood size does not pick the reference side.
        sa, ea = self._direction(va, batch["left_neighbor_distances"], batch["left_neighbor_mask"], vb, batch["right_neighbor_distances"], batch["right_neighbor_mask"])
        sb, eb = self._direction(vb, batch["right_neighbor_distances"], batch["right_neighbor_mask"], va, batch["left_neighbor_distances"], batch["left_neighbor_mask"])
        h = self.summary_proj(torch.cat([sa, sb], dim=-1))
        return h, self.logit(h).squeeze(-1), 0.5 * (ea + eb)


class ReliabilityHead(nn.Module):
    """Estimates the probability that a branch decision is correct, from the branch representation and its quality descriptors."""

    def __init__(self, rep_dim: int, quality_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rep_dim + quality_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
    def forward(self, h, q):
        return torch.sigmoid(self.net(torch.cat([h, q], dim=-1)).squeeze(-1))


class AttentionGate(nn.Module):
    def __init__(self, rep_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(rep_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, reps):
        return self.net(reps).squeeze(-1)


class ReAM(nn.Module):
    """Three-branch ReAM model with reliability, confidence, attention, or static fusion."""
    def __init__(self, cfg: ModelConfig, fusion: str = "reliability", use_quality: bool = True):
        super().__init__()
        self.cfg = cfg
        self.fusion = fusion
        self.use_quality = use_quality
        self.text = TextBranch(cfg)
        self.spatial = SpatialBranch(cfg)
        self.neighborhood = NeighborhoodBranch(cfg)
        # Quality width per branch, matching text_quality (6) and both 4-wide quality blocks.
        # Ablating quality zeroes the width rather than removing heads.
        qdims = [6 if use_quality else 0, 4 if use_quality else 0, 4 if use_quality else 0]
        self.reliability_heads = nn.ModuleList([
            ReliabilityHead(cfg.representation_dim, q, cfg.hidden_dim, cfg.dropout) for q in qdims
        ])
        self.attention_gate = AttentionGate(cfg.representation_dim, cfg.hidden_dim, cfg.dropout)
        # Identity transform until fit_effective_statistics replaces these after the warm-up epochs.
        self.register_buffer("effective_mean", torch.tensor(0.0))
        self.register_buffer("effective_std", torch.tensor(1.0))

    def set_effective_statistics(self, mean: float, std: float):
        self.effective_mean.fill_(float(mean))
        self.effective_std.fill_(max(float(std), 1e-6))

    def forward(self, batch):
        ht, zt = self.text(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
        hs, zs = self.spatial(batch["spatial_features"])
        hn, zn, effective = self.neighborhood(batch)
        reps = torch.stack([ht, hs, hn], dim=1)
        logits = torch.stack([zt, zs, zn], dim=1)
        probs = torch.sigmoid(logits)
        availability = batch["availability"].float()
        if self.fusion == "static":
            signal = availability.clone()
            reliabilities = torch.ones_like(signal)
        elif self.fusion == "confidence":
            # Distance from the 0.5 decision boundary, used as a confidence proxy in place of a learned reliability.
            signal = torch.maximum(probs, 1.0 - probs) * availability
            reliabilities = signal
        elif self.fusion == "attention":
            scores = self.attention_gate(reps).masked_fill(availability <= 0, -1e4)
            weights = torch.softmax(scores, dim=1) * availability
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps)
            final_logit = (weights * logits).sum(dim=1)
            return {"final_logit": final_logit, "probability": torch.sigmoid(final_logit), "branch_logits": logits, "branch_probabilities": probs, "reliability": weights, "weights": weights, "effective_correspondence": effective}
        else:
            # Index 2 is the effective-correspondence ratio, the one descriptor that depends on
            # the learned neighborhood representation itself.
            neighborhood_q = batch["neighborhood_quality"].clone()
            neighborhood_q[:, 2] = (effective - self.effective_mean) / self.effective_std
            qs = [batch["text_quality"], batch["spatial_quality"], neighborhood_q]
            if not self.use_quality:
                qs = [q[:, :0] for q in qs]
            reliability = torch.stack([head(h, q) for head, h, q in zip(self.reliability_heads, [ht, hs, hn], qs)], dim=1)
            reliabilities = reliability
            signal = reliability * availability
        weights = signal / signal.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps)
        final_logit = (weights * logits).sum(dim=1)
        return {"final_logit": final_logit, "probability": torch.sigmoid(final_logit), "branch_logits": logits, "branch_probabilities": probs, "reliability": reliabilities, "weights": weights, "effective_correspondence": effective}

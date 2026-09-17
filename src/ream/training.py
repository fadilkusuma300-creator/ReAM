from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .config import Config
from .metrics import select_threshold


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def branch_correctness(branch_probabilities, labels):
    """Per-branch correctness at the fixed 0.5 decision threshold, used as the reliability regression target."""
    decisions = branch_probabilities >= 0.5
    return (decisions == labels.unsqueeze(1).bool()).float()


def loss_components(outputs, batch, correctness_targets, lambda_branch: float, lambda_reliability: float, reliability_enabled: bool = True):
    """Combines the match, branch and reliability terms into `L_match + lambda_branch * L_branch + lambda_reliability * L_rel`."""
    y = batch["labels"]
    availability = batch["availability"]
    bce = nn.functional.binary_cross_entropy_with_logits(outputs["final_logit"], y)
    branch_raw = nn.functional.binary_cross_entropy_with_logits(outputs["branch_logits"], y.unsqueeze(1).expand_as(outputs["branch_logits"]), reduction="none")
    # Averaged over available branches only, so missing evidence neither contributes loss nor dilutes the normalizer.
    branch = (branch_raw * availability).sum() / availability.sum().clamp_min(1.0)
    rel = torch.zeros((), device=y.device)
    if reliability_enabled and correctness_targets is not None:
        raw = (outputs["reliability"] - correctness_targets) ** 2
        rel = (raw * availability).sum() / availability.sum().clamp_min(1.0)
    total = bce + lambda_branch * branch + lambda_reliability * rel
    return total, {"match": bce.detach(), "branch": branch.detach(), "reliability": rel.detach()}


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        for i, sample_id in enumerate(batch["ids"]):
            rows.append({
                "id": sample_id,
                "label": float(batch["labels"][i].item()),
                "probability": float(out["probability"][i].item()),
                "branch_probabilities": out["branch_probabilities"][i].detach().cpu().tolist(),
                "reliability": out["reliability"][i].detach().cpu().tolist(),
                "weights": out["weights"][i].detach().cpu().tolist(),
                "availability": batch["availability"][i].detach().cpu().tolist(),
            })
    return rows



@torch.no_grad()
def fit_effective_statistics(model, loader, device):
    """Records the mean and standard deviation of the effective-correspondence ratio over available rows."""
    model.eval()
    values = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        mask = batch["availability"][:, 2] > 0
        if mask.any():
            values.append(out["effective_correspondence"][mask].detach().float().cpu())
    if values:
        x = torch.cat(values)
        model.set_effective_statistics(float(x.mean()), float(x.std(unbiased=False).clamp_min(1e-6)))

@torch.no_grad()
def refresh_correctness(model, loader, device):
    """Recomputes branch correctness targets, which drift as the branches keep training."""
    model.eval()
    mapping = {}
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        correct = branch_correctness(out["branch_probabilities"], batch["labels"])
        for i, sample_id in enumerate(batch["ids"]):
            mapping[str(sample_id)] = correct[i].cpu()
    return mapping


def train_model(model, train_loader, refresh_loader, valid_loader, cfg: Config, output_dir: str | Path, reliability_supervision: bool = True):
    """Trains the model and writes the best checkpoint, selected on validation F1.

    `refresh_loader` re-reads the training rows without shuffling, and is used to recompute the
    branch correctness targets and the remaining statistics between epochs.
    """
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    text_params = list(model.text.encoder.parameters())
    text_ids = {id(p) for p in text_params}
    other_params = [p for p in model.parameters() if id(p) not in text_ids]
    optimizer = torch.optim.AdamW([
        {"params": text_params, "lr": cfg.training.text_lr},
        {"params": other_params, "lr": cfg.training.other_lr},
    ], weight_decay=cfg.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    correctness = None
    best = -1.0; stale = 0; best_threshold = 0.5
    history = []
    for epoch in range(cfg.training.max_epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        running = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}", leave=False):
            batch = move_batch(batch, device)
            target = None
            if correctness is not None:
                target = torch.stack([correctness[str(i)] for i in batch["ids"]]).to(device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.training.use_bf16 and device.type == "cuda")
            with amp:
                out = model(batch)
                if epoch < cfg.training.warmup_epochs:
                    # Warm-up trains the branches on their own BCE alone; the match and reliability
                    # terms only start once the branch decisions are meaningful.
                    availability = batch["availability"]
                    raw = nn.functional.binary_cross_entropy_with_logits(
                        out["branch_logits"], batch["labels"].unsqueeze(1).expand_as(out["branch_logits"]), reduction="none"
                    )
                    total = (raw * availability).sum() / availability.sum().clamp_min(1.0)
                    parts = {"match": torch.zeros((), device=device), "branch": total.detach(), "reliability": torch.zeros((), device=device)}
                else:
                    total, parts = loss_components(out, batch, target, cfg.training.lambda_branch, cfg.training.lambda_reliability, reliability_supervision)
            total.backward(); optimizer.step()
            running.append(float(total.detach().cpu()))
        if hasattr(refresh_loader.dataset, "set_epoch"):
            refresh_loader.dataset.set_epoch(epoch + 1)
        if epoch + 1 == cfg.training.warmup_epochs and hasattr(model, "set_effective_statistics"):
            # Fitted once, at the end of warm-up, because the ratio depends on the neighborhood
            # representation and is not stable before the branches have converged.
            fit_effective_statistics(model, refresh_loader, device)
        correctness = refresh_correctness(model, refresh_loader, device)
        valid = infer(model, valid_loader, device)
        probs = [r["probability"] for r in valid]; labels = [r["label"] for r in valid]
        threshold, score = select_threshold(probs, labels, cfg.training.threshold_grid_points)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(running)), "valid_f1": score, "threshold": threshold})
        if score > best + 1e-8:
            best = score; best_threshold = threshold; stale = 0
            torch.save({"model": model.state_dict(), "threshold": threshold, "config": cfg.__dict__}, output_dir / "best.pt")
        else:
            stale += 1
        if stale >= cfg.training.patience:
            break
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_threshold, history

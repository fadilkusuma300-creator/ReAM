from __future__ import annotations

import json
import math
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

NULL = "[NULL]"
TEXT_FIELDS = ("name", "address", "postal_code", "category")
NUMERIC_RE = re.compile(r"\d+")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def serialize_record(record: dict[str, Any]) -> str:
    """Renders one entity as `[COL_NAME] value ...` so the encoder sees which column a value came from."""
    pieces = []
    for field in TEXT_FIELDS:
        value = record.get(field)
        # A missing field keeps its column marker and carries the NULL token, so absence is visible.
        pieces.extend([f"[COL_{field.upper()}]", str(value) if value not in (None, "") else NULL])
    return " ".join(pieces)


def serialize_pair(example: dict[str, Any]) -> str:
    return f"{serialize_record(example['left'])} [SEP] {serialize_record(example['right'])}"


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def branch_availability(example: dict[str, Any]):
    """Returns the [text, spatial, neighborhood] availability mask; unavailable branches are excluded from fusion."""
    left, right = example["left"], example["right"]
    text = any(left.get(f) not in (None, "") or right.get(f) not in (None, "") for f in TEXT_FIELDS)
    spatial = all(x.get(k) is not None for x in (left, right) for k in ("lat", "lon"))
    neighborhood = bool(left.get("neighbors")) and bool(right.get("neighbors"))
    return np.array([text, spatial, neighborhood], dtype=np.float32)


def text_quality(example: dict[str, Any], valid_tokens: int, truncated: bool) -> np.ndarray:
    """Text quality descriptors: coverage, token count, numeric availability, house-number conflict, postal-code conflict, truncation."""
    left, right = example["left"], example["right"]
    values = [(left.get(f), right.get(f)) for f in TEXT_FIELDS]
    coverage = sum(v not in (None, "") for pair in values for v in pair) / 8.0
    numeric_available = float(any(NUMERIC_RE.search(str(v)) for pair in values for v in pair if v not in (None, "")))
    house_conflict = 0.0
    lnum = NUMERIC_RE.search(str(left.get("address") or ""))
    rnum = NUMERIC_RE.search(str(right.get("address") or ""))
    if lnum and rnum:
        house_conflict = float(lnum.group(0) != rnum.group(0))
    postal_conflict = 0.0
    if left.get("postal_code") not in (None, "") and right.get("postal_code") not in (None, ""):
        postal_conflict = float(str(left["postal_code"]) != str(right["postal_code"]))
    return np.array([coverage, float(valid_tokens), numeric_available, house_conflict, postal_conflict, float(truncated)], dtype=np.float32)


def _spatial_density(record: dict[str, Any], radius_m: float = 500.0) -> float:
    """Density for the spatial branch, read from `spatial_neighbor_count` when the degrader has frozen it.

    This keeps neighborhood deletion from disturbing the spatial descriptor.
    """
    count = record.get("spatial_neighbor_count")
    if count is None:
        count = len(record.get("neighbors") or [])
    return float(count) / (math.pi * radius_m * radius_m)


def _current_neighborhood_density(record: dict[str, Any], radius_m: float = 500.0) -> float:
    """Density of the current neighbor list, including any neighbors the degrader removed."""
    return len(record.get("neighbors") or []) / (math.pi * radius_m * radius_m)


def spatial_features(example: dict[str, Any], typical_scale_m: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Spatial encoder input followed by its quality block.

    Input: log distance, left density, right density, nearest-candidate margin, distance over typical scale.
    Quality: relative distance, nearest-candidate margin, mean density, density asymmetry.
    """
    left, right = example["left"], example["right"]
    if not all(x.get(k) is not None for x in (left, right) for k in ("lat", "lon")):
        return np.zeros(5, dtype=np.float32), np.zeros(4, dtype=np.float32)
    d = haversine_m(float(left["lat"]), float(left["lon"]), float(right["lat"]), float(right["lon"]))
    dl, dr = _spatial_density(left), _spatial_density(right)
    nearest_margin = float(example.get("nearest_candidate_margin_m", 0.0))
    rel = d / max(float(typical_scale_m), 1e-6)
    main = np.array([math.log1p(d), dl, dr, nearest_margin, rel], dtype=np.float32)
    quality = np.array([rel, nearest_margin, (dl + dr) / 2.0, abs(dl - dr)], dtype=np.float32)
    return main, quality


def neighborhood_quality(example: dict[str, Any], effective_ratio: float = 0.0) -> np.ndarray:
    """Neighborhood quality descriptors: neighbor count, density asymmetry, effective correspondence ratio, category divergence.

    Index 2 is a placeholder here and is overwritten in `ReAM.forward` by the branch's own correspondence ratio.
    """
    left, right = example["left"], example["right"]
    ln, rn = left.get("neighbors") or [], right.get("neighbors") or []
    ld, rd = _current_neighborhood_density(left), _current_neighborhood_density(right)
    lc = Counter(str(n.get("category") or "") for n in ln)
    rc = Counter(str(n.get("category") or "") for n in rn)
    keys = set(lc) | set(rc)
    total_l, total_r = max(len(ln), 1), max(len(rn), 1)
    cat_diff = 0.5 * sum(abs(lc[k] / total_l - rc[k] / total_r) for k in keys)
    return np.array([min(len(ln), len(rn)), abs(ld - rd), effective_ratio, cat_diff], dtype=np.float32)


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]


class ReAMCollator:
    def __init__(self, tokenizer, max_length=128, max_neighbors=15, max_neighbor_chars=64, typical_scales=None, text_standardizer=None, spatial_feature_standardizer=None, spatial_quality_standardizer=None, neighborhood_standardizer=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_neighbors = max_neighbors
        self.max_neighbor_chars = max_neighbor_chars
        self.typical_scales = typical_scales or {}
        self.text_standardizer = text_standardizer
        self.spatial_feature_standardizer = spatial_feature_standardizer
        self.spatial_quality_standardizer = spatial_quality_standardizer
        self.neighborhood_standardizer = neighborhood_standardizer

    def _neighbor_tensor(self, record):
        neighbors = (record.get("neighbors") or [])[: self.max_neighbors]
        chars = torch.zeros(self.max_neighbors, self.max_neighbor_chars, dtype=torch.long)
        categories = torch.zeros(self.max_neighbors, dtype=torch.long)
        distances = torch.ones(self.max_neighbors, dtype=torch.float32)
        mask = torch.zeros(self.max_neighbors, dtype=torch.bool)
        for i, n in enumerate(neighbors):
            name = str(n.get("name") or "").lower()[: self.max_neighbor_chars]
            # Byte-level character ids, shifted by one so index 0 stays reserved for padding.
            ids = [min(ord(ch), 255) + 1 for ch in name]
            if ids:
                chars[i, :len(ids)] = torch.tensor(ids)
            # Category strings are open-vocabulary, so they are hashed into a fixed-size embedding table.
            categories[i] = (int.from_bytes(hashlib.blake2b(str(n.get("category") or "").encode("utf-8"), digest_size=4).digest(), "big") % 4095) + 1
            # Normalized to [0, 1], matching the RBF centers in NeighborhoodBranch.
            distances[i] = min(float(n.get("distance_m", 500.0)) / 500.0, 1.0)
            mask[i] = True
        return chars, categories, distances, mask

    def __call__(self, batch):
        text = [serialize_pair(ex) for ex in batch]
        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt", return_length=True)
        lengths = tokens.pop("length", None)
        text_q, spatial_x, spatial_q, neigh_q, avail = [], [], [], [], []
        lchars, lcats, ldists, lmasks, rchars, rcats, rdists, rmasks = [], [], [], [], [], [], [], []
        for i, ex in enumerate(batch):
            valid = int(lengths[i]) if lengths is not None else int(tokens["attention_mask"][i].sum())
            text_q.append(text_quality(ex, valid, valid >= self.max_length))
            category_key = str(ex.get("category_key") or ex["left"].get("category") or ex["right"].get("category") or "default")
            scale = self.typical_scales.get(category_key, self.typical_scales.get("default", 100.0))
            sx, sq = spatial_features(ex, scale)
            spatial_x.append(sx); spatial_q.append(sq); neigh_q.append(neighborhood_quality(ex)); avail.append(branch_availability(ex))
            a = self._neighbor_tensor(ex["left"]); b = self._neighbor_tensor(ex["right"])
            for store, val in zip((lchars, lcats, ldists, lmasks), a): store.append(val)
            for store, val in zip((rchars, rcats, rdists, rmasks), b): store.append(val)
        def std(x, scaler):
            x = np.asarray(x, dtype=np.float32)
            return scaler.transform(x) if scaler is not None else x
        return {
            "input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
            "token_type_ids": tokens.get("token_type_ids"),
            "text_quality": torch.tensor(std(text_q, self.text_standardizer)),
            "spatial_features": torch.tensor(std(spatial_x, self.spatial_feature_standardizer)),
            "spatial_quality": torch.tensor(std(spatial_q, self.spatial_quality_standardizer)),
            "neighborhood_quality": torch.tensor(std(neigh_q, self.neighborhood_standardizer)),
            "availability": torch.tensor(np.asarray(avail), dtype=torch.float32),
            "left_neighbor_chars": torch.stack(lchars), "left_neighbor_categories": torch.stack(lcats), "left_neighbor_distances": torch.stack(ldists), "left_neighbor_mask": torch.stack(lmasks),
            "right_neighbor_chars": torch.stack(rchars), "right_neighbor_categories": torch.stack(rcats), "right_neighbor_distances": torch.stack(rdists), "right_neighbor_mask": torch.stack(rmasks),
            "labels": torch.tensor([float(ex["label"]) for ex in batch], dtype=torch.float32),
            "ids": [ex.get("id", str(i)) for i, ex in enumerate(batch)],
        }

class DegradedPairDataset(Dataset):
    """Builds a training view by degrading each row.

    The view depends only on (seed, epoch, index), so it is stable within an epoch regardless of shuffling.
    """
    def __init__(self, rows, degrader, seed: int = 13):
        self.rows = rows
        self.degrader = degrader
        self.seed = int(seed)
        self.epoch = 0
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        import random
        row = self.rows[index]
        rng = random.Random((self.seed + 1) * 1_000_003 + self.epoch * 97_409 + index)
        view, kinds = self.degrader.training_view(row, rng)
        view["id"] = row.get("id", str(index))
        view["degradation"] = kinds
        return view

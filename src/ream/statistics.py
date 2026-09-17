from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import haversine_m, text_quality, spatial_features, neighborhood_quality, serialize_pair


def _record_key(rec):
    return (str(rec.get("name") or ""), str(rec.get("address") or ""), rec.get("lat"), rec.get("lon"))


def fit_typical_scales(rows):
    """Typical spatial scale per category, taken as the median nearest-neighbor distance.

    The median keeps the scale robust to the long tail of isolated records.
    """
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        for side in ("left", "right"):
            rec = row[side]
            if rec.get("lat") is None or rec.get("lon") is None:
                continue
            key = (side,) + _record_key(rec)
            if key in seen:
                continue
            seen.add(key)
            category = str(rec.get("category") or row.get("category_key") or "default")
            groups[(side, category)].append((float(rec["lat"]), float(rec["lon"])))
    by_category = defaultdict(list)
    for (side, category), points in groups.items():
        if len(points) < 2:
            continue
        nearest = []
        for i, a in enumerate(points):
            best = min(haversine_m(a[0], a[1], b[0], b[1]) for j, b in enumerate(points) if j != i)
            nearest.append(best)
        by_category[category].extend(nearest)
    scales = {k: float(np.median(v)) for k, v in by_category.items() if v}
    all_values = [x for vals in by_category.values() for x in vals]
    scales["default"] = float(np.median(all_values)) if all_values else 100.0
    return scales


def _mean_std(values):
    x = np.asarray(values, dtype=np.float32)
    mean = x.mean(0)
    std = x.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def fit_statistics(rows, tokenizer, max_length=128):
    scales = fit_typical_scales(rows)
    text_q, spatial_x, spatial_q, neighborhood_q = [], [], [], []
    for row in rows:
        encoded = tokenizer(serialize_pair(row), truncation=True, max_length=max_length)
        length = len(encoded["input_ids"])
        text_q.append(text_quality(row, length, length >= max_length))
        cat = str(row.get("category_key") or row["left"].get("category") or row["right"].get("category") or "default")
        sx, sq = spatial_features(row, scales.get(cat, scales["default"]))
        spatial_x.append(sx); spatial_q.append(sq); neighborhood_q.append(neighborhood_quality(row))
    return {
        "typical_scales": scales,
        "text_quality": _mean_std(text_q),
        "spatial_features": _mean_std(spatial_x),
        "spatial_quality": _mean_std(spatial_q),
        "neighborhood_quality": _mean_std(neighborhood_q),
    }


def save_statistics(stats, path):
    Path(path).write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_statistics(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

class FrozenStandardizer:
    """Standardizer holding statistics fitted on the training partition, so evaluation reuses them."""
    def __init__(self, spec):
        self.mean = np.asarray(spec["mean"], dtype=np.float32)
        self.std = np.asarray(spec["std"], dtype=np.float32)
    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std


def build_collator_statistics(stats):
    return {
        "typical_scales": stats["typical_scales"],
        "text_standardizer": FrozenStandardizer(stats["text_quality"]),
        "spatial_feature_standardizer": FrozenStandardizer(stats["spatial_features"]),
        "spatial_quality_standardizer": FrozenStandardizer(stats["spatial_quality"]),
        "neighborhood_standardizer": FrozenStandardizer(stats["neighborhood_quality"]),
    }

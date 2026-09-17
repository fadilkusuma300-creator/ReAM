from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .metrics import f1, expected_calibration_error, brier_score, aurc, reliability_curve


def summarize(rows, threshold: float, ece_bins: int = 15):
    p = np.array([r["probability"] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)
    return {
        "n": int(len(rows)),
        "threshold": float(threshold),
        "f1": f1(p, y, threshold),
        "ece": expected_calibration_error(p, y, ece_bins),
        "brier": brier_score(p, y),
        "aurc": aurc(p, y),
    }


def branch_reliability_summary(rows, n_bins=15):
    result = {}
    names = ["text", "spatial", "neighborhood"]
    for m, name in enumerate(names):
        r, c = [], []
        for row in rows:
            if row["availability"][m] <= 0:
                continue
            bp = row["branch_probabilities"][m]
            r.append(row["reliability"][m])
            c.append(float((bp >= 0.5) == bool(row["label"])))
        result[name] = reliability_curve(r, c, n_bins) if r else []
    return result


def save_evaluation(rows, threshold, output_path, ece_bins=15, reliability_bins=15):
    payload = {
        "metrics": summarize(rows, threshold, ece_bins),
        "branch_reliability": branch_reliability_summary(rows, reliability_bins),
        "predictions": rows,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

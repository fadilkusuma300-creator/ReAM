from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def binary_predictions(probabilities, threshold: float = 0.5) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    return (p >= threshold).astype(np.int64)


def f1(probabilities, labels, threshold: float = 0.5) -> float:
    return float(f1_score(np.asarray(labels), binary_predictions(probabilities, threshold), zero_division=0))


def select_threshold(probabilities, labels, points: int = 181) -> tuple[float, float]:
    """Grid search for the F1-maximising threshold, over the range [0.05, 0.95]."""
    grid = np.linspace(0.05, 0.95, points)
    scores = np.array([f1(probabilities, labels, t) for t in grid])
    idx = int(scores.argmax())
    return float(grid[idx]), float(scores[idx])


def brier_score(probabilities, labels) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(probabilities, labels, n_bins: int = 15) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Bins are equal-width and left-closed; the last one is closed on the right so p == 1.0 is counted.
        mask = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))
        if not mask.any():
            continue
        confidence = p[mask].mean()
        accuracy = y[mask].mean()
        ece += mask.mean() * abs(confidence - accuracy)
    return float(ece)


def reliability_curve(reliability, correctness, n_bins: int = 15):
    r = np.asarray(reliability, dtype=float)
    c = np.asarray(correctness, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (r >= lo) & ((r < hi) if i < n_bins - 1 else (r <= hi))
        if mask.any():
            out.append({
                "bin": i,
                "count": int(mask.sum()),
                "mean_reliability": float(r[mask].mean()),
                "empirical_accuracy": float(c[mask].mean()),
            })
    return out


def risk_coverage_curve(probabilities, labels):
    """Risk as a function of coverage, measured from the most confident predictions downwards."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=np.int64)
    confidence = np.maximum(p, 1.0 - p)
    prediction = (p >= 0.5).astype(np.int64)
    errors = (prediction != y).astype(float)
    order = np.argsort(-confidence, kind="mergesort")
    errors = errors[order]
    cumulative_errors = np.cumsum(errors)
    k = np.arange(1, len(errors) + 1)
    coverage = k / len(errors)
    risk = cumulative_errors / k
    return coverage, risk


def aurc(probabilities, labels) -> float:
    """Area under the risk-coverage curve, integrated over coverage; lower is better."""
    coverage, risk = risk_coverage_curve(probabilities, labels)
    coverage = np.concatenate([[0.0], coverage])
    risk = np.concatenate([[risk[0] if len(risk) else 0.0], risk])
    return float(np.trapezoid(risk, coverage))

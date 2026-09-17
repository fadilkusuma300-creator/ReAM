import numpy as np
from ream.metrics import brier_score, expected_calibration_error, aurc, select_threshold


def test_brier_exact():
    assert abs(brier_score([0.0, 1.0], [0, 1])) < 1e-12


def test_ece_perfect_extremes():
    assert abs(expected_calibration_error([0.0, 1.0], [0, 1], 15)) < 1e-12


def test_aurc_finite():
    value = aurc([0.95, 0.8, 0.2, 0.05], [1, 1, 0, 0])
    assert np.isfinite(value)
    assert value >= 0.0


def test_threshold_selection():
    threshold, score = select_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert score == 1.0
    assert 0.2 < threshold <= 0.8

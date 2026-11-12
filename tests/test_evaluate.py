import numpy as np
import pandas as pd
import pytest

from secondorder.evaluate import (
    bootstrap_diff,
    decile_table,
    frequency_calibration,
    mae_purchases,
    poisson_deviance,
    safe_spearman,
    score_frame,
    top_share,
)


def _frames(n=500, seed=1):
    rng = np.random.default_rng(seed)
    mu = rng.gamma(2, 0.2, n)
    y = rng.poisson(mu).astype(float)
    snap = pd.DataFrame({"y_purchases": y, "y_spend": y * 20.0, "frequency": rng.integers(0, 5, n)})
    pred = pd.DataFrame({"purchases": mu, "spend": mu * 20.0})
    return snap, pred


def test_poisson_deviance_zero_at_truth():
    y = np.array([0.0, 1.0, 3.0])
    assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-6)
    assert poisson_deviance(y, y + 1) > 0


def test_top_share_bounds():
    a = np.array([10.0, 0.0, 5.0, 1.0])
    assert top_share(a, a, 0.25) == pytest.approx(10 / 16)
    assert top_share(a, -a, 0.25) < top_share(a, a, 0.25)
    assert top_share(np.zeros(4), a, 0.5) == 0.0


def test_safe_spearman_handles_constant():
    assert safe_spearman(np.ones(5), np.arange(5.0)) == 0.0


def test_score_frame_bias_sign():
    snap, pred = _frames()
    over = pred * 1.5
    assert score_frame(snap, over)["purchases_bias_pct"] > 0
    assert abs(score_frame(snap, pred)["purchases_bias_pct"]) < 15


def test_decile_table_has_ten_rows_and_monotone_scores():
    snap, pred = _frames()
    t = decile_table(snap, pred, "spend")
    assert len(t) == 10
    assert t["pred"].is_monotonic_decreasing


def test_frequency_calibration_caps_groups():
    snap, pred = _frames()
    snap["frequency"] = np.arange(len(snap)) % 12
    t = frequency_calibration(snap, pred, cap=7)
    assert t["freq"].max() == 7
    assert t["n"].sum() == len(snap)


def test_bootstrap_detects_a_clear_difference():
    snap, pred = _frames()
    bad = pd.DataFrame({"purchases": np.full(len(pred), pred["purchases"].mean()), "spend": 0.0})
    d = bootstrap_diff(snap, bad, pred, mae_purchases, n_boot=60)
    assert d["lo"] > 0

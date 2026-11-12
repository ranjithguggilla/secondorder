"""Checks on the committed run outputs, so a stale or broken run fails loudly."""

import json

import pytest

from secondorder.pipeline import ROOT

METRICS = ROOT / "outputs" / "metrics.json"


@pytest.fixture(scope="module")
def m():
    return json.loads(METRICS.read_text())


def test_no_label_window_crosses_the_scoring_origin(m):
    assert m["label_windows_clean_for_validation"] is True
    assert m["label_windows_clean_for_test"] is True


def test_every_origin_covers_every_customer(m):
    assert m["data"]["customers"] == 23570
    assert m["data"]["last_first_purchase"] < m["origins"][0]["origin"]


def test_probabilistic_model_beats_naive_baselines_on_test(m):
    t = m["test"]
    shipped = t[m["shipped_model"]]
    assert shipped["purchases_mae"] < t["population mean"]["purchases_mae"]
    assert shipped["purchases_mae"] < t["historical rate"]["purchases_mae"]


def test_charts_exist():
    for name in (
        "tracking",
        "frequency_calibration",
        "spend_deciles",
        "profit_curves",
        "model_comparison",
        "cohort_decay",
        "importance",
    ):
        assert (ROOT / "assets" / f"{name}.png").exists()

"""Run the whole pipeline on a small synthetic log, writing to a temp directory."""

import json

import pytest

from secondorder.pipeline import run


@pytest.fixture(scope="module")
def result(toy_orders, tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    path = root / "orders.txt"
    out = toy_orders.copy()
    out["date"] = out["date"].dt.strftime("%Y%m%d")
    out.to_csv(path, sep=" ", index=False)
    res = run(path, root / "out", quick=True, assets_dir=root / "assets")
    return res, root


def test_run_writes_metrics_and_params(result):
    res, root = result
    saved = json.loads((root / "out" / "metrics.json").read_text())
    assert saved["shipped_model"] == res["shipped_model"]
    assert (root / "out" / "model_params.json").exists()
    assert (root / "assets" / "tracking.png").exists()


def test_run_respects_label_windows(result):
    res, _ = result
    assert res["label_windows_clean_for_validation"]
    assert res["label_windows_clean_for_test"]


def test_selection_picks_a_known_model(result):
    res, _ = result
    assert res["selection"]["shipped"] in res["test"]
    assert set(res["economics"]["best_point"]) >= {"random", "hindsight"}

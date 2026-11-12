import numpy as np
import pandas as pd
import pytest

from secondorder.data import daily_purchases
from secondorder.models import (
    WEEKS,
    BetaGeometricModel,
    BoostedModel,
    HistoricalRate,
    MeanBaseline,
    Recalibrated,
)
from secondorder.windows import Origin, snapshot


@pytest.fixture(scope="module")
def snap(toy_orders):
    d = daily_purchases(toy_orders)
    return snapshot(d, Origin("t", pd.Timestamp("1997-12-01"), "test"))


def test_mean_baseline_is_constant(snap):
    p = MeanBaseline().fit(snap).predict(snap)
    assert p["purchases"].nunique() == 1


def test_historical_rate_formula(snap):
    p = HistoricalRate().fit(snap).predict(snap)
    row = snap.iloc[0]
    expected = row["frequency"] / max(row["T"], 1.0) * WEEKS
    assert p["purchases"].iloc[0] == pytest.approx(expected)


def test_bgnbd_gives_finite_nonnegative_forecasts(snap):
    m = BetaGeometricModel().fit(snap)
    p = m.predict(snap)
    assert np.isfinite(p.to_numpy()).all()
    assert (p["purchases"] >= 0).all()
    alive = m.p_alive(snap)
    assert alive.between(0, 1).all()


def test_more_repeat_purchases_means_higher_forecast(snap):
    m = BetaGeometricModel().fit(snap)
    lo = snap[snap["frequency"] == 0].head(30)
    hi = snap[snap["frequency"] >= 3].head(30)
    if len(hi) == 0:
        pytest.skip("toy data has no heavy buyers")
    assert m.predict(hi)["purchases"].mean() > m.predict(lo)["purchases"].mean()


def test_boosted_model_fits_and_predicts(snap):
    m = BoostedModel({"n_estimators": 30}).fit(snap)
    p = m.predict(snap)
    assert len(p) == len(snap)
    assert (p["purchases"] >= 0).all()
    assert m.importance().sum() == pytest.approx(1.0)


def test_recalibration_matches_totals(snap):
    base = HistoricalRate().fit(snap)
    r = Recalibrated(base).set_ratios(snap, base.predict(snap))
    out = r.predict(snap)
    assert out["purchases"].sum() == pytest.approx(snap["y_purchases"].sum())
    assert out["spend"].sum() == pytest.approx(snap["y_spend"].sum())

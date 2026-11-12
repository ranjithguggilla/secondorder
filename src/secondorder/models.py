"""Forecasters for next-quarter purchases and spend.

Every forecaster has the same shape: ``fit(snapshot)`` on a frame of features
at one or more origins, ``predict(snapshot)`` returning a frame with
``purchases`` and ``spend`` columns. Targets are never read by the
probabilistic models, only the frequency / recency / age summaries.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter, ParetoNBDFitter

from .windows import HORIZON_DAYS, feature_columns

WEEKS = HORIZON_DAYS / 7.0


class MeanBaseline:
    """Everyone gets the average customer's next-quarter behaviour."""

    name = "population mean"

    def fit(self, snap: pd.DataFrame) -> MeanBaseline:
        self.purchases_ = float(snap["y_purchases"].mean())
        self.spend_ = float(snap["y_spend"].mean())
        return self

    def predict(self, snap: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"purchases": self.purchases_, "spend": self.spend_}, index=snap.index)


class HistoricalRate:
    """Extrapolate each customer's own repeat rate and average order value.

    This is what most spreadsheets do. It has no notion of customers drifting
    away, which is the whole point of the comparison.
    """

    name = "historical rate"

    def fit(self, snap: pd.DataFrame) -> HistoricalRate:
        known = snap[snap["frequency"] > 0]
        self.fallback_value_ = float(known["monetary_value"].mean())
        return self

    def predict(self, snap: pd.DataFrame) -> pd.DataFrame:
        rate = snap["frequency"] / snap["T"].clip(lower=1.0)
        purchases = rate * WEEKS
        value = snap["monetary_value"].where(snap["frequency"] > 0, self.fallback_value_)
        return pd.DataFrame({"purchases": purchases, "spend": purchases * value}, index=snap.index)


class BetaGeometricModel:
    """BG/NBD for purchase counts plus Gamma-Gamma for order value."""

    name = "BG/NBD + Gamma-Gamma"

    def __init__(self, penalizer: float = 0.001):
        self.penalizer = penalizer

    def fit(self, snap: pd.DataFrame) -> BetaGeometricModel:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bgf_ = BetaGeoFitter(penalizer_coef=self.penalizer)
            self.bgf_.fit(snap["frequency"], snap["recency"], snap["T"])
            rep = snap[(snap["frequency"] > 0) & (snap["monetary_value"] > 0)]
            self.ggf_ = GammaGammaFitter(penalizer_coef=self.penalizer)
            self.ggf_.fit(rep["frequency"], rep["monetary_value"])
        p, q, v = (self.ggf_.params_[k] for k in ("p", "q", "v"))
        self.population_value_ = float(p * v / (q - 1.0)) if q > 1 else float(rep["monetary_value"].mean())
        return self

    def params(self) -> dict:
        out = {k: float(v) for k, v in self.bgf_.params_.items()}
        out.update({f"gg_{k}": float(v) for k, v in self.ggf_.params_.items()})
        return out

    def p_alive(self, snap: pd.DataFrame) -> pd.Series:
        alive = self.bgf_.conditional_probability_alive(snap["frequency"], snap["recency"], snap["T"])
        return pd.Series(np.asarray(alive), index=snap.index)

    def predict(self, snap: pd.DataFrame) -> pd.DataFrame:
        purchases = self.bgf_.conditional_expected_number_of_purchases_up_to_time(
            WEEKS, snap["frequency"], snap["recency"], snap["T"]
        )
        usable = (snap["frequency"] > 0) & (snap["monetary_value"] > 0)
        value = pd.Series(self.population_value_, index=snap.index, dtype=float)
        if usable.any():
            value.loc[usable] = self.ggf_.conditional_expected_average_profit(
                snap.loc[usable, "frequency"], snap.loc[usable, "monetary_value"]
            )
        return pd.DataFrame(
            {"purchases": np.asarray(purchases), "spend": np.asarray(purchases) * value.to_numpy()},
            index=snap.index,
        )


class ParetoModel(BetaGeometricModel):
    """Pareto/NBD, the continuous-time cousin, on the same spend model."""

    name = "Pareto/NBD + Gamma-Gamma"

    def fit(self, snap: pd.DataFrame) -> ParetoModel:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bgf_ = ParetoNBDFitter(penalizer_coef=self.penalizer)
            self.bgf_.fit(snap["frequency"], snap["recency"], snap["T"])
            rep = snap[(snap["frequency"] > 0) & (snap["monetary_value"] > 0)]
            self.ggf_ = GammaGammaFitter(penalizer_coef=self.penalizer)
            self.ggf_.fit(rep["frequency"], rep["monetary_value"])
        p, q, v = (self.ggf_.params_[k] for k in ("p", "q", "v"))
        self.population_value_ = float(p * v / (q - 1.0)) if q > 1 else float(rep["monetary_value"].mean())
        return self


class BoostedModel:
    """Gradient boosting on engineered recency, frequency and spacing features.

    Purchases use a Poisson objective, spend uses Tweedie, which handles the
    pile of exact zeros and the long right tail without a transform.
    """

    name = "LightGBM"

    def __init__(self, params: dict | None = None, seed: int = 7):
        base = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 80,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "verbose": -1,
            "random_state": seed,
            "n_jobs": 4,
        }
        base.update(params or {})
        self.params = base

    def fit(self, snap: pd.DataFrame) -> BoostedModel:
        x = snap[feature_columns()]
        self.count_ = lgb.LGBMRegressor(objective="poisson", **self.params).fit(x, snap["y_purchases"])
        self.spend_ = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.4, **self.params).fit(
            x, snap["y_spend"]
        )
        return self

    def predict(self, snap: pd.DataFrame) -> pd.DataFrame:
        x = snap[feature_columns()]
        return pd.DataFrame(
            {"purchases": self.count_.predict(x), "spend": self.spend_.predict(x)}, index=snap.index
        )

    def importance(self) -> pd.Series:
        s = pd.Series(self.spend_.booster_.feature_importance("gain"), index=feature_columns())
        return (s / s.sum()).sort_values(ascending=False)


class Recalibrated:
    """Wrap a forecaster and rescale it by last quarter's realised-to-forecast ratio.

    The ratio comes from a quarter that has already finished at the scoring
    origin, so it is information a team would really have. It exists because
    the cohort keeps ageing and a model trained on younger customers drifts high.
    """

    def __init__(self, base, ratio_purchases: float = 1.0, ratio_spend: float = 1.0):
        self.base = base
        self.ratio_purchases = ratio_purchases
        self.ratio_spend = ratio_spend
        self.name = f"{base.name} + last-quarter rescale"

    def fit(self, snap: pd.DataFrame) -> Recalibrated:
        self.base.fit(snap)
        return self

    def set_ratios(self, snap: pd.DataFrame, pred: pd.DataFrame) -> Recalibrated:
        self.ratio_purchases = float(snap["y_purchases"].sum() / pred["purchases"].sum())
        self.ratio_spend = float(snap["y_spend"].sum() / pred["spend"].sum())
        return self

    def predict(self, snap: pd.DataFrame) -> pd.DataFrame:
        p = self.base.predict(snap)
        p["purchases"] *= self.ratio_purchases
        p["spend"] *= self.ratio_spend
        return p

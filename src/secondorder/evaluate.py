"""Metrics, calibration tables and customer-level bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.mean(term - (y - mu)))


def top_share(actual: np.ndarray, score: np.ndarray, frac: float) -> float:
    """Share of all realised value held by the top ``frac`` of customers by score."""
    n = max(1, int(round(len(score) * frac)))
    order = np.argsort(-score, kind="stable")[:n]
    total = actual.sum()
    return float(actual[order].sum() / total) if total > 0 else 0.0


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(spearmanr(a, b).statistic)


def score_frame(snap: pd.DataFrame, pred: pd.DataFrame) -> dict:
    y_n, y_s = snap["y_purchases"].to_numpy(), snap["y_spend"].to_numpy()
    p_n, p_s = pred["purchases"].to_numpy(), pred["spend"].to_numpy()
    return {
        "customers": int(len(snap)),
        "purchases_actual": float(y_n.sum()),
        "purchases_predicted": float(p_n.sum()),
        "purchases_bias_pct": float(100 * (p_n.sum() / y_n.sum() - 1)),
        "purchases_mae": float(np.mean(np.abs(y_n - p_n))),
        "purchases_rmse": float(np.sqrt(np.mean((y_n - p_n) ** 2))),
        "purchases_poisson_dev": poisson_deviance(y_n, p_n),
        "purchases_spearman": safe_spearman(y_n, p_n),
        "spend_actual": float(y_s.sum()),
        "spend_predicted": float(p_s.sum()),
        "spend_bias_pct": float(100 * (p_s.sum() / y_s.sum() - 1)),
        "spend_mae": float(np.mean(np.abs(y_s - p_s))),
        "spend_rmse": float(np.sqrt(np.mean((y_s - p_s) ** 2))),
        "spend_spearman": safe_spearman(y_s, p_s),
        "top10_share": top_share(y_s, p_s, 0.10),
        "top20_share": top_share(y_s, p_s, 0.20),
        "oracle_top10_share": top_share(y_s, y_s, 0.10),
    }


def decile_table(snap: pd.DataFrame, pred: pd.DataFrame, col: str = "spend", bins: int = 10) -> pd.DataFrame:
    """Predicted versus realised mean per score decile (decile 1 = highest score)."""
    actual = snap["y_spend" if col == "spend" else "y_purchases"].to_numpy()
    p = pred[col].to_numpy()
    rank = pd.Series(p).rank(method="first", ascending=False)
    dec = pd.qcut(rank, bins, labels=range(1, bins + 1))
    t = pd.DataFrame({"decile": dec, "pred": p, "actual": actual}).groupby("decile", observed=True).mean()
    return t.reset_index()


def frequency_calibration(snap: pd.DataFrame, pred: pd.DataFrame, cap: int = 7) -> pd.DataFrame:
    """The classic check: group by past repeat count, compare mean forecast to mean actual."""
    f = snap["frequency"].clip(upper=cap).astype(int)
    t = pd.DataFrame(
        {"freq": f, "pred": pred["purchases"].to_numpy(), "actual": snap["y_purchases"].to_numpy()}
    )
    out = t.groupby("freq").agg(pred=("pred", "mean"), actual=("actual", "mean"), n=("pred", "size"))
    return out.reset_index()


def bootstrap_diff(
    snap: pd.DataFrame,
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    metric,
    n_boot: int = 400,
    seed: int = 11,
) -> dict:
    """Customer-level bootstrap of metric(a) - metric(b). Positive means a is larger."""
    rng = np.random.default_rng(seed)
    n = len(snap)
    y = snap.reset_index(drop=True)
    a = pred_a.reset_index(drop=True)
    b = pred_b.reset_index(drop=True)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(y.iloc[idx], a.iloc[idx]) - metric(y.iloc[idx], b.iloc[idx])
    point = metric(y, a) - metric(y, b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def mae_purchases(y: pd.DataFrame, p: pd.DataFrame) -> float:
    return float(np.mean(np.abs(y["y_purchases"].to_numpy() - p["purchases"].to_numpy())))


def mae_spend(y: pd.DataFrame, p: pd.DataFrame) -> float:
    return float(np.mean(np.abs(y["y_spend"].to_numpy() - p["spend"].to_numpy())))


def top10_spend(y: pd.DataFrame, p: pd.DataFrame) -> float:
    return top_share(y["y_spend"].to_numpy(), p["spend"].to_numpy(), 0.10)

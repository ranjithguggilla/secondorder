"""Rolling-origin snapshots.

At an origin date every feature is computed from purchases on or before that
date, and every target from purchases in the horizon after it. Nothing else
crosses the line, so a model trained on one origin never sees the future of the
origin it is later scored on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON_DAYS = 91  # thirteen weeks
FIRST_ORIGIN = pd.Timestamp("1997-03-31")


@dataclass(frozen=True)
class Origin:
    name: str
    date: pd.Timestamp
    role: str  # train, validation or test

    @property
    def end(self) -> pd.Timestamp:
        return self.date + pd.Timedelta(days=HORIZON_DAYS)


def make_origins(first: pd.Timestamp = FIRST_ORIGIN) -> list[Origin]:
    """Five consecutive origins, one horizon apart.

    The first three train the supervised model, the fourth is the validation
    quarter used for model selection, and the last is the untouched test quarter.
    """
    roles = ["train", "train", "train", "validation", "test"]
    return [Origin(f"o{i}", first + pd.Timedelta(days=HORIZON_DAYS * i), r) for i, r in enumerate(roles)]


def rfm(daily: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Classic frequency / recency / age summary in weeks, plus extra features."""
    d = daily[daily["date"] <= origin]
    g = d.groupby("customer_id")
    first = g["date"].min()
    last = g["date"].max()
    n = g.size()
    feat = pd.DataFrame(index=first.index)
    feat["frequency"] = (n - 1).astype(float)  # repeat purchase days
    feat["recency"] = (last - first).dt.days / 7.0
    feat["T"] = (origin - first).dt.days / 7.0
    feat["n_orders"] = n.astype(float)
    feat["total_spend"] = g["spend"].sum()
    feat["avg_spend"] = feat["total_spend"] / feat["n_orders"]
    feat["total_cds"] = g["cds"].sum()
    feat["weeks_since_last"] = (origin - last).dt.days / 7.0

    # monetary value as the Gamma-Gamma model wants it: mean over repeat days only
    d = d.assign(rank=g.cumcount())
    rep = d[d["rank"] > 0]
    rep_mean = rep.groupby("customer_id")["spend"].mean()
    feat["monetary_value"] = rep_mean.reindex(feat.index).fillna(0.0)

    # first and last order values
    feat["first_spend"] = d[d["rank"] == 0].set_index("customer_id")["spend"].reindex(feat.index)
    last_rows = d.sort_values("date").groupby("customer_id").tail(1).set_index("customer_id")
    feat["last_spend"] = last_rows["spend"].reindex(feat.index)

    # recent activity windows
    for weeks in (4, 8, 13, 26):
        cut = origin - pd.Timedelta(days=7 * weeks)
        recent = d[d["date"] > cut].groupby("customer_id")
        feat[f"orders_last_{weeks}w"] = recent.size().reindex(feat.index).fillna(0.0)
    cut = origin - pd.Timedelta(days=HORIZON_DAYS)
    feat["spend_last_13w"] = (
        d[d["date"] > cut].groupby("customer_id")["spend"].sum().reindex(feat.index).fillna(0.0)
    )

    # spacing between purchases
    gaps = d.groupby("customer_id")["date"].diff().dt.days.dropna()
    gap_g = gaps.groupby(d.loc[gaps.index, "customer_id"])
    feat["gap_mean"] = gap_g.mean().reindex(feat.index)
    feat["gap_std"] = gap_g.std().reindex(feat.index)
    feat["gap_mean"] = feat["gap_mean"].fillna(-1.0)
    feat["gap_std"] = feat["gap_std"].fillna(-1.0)
    feat["rate"] = feat["frequency"] / feat["T"].clip(lower=1.0)
    return feat


def targets(daily: pd.DataFrame, origin: Origin, index: pd.Index) -> pd.DataFrame:
    """Purchases and spend in the quarter after the origin."""
    d = daily[(daily["date"] > origin.date) & (daily["date"] <= origin.end)]
    g = d.groupby("customer_id")
    out = pd.DataFrame(index=index)
    out["y_purchases"] = g.size().reindex(index).fillna(0.0)
    out["y_spend"] = g["spend"].sum().reindex(index).fillna(0.0)
    return out


def snapshot(daily: pd.DataFrame, origin: Origin) -> pd.DataFrame:
    """Features and targets for one origin, indexed by customer."""
    feats = rfm(daily, origin.date)
    tg = targets(daily, origin, feats.index)
    snap = feats.join(tg)
    snap["origin"] = origin.name
    return snap


def stack(daily: pd.DataFrame, origins: list[Origin]) -> pd.DataFrame:
    """Concatenate snapshots for several origins."""
    parts = [snapshot(daily, o).reset_index(names="customer_id") for o in origins]
    return pd.concat(parts, ignore_index=True)


def check_no_overlap(train: list[Origin], target: Origin) -> bool:
    """True when no training label window reaches into the scored origin's future.

    Training labels may end at or before the scored origin date, because that is
    information already in the past when the scored origin is forecast.
    """
    return all(o.end <= target.date for o in train)


def feature_columns() -> list[str]:
    cols = [
        "frequency",
        "recency",
        "T",
        "n_orders",
        "total_spend",
        "avg_spend",
        "total_cds",
        "weeks_since_last",
        "monetary_value",
        "first_spend",
        "last_spend",
        "orders_last_4w",
        "orders_last_8w",
        "orders_last_13w",
        "orders_last_26w",
        "spend_last_13w",
        "gap_mean",
        "gap_std",
        "rate",
    ]
    return cols


def horizon_weeks() -> float:
    return float(np.round(HORIZON_DAYS / 7.0, 6))

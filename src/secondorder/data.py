"""Loading and cleaning the CDNOW order log."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "cdnow_master.txt"
REQUIRED = {"customer_id", "date", "number_of_cds", "dollar_value"}


def load_orders(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Read the raw order log: one row per order line, whitespace separated."""
    df = pd.read_csv(path, sep=r"\s+", dtype={"customer_id": str})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"order log is missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return df


def daily_purchases(orders: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per customer per calendar day.

    The probabilistic models count purchase occasions, not order lines, so two
    lines on the same day are one occasion. Spend and CD counts are summed.
    """
    out = (
        orders.groupby(["customer_id", "date"], as_index=False)
        .agg(cds=("number_of_cds", "sum"), spend=("dollar_value", "sum"))
        .sort_values(["customer_id", "date"], kind="stable")
        .reset_index(drop=True)
    )
    return out


def summarize(orders: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Headline facts about the log, used in the README and the test suite."""
    first = daily.groupby("customer_id")["date"].min()
    counts = daily.groupby("customer_id").size()
    return {
        "order_lines": int(len(orders)),
        "purchase_days": int(len(daily)),
        "customers": int(daily["customer_id"].nunique()),
        "first_date": str(daily["date"].min().date()),
        "last_date": str(daily["date"].max().date()),
        "last_first_purchase": str(first.max().date()),
        "one_time_buyers": int((counts == 1).sum()),
        "one_time_share": float((counts == 1).mean()),
        "zero_value_lines": int((orders["dollar_value"] <= 0).sum()),
        "same_day_extra_lines": int(len(orders) - len(daily)),
    }

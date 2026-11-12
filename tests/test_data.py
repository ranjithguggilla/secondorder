from pathlib import Path

import pandas as pd
import pytest

from secondorder.data import DEFAULT_PATH, daily_purchases, load_orders, summarize


def test_real_file_loads_with_expected_shape():
    orders = load_orders(DEFAULT_PATH)
    assert len(orders) == 69659
    assert orders["customer_id"].nunique() == 23570
    assert orders["date"].min() == pd.Timestamp("1997-01-01")
    assert orders["date"].max() == pd.Timestamp("1998-06-30")


def test_daily_collapse_sums_same_day_lines():
    df = pd.DataFrame(
        {
            "customer_id": ["a", "a", "a"],
            "date": pd.to_datetime(["1997-01-01", "1997-01-01", "1997-02-01"]),
            "number_of_cds": [1, 2, 1],
            "dollar_value": [10.0, 5.0, 7.0],
        }
    )
    d = daily_purchases(df)
    assert len(d) == 2
    assert d.loc[0, "spend"] == 15.0
    assert d.loc[0, "cds"] == 3


def test_summary_counts(toy_orders):
    d = daily_purchases(toy_orders)
    s = summarize(toy_orders, d)
    assert s["customers"] == 400
    assert s["purchase_days"] == len(d)
    assert 0 <= s["one_time_share"] <= 1


def test_missing_columns_rejected(tmp_path: Path):
    p = tmp_path / "bad.txt"
    p.write_text("customer_id date\n1 19970101\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_orders(p)

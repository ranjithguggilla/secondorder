import pandas as pd

from secondorder.data import daily_purchases
from secondorder.windows import (
    HORIZON_DAYS,
    Origin,
    check_no_overlap,
    make_origins,
    rfm,
    snapshot,
    targets,
)


def _tiny():
    df = pd.DataFrame(
        {
            "customer_id": ["a", "a", "a", "b", "c"],
            "date": pd.to_datetime(["1997-01-01", "1997-01-15", "1997-04-10", "1997-01-10", "1997-02-01"]),
            "number_of_cds": [1, 1, 2, 1, 1],
            "dollar_value": [10.0, 20.0, 30.0, 5.0, 8.0],
        }
    )
    return daily_purchases(df)


def test_rfm_matches_hand_computation():
    f = rfm(_tiny(), pd.Timestamp("1997-03-01"))
    a = f.loc["a"]
    assert a["frequency"] == 1  # only the 15 Jan repeat falls before 1 March
    assert a["recency"] == 14 / 7
    assert a["T"] == 59 / 7
    assert a["monetary_value"] == 20.0
    assert f.loc["b", "frequency"] == 0
    assert f.loc["b", "monetary_value"] == 0.0


def test_features_ignore_the_future():
    d = _tiny()
    early = rfm(d, pd.Timestamp("1997-03-01"))
    late_only = rfm(d[d["date"] <= "1997-03-01"], pd.Timestamp("1997-03-01"))
    pd.testing.assert_frame_equal(early.sort_index(), late_only.sort_index())


def test_targets_count_only_the_horizon():
    d = _tiny()
    o = Origin("x", pd.Timestamp("1997-03-01"), "test")
    t = targets(d, o, pd.Index(["a", "b", "c"]))
    assert t.loc["a", "y_purchases"] == 1  # 10 April is inside the 91 days
    assert t.loc["a", "y_spend"] == 30.0
    assert t.loc["b", "y_purchases"] == 0


def test_snapshot_has_features_and_targets():
    o = make_origins()[1]
    s = snapshot(_tiny(), o)
    assert {"frequency", "y_spend", "origin"} <= set(s.columns)


def test_origins_are_one_horizon_apart_and_ordered():
    o = make_origins()
    assert [x.role for x in o] == ["train", "train", "train", "validation", "test"]
    gaps = {(b.date - a.date).days for a, b in zip(o, o[1:], strict=False)}
    assert gaps == {HORIZON_DAYS}


def test_overlap_guard():
    o = make_origins()
    assert check_no_overlap(o[:3], o[3])
    assert not check_no_overlap(o[:4], o[3])

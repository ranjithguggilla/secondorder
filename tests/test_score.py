import json

import pytest
from pydantic import ValidationError

from secondorder import score as score_mod
from secondorder.app import describe
from secondorder.score import CustomerInput, load_scorer


@pytest.fixture(scope="module")
def scorer():
    return load_scorer(score_mod.DEFAULT_PARAMS)


def test_input_validation_rejects_impossible_history():
    with pytest.raises(ValidationError):
        CustomerInput(frequency=2, recency=50, age=10, monetary_value=20)
    with pytest.raises(ValidationError):
        CustomerInput(frequency=0, recency=3, age=10, monetary_value=0)
    with pytest.raises(ValidationError):
        CustomerInput(frequency=-1, recency=0, age=10, monetary_value=0)


def test_params_file_is_valid_json():
    cfg = json.loads(score_mod.DEFAULT_PARAMS.read_text())
    assert cfg["shipped"] in cfg["families"]
    assert cfg["break_even_quarterly_spend"] == pytest.approx(25.0)


def test_scorer_orders_customers_sensibly(scorer):
    active = scorer.score(CustomerInput(frequency=8, recency=50, age=52, monetary_value=40))
    lapsed = scorer.score(CustomerInput(frequency=8, recency=10, age=52, monetary_value=40))
    one_time = scorer.score(CustomerInput(frequency=0, recency=0, age=52, monetary_value=0))
    assert active["p_alive"] > lapsed["p_alive"]
    assert active["expected_spend"] > 5 * lapsed["expected_spend"]
    assert active["expected_spend"] > one_time["expected_spend"]
    assert "one-time" in one_time["segment"]
    assert 0 <= active["p_alive"] <= 1


def test_high_value_active_customer_is_worth_mailing(scorer):
    s = scorer.score(CustomerInput(frequency=10, recency=50, age=52, monetary_value=60))
    assert s["segment"].startswith("worth mailing")


def test_describe_returns_text_and_reports_bad_input(scorer):
    ok = describe(3, 30, 45, 35, scorer)
    assert "Expected spend" in ok
    bad = describe(3, 80, 45, 35, scorer)
    assert bad.startswith("Check the inputs")

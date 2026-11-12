import numpy as np
import pytest

from secondorder.economics import CampaignAssumptions, best_point, lift_sensitivity, profit_curve


def test_break_even_spend():
    a = CampaignAssumptions(margin=0.3, cost_per_contact=0.75, relative_lift=0.1)
    assert a.break_even_spend() == pytest.approx(25.0)


def test_profit_curve_math_on_a_tiny_list():
    spend = np.array([100.0, 10.0, 0.0, 0.0])
    a = CampaignAssumptions(margin=0.5, cost_per_contact=1.0, relative_lift=0.1)
    c = profit_curve(spend, spend, a, fractions=np.array([0.25, 1.0]))
    assert c.loc[0, "contacts"] == 1
    assert c.loc[0, "net_per_1000"] == pytest.approx((100 * 0.1 * 0.5 - 1.0) / 4 * 1000)
    assert c.loc[1, "net_per_1000"] == pytest.approx((110 * 0.05 - 4.0) / 4 * 1000)


def test_good_ranking_beats_random():
    rng = np.random.default_rng(0)
    spend = rng.gamma(0.3, 60, 3000)
    a = CampaignAssumptions()
    good = best_point(profit_curve(spend, spend, a))["net_per_1000"]
    rand = best_point(profit_curve(spend, rng.random(3000), a))["net_per_1000"]
    assert good > rand


def test_sensitivity_grid_shape():
    rng = np.random.default_rng(1)
    spend = rng.gamma(0.3, 60, 500)
    t = lift_sensitivity(spend, {"a": spend, "b": rng.random(500)}, lifts=(0.05, 0.1))
    assert len(t) == 4

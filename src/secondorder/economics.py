"""Turning a ranking into dollars, with the assumptions on the table."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CampaignAssumptions:
    margin: float = 0.30  # contribution margin on a CD order
    cost_per_contact: float = 0.75  # mailer or email plus offer cost per customer
    relative_lift: float = 0.10  # contacted customers spend this much more than they would have

    def break_even_spend(self) -> float:
        """Quarterly spend above which a contact pays for itself."""
        return self.cost_per_contact / (self.relative_lift * self.margin)

    def as_dict(self) -> dict:
        return asdict(self)


def profit_curve(
    actual_spend: np.ndarray,
    score: np.ndarray,
    assumptions: CampaignAssumptions,
    fractions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Net profit per 1,000 customers when mailing the top ``f`` of the ranked list."""
    if fractions is None:
        fractions = np.linspace(0.01, 1.0, 100)
    order = np.argsort(-score, kind="stable")
    gain = actual_spend[order] * assumptions.relative_lift * assumptions.margin
    cum_gain = np.cumsum(gain)
    n = len(score)
    rows = []
    for f in fractions:
        k = max(1, int(round(n * f)))
        net = cum_gain[k - 1] - k * assumptions.cost_per_contact
        rows.append({"fraction": float(f), "contacts": k, "net_per_1000": float(net / n * 1000)})
    return pd.DataFrame(rows)


def best_point(curve: pd.DataFrame) -> dict:
    i = int(curve["net_per_1000"].idxmax())
    return {
        "fraction": float(curve.loc[i, "fraction"]),
        "net_per_1000": float(curve.loc[i, "net_per_1000"]),
    }


def lift_sensitivity(
    actual_spend: np.ndarray,
    scores: dict[str, np.ndarray],
    lifts: tuple[float, ...] = (0.05, 0.10, 0.20),
    base: CampaignAssumptions | None = None,
) -> pd.DataFrame:
    base = base or CampaignAssumptions()
    rows = []
    for lift in lifts:
        a = CampaignAssumptions(base.margin, base.cost_per_contact, lift)
        for name, sc in scores.items():
            b = best_point(profit_curve(actual_spend, sc, a))
            rows.append({"lift": lift, "ranking": name, **b})
    return pd.DataFrame(rows)

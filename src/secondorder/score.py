"""Score a single customer from the three RFM numbers. This is what the UI calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter, ParetoNBDFitter
from pydantic import BaseModel, Field, model_validator

DEFAULT_PARAMS = Path(__file__).resolve().parents[2] / "outputs" / "model_params.json"


class CustomerInput(BaseModel):
    """Purchase history summary, in weeks and dollars, as of the scoring date."""

    frequency: int = Field(ge=0, description="repeat purchase days after the first")
    recency: float = Field(ge=0, description="weeks between first and last purchase")
    age: float = Field(gt=0, description="weeks since the first purchase")
    monetary_value: float = Field(ge=0, description="mean spend over repeat purchases, $")

    @model_validator(mode="after")
    def _check(self) -> CustomerInput:
        if self.recency > self.age:
            raise ValueError("recency cannot exceed age")
        if self.frequency == 0 and self.recency != 0:
            raise ValueError("a customer with no repeat purchases has recency 0")
        return self


@dataclass
class Scorer:
    purchase_model: BetaGeoFitter | ParetoNBDFitter
    value_model: GammaGammaFitter
    population_value: float
    horizon_weeks: float
    break_even: float
    name: str

    def score(self, c: CustomerInput) -> dict:
        f = pd.Series([c.frequency], dtype=float)
        r = pd.Series([c.recency], dtype=float)
        t = pd.Series([c.age], dtype=float)
        alive = float(self.purchase_model.conditional_probability_alive(f, r, t)[0])
        n = float(
            self.purchase_model.conditional_expected_number_of_purchases_up_to_time(
                self.horizon_weeks, f, r, t
            )[0]
        )
        if c.frequency > 0 and c.monetary_value > 0:
            value = float(
                self.value_model.conditional_expected_average_profit(
                    pd.Series([c.frequency], dtype=float), pd.Series([c.monetary_value], dtype=float)
                ).iloc[0]
            )
        else:
            value = self.population_value
        spend = n * value
        return {
            "model": self.name,
            "p_alive": alive,
            "expected_purchases": n,
            "expected_order_value": value,
            "expected_spend": spend,
            "segment": segment(c, alive, spend, self.break_even),
        }


def segment(c: CustomerInput, p_alive: float, spend: float, break_even: float) -> str:
    """Plain-language call for the marketing team."""
    if c.frequency == 0:
        return "one-time buyer: low odds of a second order, leave out of paid campaigns"
    if p_alive < 0.4:
        return "probably lapsed: a win-back offer is a long shot"
    if spend >= break_even:
        return "worth mailing: expected spend clears the campaign break-even"
    return "active but small: keep on cheap email only"


def load_scorer(path: str | Path = DEFAULT_PARAMS) -> Scorer:
    cfg = json.loads(Path(path).read_text())
    fam = cfg["families"][cfg["shipped"]]
    cls = ParetoNBDFitter if fam["kind"] == "pareto" else BetaGeoFitter
    model = cls()
    model.params_ = pd.Series(fam["purchase_params"])
    gg = GammaGammaFitter()
    gg.params_ = pd.Series(fam["value_params"])
    return Scorer(
        model,
        gg,
        fam["population_value"],
        cfg["horizon_weeks"],
        cfg["break_even_quarterly_spend"],
        cfg["shipped"],
    )

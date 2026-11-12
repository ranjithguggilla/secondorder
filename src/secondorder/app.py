"""Gradio front end: type in a customer's history, get a plain-language call."""

from __future__ import annotations

import gradio as gr

from .score import CustomerInput, load_scorer


def describe(frequency: int, recency: float, age: float, monetary_value: float, scorer=None) -> str:
    scorer = scorer or load_scorer()
    try:
        c = CustomerInput(frequency=frequency, recency=recency, age=age, monetary_value=monetary_value)
    except ValueError as exc:
        return f"Check the inputs: {exc}"
    s = scorer.score(c)
    return (
        f"Chance still active: {s['p_alive']:.0%}\n"
        f"Expected orders in the next 13 weeks: {s['expected_purchases']:.2f}\n"
        f"Expected order value: ${s['expected_order_value']:.2f}\n"
        f"Expected spend in the next 13 weeks: ${s['expected_spend']:.2f}\n"
        f"Call: {s['segment']}"
    )


def build():
    scorer = load_scorer()
    return gr.Interface(
        fn=lambda f, r, t, m: describe(int(f), r, t, m, scorer),
        inputs=[
            gr.Number(value=3, label="Repeat purchase days", precision=0),
            gr.Number(value=30, label="Weeks from first to last purchase"),
            gr.Number(value=45, label="Weeks since first purchase"),
            gr.Number(value=35, label="Average spend per repeat purchase ($)"),
        ],
        outputs=gr.Textbox(label="Forecast", lines=6),
        title="secondorder",
        description="Next-quarter orders and spend for one customer, from purchase history alone.",
        flagging_mode="never",
    )


if __name__ == "__main__":
    build().launch()

"""End-to-end run: data, validation, selection, test scoring, charts, artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from . import __version__  # noqa: E402
from .data import DEFAULT_PATH, daily_purchases, load_orders, summarize  # noqa: E402
from .economics import CampaignAssumptions, best_point, lift_sensitivity, profit_curve  # noqa: E402
from .evaluate import (  # noqa: E402
    bootstrap_diff,
    decile_table,
    frequency_calibration,
    mae_purchases,
    mae_spend,
    poisson_deviance,
    score_frame,
    top10_spend,
)
from .models import (  # noqa: E402
    WEEKS,
    BetaGeometricModel,
    BoostedModel,
    HistoricalRate,
    MeanBaseline,
    ParetoModel,
    Recalibrated,
)
from .windows import check_no_overlap, make_origins, snapshot, stack  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PALETTE = {
    "ink": "#1f2933",
    "blue": "#2f6db5",
    "orange": "#d9822b",
    "green": "#3a8f5c",
    "grey": "#8a94a0",
    "red": "#b8433a",
}
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["grey"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "font.size": 10,
    }
)

# simplest first: used by the one-standard-error rule
COMPLEXITY = [
    "population mean",
    "historical rate",
    "BG/NBD + Gamma-Gamma",
    "Pareto/NBD + Gamma-Gamma",
    "LightGBM + last-quarter rescale",
    "LightGBM",
]

BOOST_GRID = {
    "shallow": {"num_leaves": 7, "n_estimators": 250, "min_child_samples": 150},
    "medium": {"num_leaves": 15, "n_estimators": 300, "min_child_samples": 80},
    "deep": {"num_leaves": 31, "n_estimators": 400, "min_child_samples": 40},
}


def fit_boost_with_rescale(snaps: dict, train_names: list[str], rescale_on: str, params: dict):
    """Train on ``train_names``; rescale using ``rescale_on``, a finished quarter.

    The ratio model is trained on origins strictly before ``rescale_on`` so its
    forecast for that quarter is out of sample, exactly as it would have been live.
    """
    full = BoostedModel(params).fit(pd.concat([snaps[n] for n in train_names]))
    earlier = [n for n in train_names if n != rescale_on]
    probe = BoostedModel(params).fit(pd.concat([snaps[n] for n in earlier]))
    ratio_model = Recalibrated(full).set_ratios(snaps[rescale_on], probe.predict(snaps[rescale_on]))
    return full, ratio_model


def run(
    data_path: Path = DEFAULT_PATH,
    out_dir: Path | None = None,
    quick: bool = False,
    assets_dir: Path | None = None,
) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir or ROOT / "outputs")
    assets = Path(assets_dir or ROOT / "assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    orders = load_orders(data_path)
    daily = daily_purchases(orders)
    origins = make_origins()
    by_name = {o.name: o for o in origins}
    snaps = {o.name: snapshot(daily, o) for o in origins}
    results: dict = {"version": __version__, "data": summarize(orders, daily)}
    results["origins"] = [
        {"name": o.name, "origin": str(o.date.date()), "target_ends": str(o.end.date()), "role": o.role}
        for o in origins
    ]
    train_origins = [o for o in origins if o.role == "train"]
    results["label_windows_clean_for_validation"] = check_no_overlap(train_origins, by_name["o3"])
    results["label_windows_clean_for_test"] = check_no_overlap(train_origins + [by_name["o3"]], by_name["o4"])

    # ---------------- validation quarter (o3) ----------------
    val = snaps["o3"]
    train_names = ["o0", "o1", "o2"]
    preds_val: dict[str, pd.DataFrame] = {}
    mean_m = MeanBaseline().fit(pd.concat([snaps[n] for n in train_names]))
    preds_val[mean_m.name] = mean_m.predict(val)
    hist = HistoricalRate().fit(val)
    preds_val[hist.name] = hist.predict(val)
    bg = BetaGeometricModel().fit(val)
    preds_val[bg.name] = bg.predict(val)
    pareto = ParetoModel().fit(val)
    preds_val[pareto.name] = pareto.predict(val)

    grid_scores = {}
    best_cfg, best_dev = None, np.inf
    grid = {"medium": BOOST_GRID["medium"]} if quick else BOOST_GRID
    for cfg, params in grid.items():
        raw, rescaled = fit_boost_with_rescale(snaps, train_names, "o2", params)
        p_raw, p_res = raw.predict(val), rescaled.predict(val)
        dev = poisson_deviance(val["y_purchases"].to_numpy(), p_res["purchases"].to_numpy())
        grid_scores[cfg] = {
            "poisson_dev_rescaled": dev,
            "poisson_dev_raw": poisson_deviance(val["y_purchases"].to_numpy(), p_raw["purchases"].to_numpy()),
        }
        if dev < best_dev:
            best_cfg, best_dev, _, best_res, best_p_raw, best_p_res = (
                cfg,
                dev,
                raw,
                rescaled,
                p_raw,
                p_res,
            )
    preds_val["LightGBM"] = best_p_raw
    preds_val["LightGBM + last-quarter rescale"] = best_p_res
    results["boost_grid_validation"] = grid_scores
    results["boost_config_chosen"] = best_cfg
    results["boost_rescale_ratios_validation"] = {
        "purchases": best_res.ratio_purchases,
        "spend": best_res.ratio_spend,
    }

    val_scores = {k: score_frame(val, v) for k, v in preds_val.items()}
    results["validation"] = val_scores

    # one-standard-error selection on purchase MAE
    maes = {k: v["purchases_mae"] for k, v in val_scores.items()}
    best_name = min(maes, key=maes.get)
    ses = {}
    for k in preds_val:
        if k == best_name:
            ses[k] = 0.0
            continue
        d = bootstrap_diff(val, preds_val[k], preds_val[best_name], mae_purchases, n_boot=150)
        ses[k] = float((d["hi"] - d["lo"]) / 3.92)
    shipped = next(k for k in COMPLEXITY if k in maes and maes[k] - maes[best_name] <= ses[k] + 1e-12)
    results["selection"] = {
        "rule": "simplest model whose validation purchase MAE is within one bootstrap SE of the best",
        "best_by_mae": best_name,
        "shipped": shipped,
        "mae": maes,
        "se_vs_best": ses,
    }

    # ---------------- test quarter (o4) ----------------
    test = snaps["o4"]
    all_train = ["o0", "o1", "o2", "o3"]
    cfg_params = BOOST_GRID[best_cfg]
    full_raw, full_res = fit_boost_with_rescale(snaps, all_train, "o3", cfg_params)
    bg_t = BetaGeometricModel().fit(test)
    pareto_t = ParetoModel().fit(test)
    hist_t = HistoricalRate().fit(test)
    mean_t = MeanBaseline().fit(pd.concat([snaps[n] for n in all_train]))
    preds_test = {
        mean_t.name: mean_t.predict(test),
        hist_t.name: hist_t.predict(test),
        bg_t.name: bg_t.predict(test),
        pareto_t.name: pareto_t.predict(test),
        "LightGBM": full_raw.predict(test),
        "LightGBM + last-quarter rescale": full_res.predict(test),
    }
    test_scores = {k: score_frame(test, v) for k, v in preds_test.items()}
    results["test"] = test_scores
    results["test_boost_rescale_ratios"] = {
        "purchases": full_res.ratio_purchases,
        "spend": full_res.ratio_spend,
    }
    ship_pred = preds_test[shipped]
    ci = {}
    for other in preds_test:
        if other == shipped:
            continue
        ci[other] = {
            "purchases_mae_diff_vs_shipped": bootstrap_diff(
                test, preds_test[other], ship_pred, mae_purchases, n_boot=200
            ),
            "spend_mae_diff_vs_shipped": bootstrap_diff(
                test, preds_test[other], ship_pred, mae_spend, n_boot=200
            ),
            "top10_share_diff_vs_shipped": bootstrap_diff(
                test, preds_test[other], ship_pred, top10_spend, n_boot=200
            ),
        }
    results["test_paired_bootstrap"] = ci
    results["shipped_model"] = shipped

    # ---------------- where does next quarter's money come from ----------------
    ot = test["frequency"] == 0
    results["one_time_buyers_at_test_origin"] = {
        "share_of_customers": float(ot.mean()),
        "share_of_next_quarter_spend": float(test.loc[ot, "y_spend"].sum() / test["y_spend"].sum()),
        "share_who_bought_again": float((test.loc[ot, "y_purchases"] > 0).mean()),
        "repeat_buyers_bought_again": float((test.loc[~ot, "y_purchases"] > 0).mean()),
    }
    prob_models = {bg_t.name: bg_t, pareto_t.name: pareto_t}
    shipped_prob = prob_models.get(shipped, pareto_t)
    alive = shipped_prob.p_alive(test)
    rep_mask = test["frequency"] > 0
    bands = pd.cut(alive[rep_mask], [0, 0.2, 0.4, 0.6, 0.8, 1.0001], include_lowest=True)
    seg = (
        pd.DataFrame(
            {
                "band": bands,
                "bought": test.loc[rep_mask, "y_purchases"] > 0,
                "spend": test.loc[rep_mask, "y_spend"],
            }
        )
        .groupby("band", observed=True)
        .agg(customers=("bought", "size"), share_bought=("bought", "mean"), mean_spend=("spend", "mean"))
        .reset_index()
    )
    seg["band"] = seg["band"].astype(str)
    results["p_alive_bands_test"] = seg.to_dict(orient="records")
    results["bgnbd_params_test"] = bg_t.params()
    results["pareto_params_test"] = pareto_t.params()
    results["bgnbd_params_validation"] = bg.params()

    # ---------------- evaluation traps ----------------
    pooled = stack(daily, [by_name[n] for n in train_names + ["o3"]])
    kf = KFold(5, shuffle=True, random_state=3)
    cv_dev = []
    cfg = {**cfg_params, "n_estimators": 150}
    for tr_idx, te_idx in kf.split(pooled):
        m = BoostedModel(cfg).fit(pooled.iloc[tr_idx])
        p = m.predict(pooled.iloc[te_idx])
        cv_dev.append(
            poisson_deviance(pooled.iloc[te_idx]["y_purchases"].to_numpy(), p["purchases"].to_numpy())
        )
    fwd = BoostedModel(cfg).fit(pd.concat([snaps[n] for n in train_names]))
    fwd_dev = poisson_deviance(val["y_purchases"].to_numpy(), fwd.predict(val)["purchases"].to_numpy())
    results["shuffled_cv_vs_forward"] = {
        "shuffled_cv_poisson_dev": float(np.mean(cv_dev)),
        "forward_poisson_dev": float(fwd_dev),
        "note": "shuffled rows scatter one customer across train and test folds and mix quarters",
    }
    results["cohort_decay"] = {
        n: {
            "mean_purchases": float(snaps[n]["y_purchases"].mean()),
            "mean_spend": float(snaps[n]["y_spend"].mean()),
            "mean_T_weeks": float(snaps[n]["T"].mean()),
        }
        for n in snaps
    }
    results["lightgbm_T_range"] = {
        "train_max_T": float(max(snaps[n]["T"].max() for n in train_names)),
        "validation_max_T": float(val["T"].max()),
    }

    # ---------------- tracking: cumulative purchases week by week ----------------
    weeks = np.arange(1, 27)
    bg_o3 = bg  # fitted on history to o3
    track_bg = [
        float(
            np.sum(
                bg_o3.bgf_.conditional_expected_number_of_purchases_up_to_time(
                    w, val["frequency"], val["recency"], val["T"]
                )
            )
        )
        for w in weeks
    ]
    rate = val["frequency"] / val["T"].clip(lower=1.0)
    track_hist = [float(rate.sum() * w) for w in weeks]
    d2 = daily[
        (daily["date"] > by_name["o3"].date) & (daily["date"] <= by_name["o3"].date + pd.Timedelta(days=182))
    ]
    wk = ((d2["date"] - by_name["o3"].date).dt.days - 1) // 7 + 1
    actual = d2.groupby(wk).size().reindex(weeks).fillna(0).cumsum().to_numpy()
    results["tracking_26w"] = {
        "actual_final": float(actual[-1]),
        "bgnbd_final": track_bg[-1],
        "historical_final": track_hist[-1],
        "bgnbd_error_pct": float(100 * (track_bg[-1] / actual[-1] - 1)),
        "historical_error_pct": float(100 * (track_hist[-1] / actual[-1] - 1)),
    }

    # ---------------- economics on the test quarter ----------------
    a = CampaignAssumptions()
    y = test["y_spend"].to_numpy()
    rng = np.random.default_rng(5)
    rankings = {
        "random": rng.random(len(test)),
        "historical rate": preds_test["historical rate"]["spend"].to_numpy(),
        shipped: preds_test[shipped]["spend"].to_numpy(),
        "LightGBM + last-quarter rescale": preds_test["LightGBM + last-quarter rescale"]["spend"].to_numpy(),
        "hindsight": y,
    }
    curves = {k: profit_curve(y, v, a) for k, v in rankings.items()}
    results["economics"] = {
        "assumptions": a.as_dict(),
        "break_even_quarterly_spend": a.break_even_spend(),
        "mail_everyone_net_per_1000": float(curves["random"]["net_per_1000"].iloc[-1]),
        "best_point": {k: best_point(c) for k, c in curves.items()},
        "sensitivity": lift_sensitivity(y, rankings).to_dict(orient="records"),
        "customers_in_base": int(len(test)),
    }

    # ---------------- artifacts ----------------
    (out_dir / "model_params.json").write_text(
        json.dumps(
            {
                "shipped": shipped,
                "horizon_weeks": WEEKS,
                "families": {
                    m.name: {
                        "kind": "pareto" if isinstance(m, ParetoModel) else "bgnbd",
                        "purchase_params": {k: float(v) for k, v in m.bgf_.params_.items()},
                        "value_params": {k: float(v) for k, v in m.ggf_.params_.items()},
                        "population_value": m.population_value_,
                    }
                    for m in (bg_t, pareto_t)
                },
                "break_even_quarterly_spend": a.break_even_spend(),
            },
            indent=2,
        )
    )
    snaps["o4"].reset_index(names="customer_id").assign(
        p_alive=alive.to_numpy(),
        pred_purchases=preds_test[shipped]["purchases"].to_numpy(),
        pred_spend=preds_test[shipped]["spend"].to_numpy(),
    ).sort_values("pred_spend", ascending=False).to_csv(out_dir / "scored_customers.csv", index=False)

    make_charts(
        assets,
        val,
        test,
        preds_val,
        preds_test,
        shipped,
        curves,
        weeks,
        actual,
        track_bg,
        track_hist,
        results,
        full_raw,
    )
    results["runtime_seconds"] = round(time.time() - t0, 1)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    return results


def make_charts(
    assets,
    val,
    test,
    preds_val,
    preds_test,
    shipped,
    curves,
    weeks,
    actual,
    track_bg,
    track_hist,
    results,
    boost,
):
    c = PALETTE
    # 1 cohort tracking
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(weeks, actual, color=c["ink"], lw=2.2, label="actual")
    ax.plot(weeks, track_bg, color=c["blue"], lw=2, label="BG/NBD")
    ax.plot(weeks, track_hist, color=c["orange"], lw=2, ls="--", label="historical rate")
    ax.set_xlabel("weeks after the forecast date")
    ax.set_ylabel("cumulative purchases, all customers")
    ax.set_title("Forecast made once, tracked for 26 weeks", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(assets / "tracking.png")
    plt.close(fig)

    # 2 frequency calibration (test)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name in zip(axs, ["historical rate", shipped], strict=True):
        t = frequency_calibration(test, preds_test[name])
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["actual"], 0.4, color=c["ink"], label="actual")
        ax.bar(
            x + 0.2,
            t["pred"],
            0.4,
            color=c["orange"] if name == "historical rate" else c["blue"],
            label="forecast",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(f)}+" if f == t["freq"].max() else str(int(f)) for f in t["freq"]])
        ax.set_xlabel("repeat purchases before the forecast date")
        ax.set_title(name, loc="left")
    axs[0].set_ylabel("purchases in the next 13 weeks, per customer")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(assets / "frequency_calibration.png")
    plt.close(fig)

    # 3 spend deciles for the shipped model
    d = decile_table(test, preds_test[shipped], "spend")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = d["decile"].astype(int)
    ax.bar(x - 0.2, d["actual"], 0.4, color=c["ink"], label="actual")
    ax.bar(x + 0.2, d["pred"], 0.4, color=c["blue"], label="forecast")
    ax.set_xlabel("customer decile by forecast spend (1 = highest)")
    ax.set_ylabel("mean spend in next 13 weeks ($)")
    ax.set_title(f"Test quarter, {shipped}", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(assets / "spend_deciles.png")
    plt.close(fig)

    # 4 profit curves
    fig, ax = plt.subplots(figsize=(7, 4))
    style = {
        "random": (c["grey"], "-"),
        "historical rate": (c["orange"], "-"),
        shipped: (c["blue"], "-"),
        "LightGBM + last-quarter rescale": (c["green"], "-"),
        "hindsight": (c["ink"], ":"),
    }
    for k, cur in curves.items():
        ax.plot(cur["fraction"] * 100, cur["net_per_1000"], color=style[k][0], ls=style[k][1], lw=2, label=k)
    ax.axhline(0, color=c["grey"], lw=0.8)
    ax.set_xlabel("share of customers mailed (%)")
    ax.set_ylabel("net profit per 1,000 customers ($)")
    ax.set_title("Who to mail, test quarter", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(assets / "profit_curves.png")
    plt.close(fig)

    # 5 model comparison on test, spend MAE and top-decile share
    names = list(preds_test)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    mae = [results["test"][n]["purchases_mae"] for n in names]
    top = [results["test"][n]["top10_share"] * 100 for n in names]
    cols = [c["red"] if n == shipped else c["grey"] for n in names]
    short = [n.replace(" + Gamma-Gamma", "").replace(" + last-quarter rescale", "\n+ rescale") for n in names]
    axs[0].barh(short, mae, color=cols)
    axs[0].invert_yaxis()
    axs[0].set_xlabel("purchase MAE per customer (lower is better)")
    axs[1].barh(short, top, color=cols)
    axs[1].invert_yaxis()
    axs[1].set_yticklabels([])
    axs[1].set_xlabel("share of next-quarter spend held by top 10% (%)")
    axs[1].axvline(results["test"][names[0]]["oracle_top10_share"] * 100, color=c["ink"], ls=":", lw=1)
    fig.suptitle(
        "Test quarter (shipped model in red, dotted line = hindsight)", x=0.01, ha="left", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(assets / "model_comparison.png")
    plt.close(fig)

    # 6 cohort decay and boosted drift
    quarters = ["o0", "o1", "o2", "o3", "o4"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        range(5),
        [results["cohort_decay"][q]["mean_purchases"] for q in quarters],
        color=c["ink"],
        lw=2.2,
        marker="o",
        label="actual purchases per customer",
    )
    ax.plot(
        [3, 4],
        [preds_val["LightGBM"]["purchases"].mean(), preds_test["LightGBM"]["purchases"].mean()],
        color=c["red"],
        lw=2,
        marker="s",
        ls="--",
        label="LightGBM, no rescale",
    )
    ax.plot(
        [3, 4],
        [
            preds_val["LightGBM + last-quarter rescale"]["purchases"].mean(),
            preds_test["LightGBM + last-quarter rescale"]["purchases"].mean(),
        ],
        color=c["green"],
        lw=2,
        marker="s",
        label="LightGBM, rescaled",
    )
    ax.plot(
        [3, 4],
        [preds_val[shipped]["purchases"].mean(), preds_test[shipped]["purchases"].mean()],
        color=c["blue"],
        lw=2,
        marker="s",
        label=shipped.split(" +")[0],
    )
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Apr-Jun 97", "Jul-Sep 97", "Oct-Dec 97", "Jan-Mar 98", "Apr-Jun 98"])
    ax.set_ylabel("purchases per customer per quarter")
    ax.set_title("A single cohort keeps cooling off", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(assets / "cohort_decay.png")
    plt.close(fig)

    # 7 importance
    imp = boost.importance().head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp.index, imp.values, color=c["blue"])
    ax.set_xlabel("share of split gain, spend model")
    ax.set_title("What the boosted model leans on", loc="left")
    fig.tight_layout()
    fig.savefig(assets / "importance.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full secondorder pipeline")
    ap.add_argument("--data", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quick", action="store_true", help="skip the boosting grid")
    args = ap.parse_args()
    res = run(args.data, args.out, args.quick)
    print(
        json.dumps({k: res[k] for k in ("shipped_model", "boost_config_chosen", "runtime_seconds")}, indent=2)
    )


if __name__ == "__main__":
    main()

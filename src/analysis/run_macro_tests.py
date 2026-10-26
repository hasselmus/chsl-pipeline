"""Monte Carlo observational cross-tests for CHSL-v1."""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxTimeVaryingFitter
from src.schema import VERSION, assert_contract
from src.transform.build_vectors import sample_compatible_states

def prepare_risk(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    frame = sample_compatible_states(df[df["n0_id"].notna()].copy(), rng)
    frame = frame.sort_values(["n0_id", "t_obs"])
    frame["over_s_ratio"] = (frame["a_overhead_index"] /
                           frame["s_phys_index"].where(frame["s_phys_index"] > 0))
    frame["vigilance_s_ratio"] = (frame["p_vig_index"] /
                                frame["s_phys_index"].where(frame["s_phys_index"] > 0))
    years = {}
    for name, g in frame.groupby("n0_id"):
        if g["event_observed"].astype(bool).any():
            left, right = int(g["sigma_low"].iloc[0]), int(g["sigma_high"].iloc[0])
            if left > right:
                raise ValueError("CHSL-v1 invalid termination compatible states")
            years[name] = int(rng.integers(left, right + 1))
    frame["sigma_draw"] = frame["n0_id"].map(years)
    frame["event_decade"] = ((frame["sigma_draw"] >= frame["t_obs"]) &
                             (frame["sigma_draw"] < frame["t_obs"] + 10)).fillna(False)
    frame["time_since_start"] = (frame["t_obs"] - frame["n0_start_yr"]) / 100
    frame["start"] = (frame["t_obs"] - frame["n0_start_yr"]).clip(lower=0).astype(float)
    frame["stop"] = frame["start"] + 10
    frame.loc[frame["event_decade"], "stop"] = (
        frame.loc[frame["event_decade"], "sigma_draw"] -
        frame.loc[frame["event_decade"], "n0_start_yr"] + 1
    ).clip(lower=frame.loc[frame["event_decade"], "start"] + 1)
    frame = frame[(frame["sigma_draw"].isna()) | (frame["t_obs"] <= frame["sigma_draw"])]
    frame["urban_contraction"] = np.nan
    for _, g in frame.groupby("n0_id", sort=False):
        last = g["b_urban_max"].shift(1)
        frame.loc[g.index, "urban_contraction"] = np.where(
            g["b_urban_max"].notna() & last.notna(),
            (g["b_urban_max"] < last).astype(float), np.nan
        )
    frame["interaction"] = frame["over_s_ratio"] * frame["u_shock_freq"]
    return frame

def fit_logit(df: pd.DataFrame, response: str, terms: list[str],
              test: str, model: str, draw: int) -> list[dict]:
    cols = [response, "n0_id", "basin_id"] + terms
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    n, events = len(sub), int(sub[response].sum()) if n else 0
    k = len(terms) + 1
    if n < max(20, 5 * k) or events < 3 or (n - events) < 3:
        return [{"draw": draw, "test": test, "model": model, "family": "panel_logit",
                 "term": "", "status": "insufficient_observed_events", "n": n, "events": events}]
    design = sm.add_constant(sub[terms].astype(float), has_constant="add")
    try:
        fit = sm.GLM(sub[response].astype(float), design,
                     family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": sub["n0_id"]}
        )
        return [{
            "draw": draw, "test": test, "model": model, "family": "panel_logit",
            "term": term, "estimate": float(fit.params[term]),
            "odds_ratio": float(np.exp(np.clip(fit.params[term], -700, 700))),
            "p_value": float(fit.pvalues[term]), "aic": float(fit.aic),
            "bic": float(-2 * fit.llf + k * np.log(n)),
            "n": n, "events": events, "status": "ok"
        } for term in terms]
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as error:
        return [{"draw": draw, "test": test, "model": model, "family": "panel_logit",
                 "term": "", "status": type(error).__name__, "n": n, "events": events}]

def fit_cox(df: pd.DataFrame, terms: list[str],
            test: str, model: str, draw: int) -> list[dict]:
    cols = ["n0_id", "start", "stop", "event_decade"] + terms
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    sub = sub[sub["stop"] > sub["start"]]
    n, events = len(sub), int(sub["event_decade"].sum())
    if n < max(25, 6 * len(terms)) or events < max(4, len(terms) + 1):
        return [{"draw": draw, "test": test, "model": model, "family": "cox_tv",
                 "term": "", "status": "insufficient_observed_events", "n": n, "events": events}]
    sub["event_decade"] = sub["event_decade"].astype(int)
    try:
        fitter = CoxTimeVaryingFitter(penalizer=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitter.fit(sub, id_col="n0_id", event_col="event_decade",
                       start_col="start", stop_col="stop")
        return [{
            "draw": draw, "test": test, "model": model, "family": "cox_tv",
            "term": term, "estimate": float(fitter.params_[term]),
            "hazard_ratio": float(np.exp(np.clip(fitter.params_[term], -700, 700))),
            "p_value": float(fitter.summary.loc[term, "p"]),
            "aic": float(fitter.AIC_partial_), "n": n, "events": events, "status": "ok"
        } for term in terms]
    except (ValueError, np.linalg.LinAlgError, RuntimeError, KeyError) as error:
        return [{"draw": draw, "test": test, "model": model, "family": "cox_tv",
                 "term": "", "status": type(error).__name__, "n": n, "events": events}]

def event_study(frame: pd.DataFrame, draw: int) -> list[dict]:
    event = frame[frame["sigma_draw"].notna()].copy()
    event["relative_year"] = event["t_obs"] - event["sigma_draw"]
    event = event[(event["relative_year"] >= -150) & (event["relative_year"] < 0)]
    event = event.dropna(subset=["vigilance_s_ratio"])
    intervals = [-151, -101, -51, -1]
    categories = ["-150:-101", "-100:-51", "-50:-1"]
    event["event_bin"] = pd.cut(event["relative_year"],
                                 bins=intervals, labels=categories)
    event = event.dropna(subset=["event_bin"])
    if (event["n0_id"].nunique() < 3 or event["event_bin"].nunique() < 2 or len(event) < 20):
        return [{"draw": draw, "test": "T3", "model": "pre_termination", "family": "event_study",
                 "term": "", "status": "insufficient_event_window_coverage", "n": len(event)}]
    try:
        model = smf.ols(
            'vigilance_s_ratio ~ C(event_bin, Treatment(reference="-150:-101")) + C(n0_id)',
            data=event
        ).fit(cov_type="cluster", cov_kwds={"groups": event["n0_id"]})
        return [{"draw": draw, "test": "T3", "model": "pre_termination",
                 "family": "event_study", "term": name,
                 "estimate": float(value), "p_value": float(model.pvalues[name]),
                 "n": len(event), "status": "ok"}
                for name, value in model.params.items() if "event_bin" in name]
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as error:
        return [{"draw": draw, "test": "T3", "model": "pre_termination", "family": "event_study",
                 "term": "", "status": type(error).__name__, "n": len(event)}]

def decoupling_audit(frame: pd.DataFrame, full_data: pd.DataFrame, draw: int) -> list[dict]:
    rows = []
    for name, g in frame[frame["sigma_draw"].notna()].groupby("n0_id"):
        sigma = int(g["sigma_draw"].iloc[0])
        basin = str(g["basin_id"].iloc[0])
        before = g[g["t_obs"] < sigma].sort_values("t_obs")
        regional = full_data[full_data["basin_id"] == basin].drop_duplicates("t_obs")
        regional = regional.sort_values("t_obs")
        pre_regional = regional[regional["t_obs"] < sigma]
        after = regional[regional["t_obs"] >= sigma]
        row = {"draw": draw, "n0_id": name, "basin_id": str(g["basin_id"].iloc[0]),
               "sigma_draw": sigma,
               "admin_last_pre": float(before["b_admin_pop"].dropna().iloc[-1])
                                 if before["b_admin_pop"].notna().any() else np.nan,
               "regional_last_pre": float(pre_regional["b_regional_hyde_pop"].dropna().iloc[-1])
                                 if pre_regional["b_regional_hyde_pop"].notna().any() else np.nan,
               "regional_first_post": float(after["b_regional_hyde_pop"].dropna().iloc[0])
                                 if after["b_regional_hyde_pop"].notna().any() else np.nan}
        if (np.isfinite(row["regional_last_pre"]) and row["regional_last_pre"] > 0 and
            np.isfinite(row["regional_first_post"])):
            row["regional_retention_ratio"] = row["regional_first_post"] / row["regional_last_pre"]
        else:
            row["regional_retention_ratio"] = np.nan
        row["status"] = ("paired_regional_observations" if
                         np.isfinite(row["regional_retention_ratio"]) else "post_transition_unobserved")
        rows.append(row)
    return rows

def draw_figure(frame: pd.DataFrame, out: Path):
    sample = frame.dropna(subset=["s_phys_index", "a_overhead_index", "u_shock_freq"])
    if sample.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    plotted = ax.scatter(sample["s_phys_index"], sample["a_overhead_index"],
                         c=sample["u_shock_freq"], s=20, cmap="viridis")
    ax.set_xlabel("CHSL-v1 physical provisioning indicator")
    ax.set_ylabel("CHSL-v1 administrative indicator")
    fig.colorbar(plotted, ax=ax, label="Multi-year shock years per 50 years")
    fig.tight_layout()
    fig.savefig(out / "CHSL-v1_phase_plane.png", dpi=160)
    plt.close(fig)

def test_draw(frame: pd.DataFrame, draw: int) -> list[dict]:
    result = []
    t1 = ["over_s_ratio", "u_shock_freq", "interaction"]
    result += fit_logit(frame, "event_decade", t1, "T1", "overhead_shock", draw)
    result += fit_cox(frame, t1, "T1", "overhead_shock", draw)
    for label, terms in [
        ("drift", ["u_mean_50yr"]),
        ("shocks", ["u_shock_freq"]),
        ("joint", ["u_mean_50yr", "u_shock_freq"])
    ]:
        result += fit_logit(frame, "event_decade", terms, "T2", label + "_termination", draw)
        result += fit_logit(frame, "urban_contraction", terms, "T2", label + "_urban", draw)
        result += fit_cox(frame, terms, "T2", label + "_termination", draw)
    result += event_study(frame, draw)
    return result

def main(argv=None):
    p = argparse.ArgumentParser(description="CHSL-v1 Monte Carlo macro tests")
    p.add_argument("--input", type=Path, default=Path("data/processed/chsl_v1.parquet"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--draws", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(argv)
    if args.draws < 1:
        p.error("--draws must be positive")
    data = pd.read_parquet(args.input)
    assert_contract(data)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fits, audits = [], []
    for draw in range(args.draws):
        frame = prepare_risk(data, rng)
        fits.extend(test_draw(frame, draw))
        audits.extend(decoupling_audit(frame, data, draw))
        if draw == 0:
            draw_figure(frame, args.out)
    raw = pd.DataFrame(fits)
    raw.to_csv(args.out / "CHSL-v1_draw_models.csv", index=False)
    if audits:
        pd.DataFrame(audits).to_csv(args.out / "CHSL-v1_demographic_audit.csv", index=False)
    complete = raw[raw["status"] == "ok"]
    if len(complete):
        summary = complete.groupby(["test", "model", "family", "term"], dropna=False).agg(
            draws_fitted=("draw", "nunique"), estimate_median=("estimate", "median"),
            estimate_low=("estimate", lambda x: x.quantile(0.025)),
            estimate_high=("estimate", lambda x: x.quantile(0.975)),
            p_median=("p_value", "median"), aic_median=("aic", "median")
        ).reset_index()
    else:
        summary = pd.DataFrame(columns=["test","model","family","term","draws_fitted",
                                        "estimate_median","estimate_low","estimate_high",
                                        "p_median","aic_median"])
    summary.to_csv(args.out / "CHSL-v1_model_summary.csv", index=False)
    (args.out / "CHSL-v1_model_summary.md").write_text(
        "# CHSL-v1 macro-statistical cross-tests\n\n"
        "Compatible-state Monte Carlo interval summaries; no missing data are assigned midpoints.\n\n"
        + summary.to_markdown(index=False) +
        "\n\nInterpretation: coefficients are observational; p-value medians are not "
        "pooled hypothesis-test probabilities. Event counts and model feasibility "
        "are recorded in the draw-level CSV. Full buffer coverage is not observed.\n"
    )
    print(f"{VERSION} completed {args.draws} compatible-state draws; "
          f"{len(complete)} model terms successfully fitted")

if __name__ == "__main__":
    main()

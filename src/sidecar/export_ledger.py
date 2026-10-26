"""CHSL-v1 Chapter 14 Observation Ledger; only whitelisted state fields leave."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.schema import (
    ADMINISTRATIVE, EXOGENOUS, POPULATION, REGIONAL, STRUCTURAL,
    VIGILANCE, VERSION, assert_contract, quote, safe_outbound_text
)

def value(x: object):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if x is None:
        return None
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x

def coordinates(df: pd.DataFrame, target: str) -> str:
    ids = df[["basin_id", "lat", "lon"]].drop_duplicates("basin_id")
    if target in ids["basin_id"].values:
        return target
    try:
        lat, lon = [float(x.strip()) for x in target.split(",")]
    except (ValueError, AttributeError) as error:
        raise ValueError("CHSL-v1 basin must be an internal identifier or lat,lon") from error
    radius = np.sqrt((ids["lat"].astype(float) - lat)**2 +
                     ((ids["lon"].astype(float) - lon)**2)
    i = radius.idxmin()
    if float(radius.loc[i]) > 1:
        raise ValueError("CHSL-v1 no basin centroid lies within one degree")
    return str(ids.loc[i, "basin_id"])

def observations(data: pd.DataFrame, basin: str, start: int, end: int) -> dict:
    sub = data[data["basin_id"] == basin].copy()
    sub = sub[((sub["t_end"].astype(float) >= start) &
               (sub["t_start"].astype(float) <= end)) |
              (sub["t_obs"].between(start, end))]
    if sub.empty:
        raise ValueError("CHSL-v1 no observations in selected temporal boundary")
    lat = float(sub["lat"].iloc[0])
    lon = float(sub["lon"].iloc[0])
    cfg = []
    for name, group in sub[sub["n0_id"].notna()].groupby("n0_id"):
        first = group.iloc[0]
        cfg.append({
            "n0_id": name, "n0_start_yr": value(first["n0_start_yr"]),
            "n0_start_estimated_flag": bool(first["n0_start_estimated_flag"]),
            "sigma_window": [value(first["sigma_low"]), value(first["sigma_high"])],
            "termination_observed": bool(first["event_observed"])
        })
    climate = []
    for _, row in sub.drop_duplicates(["t_obs"]).sort_values("t_obs").iterrows():
        if not any(pd.notna(row.get(k)) for k in EXOGENOUS):
            continue
        climate.append({"t_obs": int(row["t_obs"]), **{
            k: value(row.get(k)) for k in EXOGENOUS}})
    inventory = []
    domains = {
        "s(t)": STRUCTURAL, "a(t)": ADMINISTRATIVE, "P_vig": VIGILANCE,
        "b_admin(t)": POPULATION
    }
    for _, row in sub[sub["n0_id"].notna()].sort_values(["n0_id", "t_obs"]).iterrows():
        for domain, fields in domains.items():
            for field in fields:
                low, high = row.get(field + "_low"), row.get(field + "_high")
                if pd.isna(low) and pd.isna(high):
                    continue
                inventory.append({
                    "n0_id": str(row["n0_id"]), "domain": domain,
                    "variable": field, "t_start": value(row["t_start"]),
                    "t_end": value(row["t_end"]),
                    "compatible_low": value(low), "compatible_high": value(high),
                    "disputed_flag": bool(row.get(field + "_disputed_flag", False)),
                    "source_refs": str(row.get("source_refs", ""))
                })
    audit = []
    for _, row in sub.sort_values(["t_obs", "n0_id"]).iterrows():
        if not any(pd.notna(row.get(k)) for k in
                   ("b_admin_pop_low", "b_urban_max_low", "b_regional_hyde_pop")):
            continue
        audit.append({
            "t_obs": value(row["t_obs"]),
            "n0_id": str(row["n0_id"]) if pd.notna(row["n0_id"]) else None,
            "b_admin_pop_min": value(row.get("b_admin_pop_low")),
            "b_admin_pop_max": value(row.get("b_admin_pop_high")),
            "b_urban_max_min": value(row.get("b_urban_max_low")),
            "b_urban_max_max": value(row.get("b_urban_max_high")),
            "b_regional_hyde_pop": value(row.get("b_regional_hyde_pop")),
            "m_cropland_km2": value(row.get("m_cropland_km2")),
            "m_grazing_km2": value(row.get("m_grazing_km2"))
        })
    return {
        "ledger": VERSION, "chapter": 14,
        "boundary": {"basin_id": basin, "lat": lat, "lon": lon,
                     "t_start": start, "t_end": end},
        "configurations": cfg, "forcing": climate,
        "inventory": inventory, "decoupling_audit": audit,
        "caveats": [
            "Institutional termination and regional demographic persistence are distinct observations.",
            "Administrative populations and geographically bounded populations have different boundaries.",
            "Absence of a post-termination administrative record is not a population count of zero.",
            "Physical features do not directly measure stored exergy or buffer coverage.",
            "Temporal gaps and missing compatible-state bounds are not imputed."
        ]
    }

def table(rows: list[dict], keys: list[str]) -> str:
    if not rows:
        return "_Not observed within the declared boundary._\n"
    header = "| " + " | ".join(keys) + " |\n"
    rule = "| " + " | ".join("---" for _ in keys) + " |\n"
    contents = ""
    for row in rows:
        contents += "| " + " | ".join(
            quote(row.get(key)).replace("|", "/").replace("\n", " ")
            for key in keys
        ) + " |\n"
    return header + rule + contents

def to_markdown(ledger: dict) -> str:
    b = ledger["boundary"]
    text = (f"# {VERSION} — Chapter 14 Observation Ledger\n\n"
            f"## Boundary and temporal declaration\n\n"
            f"Basin: `{b['basin_id']}`; centroid: "
            f"({b['lat']:.4f}, {b['lon']:.4f}); "
            f"observation window: [{b['t_start']}, {b['t_end']}].\n\n")
    text += table(ledger["configurations"],
                  ["n0_id", "n0_start_yr", "n0_start_estimated_flag",
                   "sigma_window", "termination_observed"])
    text += "\n## Exogenous forcing u(t)\n\n"
    text += ("Trailing 50-year means and counts cover only observed annual values. "
             "Recurrence is a proxy, not a buffer measurement.\n\n")
    text += table(ledger["forcing"], ["t_obs", "u_temp_anomaly", "u_drought_index",
                                      "u_mean_50yr", "u_shock_freq", "u_coverage_years",
                                      "u_shock_events_50yr", "u_t_recur_proxy"])
    text += "\n## Physical and institutional inventory s(t), a(t), P_vig\n\n"
    text += table(ledger["inventory"], ["n0_id","domain","variable","t_start","t_end",
                                        "compatible_low","compatible_high",
                                        "disputed_flag","source_refs"])
    text += "\n## Institutional-to-regional decoupling audit\n\n"
    text += table(ledger["decoupling_audit"],
                  ["t_obs","n0_id","b_admin_pop_min","b_admin_pop_max",
                   "b_urban_max_min","b_urban_max_max","b_regional_hyde_pop",
                   "m_cropland_km2","m_grazing_km2"])
    text += "\n## Observation limits\n\n"
    for item in ledger["caveats"]:
        text += "- " + item + "\n"
    return text

def export(data: pd.DataFrame, basin: str, start: int, end: int, out: Path):
    assert_contract(data)
    if start > end:
        raise ValueError("CHSL-v1 reversed temporal boundary")
    target = coordinates(data, basin)
    record = observations(data, target, start, end)
    if out.suffix.lower() == ".json":
        content = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    elif out.suffix.lower() == ".md":
        content = to_markdown(record)
    else:
        raise ValueError("CHSL-v1 output must be .md or .json")
    safe_outbound_text(content)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out

def main(argv=None):
    p = argparse.ArgumentParser(description="CHSL-v1 Chapter 14 ledger")
    p.add_argument("--input", type=Path, default=Path("data/processed/chsl_v1.parquet"))
    p.add_argument("--basin", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    export(pd.read_parquet(args.input), args.basin, args.start, args.end, args.out)
    print(f"{VERSION} observation ledger exported")

if __name__ == "__main__":
    main()

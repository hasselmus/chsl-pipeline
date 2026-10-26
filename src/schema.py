"""CHSL-v1 data contract. Only state-vector terminology is permitted here."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

VERSION = "CHSL-v1"
STRUCTURAL = (
    "s_irrigation", "s_water_supply", "s_food_storage", "s_roads",
    "s_bridges", "s_ports", "s_mines_foundry"
)
ADMINISTRATIVE = ("a_admin_levels", "a_spec_operators", "a_legal_formal", "a_claims_abstraction")
VIGILANCE = (
    "p_standing_army", "p_mercenary_reliance", "p_fortifications_tier",
    "p_internal_policing", "p_ideological_enforcement"
)
POPULATION = ("b_admin_pop", "b_urban_max")
BOUNDED = STRUCTURAL + ADMINISTRATIVE + VIGILANCE + POPULATION
EXOGENOUS = (
    "u_temp_anomaly", "u_drought_index", "u_mean_50yr", "u_shock_freq",
    "u_coverage_years", "u_shock_events_50yr", "u_t_recur_proxy"
)
REGIONAL = (
    "b_regional_hyde_pop", "m_cropland_km2", "m_grazing_km2"
)
BASE = (
    "chsl_version", "basin_id", "lat", "lon", "n0_id", "observation_scope",
    "n0_start_yr", "n0_start_estimated_flag", "t_obs", "t_start", "t_end",
    "sigma_low", "sigma_high", "event_observed", "source_refs"
)
POP_ALIASES = ("b_admin_pop_min", "b_admin_pop_max", "b_urban_max_min", "b_urban_max_max")
DERIVED = (
    "s_phys_index", "a_overhead_index", "p_vig_index",
    "s_phys_coverage", "a_overhead_coverage", "p_vig_coverage"
)
EXTRA = tuple(
    f"{name}_{suffix}" for name in BOUNDED + DERIVED
    for suffix in ("low", "high", "disputed_flag")
)
ALLOWLIST = frozenset(BASE + EXOGENOUS + REGIONAL + DERIVED + POP_ALIASES + EXTRA + tuple(
    f"{name}_{suffix}" for name in REGIONAL for suffix in ("low", "high", "disputed_flag")
))

def assert_contract(df: pd.DataFrame) -> None:
    unexpected = set(df.columns) - ALLOWLIST
    if unexpected:
        raise ValueError("CHSL-v1 rejected unexpected columns: " + ", ".join(sorted(unexpected)))
    missing = set(BASE) - set(df.columns)
    if missing:
        raise ValueError("CHSL-v1 missing mandatory columns: " + ", ".join(sorted(missing)))
    if not (df["chsl_version"].astype(str) == VERSION).all():
        raise ValueError("CHSL-v1 version mismatch")
    if not df["basin_id"].astype(str).str.fullmatch(r"BASIN_\d{2,}").all():
        raise ValueError("CHSL-v1 invalid geographic key")
    if not df["observation_scope"].isin(["institutional", "regional"]).all():
        raise ValueError("CHSL-v1 invalid observation scope")
    inst = df["observation_scope"].eq("institutional")
    if not df.loc[inst, "n0_id"].astype(str).str.fullmatch(r"N0_BASIN_\d{2,}_[+-]?\d+").all():
        raise ValueError("CHSL-v1 invalid configuration key")
    if df.loc[~inst, "n0_id"].notna().any():
        raise ValueError("CHSL-v1 regional-only row has an institutional identity")
    if (pd.to_numeric(df["t_start"]) > pd.to_numeric(df["t_end"])).any():
        raise ValueError("CHSL-v1 reversed observation interval")
    for name in BOUNDED + DERIVED + REGIONAL:
        lo, hi = name + "_low", name + "_high"
        if lo in df and hi in df:
            invalid = (pd.to_numeric(df[lo], errors="coerce") >
                       pd.to_numeric(df[hi], errors="coerce"))
            if invalid.any():
                raise ValueError(f"CHSL-v1 inconsistent compatible states: {name}")

def quote(value: object) -> str:
    if isinstance(value, (list, tuple, dict)):
        return str(value)
    if pd.isna(value):
        return "not observed"
    if isinstance(value, (float, np.floating)):
        return f"{value:.5g}"
    return str(value)

def bounded(df: pd.DataFrame, name: str, rng: np.random.Generator) -> pd.Series:
    low = pd.to_numeric(df[name + "_low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df[name + "_high"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(low) & np.isfinite(high)
    draws = np.full(len(low), np.nan)
    draws[mask] = rng.uniform(low[mask], high[mask])
    return pd.Series(draws, index=df.index)

def compatible_mean(row: pd.Series, names: tuple[str, ...]) -> tuple[float, float, float, bool]:
    lo, hi = [], []
    disputed = False
    for name in names:
        a, b = row.get(name + "_low"), row.get(name + "_high")
        if pd.notna(a) and pd.notna(b):
            lo.append(float(a))
            hi.append(float(b))
            disputed = disputed or bool(row.get(name + "_disputed_flag", False))
    coverage = len(lo) / len(names)
    if not lo:
        return np.nan, np.nan, coverage, disputed
    return float(np.mean(lo)), float(np.mean(hi)), coverage, disputed

def safe_outbound_text(text: str) -> None:
    # Only internal identifiers may appear in exported material.
    tokens = re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", text)
    if any(t.startswith(("RAW_", "EXT_")) for t in tokens):
        raise ValueError("CHSL-v1 rejected an upstream token")

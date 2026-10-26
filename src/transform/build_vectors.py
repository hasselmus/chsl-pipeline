"""Only CHSL-v1-labelled frames cross this boundary."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.schema import (
    ADMINISTRATIVE, BOUNDED, DERIVED, EXOGENOUS, REGIONAL,
    STRUCTURAL, VIGILANCE, VERSION, assert_contract, compatible_mean
)

def build_vectors(df: pd.DataFrame, forcing: pd.DataFrame | None = None,
                  regional: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    df["observation_scope"] = "institutional"
    for name in BOUNDED:
        for part in ("low", "high"):
            key = f"{name}_{part}"
            if key not in df:
                df[key] = np.nan
        flag = name + "_disputed_flag"
        if flag not in df:
            df[flag] = False
        df[flag] = df[flag].fillna(False).astype(bool)
    groups = {
        "s_phys_index": STRUCTURAL,
        "a_overhead_index": ADMINISTRATIVE,
        "p_vig_index": VIGILANCE
    }
    for index, components in groups.items():
        values = df.apply(compatible_mean, axis=1, names=components, result_type="expand")
        df[index + "_low"] = values[0].astype(float)
        df[index + "_high"] = values[1].astype(float)
        df[index.removesuffix("_index") + "_coverage"] = values[2].astype(float)
        df[index + "_disputed_flag"] = values[3].astype(bool)
    if forcing is not None:
        if forcing.duplicated(["basin_id", "t_obs"]).any():
            raise ValueError("CHSL-v1 forcing contains duplicate region-time cells")
        df = df.merge(forcing[["basin_id", "t_obs"] + [x for x in EXOGENOUS if x in forcing]],
                      how="left", on=["basin_id", "t_obs"], validate="many_to_one")
    if regional is not None:
        if regional.duplicated(["basin_id", "t_obs"]).any():
            raise ValueError("CHSL-v1 regional grid contains duplicate region-time cells")
        keys = [x for x in regional if x in REGIONAL or any(
            x == f"{r}_{suffix}" for r in REGIONAL
            for suffix in ("low", "high", "disputed_flag"))]
        df = df.merge(regional[["basin_id", "t_obs"] + keys],
                      how="left", on=["basin_id", "t_obs"], validate="many_to_one")
    for field in EXOGENOUS + REGIONAL:
        if field not in df:
            df[field] = np.nan
    for field in REGIONAL:
        for suffix in ("low", "high", "disputed_flag"):
            k = field + "_" + suffix
            if k not in df:
                df[k] = False if suffix == "disputed_flag" else np.nan
    df["chsl_version"] = VERSION
    # Preserve physical and demographic observations after the mapped configuration ends.
    physical_keys = []
    for source in (forcing, regional):
        if source is not None and not source.empty:
            physical_keys.append(source[["basin_id", "t_obs"]])
    if physical_keys:
        keys = pd.concat(physical_keys, ignore_index=True).drop_duplicates()
        missing = keys.merge(df[["basin_id", "t_obs"]].drop_duplicates(),
                             on=["basin_id", "t_obs"], how="left", indicator=True)
        missing = missing[missing["_merge"] == "left_only"]
        if not missing.empty:
            positions = df[["basin_id", "lat", "lon"]].drop_duplicates("basin_id")
            extra = missing[["basin_id", "t_obs"]].merge(positions, on="basin_id", validate="many_to_one")
            extra["observation_scope"] = "regional"
            extra["n0_id"] = None
            extra["n0_start_yr"] = np.nan
            extra["n0_start_estimated_flag"] = False
            extra["t_start"] = extra["t_obs"]
            extra["t_end"] = extra["t_obs"]
            extra["sigma_low"] = np.nan
            extra["sigma_high"] = np.nan
            extra["event_observed"] = False
            extra["source_refs"] = ""
            extra["chsl_version"] = VERSION
            for source in (forcing, regional):
                if source is not None and not source.empty:
                    extra = extra.merge(source, on=["basin_id", "t_obs"],
                                        how="left", validate="one_to_one")
            df = pd.concat([df, extra], ignore_index=True)
    for field in REGIONAL:
        for suffix in ("low", "high", "disputed_flag"):
            key = field + "_" + suffix
            if key not in df:
                df[key] = np.nan if suffix != "disputed_flag" else False
    df = df.loc[:, list(dict.fromkeys(df.columns))]
    assert_contract(df)
    return df.sort_values(["basin_id", "n0_id", "t_obs", "t_start"]).reset_index(drop=True)

def sample_compatible_states(df: pd.DataFrame, rng: np.random.Generator,
                             coverage_required: float = 0.5) -> pd.DataFrame:
    out = df[["basin_id", "n0_id", "n0_start_yr", "t_obs", "t_start",
              "t_end", "sigma_low", "sigma_high", "event_observed"]].copy()
    for name in BOUNDED:
        low = pd.to_numeric(df[name + "_low"], errors="coerce").to_numpy(float)
        high = pd.to_numeric(df[name + "_high"], errors="coerce").to_numpy(float)
        out[name] = rng.uniform(low, high)
    for index, names in {
        "s_phys_index": STRUCTURAL, "a_overhead_index": ADMINISTRATIVE,
        "p_vig_index": VIGILANCE
    }.items():
        matrix = out[list(names)]
        count = matrix.notna().sum(axis=1)
        out[index] = matrix.mean(axis=1).where(count >= np.ceil(len(names) * coverage_required))
    for name in EXOGENOUS + REGIONAL:
        out[name] = pd.to_numeric(df.get(name), errors="coerce")
    return out

"""Raw workbook adapter. External identifiers never pass to the transformed table."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from src.schema import BOUNDED, VERSION
from src.transform.build_vectors import build_vectors
from src.raw_ingest.regional_grid import geometry_definition

HERE = Path(__file__).parent
MAPPING = json.loads((HERE / "quarantine_map.json").read_text())

def normal(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())

ALIASES = {normal(label): field for field, labels in MAPPING["column_aliases"].items()
           for label in labels}
VAR_ALIASES = {normal(label): key for key, labels in MAPPING["variables"].items()
               for label in labels}
EXCLUDED = [re.compile(x, re.IGNORECASE) for x in MAPPING["exclude_patterns"]]

def read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        book = pd.ExcelFile(path)
        for sheet in book.sheet_names:
            frame = pd.read_excel(book, sheet_name=sheet)
            if len(set(normal(c) for c in frame) & set(ALIASES)) >= 2:
                return frame
        raise ValueError("CHSL-v1 source workbook lacks identifiable observation columns")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("CHSL-v1 unsupported raw table format")

def to_year(value: object) -> int:
    if pd.isna(value):
        raise ValueError("CHSL-v1 observation without a dated interval")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(float(value))
    s = str(value).strip()
    found = re.search(r"-?\d{1,5}", s)
    if not found:
        raise ValueError("CHSL-v1 cannot parse time interval")
    year = int(found.group())
    if re.search(r"\b(BCE|BC)\b", s, re.IGNORECASE):
        year = -abs(year)
    return year

def numeric_bounds(value: object, name: str,
                   low: object = None, high: object = None) -> tuple[float,float,bool]:
    s = "" if pd.isna(value) else str(value).strip().lower()
    disputed = "?" in s or "disput" in s or "uncertain" in s
    if pd.notna(low) or pd.notna(high):
        if pd.isna(low) or pd.isna(high):
            return np.nan, np.nan, True
        a, b = float(low), float(high)
        if a > b:
            raise ValueError("CHSL-v1 source bounds are reversed")
        return a, b, disputed
    if name == "a_claims_abstraction":
        forms = {"none": 0, "no formal medium": 0, "barter": 0, "bullion": 1,
                 "articles": 1, "coin": 2, "coinage": 2, "paper": 3,
                 "paper currency": 3, "credit": 4, "state debt": 4}
        if s in forms:
            return float(forms[s]), float(forms[s]), disputed
        if s in ("yes", "present", "true"):
            return np.nan, np.nan, True
    missing = {"", "unknown", "not known", "uncoded", "na", "nan", "?", "not applicable"}
    if s in missing:
        return np.nan, np.nan, False
    if name != "a_claims_abstraction":
        binary = {"yes": 1, "present": 1, "true": 1, "no": 0,
                  "absent": 0, "false": 0}
        if s in binary:
            return float(binary[s]), float(binary[s]), disputed
    # Brackets are compatible states, not estimated midpoints.
    bracket = re.fullmatch(r"\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]\??", s)
    interval = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*(?:to|–|-)\s*([-+]?\d+(?:\.\d+)?)\??", s)
    match = bracket or interval
    if match:
        a, b = float(match.group(1)), float(match.group(2))
        return min(a, b), max(a, b), disputed
    point = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\??", s)
    if point:
        number = float(point.group(1))
        return number, number, disputed
    return np.nan, np.nan, True

def rename_meta(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={col: ALIASES[normal(col)] for col in frame
                                 if normal(col) in ALIASES})

def source_rows(raw: pd.DataFrame) -> pd.DataFrame:
    raw = rename_meta(raw)
    required = {"raw_region", "n0_label"}
    if not required <= set(raw):
        raise ValueError("CHSL-v1 requires explicit geography and configuration columns")
    if "year" not in raw and "t_start" not in raw:
        raise ValueError("CHSL-v1 requires dated observations")
    rows = []
    for _, item in raw.iterrows():
        start = to_year(item.get("t_start", item.get("year")))
        end = to_year(item.get("t_end", item.get("year", start)))
        if end < start:
            raise ValueError("CHSL-v1 source observation has reversed interval")
        label = str(item["n0_label"])
        region = str(item["raw_region"])
        if not label.strip() or not region.strip():
            raise ValueError("CHSL-v1 missing configuration/geography mapping")
        citation = str(item.get("citation")).strip() if pd.notna(item.get("citation")) else ""
        cit_id = "CIT_" + hashlib.sha256(citation.encode()).hexdigest()[:14] if citation else ""
        if "variable" in raw and "value" in raw:
            pairs = [(item["variable"], item["value"])]
        else:
            pairs = [(col, item[col]) for col in raw.columns if normal(col) in VAR_ALIASES]
        for col, value in pairs:
            if any(p.search(str(col)) for p in EXCLUDED):
                continue
            target = VAR_ALIASES.get(normal(col))
            if not target:
                continue
            low, high, disputed = numeric_bounds(value, target,
                item.get("lower", np.nan), item.get("upper", np.nan))
            if pd.isna(low) and pd.isna(high):
                continue
            rows.append({"raw_region": region, "n0_label": label, "t_start": start,
                         "t_end": end, "target": target, "low": low, "high": high,
                         "disputed": disputed or str(item.get("disputed", "")).strip().lower() in
                         ("true", "yes", "1", "disputed"),
                         "citation_ref": cit_id})
    if not rows:
        raise ValueError("CHSL-v1 found no admissible disaggregated observations; review raw mapping")
    return pd.DataFrame(rows)

def anonymise(rows: pd.DataFrame, basins: pd.DataFrame,
              terminations: pd.DataFrame | None) -> pd.DataFrame:
    expected = {"basin_id", "lat", "lon", "raw_region"}
    if not expected <= set(basins):
        raise ValueError("CHSL-v1 basin geometry requires explicit identifiers and bounding boxes")
    if basins["raw_region"].duplicated().any():
        raise ValueError("CHSL-v1 ambiguous geographic identity mapping")
    if not basins["basin_id"].astype(str).str.fullmatch(r"BASIN_\d{2,}").all():
        raise ValueError("CHSL-v1 invalid basin names")
    basins = basins.copy()
    info = basins.apply(geometry_definition, axis=1)
    for key in ("rj_id", "rj_kind", "rj_geometry_hash", "rj_radius_km", "grid_data_kind"):
        basins[key] = info.map(lambda record: record[1][key])
    if "admin_population_scope" not in basins:
        basins["admin_population_scope"] = "unknown"
    scope = {"rj_exact": "matched", "jurisdiction": "unmatched", "unknown": "unknown"}
    if not basins["admin_population_scope"].isin(scope).all():
        raise ValueError("CHSL-v1 invalid reviewed administrative coverage classification")
    basins["b_admin_boundary_scope"] = basins["admin_population_scope"].map(scope)
    merged = rows.merge(basins, on="raw_region", validate="many_to_one")
    if len(merged) != len(rows):
        raise ValueError("CHSL-v1 raw geography lacks a reviewed basin assignment")
    if terminations is not None:
        keys = {"raw_region", "n0_label", "sigma_low", "sigma_high", "event_observed"}
        if not keys <= set(terminations):
            raise ValueError("CHSL-v1 termination annotation incomplete")
        if terminations.duplicated(["raw_region", "n0_label"]).any():
            raise ValueError("CHSL-v1 duplicate termination annotation")
        merged = merged.merge(terminations[list(keys)],
                              on=["raw_region", "n0_label"], how="left", validate="many_to_one")
    else:
        merged["sigma_low"] = np.nan
        merged["sigma_high"] = np.nan
        merged["event_observed"] = False
    merged["event_observed"] = merged["event_observed"].fillna(False).astype(bool)
    if (merged["event_observed"] & (merged["sigma_low"].isna() | merged["sigma_high"].isna())).any():
        raise ValueError("CHSL-v1 events require a compatible termination interval")
    merged["n0_start_yr"] = merged.groupby(["basin_id", "n0_label"])["t_start"].transform("min").astype(int)
    merged["n0_id"] = ("N0_" + merged["basin_id"] + "_" +
                       merged["n0_start_yr"].astype(str))
    collided = merged[["n0_id", "n0_label"]].drop_duplicates().groupby("n0_id").size()
    if (collided > 1).any():
        raise ValueError("CHSL-v1 configuration key collision; supply distinct reviewed start dates")
    group = ["basin_id", "lat", "lon", "n0_id", "n0_start_yr",
             "t_start", "t_end", "sigma_low", "sigma_high", "event_observed"]
    result = []
    for keys, frame in merged.groupby(group, dropna=False):
        record = dict(zip(group, keys))
        record["n0_start_estimated_flag"] = True
        record["t_obs"] = (int(record["t_start"]) // 10) * 10
        record["source_refs"] = "|".join(sorted(set(frame["citation_ref"]) - {""}))
        for name in BOUNDED:
            matching = frame[frame["target"] == name]
            record[name + "_low"] = matching["low"].min() if len(matching) else np.nan
            record[name + "_high"] = matching["high"].max() if len(matching) else np.nan
            record[name + "_disputed_flag"] = (
                bool(matching["disputed"].any() or len(matching) > 1) if len(matching) else False
            )
        result.append(record)
    output = pd.DataFrame(result)
    columns = ["basin_id", "rj_id", "rj_kind", "rj_geometry_hash",
               "rj_radius_km", "grid_data_kind", "b_admin_boundary_scope"]
    output = output.merge(basins[columns].drop_duplicates("basin_id"),
                          on="basin_id", validate="many_to_one")
    if output.duplicated(["n0_id", "t_start", "t_end"]).any():
        raise ValueError("CHSL-v1 repeated configuration interval")
    return output

def main(argv=None):
    p = argparse.ArgumentParser(description="Write the quarantined CHSL-v1 dataset")
    p.add_argument("--institutional", type=Path, required=True)
    p.add_argument("--basins", type=Path, required=True)
    p.add_argument("--terminations", type=Path)
    p.add_argument("--forcing-csv", type=Path)
    p.add_argument("--regional-csv", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/processed/chsl_v1.parquet"))
    args = p.parse_args(argv)
    raw = read_raw(args.institutional)
    rows = source_rows(raw)
    raw_citations = raw.rename(columns={col: ALIASES[normal(col)] for col in raw
                                       if normal(col) in ALIASES})
    if "citation" in raw_citations:
        references = sorted(set(raw_citations["citation"].dropna().astype(str)) - {""})
        crosswalk = {"CIT_" + hashlib.sha256(c.encode()).hexdigest()[:14]: c for c in references}
        archive = Path("data/raw/citation_crosswalk.json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(json.dumps(crosswalk, ensure_ascii=False, indent=2))
    basins = pd.read_csv(args.basins)
    terms = pd.read_csv(args.terminations) if args.terminations else None
    translated = anonymise(rows, basins, terms)
    forcing = pd.read_csv(args.forcing_csv) if args.forcing_csv else None
    regional = pd.read_csv(args.regional_csv) if args.regional_csv else None
    result = build_vectors(translated, forcing, regional)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)
    print(f"{VERSION} wrote {len(result)} observations with {result['basin_id'].nunique()} basins")

if __name__ == "__main__":
    main()

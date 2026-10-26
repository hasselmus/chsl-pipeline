"""Software-contract tests use fabricated records, never historical observations."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.schema import assert_contract, BOUNDED, VERSION
from src.raw_ingest.build_chsl_parquet import anonymise, source_rows
from src.transform.build_vectors import build_vectors
from src.sidecar.export_ledger import export
from src.analysis.run_macro_tests import prepare_risk, test_draw as evaluate_draw


def synthetic_dataset():
    raw = pd.DataFrame([
        {"raw_region": "Geography Alpha", "n0_label": "Configuration Alpha",
         "year": 100, "irrigation": 1, "Administrative levels": 2,
         "Population": 1000, "largest_settlement": 200},
        {"raw_region": "Geography Alpha", "n0_label": "Configuration Alpha",
         "year": 110, "irrigation": 1, "Administrative levels": 3,
         "Population": 800, "largest_settlement": 160},
        {"raw_region": "Geography Beta", "n0_label": "Configuration Beta",
         "year": 100, "irrigation": 0, "Administrative levels": 1,
         "Population": 500, "largest_settlement": 80},
    ])
    basins = pd.DataFrame([
        {"raw_region":"Geography Alpha","basin_id":"BASIN_01",
         "lat":45.0,"lon":12.0,"lat_min":44,"lat_max":46,"lon_min":11,"lon_max":13},
        {"raw_region":"Geography Beta","basin_id":"BASIN_02",
         "lat":30.0,"lon":42.0,"lat_min":29,"lat_max":31,"lon_min":41,"lon_max":43},
    ])
    ends = pd.DataFrame([
        {"raw_region":"Geography Alpha","n0_label":"Configuration Alpha",
         "sigma_low":160,"sigma_high":170,"event_observed":True}
    ])
    forcing = pd.DataFrame([
        {"basin_id":"BASIN_01","t_obs":100,"u_temp_anomaly":0.1,
         "u_drought_index":-0.2,"u_mean_50yr":0.12,"u_shock_freq":2,
         "u_coverage_years":50},
        {"basin_id":"BASIN_01","t_obs":110,"u_temp_anomaly":0.0,
         "u_drought_index":-0.5,"u_mean_50yr":0.1,"u_shock_freq":4,
         "u_coverage_years":50},
    ])
    regional = pd.DataFrame([
        {"basin_id":"BASIN_01","t_obs":100,"b_regional_hyde_pop":2000,
         "m_cropland_km2":50.0,"m_grazing_km2":10.0},
        {"basin_id":"BASIN_01","t_obs":110,"b_regional_hyde_pop":2100,
         "m_cropland_km2":50.0,"m_grazing_km2":10.0},
        {"basin_id":"BASIN_01","t_obs":180,"b_regional_hyde_pop":1950,
         "m_cropland_km2":45.0,"m_grazing_km2":9.0},
    ])
    return build_vectors(anonymise(source_rows(raw), basins, ends), forcing, regional)


def test_anonymisation_and_post_transition_regional_state():
    data = synthetic_dataset()
    assert_contract(data)
    assert set(data["chsl_version"]) == {VERSION}
    assert len(data) == 4
    post = data[(data["basin_id"]=="BASIN_01") & (data["t_obs"]==180)].iloc[0]
    assert post["observation_scope"] == "regional"
    assert pd.isna(post["n0_id"])
    assert post["b_regional_hyde_pop"] == 1950
    assert not set(data.columns).intersection({"raw_region","n0_label","polity_id"})
    assert (data["b_admin_pop_low"].dropna() <= data["b_admin_pop_high"].dropna()).all()


def test_scenario_sidecar_two_formats(tmp_path):
    data = synthetic_dataset()
    for extension in ("md", "json"):
        dest = tmp_path / f"ledger.{extension}"
        export(data, "BASIN_01", 100, 180, dest)
        rendered = dest.read_text()
        assert VERSION in rendered
        assert "b_regional_hyde_pop" in rendered
        assert "Configuration Alpha" not in rendered
        if extension == "json":
            parsed = json.loads(rendered)
            assert len(parsed["configurations"]) == 1
            assert any(r["t_obs"] == 180 and r["n0_id"] is None
                       for r in parsed["decoupling_audit"])


def test_missing_brackets_never_become_midpoints():
    data = synthetic_dataset()
    row = data.iloc[0]
    assert np.isnan(row["a_claims_abstraction_low"])
    assert np.isnan(row["a_claims_abstraction_high"])
    assert not row["a_claims_abstraction_disputed_flag"]


def test_analysis_reports_insufficiency_without_claiming_empirical_significance():
    data = synthetic_dataset()
    frame = prepare_risk(data, np.random.default_rng(11))
    outcome = evaluate_draw(frame, 0)
    assert outcome
    assert any(x["status"] == "insufficient_observed_events" for x in outcome)


def test_boundary_rejects_external_columns():
    data = synthetic_dataset()
    data["extraneous_identifier"] = "do not export"
    with pytest.raises(ValueError, match="unexpected"):
        assert_contract(data)


def test_lexical_quarantine_on_tracked_project_files():
    root = Path(__file__).resolve().parents[1]
    raw_map = json.loads((root / "src/raw_ingest/quarantine_map.json").read_text())
    patterns = [re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])",re.I)
                for term in raw_map["prohibited_export_tokens"]]
    for item in root.rglob("*"):
        if not item.is_file() or item.suffix not in {".py",".md",".json",".toml",".yml",".yaml"}:
            continue
        if "raw_ingest" in item.parts or ".pytest_cache" in item.parts:
            continue
        if ".git" in item.parts:
            continue
        text = item.read_text(errors="ignore")
        leaked = [p.pattern for p in patterns if p.search(text)]
        assert not leaked, f"CHSL-v1 quarantine failed: {item.relative_to(root)}"

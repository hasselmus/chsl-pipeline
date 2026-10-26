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
from src.raw_ingest.regional_grid import geometry_definition, extract_regional as extract_masked
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
        {"basin_id":"BASIN_01","t_obs":100,"b_regional_grid_pop":2000,
         "m_cropland_km2":50.0,"m_grazing_km2":10.0},
        {"basin_id":"BASIN_01","t_obs":110,"b_regional_grid_pop":2100,
         "m_cropland_km2":50.0,"m_grazing_km2":10.0},
        {"basin_id":"BASIN_01","t_obs":180,"b_regional_grid_pop":1950,
         "m_cropland_km2":45.0,"m_grazing_km2":9.0},
    ])
    keys = [geometry_definition(row)[1] for _, row in basins.iterrows()]
    region_meta = pd.DataFrame([{"basin_id": str(basins.iloc[i]["basin_id"]), **m}
                                for i, m in enumerate(keys)])
    regional = regional.merge(region_meta, on="basin_id", validate="many_to_one")
    return build_vectors(anonymise(source_rows(raw), basins, ends), forcing, regional)


def test_anonymisation_and_post_transition_regional_state():
    data = synthetic_dataset()
    assert_contract(data)
    assert set(data["chsl_version"]) == {VERSION}
    assert len(data) == 4
    post = data[(data["basin_id"]=="BASIN_01") & (data["t_obs"]==180)].iloc[0]
    assert post["observation_scope"] == "regional"
    assert pd.isna(post["n0_id"])
    assert post["b_regional_grid_pop"] == 1950
    assert not set(data.columns).intersection({"raw_region","n0_label","polity_id"})
    assert (data["b_admin_pop_low"].dropna() <= data["b_admin_pop_high"].dropna()).all()


def test_scenario_sidecar_two_formats(tmp_path):
    data = synthetic_dataset()
    for extension in ("md", "json"):
        dest = tmp_path / f"ledger.{extension}"
        export(data, "BASIN_01", 100, 180, dest)
        rendered = dest.read_text()
        assert VERSION in rendered
        assert "b_regional_grid_pop" in rendered
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
    # Check version-controlled project files only, not installed dependencies.
    import subprocess
    root = Path(__file__).resolve().parents[1]
    raw_map = json.loads((root / "src/raw_ingest/quarantine_map.json").read_text())
    patterns = [re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", re.I)
                for term in raw_map["prohibited_export_tokens"]]
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--cached", "-z"]
    ).split(b"\0")
    for entry in tracked:
        if not entry:
            continue
        relative = Path(entry.decode("utf-8", errors="surrogateescape"))
        item = root / relative
        if not item.is_file() or item.suffix not in {".py", ".md", ".json", ".toml", ".yml", ".yaml"}:
            continue
        if relative.parts[:2] == ("src", "raw_ingest") or relative.as_posix() == "PROVENANCE_AND_LICENSE.md":
            continue
        text = item.read_text(encoding="utf-8", errors="replace")
        leaked = [p.pattern for p in patterns if p.search(text)]
        assert not leaked, f"CHSL-v1 quarantine failed: {relative}"


def test_historical_bounds_parser():
    from src.raw_ingest.build_chsl_parquet import numeric_bounds
    assert numeric_bounds("[1, 3]", "a_admin_levels")[:2] == (1.0, 3.0)
    assert numeric_bounds("-2--1", "a_admin_levels")[:2] == (-2.0, -1.0)
    assert numeric_bounds("none", "a_claims_abstraction")[:2] == (0.0, 0.0)
    assert np.isnan(numeric_bounds("present", "a_claims_abstraction")[0])


def test_end_to_end_command_line_outputs(tmp_path):
    from src.raw_ingest.build_chsl_parquet import main as build_main
    from src.analysis.run_macro_tests import main as analysis_main
    from src.sidecar.export_ledger import main as ledger_main
    raw = pd.DataFrame([
        {"raw_region":"Basin Source A", "n0_label":"Conf A", "year":100,
         "irrigation":1, "Administrative levels":"[2,3]",
         "Population":1000, "largest_settlement":200},
        {"raw_region":"Basin Source A", "n0_label":"Conf A", "year":110,
         "irrigation":1, "Administrative levels":"[3,4]",
         "Population":900, "largest_settlement":180}
    ])
    geo = pd.DataFrame([{"raw_region":"Basin Source A","basin_id":"BASIN_01",
                         "lat":45,"lon":12,"lat_min":44,"lat_max":46,
                         "lon_min":11,"lon_max":13}])
    annotation = pd.DataFrame([{"raw_region":"Basin Source A", "n0_label":"Conf A",
                                "sigma_low":160,"sigma_high":170,"event_observed":True}])
    forcing = pd.DataFrame([{"basin_id":"BASIN_01","t_obs":100,
                             "u_mean_50yr":0.1,"u_shock_freq":2}])
    demographic = pd.DataFrame([
        {"basin_id":"BASIN_01","t_obs":100,"b_regional_grid_pop":2000},
        {"basin_id":"BASIN_01","t_obs":180,"b_regional_grid_pop":1800}
    ])
    geo_meta = geometry_definition(geo.iloc[0])[1]
    for key, value in geo_meta.items():
        demographic[key] = value
    files = {"i":raw,"g":geo,"t":annotation,"f":forcing,"r":demographic}
    for key, frame in files.items():
        frame.to_csv(tmp_path / f"{key}.csv", index=False)
    parquet = tmp_path / "chsl_v1.parquet"
    build_main(["--institutional",str(tmp_path/"i.csv"),
                "--basins",str(tmp_path/"g.csv"),
                "--terminations",str(tmp_path/"t.csv"),
                "--forcing-csv",str(tmp_path/"f.csv"),
                "--regional-csv",str(tmp_path/"r.csv"),
                "--out",str(parquet)])
    assert parquet.is_file()
    result = pd.read_parquet(parquet)
    assert len(result) == 3
    assert result.iloc[0]["b_admin_pop_min"] == 1000
    destination = tmp_path / "ledger.json"
    ledger_main(["--input",str(parquet),"--basin","BASIN_01",
                 "--start","100","--end","180","--out",str(destination)])
    assert destination.is_file()
    analysis_main(["--input",str(parquet),"--draws","2",
                   "--seed","17","--out",str(tmp_path/"models")])
    assert (tmp_path/"models"/"CHSL-v1_draw_models.csv").is_file()


def test_spatial_aggregation_uses_cells_and_geographic_bounds(tmp_path):
    import xarray as xr
    from src.raw_ingest.physical_grids import extract_regional, compute_forcing
    cube = xr.DataArray(np.ones((2, 2, 2))*200.0,
                        coords={"time":[100,110],"lat":[44.,45.],"lon":[11.,12.]},
                        dims=("time","lat","lon"),name="population")
    path = tmp_path/"grid.nc"
    cube.to_dataset().to_netcdf(path)
    basin = pd.DataFrame([{"basin_id":"BASIN_01","lat_min":44,"lat_max":45,
                           "lon_min":11,"lon_max":12}])
    result = extract_regional(basin,{"population":{"path":str(path),"variable":"population",
                                                  "unit":"persons_per_cell"}})
    assert result["b_regional_grid_pop"].tolist() == [800.0,800.0]
    forcing = compute_forcing(pd.Series(np.linspace(0,1,60), index=np.arange(1,61)),
                              None,"BASIN_01",50,59)
    assert forcing[-1]["u_coverage_years"] == 50


def test_radius_mask_and_same_boundary_across_rasters(tmp_path):
    import xarray as xr
    lat, lon = np.array([0., 1., 2.]), np.array([0., 1., 2.])
    grid = xr.DataArray(np.ones((2, 3, 3))*100., dims=("time", "lat", "lon"),
                        coords={"time":[100, 110], "lat":lat, "lon":lon}, name="pop")
    path = tmp_path/"input.nc"
    grid.to_dataset().to_netcdf(path)
    basin = pd.DataFrame([{"basin_id":"BASIN_01", "lat":1., "lon":1., "radius_km":150.}])
    result = extract_masked(basin, {"population":{"path":str(path),
                              "variable":"pop","unit":"persons_per_cell"}})
    # Five cell centres fall within 150 km, whereas the four corners do not.
    assert result["b_regional_grid_pop"].tolist() == [500., 500.]
    assert set(result["rj_kind"]) == {"centroid_buffer"}
    assert set(result["grid_data_kind"]) == {"gridded_cells"}
    assert len(set(result["rj_geometry_hash"])) == 1
    assert result["b_regional_grid_pop_grid_cells"].tolist() == [5, 5]


def test_polygon_mask_and_fractional_cropland(tmp_path):
    import xarray as xr
    geojson = {"type":"Polygon", "coordinates":[[
        [0.4,0.4],[1.6,0.4],[1.6,1.6],[0.4,1.6],[0.4,0.4]
    ]]}
    location = tmp_path / "basin.geojson"
    location.write_text(json.dumps(geojson))
    lat = np.array([0., 1., 2.])
    lon = np.array([0., 1., 2.])
    crop = xr.DataArray(np.full((1,3,3),0.5),dims=("time","lat","lon"),
                        coords={"time":[100],"lat":lat,"lon":lon},name="cropland")
    path=tmp_path/"crop.nc"
    crop.to_dataset().to_netcdf(path)
    basin=pd.DataFrame([{"basin_id":"BASIN_01","polygon_path":str(location)}])
    result=extract_masked(basin,{"cropland":{"path":str(path),"variable":"cropland",
                                           "unit":"fraction_of_cell"}})
    from src.raw_ingest.regional_grid import cell_area_km2
    expected=0.5*cell_area_km2(lat,lon)[1,1]
    assert result.iloc[0]["m_cropland_km2"] == pytest.approx(expected)
    assert result.iloc[0]["m_cropland_km2_grid_cells"] == 1


def test_dated_asc_grid_cells_and_nodata(tmp_path):
    raster = "\n".join([
        "ncols 2","nrows 2","xllcorner 0","yllcorner 0",
        "cellsize 1","NODATA_value -9999","1 2","3 4"]) + "\n"
    path=tmp_path/"100.asc"
    path.write_text(raster)
    basin=pd.DataFrame([{"basin_id":"BASIN_01","lat_min":0,"lat_max":2,
                         "lon_min":0,"lon_max":2}])
    observed=extract_masked(basin,{"population":{"files":{"100":str(path)},
                                                    "unit":"persons_per_cell"}})
    assert observed["b_regional_grid_pop"].iloc[0] == pytest.approx(10.)
    path.write_text(raster.replace("3 4","-9999 4"))
    incomplete=extract_masked(basin,{"population":{"files":{"100":str(path)},
                                                      "unit":"persons_per_cell"}})
    assert np.isnan(incomplete["b_regional_grid_pop"].iloc[0])


def test_reviewed_admin_scope_and_grid_mismatch():
    data=synthetic_dataset()
    assert (data[data["n0_id"].notna()]["b_admin_boundary_scope"] == "unknown").all()
    bad=data.copy()
    bad["rj_geometry_hash"]="0"*20
    with pytest.raises(ValueError, match="(mismatch|boundary)"):
        build_vectors(
            data[data["n0_id"].notna()].drop(columns=["observation_scope"]),
            pd.DataFrame({"basin_id":["BASIN_01"],"t_obs":[100],
                          "u_shock_freq":[2]}),
            pd.DataFrame({"basin_id":["BASIN_01"],"t_obs":[100],
                          "b_regional_grid_pop":[2000],
                          "rj_id":["R_BASIN_01"],
                          "rj_kind":["box"],
                          "rj_geometry_hash":["0"*20],
                          "grid_data_kind":["gridded_cells"]})
        )


def test_ledger_reports_boundary_mismatch(tmp_path):
    data=synthetic_dataset()
    target=tmp_path/"ledger.json"
    export(data,"BASIN_01",100,180,target)
    output=json.loads(target.read_text())
    assert output["boundary"]["rj_id"] == "R_BASIN_01"
    assert output["configurations"][0]["b_admin_boundary_scope"] == "unknown"
    assert any(x["b_admin_boundary_scope"] == "not_applicable" for x in
               output["decoupling_audit"] if x["t_obs"] == 180)


def test_two_dimensional_dated_netcdf(tmp_path):
    import xarray as xr
    cube = xr.DataArray(np.array([[1., 2.], [3., 4.]]),
                        coords={"lat":[0.5,1.5], "lon":[0.5,1.5]},
                        dims=("lat","lon"), name="headcount")
    source = tmp_path / "year_100.nc"
    cube.to_dataset().to_netcdf(source)
    basin = pd.DataFrame([{"basin_id":"BASIN_01", "lat_min":0.,
                           "lat_max":2., "lon_min":0., "lon_max":2.}])
    extracted = extract_masked(basin, {
        "population": {"files": {"100":str(source)},
                       "variable":"headcount", "unit":"persons_per_cell"}
    })
    assert extracted.iloc[0]["b_regional_grid_pop"] == pytest.approx(10.0)
    assert extracted.iloc[0]["b_regional_grid_pop_grid_cells"] == 4


def test_population_boundary_requires_explicit_review():
    raw = pd.DataFrame([{"raw_region":"Region A","n0_label":"Conf A",
                         "year":100,"Population":1000}])
    geometry = pd.DataFrame([{"raw_region":"Region A","basin_id":"BASIN_01",
                              "lat":12.,"lon":14., "radius_km":200.,
                              "admin_population_scope":"rj_exact"}])
    result = anonymise(source_rows(raw), geometry, None)
    assert set(result["b_admin_boundary_scope"]) == {"matched"}
    geometry.loc[0, "admin_population_scope"] = "jurisdiction"
    result2 = anonymise(source_rows(raw), geometry, None)
    assert set(result2["b_admin_boundary_scope"]) == {"unmatched"}

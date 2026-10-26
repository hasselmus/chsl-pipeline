# CHSL-v1 — Local gridded extraction under WSL

This repository deliberately does **not** ship raw or derived historical data. Every `data/` path is ignored by Git. Full upstream attribution is isolated in the root provenance register, which is excluded from all analysis and scenario exports.

## 1. Install and activate

```bash
cd ~/chsl-pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
```

## 2. Declare a fixed geographical region

Example local `data/raw/basins.csv` (the names here are fabricated):

```csv
basin_id,lat,lon,raw_region,radius_km,admin_population_scope
BASIN_01,38.0,28.0,Reference Area One,200,jurisdiction
BASIN_02,42.0,18.0,Reference Area Two,175,unknown
```

A centre plus `radius_km` defines a geodesic circle. Alternatively provide `lat_min,lat_max,lon_min,lon_max` for a rectangle, or `polygon_path` pointing to a **local** GeoJSON polygon. Use the same basin input file for every regional layer and for institutional ingestion. Boundary digests are checked at the join.

The `admin_population_scope` values are `rj_exact`, `jurisdiction`, and `unknown`. Use `rj_exact` **only** when independent evidence demonstrates the administrative population observation refers to precisely this geographical boundary. A polity-wide estimate cannot be converted into a regional observation merely by applying a mask to a separate population raster.

## 3. Specify local 5-arc-minute gridded releases

Create `data/raw/regional_spec.json` with local paths and explicit units:

```json
{
  "population": {
    "files": {
      "1000": "data/raw/regional/pop_1000.asc",
      "1010": "data/raw/regional/pop_1010.asc"
    },
    "unit": "persons_per_cell"
  },
  "cropland": {
    "files": {
      "1000": "data/raw/regional/crop_1000.nc",
      "1010": "data/raw/regional/crop_1010.nc"
    },
    "variable": "cropland",
    "unit": "fraction_of_cell"
  }
}
```

Filenames above are illustrative. Each listed file must exist and correspond to the declared year. Dated NetCDF rasters must have two dimensions named `lat` and `lon`; ASC files must use a standard six-line ESRI ASCII header. For a three-dimensional time-series NetCDF cube instead, use `"path": "data/raw/regional/pop.nc"`, `"variable": "population"`, and a declared `"unit"`.

Accepted units:

- Population: `persons_per_cell`, `persons_per_km2`.
- Cropland or grazing: `km2_per_cell`, `fraction_of_cell`, `percent_of_cell`.

Fractions are converted using spherical cell area. No macro-regional totals, synthetic rescaling or arbitrary downscaling is accepted. Missing values inside a selected region leave that field unobserved. Cell-centre inclusion is used, not fractional intersection with boundary edges. The geographic boundary remains the same but resolution-specific edge effects must be assessed separately.

## 4. Extract raster totals and attach observations

```bash
python -m src.raw_ingest.physical_grids \
  --basins data/raw/basins.csv \
  --regional-spec data/raw/regional_spec.json \
  --out-regional data/raw/regional_decades.csv

python -m src.raw_ingest.build_chsl_parquet \
  --institutional data/raw/institutional.csv \
  --basins data/raw/basins.csv \
  --terminations data/raw/terminations.csv \
  --regional-csv data/raw/regional_decades.csv \
  --out data/processed/chsl_v1.parquet
```

The geographic boundary is recorded in the processed panel as `rj_id`, `rj_kind` and `rj_geometry_hash`. Administrative-to-regional population comparisons are explicitly qualified by `b_admin_boundary_scope`.

## 5. Run analyses and export a scenario sidecar

```bash
python -m src.analysis.run_macro_tests \
  --input data/processed/chsl_v1.parquet \
  --draws 500 --out results

python -m src.sidecar.export_ledger \
  --input data/processed/chsl_v1.parquet \
  --basin BASIN_01 --start 1000 --end 1100 \
  --out results/CHSL-v1_ledger.md
```

No upstream names, source files or licence text are inserted into either downstream product. The local `data/raw/citation_crosswalk.json` and root provenance register remain separate from the context-window export.

## Interpretation constraint

`s_food_storage` measures coded infrastructure. Its interaction with `u_shock_freq` provides an observational test of whether material-storage facilities modify the response to hydroclimatic volatility. It is **not** a measurement of stored exergy `B(t)`, exergy-flow deficits, actual buffer coverage or refill duration; testing those directly requires independent reserve and replenishment data.

# CHSL-v1 — Coupled Historical State-Space Ledger

Reproducible research pipeline for historical institutional, physical and demographic observations. Two downstream commands are provided:

- `python -m src.analysis.run_macro_tests --input data/processed/chsl_v1.parquet --out results`
- `python -m src.sidecar.export_ledger --basin BASIN_01 --start 1200 --end 1600 --out results/ledger.md`

## Install

Python 3.11+. Run `python -m pip install -e ".[test]"`.

## Private research and provenance

All `data/` is ignored by Git, including raw basins, cached downloads, local rasters and generated Parquet products. Only the root `PROVENANCE_AND_LICENSE.md` registers original authors, releases, licences and ambiguity; statistical outputs and Chapter 14 ledgers never import that file.

## Inputs and ingestion

`python -m src.raw_ingest.fetch_sources --all` downloads/cache-checks the verified public historical workbook and three drought atlases plus the gridded temperature reconstruction. The regional demographic grid requires an explicit licensed release URL with `--hyde-url` or a locally acquired NetCDF dataset. Downloads remain in `data/raw/`, which is ignored by Git.

Raw geographic identities are supplied by `data/raw/basins.csv`. Required columns are `basin_id,lat,lon,raw_region`, plus **one** of (i) `lat_min,lat_max,lon_min,lon_max`, (ii) `radius_km` (e.g. 200), or (iii) `polygon_path` pointing to local GeoJSON. For the administrative population, `admin_population_scope` accepts `rj_exact` (reviewed spatial match), `jurisdiction` (mismatch), or `unknown` (default). An unverified polity-wide population never becomes a geographic-basin count. The resulting internal `R_BASIN_XX` boundary key and geometry hash are stable across grid layers. The upstream-to-state mapping exists only in `src/raw_ingest/quarantine_map.json`. No automatic geographical identity inference is made from names. A reviewed `data/raw/terminations.csv` must specify `raw_region,n0_label,sigma_low,sigma_high,event_observed` to identify termination events; a missing record is right-censored.

Build the institutional panel with `python -m src.raw_ingest.build_chsl_parquet --institutional data/raw/institutional.xlsx --basins data/raw/basins.csv --terminations data/raw/terminations.csv`. Optional NetCDF forcing and **gridded-only** regional demographic inputs are passed as arguments; run `--help` for details. To create `data/raw/regional_decades.csv`, pass `--regional-spec data/raw/regional_spec.json` to `python -m src.raw_ingest.physical_grids`. The local JSON can specify NetCDF `path`, `variable`, and `unit`, or year-indexed ASC `files`, with explicit per-cell units. Population accepts `persons_per_cell` or `persons_per_km2`. Cropland and grazing accept `km2_per_cell`, `fraction_of_cell`, or `percent_of_cell`. All observed cells inside the declared geographic boundary must be present; absent cells produce missing totals, not zero-filled estimates. Unavailable observations remain missing. The builder fails when an input cannot be mapped without invention.

## Empirical limitations

* `s_phys_index` is a composite of recorded physical features, **not** joules of stored exergy, buffer coverage time or maintenance service capacity.
* A ratio of administrative codings to material codings is a descriptive, scale-dependent proxy. Results depend on component coverage and weighting; inspect the disaggregated series.
* `s_food_storage × u_shock_freq` operationalises a candidate interaction between material storage facilities and hydroclimatic variance, not direct measurement of `B(t)` or its refill interval. A drought-recurrence proxy does not test a physical buffer-exhaustion inequality without independent reserve and replenishment measurements.
* An empty institutional record is **not** evidence of institutional termination. Terminations require explicit historical annotation.
* The population raster is masked by the same declared geometric region `R_j` as land-use rasters. Cell-centre inclusion is an approximation at raster resolution, not fractional pixel overlap. Administrative totals are comparable only when a reviewed `rj_exact` alignment is explicitly supplied. Neither a post-transition grid count nor a jurisdictional total alone establishes migration or extinction.
* Cross-tests are associational and observational; model fit and coefficients are reported without asserting causal identification.
* Climate records have different geographic and temporal extents. No unavailable annual values are silently interpolated.
* Underlying institutional source files can carry non-commercial redistribution requirements. Raw releases are not committed.

## Quarantine

Downstream data, figure annotations and ledgers use only CHSL-v1 terminology. The raw translation map, raw cached content, and the isolated root provenance register are the only places where external source names and identifiers are kept. Export validates the allowlist and rejects non-conforming columns. Tests check forbidden lexical material outside the raw adapter and excluded provenance register.

## Tests

`python -m pytest -q`

The committed synthetic fixture and tests are solely for software validation; no historical inference is represented as an empirical result.

# CHSL-v1 — Coupled Historical State-Space Ledger

Reproducible research pipeline for historical institutional, physical and demographic observations. Two downstream commands are provided:

- `python -m src.analysis.run_macro_tests --input data/processed/chsl_v1.parquet --out results`
- `python -m src.sidecar.export_ledger --basin BASIN_01 --start 1200 --end 1600 --out results/ledger.md`

## Install

Python 3.11+. Run `python -m pip install -e ".[test]"`.

## Inputs and ingestion

`python -m src.raw_ingest.fetch_sources --all` downloads/cache-checks the verified public historical workbook and three drought atlases plus the gridded temperature reconstruction. The regional demographic grid requires an explicit licensed release URL with `--hyde-url` or a locally acquired NetCDF dataset. Downloads remain in `data/raw/`, which is ignored by Git.

Geographical identities and bounding boxes are supplied through `data/raw/basins.csv` (columns: `basin_id,lat,lon,lat_min,lat_max,lon_min,lon_max,raw_region`). The upstream-to-state mapping exists only in `src/raw_ingest/quarantine_map.json`. No automatic geographical identity inference is made from names. A reviewed `data/raw/terminations.csv` must specify `raw_region,n0_label,sigma_low,sigma_high,event_observed` to identify termination events; a missing record is right-censored.

Build the institutional panel with `python -m src.raw_ingest.build_chsl_parquet --institutional data/raw/institutional.xlsx --basins data/raw/basins.csv --terminations data/raw/terminations.csv`. Optional NetCDF forcing and regional demographic inputs are passed as arguments; run `--help` for details. Unavailable observations remain missing. The builder fails when an input cannot be mapped without invention.

## Empirical limitations

* `s_phys_index` is a composite of recorded physical features, **not** joules of stored exergy, buffer coverage time or maintenance service capacity.
* A ratio of administrative codings to material codings is a descriptive, scale-dependent proxy. Results depend on component coverage and weighting; inspect the disaggregated series.
* A drought-recurrence proxy does not test a buffer-exhaustion inequality without independent reserve and replenishment measurements.
* An empty institutional record is **not** evidence of institutional termination. Terminations require explicit historical annotation.
* A regional land-use grid is geographically anchored, whereas administrative population covers a historically changing jurisdiction. Their difference is not a directly estimated migration flow, nor proof of demographic persistence without post-transition observations.
* Cross-tests are associational and observational; model fit and coefficients are reported without asserting causal identification.
* Climate records have different geographic and temporal extents. No unavailable annual values are silently interpolated.
* Underlying institutional source files can carry non-commercial redistribution requirements. Raw releases are not committed.

## Quarantine

Downstream data, figure annotations and ledgers use only CHSL-v1 terminology. The raw translation map and raw cached content are the only places where external variable names and original identifiers are kept. Export validates the allowlist and rejects non-conforming columns. Tests check forbidden lexical material outside the raw adapter.

## Tests

`python -m pytest -q`

The committed synthetic fixture and tests are solely for software validation; no historical inference is represented as an empirical result.

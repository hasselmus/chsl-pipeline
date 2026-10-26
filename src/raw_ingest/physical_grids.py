"""Spatial extraction from raw annual gridded reconstructions and land-use archives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

LAT = ("lat", "latitude", "Latitude", "LAT")
LON = ("lon", "longitude", "Longitude", "LON")
TIME = ("time", "year", "Year", "years", "TIME")

def dimension(da: xr.DataArray, options: tuple[str, ...]) -> str:
    for name in options:
        if name in da.dims:
            return name
    raise ValueError("CHSL-v1 grid lacks a required coordinate dimension")

def open_variable(path: Path, requested: str | None = None) -> tuple[xr.Dataset, xr.DataArray, tuple[str,str,str]]:
    ds = xr.open_dataset(path, decode_times=False)
    if requested:
        if requested not in ds:
            raise ValueError("CHSL-v1 requested physical grid variable is unavailable")
        da = ds[requested]
    else:
        candidates = []
        for name, arr in ds.data_vars.items():
            try:
                dimensions = [dimension(arr, opts) for opts in (TIME, LAT, LON)]
                if arr.ndim == 3 and np.issubdtype(arr.dtype, np.number):
                    candidates.append(name)
            except ValueError:
                pass
        if len(candidates) != 1:
            ds.close()
            raise ValueError("CHSL-v1 grid selection ambiguous; specify the variable explicitly")
        da = ds[candidates[0]]
    dims = tuple(dimension(da, opts) for opts in (TIME, LAT, LON))
    extra = [d for d in da.dims if d not in dims]
    if extra:
        ds.close()
        raise ValueError("CHSL-v1 refuses to collapse unreviewed ensemble or extra grid dimensions")
    return ds, da, dims

def years(time: xr.DataArray) -> np.ndarray:
    values = np.asarray(time.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.DatetimeIndex(values).year.to_numpy()
    if values.dtype.kind in "iuf":
        units = str(time.attrs.get("units", "")).lower()
        if "before present" in units or "years bp" in units or units == "bp":
            return np.rint(1950 - values.astype(float)).astype(int)
        if "since" in units:
            base = units.split("since", 1)[1].strip()
            if base[:4].isdigit():
                return np.rint(int(base[:4]) + values.astype(float)).astype(int)
            raise ValueError("CHSL-v1 unparsed time reference")
        return np.rint(values.astype(float)).astype(int)
    try:
        return np.asarray([int(v.year) for v in values])
    except (AttributeError, TypeError) as error:
        raise ValueError("CHSL-v1 unparseable grid dates") from error

def longitude_wrap(longitude: float, values: np.ndarray) -> float:
    if float(np.nanmin(values)) >= 0 and longitude < 0:
        return longitude % 360
    if float(np.nanmax(values)) <= 180 and longitude > 180:
        return ((longitude + 180) % 360) - 180
    return longitude

def nearest_series(da: xr.DataArray, dims: tuple[str,str,str],
                   lat: float, lon: float) -> pd.Series | None:
    td, yd, xd = dims
    ycoords = np.asarray(da[yd], dtype=float)
    xcoords = np.asarray(da[xd], dtype=float)
    lon = longitude_wrap(lon, xcoords)
    if not (np.nanmin(ycoords) <= lat <= np.nanmax(ycoords) and
            np.nanmin(xcoords) <= lon <= np.nanmax(xcoords)):
        return None
    iy = int(np.abs(ycoords - lat).argmin())
    ix = int(np.abs(xcoords - lon).argmin())
    # A masked grid cell has no admissible substitute at another location.
    subset = da.isel({yd: iy, xd: ix})
    arr = np.asarray(subset.values, dtype=float).ravel()
    calendar = years(da[td])
    if len(calendar) != len(arr):
        raise ValueError("CHSL-v1 grid/date size mismatch")
    return pd.Series(arr, index=calendar).replace([np.inf, -np.inf], np.nan).groupby(level=0).mean()

def compute_forcing(temp: pd.Series | None, dry: pd.Series | None,
                    basin_id: str, start: int, end: int) -> list[dict]:
    temp = temp if temp is not None else pd.Series(dtype=float)
    dry = dry if dry is not None else pd.Series(dtype=float)
    temp, dry = temp.astype(float), dry.astype(float)
    output = []
    def mark(series: pd.Series) -> pd.Series:
        if series.dropna().size < 30 or series.std() <= 0:
            return pd.Series(dtype=bool)
        z = (series - series.mean()) / series.std(ddof=1)
        return z < -2
    cold, drought = mark(temp), mark(dry)
    for year in range((start // 10) * 10, end + 1, 10):
        decade = list(range(year, year + 10))
        t = temp.reindex(decade)
        d = dry.reindex(decade)
        # Decade averages do not fill an unobserved year.
        v_temp = float(t.mean()) if t.notna().all() else np.nan
        v_dry = float(d.mean()) if d.notna().all() else np.nan
        window = list(range(year - 49, year + 1))
        temp50 = temp.reindex(window)
        u_mean = float(temp50.mean()) if temp50.notna().all() else np.nan
        c = cold.reindex(window)
        h = drought.reindex(window)
        covered = (temp.reindex(window).notna() | dry.reindex(window).notna())
        count = int(covered.sum())
        shock_years = np.zeros(50, dtype=bool)
        for events in (c, h):
            mask = events.fillna(False).to_numpy(bool)
            for i in range(49):
                if mask[i] and mask[i + 1]:
                    shock_years[i:i + 2] = True
        # Distinct runs, rather than all anomalous single years, give a recurrence proxy.
        event_count = int(sum(shock_years[i] and (i == 0 or not shock_years[i - 1])
                              for i in range(50)))
        valid = bool(covered.all())
        output.append({
            "basin_id": basin_id, "t_obs": year,
            "u_temp_anomaly": v_temp, "u_drought_index": v_dry,
            "u_mean_50yr": u_mean,
            "u_shock_freq": int(shock_years.sum()) if valid else np.nan,
            "u_coverage_years": count,
            "u_shock_events_50yr": event_count if valid else np.nan,
            "u_t_recur_proxy": float(50 / event_count) if valid and event_count else np.nan
        })
    return output

def extract_forcing(basins: pd.DataFrame, temperature: Path | None,
                    drought: list[tuple[Path, str | None]],
                    start: int, end: int, temperature_var: str | None = None) -> pd.DataFrame:
    t_source = open_variable(temperature, temperature_var) if temperature else None
    d_sources = [(open_variable(path, var), path) for path, var in drought]
    records = []
    try:
        for _, basin in basins.iterrows():
            lat, lon = float(basin["lat"]), float(basin["lon"])
            ts = nearest_series(t_source[1], t_source[2], lat, lon) if t_source else None
            ds = []
            for (resource, _) in d_sources:
                fragment = nearest_series(resource[1], resource[2], lat, lon)
                if fragment is not None and fragment.notna().any():
                    ds.append(fragment)
            if len(ds) > 1:
                # Geographic overlap is not a license to average distinct reconstructions.
                raise ValueError("CHSL-v1 conflicting climate reconstructions for one basin")
            records.extend(compute_forcing(ts, ds[0] if ds else None,
                                           str(basin["basin_id"]), start, end))
    finally:
        if t_source:
            t_source[0].close()
        for resource, _ in d_sources:
            resource[0].close()
    return pd.DataFrame(records)

def extract_regional(basins: pd.DataFrame, spec: dict) -> pd.DataFrame:
    from src.raw_ingest.regional_grid import extract_regional as extract_masked
    return extract_masked(basins, spec)

def main(argv=None):
    p = argparse.ArgumentParser(description="CHSL-v1 spatial extraction")
    p.add_argument("--basins", type=Path, required=True)
    p.add_argument("--temperature-nc", type=Path)
    p.add_argument("--temperature-var")
    p.add_argument("--dry-nc", type=Path, action="append")
    p.add_argument("--dry-var", action="append", help="One explicit variable per atlas file")
    p.add_argument("--regional-spec", type=Path, help="Raw local JSON with NetCDF paths, variables and units")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=2000)
    p.add_argument("--out-forcing", type=Path, default=Path("data/raw/forcing_decades.csv"))
    p.add_argument("--out-regional", type=Path, default=Path("data/raw/regional_decades.csv"))
    args = p.parse_args(argv)
    basins = pd.read_csv(args.basins)
    if args.temperature_nc or args.dry_nc:
        dry = args.dry_nc or []
        vars_ = args.dry_var or [None] * len(dry)
        if len(vars_) != len(dry):
            p.error("Provide the same number of --dry-nc and --dry-var options")
        forcing = extract_forcing(basins, args.temperature_nc, list(zip(dry, vars_)),
                                  args.start, args.end, args.temperature_var)
        args.out_forcing.parent.mkdir(parents=True, exist_ok=True)
        forcing.to_csv(args.out_forcing, index=False)
        print("CHSL-v1 wrote physical forcing observations")
    if args.regional_spec:
        regional = extract_regional(basins, json.loads(args.regional_spec.read_text()))
        args.out_regional.parent.mkdir(parents=True, exist_ok=True)
        regional.to_csv(args.out_regional, index=False)
        print("CHSL-v1 wrote bounded regional substrate observations")

if __name__ == "__main__":
    main()

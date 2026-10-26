"""CHSL-v1 cell-level regional aggregation inside fixed geographic boundaries."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point, shape
from src.raw_ingest.physical_grids import open_variable, years

EARTH_KM = 6371.0088
FIELDS = {
    "population": "b_regional_grid_pop",
    "cropland": "m_cropland_km2",
    "grazing": "m_grazing_km2",
}
UNITS = {
    "population": ("persons_per_cell", "persons_per_km2"),
    "cropland": ("km2_per_cell", "fraction_of_cell", "percent_of_cell"),
    "grazing": ("km2_per_cell", "fraction_of_cell", "percent_of_cell"),
}

def present(value):
    return value is not None and not pd.isna(value) and str(value).strip() != ""

def geometry_definition(basin: pd.Series):
    if present(basin.get("polygon_path")):
        path = Path(str(basin["polygon_path"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") == "FeatureCollection":
            selected = [f for f in payload["features"] if
                        str(f.get("properties", {}).get("basin_id", "")) == str(basin["basin_id"])]
            if len(selected) != 1:
                raise ValueError("CHSL-v1 requires exactly one reviewed polygon per basin")
            payload = selected[0]["geometry"]
        elif payload.get("type") == "Feature":
            payload = payload["geometry"]
        polygon = shape(payload)
        if polygon.geom_type not in ("Polygon", "MultiPolygon") or not polygon.is_valid:
            raise ValueError("CHSL-v1 invalid basin polygon")
        kind = "polygon"
        desc = {"type": kind, "wkb": polygon.wkb_hex}
        parameter = polygon
        radius = np.nan
    elif present(basin.get("radius_km")):
        radius = float(basin["radius_km"])
        if not 0 < radius <= 1000:
            raise ValueError("CHSL-v1 radius must be 0–1000 kilometres")
        kind = "centroid_buffer"
        parameter = (float(basin["lat"]), float(basin["lon"]), radius)
        desc = {"type": kind, "lat": parameter[0], "lon": parameter[1], "radius_km": radius}
    else:
        required = ["lat_min", "lat_max", "lon_min", "lon_max"]
        if any(not present(basin.get(k)) for k in required):
            raise ValueError("CHSL-v1 requires a polygon, radius, or explicit geographic box")
        vals = tuple(float(basin[k]) for k in required)
        if vals[0] > vals[1] or vals[2] > vals[3]:
            raise ValueError("CHSL-v1 invalid geographic box")
        kind = "box"
        parameter = vals
        desc = {"type": kind, "bounds": vals}
        radius = np.nan
    digest = hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest()[:20]
    return parameter, {
        "rj_id": "R_" + str(basin["basin_id"]),
        "rj_kind": kind,
        "rj_geometry_hash": digest,
        "rj_radius_km": radius,
        "grid_data_kind": "gridded_cells",
    }

def normalise_lon(lon):
    return (np.asarray(lon, dtype=float) + 180) % 360 - 180

def haversine(lat, lon, centre_lat, centre_lon):
    dlat = np.deg2rad(lat - centre_lat)
    dlon = np.deg2rad((lon - centre_lon + 180) % 360 - 180)
    p = np.deg2rad(lat)
    pc = np.deg2rad(centre_lat)
    a = np.sin(dlat / 2)**2 + np.cos(p) * np.cos(pc) * np.sin(dlon / 2)**2
    return 2 * EARTH_KM * np.arcsin(np.minimum(1, np.sqrt(a)))

def cell_mask(basin: pd.Series, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, dict]:
    parameter, meta = geometry_definition(basin)
    yy, xx = np.meshgrid(np.asarray(lats, float), normalise_lon(lons), indexing="ij")
    if meta["rj_kind"] == "centroid_buffer":
        centre_y, centre_x, radius = parameter
        within = haversine(yy, xx, centre_y, centre_x) <= radius
    elif meta["rj_kind"] == "box":
        a, b, c, d = parameter
        x = normalise_lon(xx)
        c, d = float(normalise_lon(c)), float(normalise_lon(d))
        if c > d:
            within = (yy >= a) & (yy <= b) & ((x >= c) | (x <= d))
        else:
            within = (yy >= a) & (yy <= b) & (x >= c) & (x <= d)
    else:
        polygon = parameter
        west, south, east, north = polygon.bounds
        if east - west > 180:
            raise ValueError("CHSL-v1 polygon crossing longitude discontinuity needs preprocessing")
        candidate = (yy >= south) & (yy <= north) & (xx >= west) & (xx <= east)
        within = np.zeros(yy.shape, bool)
        for i, j in zip(*np.where(candidate)):
            within[i, j] = polygon.covers(Point(float(xx[i, j]), float(yy[i, j])))
    if not within.any():
        raise ValueError("CHSL-v1 selected boundary has no grid-cell centres")
    return within, meta

def cell_area_km2(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lats, lons = np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
    if lats.size < 2 or lons.size < 2:
        raise ValueError("CHSL-v1 requires at least two grid coordinates per axis for area conversion")
    dy = float(np.median(np.abs(np.diff(lats))))
    dx = float(np.median(np.abs(np.diff(lons))))
    if not 0 < dy <= 180 or not 0 < dx <= 360:
        raise ValueError("CHSL-v1 invalid grid resolution")
    lo = np.clip(lats - dy / 2, -90, 90)
    hi = np.clip(lats + dy / 2, -90, 90)
    per_lat = EARTH_KM**2 * np.deg2rad(dx) * (
        np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo))
    )
    return np.broadcast_to(per_lat[:, None], (lats.size, lons.size))

def read_asc(path: Path):
    with path.open(encoding="utf-8") as f:
        header = [f.readline().split() for _ in range(6)]
    params = {row[0].lower(): float(row[1]) for row in header if len(row) >= 2}
    ncols, nrows = int(params["ncols"]), int(params["nrows"])
    step = float(params["cellsize"])
    xstart = params.get("xllcenter", params.get("xllcorner", np.nan) + step / 2)
    ystart = params.get("yllcenter", params.get("yllcorner", np.nan) + step / 2)
    raw = np.loadtxt(path, skiprows=6)
    if raw.shape != (nrows, ncols):
        raise ValueError("CHSL-v1 invalid ASC raster dimensions")
    nodata = params.get("nodata_value")
    if nodata is not None:
        raw = np.where(raw == nodata, np.nan, raw)
    lons = xstart + np.arange(ncols) * step
    lats = ystart + np.arange(nrows - 1, -1, -1) * step
    return raw, lats, lons

def read_layers(cfg: dict):
    if "files" in cfg:
        if "path" in cfg:
            raise ValueError("CHSL-v1 select either dated ASC files or a NetCDF cube")
        for year, path in sorted(cfg["files"].items(), key=lambda v: int(v[0])):
            if Path(path).suffix.lower() != ".asc":
                raise ValueError("CHSL-v1 dated files must be ASC rasters")
            arr, lat, lon = read_asc(Path(path))
            yield int(year), arr, lat, lon
        return
    if "path" not in cfg:
        raise ValueError("CHSL-v1 missing gridded raster path")
    if Path(cfg["path"]).suffix.lower() == ".asc":
        if "year" not in cfg:
            raise ValueError("CHSL-v1 single ASC raster requires an explicit year")
        arr, lat, lon = read_asc(Path(cfg["path"]))
        yield int(cfg["year"]), arr, lat, lon
        return
    ds, da, (td, yd, xd) = open_variable(Path(cfg["path"]), cfg.get("variable"))
    try:
        calendar = years(da[td])
        data = da.transpose(td, yd, xd)
        lat, lon = np.asarray(data[yd].values, dtype=float), np.asarray(data[xd].values, dtype=float)
        for i, year in enumerate(calendar):
            yield int(year), np.asarray(data.isel({td: i}).values, dtype=float), lat, lon
    finally:
        ds.close()

def aggregate(values: np.ndarray, mask: np.ndarray, unit: str,
              field: str, lats: np.ndarray, lons: np.ndarray) -> float:
    if values.shape != mask.shape:
        raise ValueError("CHSL-v1 grid shape and selected boundary disagree")
    selected = np.asarray(values, dtype=float)[mask]
    if not np.isfinite(selected).all():
        return np.nan  # Incomplete area is never silently treated as empty.
    if (selected < 0).any():
        raise ValueError("CHSL-v1 regional count/area cannot be negative")
    if unit in ("fraction_of_cell", "percent_of_cell", "persons_per_km2"):
        area = cell_area_km2(lats, lons)[mask]
        if unit == "fraction_of_cell" and (selected > 1).any():
            raise ValueError("CHSL-v1 fractional area exceeds 1")
        if unit == "percent_of_cell" and (selected > 100).any():
            raise ValueError("CHSL-v1 percentage area exceeds 100")
        factor = 0.01 if unit == "percent_of_cell" else 1
        return float(np.sum(selected * area * factor))
    return float(np.sum(selected))

def extract_regional(basins: pd.DataFrame, spec: dict) -> pd.DataFrame:
    if not spec or any(key not in FIELDS for key in spec):
        raise ValueError("CHSL-v1 requires raw gridded population/land-use components")
    if basins["basin_id"].duplicated().any():
        raise ValueError("CHSL-v1 duplicate basin boundaries")
    results: dict[tuple[str, int], dict] = {}
    for field, cfg in spec.items():
        if cfg.get("unit") not in UNITS[field]:
            raise ValueError("CHSL-v1 unsupported raster unit for this physical field")
        target = FIELDS[field]
        for year, data, lat, lon in read_layers(cfg):
            if year % 10:
                continue
            for _, basin in basins.iterrows():
                mask, metadata = cell_mask(basin, lat, lon)
                ident = (str(basin["basin_id"]), year)
                if ident not in results:
                    results[ident] = {"basin_id": ident[0], "t_obs": year, **metadata}
                elif results[ident]["rj_geometry_hash"] != metadata["rj_geometry_hash"]:
                    raise ValueError("CHSL-v1 source layers do not share the same R_j boundary")
                if target in results[ident]:
                    raise ValueError("CHSL-v1 duplicate raster field for one time cell")
                count = int(mask.sum())
                observation = aggregate(data, mask, cfg["unit"], field, lat, lon)
                results[ident][target] = observation
                results[ident][target + "_low"] = observation
                results[ident][target + "_high"] = observation
                results[ident][target + "_disputed_flag"] = False
                results[ident][target + "_grid_cells"] = count
    if not results:
        raise ValueError("CHSL-v1 no dated gridded cells within requested boundaries")
    return pd.DataFrame(results.values()).sort_values(["basin_id", "t_obs"]).reset_index(drop=True)

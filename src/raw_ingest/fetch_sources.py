"""Fetch and cache public upstream artefacts before translation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

RAW = Path("data/raw")
SOURCES = {
    "institutional": "https://raw.githubusercontent.com/seshatdb/Equinox_Data/master/Equinox_on_GitHub_June9_2022.xlsx",
    "temperature": "https://www.ncei.noaa.gov/pub/data/paleo/pages2k/neukom2019temp/pages2k_ngeo19_recons.nc",
    "dry_europe": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/europe/owda.nc",
    "dry_america": "https://www.ncei.noaa.gov/pub/data/paleo/drought/NAmericanDroughtAtlas.v2/NADAv2-2008.nc",
    "dry_asia": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/asia/cook2010pdsi/MADApdsi.nc",
}

def fetch(url: str, target: Path, refresh: bool = False) -> dict:
    if not url.startswith("https://"):
        raise ValueError("CHSL-v1 accepts HTTPS downloads only")
    target.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not target.is_file():
        partial = target.with_name(target.name + ".part")
        partial.unlink(missing_ok=True)
        try:
            with urlopen(Request(url, headers={"User-Agent": "CHSL-v1/0.1"}), timeout=90) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    h = hashlib.sha256()
    with target.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return {"path": str(target), "sha256": h.hexdigest(), "bytes": target.stat().st_size}

def main(argv=None):
    p = argparse.ArgumentParser(description="CHSL-v1 raw cache")
    p.add_argument("--all", action="store_true")
    p.add_argument("--source", choices=tuple(SOURCES), action="append")
    p.add_argument("--hyde-url", help="Verified gridded regional release; no global summaries")
    p.add_argument("--institutional-url", help="Override with reviewed later tabular release")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--raw-dir", type=Path, default=RAW)
    args = p.parse_args(argv)
    keys = list(SOURCES) if args.all or not args.source else args.source
    links = dict(SOURCES)
    if args.institutional_url:
        links["institutional"] = args.institutional_url
    if args.hyde_url:
        links["regional_grid"] = args.hyde_url
        keys.append("regional_grid")
    records = {}
    for key in dict.fromkeys(keys):
        url = links[key]
        name = url.split("?")[0].rsplit("/", 1)[-1]
        if not name or "." not in name:
            raise ValueError("CHSL-v1 download requires a named file")
        records[key] = fetch(url, args.raw_dir / key / name, args.refresh)
        records[key]["url"] = url
        print(f"CHSL-v1 cached {key}: {records[key]['bytes']} bytes")
    (args.raw_dir / "fetch_manifest.json").write_text(json.dumps(records, indent=2))
    print("CHSL-v1 raw cache manifest written")

if __name__ == "__main__":
    main()

"""Download optional public FRED context series into data/raw.

The core project runs from the local GSW and Treasury panel workbooks.  This
helper supports extensions that condition relative-value signals on market
stress, credit, and curve regimes.  Downloaded CSV files are ignored by git.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


DEFAULT_SERIES = [
    "DGS1",
    "DGS2",
    "DGS5",
    "DGS10",
    "DGS30",
    "T10Y2Y",
    "BAMLC0A0CM",
    "VIXCLS",
]


def fetch_series(series_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    target = output_dir / f"{series_id}.csv"
    with urlopen(url, timeout=30) as response:
        target.write_bytes(response.read())
    return target


def main() -> None:
    output_dir = Path("data/raw/fred")
    for series_id in DEFAULT_SERIES:
        path = fetch_series(series_id, output_dir)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

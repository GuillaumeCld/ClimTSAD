"""Download Bordeaux ERA5-Land precipitation and create the daily NPY file."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from threading import local

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr


DOWNLOAD_DIR = Path("/home/guillaume/Devel/Data/ERA5_land/tp_1970_2023")
OUTPUT = Path(__file__).resolve().parents[1] / "data/preci_time_series.npy"
YEARS = range(1970, 2024)
MONTHS = [f"{month:02d}" for month in range(1, 13)]
MONTH_CHUNKS = [MONTHS[start:start + 4] for start in range(0, 12, 4)]
MONTH_DAYS = [f"{day:02d}" for day in range(1, 32)]
HOURS = [f"{hour:02d}:00" for hour in range(24)]
BORDEAUX = [44.8, -0.6, 44.8, -0.6]  # north, west, south, east
MAX_WORKERS = int(os.environ.get("TP_DOWNLOAD_WORKERS", "1"))


DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
files = []
downloads = []


_worker = local()


def download(target, request):
    """Download one request without exposing a partially written NetCDF file."""
    if not hasattr(_worker, "client"):
        _worker.client = cdsapi.Client()

    partial = target.with_suffix(f"{target.suffix}.{os.getpid()}.part")
    print(f"Downloading ERA5-Land precipitation to {target.name}...")
    try:
        _worker.client.retrieve(
            "reanalysis-era5-land", request, str(partial)
        )
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


for year in YEARS:
    annual = DOWNLOAD_DIR / f"tp_{year}.nc"
    remaining = DOWNLOAD_DIR / f"tp_{year}_remaining.nc"
    legacy_months = {
        month: DOWNLOAD_DIR / f"tp_{year}_{month}.nc" for month in MONTHS
    }
    existing_months = [
        path for path in legacy_months.values() if path.exists()
    ]

    if annual.exists():
        files.append(annual)
        continue

    # Reuse files created by the old monthly downloader. If a year is only
    # partially present, reuse a completion file made by the previous annual
    # batching implementation if one exists.
    files.extend(existing_months)
    if remaining.exists():
        files.append(remaining)
        continue

    # Limit new requests to four-month chunks. Months already available as
    # legacy files are omitted from their chunk's request.
    for chunk in MONTH_CHUNKS:
        missing_months = [
            month for month in chunk if not legacy_months[month].exists()
        ]
        if not missing_months:
            continue

        target = DOWNLOAD_DIR / f"tp_{year}_{chunk[0]}_{chunk[-1]}.nc"
        files.append(target)
        if not target.exists():
            downloads.append(
                (
                    target,
                    {
                        "variable": ["total_precipitation"],
                        "year": str(year),
                        "month": missing_months,
                        # CDS ignores invalid dates in shorter months.
                        "day": MONTH_DAYS,
                        "time": HOURS,
                        "area": BORDEAUX,
                        "data_format": "netcdf",
                        "download_format": "unarchived",
                    },
                )
            )

# The following midnight contains the total for 2023-12-31.
boundary = DOWNLOAD_DIR / "tp_2024_01_01.nc"
files.append(boundary)
if not boundary.exists():
    downloads.append(
        (
            boundary,
            {
                "variable": ["total_precipitation"],
                "year": "2024",
                "month": "01",
                "day": ["01"],
                "time": ["00:00"],
                "area": BORDEAUX,
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
        )
    )

if downloads:
    print(f"Downloading {len(downloads)} files with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(lambda job: download(*job), downloads))

series = []
for path in files:
    with xr.open_dataset(path) as ds:
        tp = ds["tp"].squeeze().load()
        time = "time" if "time" in tp.coords else "valid_time"
        values = np.atleast_1d(tp.values)
        timestamps = np.atleast_1d(tp[time].values)
        series.append(
            pd.Series(values, index=pd.DatetimeIndex(timestamps))
        )

hourly = pd.concat(series).sort_index()
daily = hourly[hourly.index.hour == 0]
daily.index = daily.index - pd.Timedelta(days=1)
daily = daily.loc["1970-01-01":"2023-12-31"]
daily = daily[~((daily.index.month == 2) & (daily.index.day == 29))]

if len(daily) != 54 * 365:
    raise ValueError(f"Expected {54 * 365} daily values, found {len(daily)}")

np.save(OUTPUT, np.maximum(daily.to_numpy(dtype=float), 0.0))
print(f"Saved {len(daily)} daily precipitation values to {OUTPUT}")

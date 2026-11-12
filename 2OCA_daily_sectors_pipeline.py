"""
OCA (Optimal Cloud Analysis, SEVIRI) 07:00-17:00 local daily means, by longitude sector -> 0.25 deg TNA files
on the CLARA-A3 grid
Credits: Developed for CloudF project.
License: This code is offered as free-to-use in the public domain, with no warranty.

Source : EO:EUM:DAT:0617, OCA CDR, 15-min full-disc scans, MSG1-4, 2004-01-19 -> 2019-08-31
Output : one netCDF per year in oca_sectors_025_years/, May-Aug only, same box and 0.25 deg cells as the
         CLARA CFC/OLR and SARAH CAL stores (lat -4.875..19.875, lon -69.875..-20.125), so arrays align 1:1.

Local time window, by sector:
    The box is split into three equal longitude sectors. Each sector uses the hourly scans from 07:00
    to 17:00 local (11 scans), with local time taken at the sector centre and rounded to the nearest
    whole hour of GMT offset:
        west   70.0-53.3W  centre 61.7W  GMT = local + 4 h  ->  11:00-21:00 GMT
        centre 53.3-36.7W  centre 45.0W  GMT = local + 3 h  ->  10:00-20:00 GMT
        east   36.7-20.0W  centre 28.3W  GMT = local + 2 h  ->  09:00-19:00 GMT
    Each 0.25 deg cell belongs to the sector its centre falls in.

Variables per day and 0.25 deg cell:
    cf_scene   : % of scans OCA's scene_classification marks cloudy (1, 2, 3), per pixel, then mean over
                 pixels in the cell; failed retrievals (10) are left out
    valid_frac : fraction of OCA pixels in the cell with a daily cf_scene value (cells below 0.5 are NaN)
    n_scans    : number of scans read for the day (max 13); each cell uses its sector's 11
    platform   : Meteosat satellite of the day's scans (e.g. MET-09)

Steps (run in order, each is resumable):
    build   : per year, download each scan, crop + bin it, delete the raw file, save one netCDF per year.
              Reruns only fetch missing/failed days.
    combine : join the yearly files into pre- and post-2011 zarr stores

Usage:
    python 2OCA_daily_sectors_pipeline.py build [first_year last_year]
    python 2OCA_daily_sectors_pipeline.py combine
"""
import os
import re
import sys
import shutil
import tempfile
import warnings
from datetime import datetime, date, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

warnings.filterwarnings("ignore")

# --- Paths and settings ---
COLLECTION_ID = "EO:EUM:DAT:0617"
ROOT          = Path("/home/nandii/cloudML")
YEAR_DIR      = ROOT / "oca_sectors_025_years"
TMP_DIR       = ROOT / "oca_tmp_download"          # raw scans live here only while being read
ZARR_PRE      = ROOT / "OCA_sectors_ss_2004_2010.zarr"
ZARR_POST     = ROOT / "OCA_sectors_ss_2011_2019.zarr"

FIRST_DAY     = date(2004, 1, 19)
LAST_DAY      = date(2019, 8, 31)
MONTHS        = [5, 6, 7, 8]

# Sectors: west edge, east edge, hours added to local time to get GMT
SECTORS       = [(-70.0, -160 / 3, 4), (-160 / 3, -110 / 3, 3), (-110 / 3, -20.0, 2)]
LOCAL_TIMES   = [time(h, 0) for h in range(7, 18)]                                         # 07:00 ... 17:00
SECTOR_SCANS  = [{time(t.hour + shift, t.minute) for t in LOCAL_TIMES} for _, _, shift in SECTORS]
ALL_SCANS     = set().union(*SECTOR_SCANS)                                                 # 09:00 ... 21:00 GMT

MIN_COVERAGE   = 0.5    # fraction of the sector's scans a pixel needs valid data in
MIN_VALID_FRAC = 0.5    # fraction of OCA pixels in a 0.25 deg cell that must be valid

N_WORKERS     = 6
MAX_RETRIES   = 3
CHUNK_SIZE    = 1024 * 1024

# Target grid: CLARA-A3 0.25 deg cell centres over the TNA box (-5..20N, -70..-20E)
LAT_MIN, LAT_MAX = -5.0, 20.0
LON_MIN, LON_MAX = -70.0, -20.0
RES     = 0.25
LAT_025 = np.round(np.arange(-4.875, 20.0, 0.25), 3)        # 100
LON_025 = np.round(np.arange(-69.875, -20.0, 0.25), 3)      # 200
N_CELLS = len(LAT_025) * len(LON_025)

# Sector of every cell, from the cell-centre longitude
CELL_SECTOR = np.tile(np.digitize(LON_025, [w for w, _, _ in SECTORS[1:]]), len(LAT_025))
# ---------------------------------------------------------------------------


def datastore_from_credentials():
    """EUMDAC DataStore from ~/.eumdac/credentials ('key,secret'), as written by `eumdac set-credentials`."""
    import eumdac
    cred_file = Path.home() / ".eumdac" / "credentials"
    if not cred_file.exists():
        sys.exit("No ~/.eumdac/credentials found: run `eumdac set-credentials` once.")
    key, secret = cred_file.read_text().strip().split(",")
    return eumdac.DataStore(eumdac.AccessToken((key, secret)))


def scan_time(product):
    return datetime.strptime(re.search(r"_(\d{14})_", str(product)).group(1), "%Y%m%d%H%M%S")


def daily_scans(datastore, year):
    """The 09:00-21:00 GMT hourly scans of each May-Aug day in `year`, as {date: [products]}."""
    collection = datastore.get_collection(COLLECTION_ID)
    by_day = {}
    for month in MONTHS:
        start = max(date(year, month, 1), FIRST_DAY)
        end   = min((pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).date(), LAST_DAY)
        if start > end:
            continue
        for product in collection.search(dtstart = f"{start}T08:45:00", dtend = f"{end}T21:15:00"):
            t = scan_time(product)
            if t.time() in ALL_SCANS and t.month in MONTHS:
                by_day.setdefault(t.date(), {})[t.time()] = product
    return {d: [p for _, p in sorted(s.items())] for d, s in by_day.items()}


def read_scan(product):
    """Download one scan, crop it to the box, return per-pixel fields and the 0.25 deg cell index; delete the file."""
    entry = next(e for e in product.entries if e.endswith(".nc"))
    tmp = Path(tempfile.mkdtemp(dir = TMP_DIR))
    path = tmp / entry
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with product.open(entry = entry) as fsrc, open(path, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst, CHUNK_SIZE)
                break
            except Exception:
                path.unlink(missing_ok = True)
                if attempt == MAX_RETRIES:
                    raise

        with netCDF4.Dataset(path) as nc:
            nc.set_auto_maskandscale(True)
            lat = np.ma.filled(nc["latitude"][:].astype(np.float32), np.nan)
            lon = np.ma.filled(nc["longitude"][:].astype(np.float32), np.nan)
            # Off-disc pixels have NaN coordinates and fail this test
            in_box = (lat >= LAT_MIN) & (lat < LAT_MAX) & (lon >= LON_MIN) & (lon < LON_MAX)
            rows = np.where(in_box.any(axis = 1))[0]
            cols = np.where(in_box.any(axis = 0))[0]
            rs, cs = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
            in_box = in_box[rs, cs]

            fields = {"scene": np.ma.filled(nc["scene_classification"][rs, cs].astype(np.float32), np.nan)[in_box]}
            platform = str(nc.getncattr("platform"))
            lat, lon = lat[rs, cs][in_box], lon[rs, cs][in_box]

        cell = (np.floor((lat - LAT_MIN) / RES).astype(np.int32) * len(LON_025)
                + np.floor((lon - LON_MIN) / RES).astype(np.int32))
        return fields, cell, platform
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


def cell_mean(values, cell, valid):
    """Mean of `values` per 0.25 deg cell over pixels where `valid`, plus the count."""
    n = np.bincount(cell[valid], minlength = N_CELLS)
    s = np.bincount(cell[valid], weights = values[valid], minlength = N_CELLS)
    with np.errstate(invalid = "ignore", divide = "ignore"):
        return s / n, n


def process_day(day, products):
    """All scans of one day -> 0.25 deg daily fields. Returns (day, fields) or (day, error)."""
    try:
        cell = None
        platforms = set()
        n_read = 0

        # Running per-pixel totals, so only one scan is held in memory at a time
        for p in products:
            try:
                fields, c, plat = read_scan(p)
            except Exception as e:
                print(f"  {day}: scan {scan_time(p):%H%M} skipped ({type(e).__name__})", flush = True)
                continue
            if cell is None:
                cell = c
                px_sector = CELL_SECTOR[cell]
                n_win = np.zeros(cell.size, np.int16)
                n_sc  = np.zeros(cell.size, np.int16)
                sc_s  = np.zeros(cell.size, np.float32)
            elif not np.array_equal(c, cell):
                raise RuntimeError("scans of the day are on different grids")
            platforms.add(plat)
            n_read += 1

            # Pixels whose sector uses this scan time
            t = scan_time(p).time()
            use = np.isin(px_sector, [k for k, s in enumerate(SECTOR_SCANS) if t in s])
            n_win += use

            # OCA's own cloud mask: scene classes 1-3 cloudy, 0 clear, 10 (failed) left out
            sc = fields["scene"]
            ok = use & np.isfinite(sc) & (sc != 10)
            n_sc += ok
            sc_s += ok & np.isin(sc, [1, 2, 3])

        if cell is None:
            raise RuntimeError("no readable scans")

        with np.errstate(invalid = "ignore", divide = "ignore"):
            keep  = (n_win > 0) & (n_sc >= MIN_COVERAGE * n_win)
            sc_px = np.where(keep, sc_s / n_sc * 100, np.nan)

            n_all = np.bincount(cell, minlength = N_CELLS)
            sc_c, n_valid = cell_mean(sc_px, cell, np.isfinite(sc_px))
            valid_frac = n_valid / n_all
            good = (n_all > 0) & (valid_frac >= MIN_VALID_FRAC)

            shape = (len(LAT_025), len(LON_025))
            out = {
                "cf_scene":   np.where(good, sc_c, np.nan).reshape(shape).astype(np.float32),
                "valid_frac": np.where(n_all > 0, valid_frac, np.nan).reshape(shape).astype(np.float32),
            }
        return day, (out, n_read, ",".join(sorted(platforms)))
    except Exception as e:
        return day, e


def year_path(year):
    return YEAR_DIR / f"OCAsec025_{year}.nc"


GRID_VARS = ("cf_scene", "valid_frac")


def build_year(datastore, year):
    available = daily_scans(datastore, year)
    if not available:
        print(f"{year}: no May-Aug OCA scans on the Data Store")
        return

    existing = None
    if year_path(year).exists():
        existing = xr.load_dataset(year_path(year))
        done = set(pd.DatetimeIndex(existing["time"].values).date)
    else:
        done = set()

    todo = {d: p for d, p in available.items() if d not in done}
    if not todo:
        print(f"{year}: complete ({len(done)} days)")
        return

    print(f"{year}: {len(todo)} days to fetch ({len(done)} already done, {len(available)} available)", flush = True)
    results, failed = {}, []
    with ThreadPoolExecutor(N_WORKERS) as pool:
        futures = [pool.submit(process_day, d, p) for d, p in todo.items()]
        for f in as_completed(futures):
            day, res = f.result()
            if isinstance(res, Exception):
                failed.append(day)
                print(f"  FAILED {day}: {type(res).__name__}: {res}", flush = True)
            else:
                results[day] = res

    if results:
        days = sorted(results)
        new = xr.Dataset(
            {name: (("time", "lat", "lon"), np.stack([results[d][0][name] for d in days])) for name in GRID_VARS},
            coords = {"time": pd.to_datetime(days), "lat": LAT_025, "lon": LON_025})
        new["n_scans"]  = ("time", np.array([results[d][1] for d in days], dtype = np.int8))
        new["platform"] = ("time", np.array([results[d][2] for d in days], dtype = object))
        ds = xr.concat([existing, new], dim = "time").sortby("time") if existing is not None else new
    else:
        ds = existing

    if ds is None:
        print(f"{year}: nothing written, all {len(failed)} days failed")
        return

    ds["cf_scene"].attrs   = {"long_name": "Percentage of the sector's 07:00-17:00 local scans classed cloudy by OCA scene_classification (1-3; 10 = failed left out)", "units": "%"}
    ds["valid_frac"].attrs = {"long_name": "Fraction of OCA pixels in the cell with a daily cf_scene value", "units": "1"}
    ds["n_scans"].attrs    = {"long_name": "Number of hourly scans read for the day (max 13); each cell uses its sector's 11"}
    ds["platform"].attrs   = {"long_name": "Meteosat platform of the day's scans"}
    ds.attrs = {"source": f"OCA CDR {COLLECTION_ID}, SEVIRI full-disc scans, hourly 07:00-17:00 local by longitude sector",
                "sectors": "70.0-53.3W: GMT = local + 4 h; 53.3-36.7W: + 3 h; 36.7-20.0W: + 2 h (cell centre decides the sector)",
                "grid": "CLARA-A3 0.25 deg cell centres; OCA pixels binned by their own lat/lon",
                "failed_days": ",".join(str(d) for d in sorted(failed))}

    YEAR_DIR.mkdir(parents = True, exist_ok = True)
    tmp_out = year_path(year).with_suffix(".nc.tmp")
    enc = {v: {"zlib": True, "complevel": 4} for v in GRID_VARS}
    ds.to_netcdf(tmp_out, encoding = enc)
    os.replace(tmp_out, year_path(year))
    print(f"{year}: wrote {ds.sizes['time']} days" + (f", {len(failed)} failed (rerun to retry)" if failed else ""),
          flush = True)


def build(datastore, first_year, last_year):
    TMP_DIR.mkdir(parents = True, exist_ok = True)
    for year in range(first_year, last_year + 1):
        build_year(datastore, year)
    shutil.rmtree(TMP_DIR, ignore_errors = True)


def combine():
    files = sorted(YEAR_DIR.glob("OCAsec025_*.nc"))
    ds = xr.concat([xr.load_dataset(f) for f in files], dim = "time").sortby("time")
    ds.attrs.pop("failed_days", None)
    ds["platform"] = ds["platform"].astype(str)
    times = pd.DatetimeIndex(ds["time"].values)
    print(f"{len(times)} days, {times.min().date()} to {times.max().date()}")

    for path, sub in ((ZARR_PRE, ds.sel(time = slice(None, "2010-12-31"))),
                      (ZARR_POST, ds.sel(time = slice("2011-01-01", None)))):
        sub = sub.chunk({"time": 123, "lat": 25, "lon": 100})
        for v in sub.data_vars:
            sub[v].encoding = {}
        sub.to_zarr(path, mode = "w")
        print(f"wrote {path.name}: {sub.sizes['time']} days")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "build"
    if step == "build":
        first, last = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (FIRST_DAY.year, LAST_DAY.year)
        build(datastore_from_credentials(), first, last)
    elif step == "combine":
        combine()
    else:
        sys.exit(f"unknown step {step}: use build or combine")

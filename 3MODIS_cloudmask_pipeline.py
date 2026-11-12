"""
MODIS (Terra MOD35 / Aqua MYD35) daytime cloud fraction, daily -> 0.25 deg TNA files on the CLARA-A3 grid
Credits: Developed for CloudF project.
License: This code is offered as free-to-use in the public domain, with no warranty.

Source : NASA LAADS, MOD35_L2 and MYD35_L2 collection 6.1 cloud mask, 5-min granules
         Terra from 2000-02-24, Aqua from 2002-07-04
Output : one netCDF per year in modis_cf_025_years/, May-Aug only, same box and 0.25 deg cells as the
         CLARA CFC/OLR, SARAH CAL and OCA stores (lat -4.875..19.875, lon -69.875..-20.125), so arrays align 1:1.

Sampling:
    Daytime only (Terra ~10:30, Aqua ~13:30 local), from MOD35's own day/night flag.
    MOD35 gives latitude/longitude every 5 km, so the 1 km cloud-mask pixel at the centre of each 5 x 5 block
    is used, with the 5 km latitude/longitude.

Cloud mask (Cloud_Mask byte 0):
    bit 0     : 1 = mask determined (others left out)
    bit 3     : 1 = day (others left out)
    bits 1-2  : 00 cloudy, 01 probably cloudy  -> cloudy
                10 probably clear, 11 confident clear -> clear

Variables per day and 0.25 deg cell:
    cf_terra, cf_aqua : % of valid daytime pixels that are cloudy, all of that satellite's granules for the day
    n_terra, n_aqua   : number of valid pixels behind each value (use it to drop thinly sampled cells)

Steps (run in order, each is resumable):
    build   : per year and satellite, download each granule, bin it, delete the raw file, save one netCDF per year.
              Reruns only fetch missing/failed days.
    combine : join the yearly files into pre- and post-2011 zarr stores

Usage:
    python 3MODIS_cloudmask_pipeline.py build [first_year last_year]
    python 3MODIS_cloudmask_pipeline.py combine

NASA Earthdata credentials come from EARTHDATA_USERNAME / EARTHDATA_PASSWORD if set, otherwise ~/.netrc.
"""
import os
import sys
import shutil
import tempfile
import warnings
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# --- Paths and settings ---
ROOT      = Path("/home/nandii/cloudML")
YEAR_DIR  = ROOT / "modis_cf_025_years"
TMP_DIR   = ROOT / "modis_tmp_download"         # raw granules live here only while being read
ZARR_PRE  = ROOT / "MODIS_CF_2000_2010.zarr"
ZARR_POST = ROOT / "MODIS_CF_2011_2026.zarr"

SATELLITES = {"terra": ("MOD35_L2", date(2000, 2, 24)), "aqua": ("MYD35_L2", date(2002, 7, 4))}
VERSION    = "6.1"
LAST_YEAR  = 2026
MONTHS     = [5, 6, 7, 8]

N_WORKERS   = 6
MAX_RETRIES = 3

# Target grid: CLARA-A3 0.25 deg cell centres over the TNA box (-5..20N, -70..-20E)
LAT_MIN, LAT_MAX = -5.0, 20.0
LON_MIN, LON_MAX = -70.0, -20.0
RES     = 0.25
LAT_025 = np.round(np.arange(-4.875, 20.0, 0.25), 3)        # 100
LON_025 = np.round(np.arange(-69.875, -20.0, 0.25), 3)      # 200
N_CELLS = len(LAT_025) * len(LON_025)
SHAPE   = (len(LAT_025), len(LON_025))
# ---------------------------------------------------------------------------


def login():
    import earthaccess
    # EARTHDATA_USERNAME / EARTHDATA_PASSWORD if set, otherwise ~/.netrc
    strategy = "environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc"
    auth = earthaccess.login(strategy = strategy)
    if not auth.authenticated:
        sys.exit("Earthdata login failed: set EARTHDATA_USERNAME/EARTHDATA_PASSWORD or ~/.netrc")
    return earthaccess


def daily_granules(earthaccess, short_name, first_day, year):
    """Daytime granules over the box for each May-Aug day in `year`, as {date: [granules]}."""
    out = {}
    for month in MONTHS:
        start = max(date(year, month, 1), first_day)
        end   = (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).date()
        if start > end:
            continue
        results = earthaccess.search_data(short_name = short_name, version = VERSION,
                                          bounding_box = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
                                          temporal = (f"{start}T00:00:00", f"{end}T23:59:59"),
                                          day_night_flag = "day")
        for g in results:
            day = pd.Timestamp(g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]).date()
            out.setdefault(day, []).append(g)
    return dict(sorted(out.items()))


def read_granule(path):
    """One MOD35/MYD35 file -> (cell index, cloudy flag) for valid daytime pixels in the box."""
    from pyhdf.SD import SD, SDC
    hdf = SD(str(path), SDC.READ)
    try:
        lat = hdf.select("Latitude")[:].astype(np.float32)
        lon = hdf.select("Longitude")[:].astype(np.float32)
        byte0 = hdf.select("Cloud_Mask")[0].astype(np.uint8)
    finally:
        hdf.end()

    # Centre 1 km pixel of each 5 x 5 block, matching the 5 km latitude/longitude grid
    rows, cols = lat.shape
    byte0 = byte0[2:5 * rows:5, 2:5 * cols:5][:rows, :cols]

    determined = (byte0 & 0b1) == 1
    day        = ((byte0 >> 3) & 0b1) == 1
    cloudy     = ((byte0 >> 1) & 0b11) <= 1
    in_box = (lat >= LAT_MIN) & (lat < LAT_MAX) & (lon >= LON_MIN) & (lon < LON_MAX)
    ok = determined & day & in_box & (np.abs(lat) <= 90)

    cell = (np.floor((lat[ok] - LAT_MIN) / RES).astype(np.int32) * len(LON_025)
            + np.floor((lon[ok] - LON_MIN) / RES).astype(np.int32))
    return cell, cloudy[ok]


def process_day(earthaccess, day, granules):
    """All granules of one day -> 0.25 deg cloud fraction and pixel count. Returns (day, result) or (day, error)."""
    tmp = Path(tempfile.mkdtemp(dir = TMP_DIR))
    try:
        n_cloud = np.zeros(N_CELLS, np.int32)
        n_valid = np.zeros(N_CELLS, np.int32)
        n_read = 0
        for g in granules:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    files = earthaccess.download([g], str(tmp), show_progress = False, threads = 1)
                    if not files:
                        raise RuntimeError("no file returned")
                    cell, cloudy = read_granule(files[0])
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        print(f"  {day}: granule {g['meta']['native-id']} skipped ({type(e).__name__})", flush = True)
                        cell = None
                finally:
                    for f in tmp.iterdir():
                        f.unlink()
            if cell is None:
                continue
            n_valid += np.bincount(cell, minlength = N_CELLS)
            n_cloud += np.bincount(cell, weights = cloudy, minlength = N_CELLS).astype(np.int32)
            n_read += 1
        if n_read == 0:
            raise RuntimeError("no readable granules")
        with np.errstate(invalid = "ignore", divide = "ignore"):
            cf = np.where(n_valid > 0, n_cloud / n_valid * 100, np.nan)
        return day, (cf.reshape(SHAPE).astype(np.float32), n_valid.reshape(SHAPE))
    except Exception as e:
        return day, e
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


def year_path(sat, year):
    return YEAR_DIR / f"MODIScf025_{sat}_{year}.nc"


def build_year(earthaccess, sat, year):
    short_name, first_day = SATELLITES[sat]
    if year < first_day.year:
        return
    available = daily_granules(earthaccess, short_name, first_day, year)
    if not available:
        print(f"{sat} {year}: no May-Aug granules")
        return

    done = set()
    if year_path(sat, year).exists():
        existing = xr.load_dataset(year_path(sat, year))
        done = set(pd.DatetimeIndex(existing["time"].values).date)
    todo = {d: g for d, g in available.items() if d not in done}
    if not todo:
        print(f"{sat} {year}: complete ({len(done)} days)")
        return
    print(f"{sat} {year}: {len(todo)} days to fetch ({len(done)} already done, {len(available)} available)", flush = True)

    results, failed = {}, []
    with ThreadPoolExecutor(max_workers = N_WORKERS) as pool:
        futures = [pool.submit(process_day, earthaccess, d, g) for d, g in todo.items()]
        for fut in as_completed(futures):
            day, res = fut.result()
            if isinstance(res, Exception):
                print(f"  {day}: failed ({type(res).__name__}: {res})", flush = True)
                failed.append(day)
            else:
                results[day] = res

    if not results:
        print(f"{sat} {year}: nothing written, all {len(failed)} days failed")
        return

    days = sorted(results)
    new = xr.Dataset({f"cf_{sat}": (("time", "lat", "lon"), np.stack([results[d][0] for d in days])),
                      f"n_{sat}":  (("time", "lat", "lon"), np.stack([results[d][1] for d in days]))},
                     coords = {"time": pd.DatetimeIndex(days), "lat": LAT_025, "lon": LON_025})
    ds = xr.concat([existing, new], dim = "time").sortby("time") if done else new
    ds.attrs = {"source": f"NASA {short_name} C6.1 cloud mask, daytime, centre pixel of each 5 km block",
                "cloudy": "cloudy + probably cloudy"}

    YEAR_DIR.mkdir(parents = True, exist_ok = True)
    tmp_out = year_path(sat, year).with_suffix(".nc.tmp")
    ds.to_netcdf(tmp_out)
    os.replace(tmp_out, year_path(sat, year))
    print(f"{sat} {year}: wrote {ds.sizes['time']} days" + (f", {len(failed)} failed (rerun to retry)" if failed else ""),
          flush = True)


def build(earthaccess, first_year, last_year):
    TMP_DIR.mkdir(parents = True, exist_ok = True)
    for year in range(first_year, last_year + 1):
        for sat in SATELLITES:
            build_year(earthaccess, sat, year)
    shutil.rmtree(TMP_DIR, ignore_errors = True)


def combine():
    per_sat = []
    for sat in SATELLITES:
        files = sorted(YEAR_DIR.glob(f"MODIScf025_{sat}_*.nc"))
        if files:
            per_sat.append(xr.concat([xr.load_dataset(f) for f in files], dim = "time").sortby("time"))
    ds = xr.merge(per_sat)
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
        first, last = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2000, LAST_YEAR)
        build(login(), first, last)
    elif step == "combine":
        combine()
    else:
        sys.exit(f"unknown step {step}: use build or combine")

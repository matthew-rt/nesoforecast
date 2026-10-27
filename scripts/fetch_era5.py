"""Download hourly ERA5 single-level fields over GB + offshore wind zones, one NetCDF per month.
Run with a month range, e.g. `python fetch_era5.py 2021-01 2026-10`; already-downloaded months are skipped."""
import pathlib, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
import cdsapi, pandas as pd

# permanent shared store (see /Users/matt/SCORESdata/DATA_INDEX.md); data/raw/era5 is a symlink to it
OUT = pathlib.Path("/Users/matt/SCORESdata/era5_gb_hourly_2021-2026")
OUT.mkdir(parents=True, exist_ok=True)
AREA = [61.5, -8.5, 49.5, 3.0]  # N, W, S, E: covers Shetland (Viking) to Dogger Bank / East Anglia
VARS = ["100m_u_component_of_wind", "100m_v_component_of_wind",
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "2m_temperature", "2m_dewpoint_temperature", "surface_pressure",
        "sea_surface_temperature", "boundary_layer_height", "instantaneous_10m_wind_gust"]

def month(ym):
    f = OUT / f"era5_{ym}.nc"
    if f.exists():
        return
    p = pd.Period(ym)
    req = {"product_type": ["reanalysis"], "variable": VARS, "year": [str(p.year)], "month": [f"{p.month:02d}"],
           "day": [f"{d:02d}" for d in range(1, p.days_in_month + 1)], "time": [f"{h:02d}:00" for h in range(24)],
           "area": AREA, "data_format": "netcdf", "download_format": "unarchived"}
    tmp = f.with_suffix(".part")
    cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels", req, str(tmp))
    # CDS may zip multi-stream output (instantaneous vs accumulated) despite "unarchived"
    if zipfile.is_zipfile(tmp):
        d = OUT / f"era5_{ym}"; zipfile.ZipFile(tmp).extractall(d); tmp.unlink()
        d.rename(f)  # directory of .nc files
    else:
        tmp.rename(f)
    print("done", ym, flush=True)

if __name__ == "__main__":
    months = [str(p) for p in pd.period_range(sys.argv[1], sys.argv[2], freq="M")]
    with ThreadPoolExecutor(4) as ex:  # CDS runs a few requests per user concurrently
        for r in ex.map(lambda m: (m, month(m)), months):
            pass

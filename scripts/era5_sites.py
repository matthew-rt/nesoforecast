"""Interpolate monthly ERA5 grids to each wind-site location -> data/era5_sites.parquet (hourly, UTC).
Speeds are bilinearly interpolated as speeds (interpolating u/v first biases speed low where direction varies);
direction comes from interpolated u/v. Derived: shear exponent 10->100 m, air density, SST - T2m."""
import pathlib
import numpy as np, pandas as pd, xarray as xr

ROOT = pathlib.Path(__file__).resolve().parent.parent
ERA5, DATA = pathlib.Path("/Users/matt/SCORESdata/era5_gb_hourly_2021-2026"), ROOT / "data"

def locations():
    s = pd.read_csv(DATA / "sites.csv").dropna(subset=["lat"])
    s["loc"] = s.osm_id.fillna("repd" + s.repd_id.astype(str))
    return s.groupby("loc")[["lat", "lon"]].first()

def open_month(f):
    # CDS sometimes returns a directory of per-stream files instead of one NetCDF
    ds = xr.open_mfdataset(sorted(f.glob("*.nc"))) if f.is_dir() else xr.open_dataset(f)
    return ds.rename({"valid_time": "time"}).drop_vars(["number", "expver"], errors="ignore")

def derived(ds):
    ws100, ws10 = np.hypot(ds.u100, ds.v100), np.hypot(ds.u10, ds.v10)
    t, td, p = ds.t2m, ds.d2m, ds.sp
    # SST is NaN on land cells; fill from adjacent sea cells so coastal/offshore sites interpolate cleanly
    sst = ds.sst.ffill("longitude").bfill("longitude").ffill("latitude").bfill("latitude")
    e = 610.94 * np.exp(17.625 * (td - 273.15) / (td - 30.11))  # vapour pressure (Pa), Magnus
    rho = (p - e) / (287.05 * t) + e / (461.5 * t)
    return xr.Dataset({"ws100": ws100, "ws10": ws10, "u100": ds.u100, "v100": ds.v100,
                       "t2m": t - 273.15, "rho": rho, "sp": p / 100, "sst": sst - 273.15,
                       "blh": ds.blh, "gust": ds.i10fg if "i10fg" in ds else np.nan * ds.blh})  # 2026-08 file lacks i10fg

CACHE = DATA / "cache" / "era5_sites"  # one parquet per month, so reruns only process new months

def month_sites(f, locs):
    out = CACHE / (f.stem + ".parquet")
    if out.exists() and set(pd.read_parquet(out, columns=["loc"])["loc"].unique()) == set(locs.index):
        return pd.read_parquet(out)
    lat = xr.DataArray(locs.lat.values, dims="loc"); lon = xr.DataArray(locs.lon.values, dims="loc")
    x = derived(open_month(f)).interp(latitude=lat, longitude=lon, method="linear").load()
    d = x.to_dataframe().reset_index()
    d["loc"] = locs.index.values[d["loc"].values] if d["loc"].dtype != object else d["loc"]
    d = d.drop(columns=["latitude", "longitude"], errors="ignore")
    d.to_parquet(out, index=False)
    print(f.name, flush=True)
    return d

def main():
    locs = locations()
    CACHE.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(ERA5.glob("era5_*")) if f.suffix != ".part" and not f.name.endswith(".md")]
    d = pd.concat([month_sites(f, locs) for f in files])
    d["wd100"] = (np.degrees(np.arctan2(-d.u100, -d.v100)) % 360).round(1)  # meteorological: from-direction
    d["alpha"] = np.log(d.ws100.clip(lower=0.5) / d.ws10.clip(lower=0.5)) / np.log(10)
    d["sst_t2m"] = d.sst - d.t2m  # >0: sea warmer than air -> unstable, mixed; <0: stable, high shear
    d["time"] = pd.to_datetime(d.time).dt.tz_localize("UTC")
    d = d.drop(columns=["u100", "v100"]).sort_values(["loc", "time"])
    d.to_parquet(DATA / "era5_sites.parquet", index=False)
    print(d.describe().T.round(3))

if __name__ == "__main__":
    main()

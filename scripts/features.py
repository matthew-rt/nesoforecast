"""Hourly modelling table per BMU: metered output (clean-hour mask) joined to weather at the site, from either
ERA5 (hindcast, curve fitting) or archived ECMWF day-ahead forecasts (the operational input).

Both sources are mapped to the same columns so one curve / residual model can be driven by either:
  ws100, ws10, wd100, t2m, rho, sp, alpha  (+ sst, sst_t2m, blh, gust for ERA5; ws100_d2, wd100_d2 for NWP)
  ws_hub = k_gwa_100 * ws100 * (hub / 100) ** alpha        GWA horizontal downscaling + time-varying shear
  ws_eq  = ws_hub * (rho / RHO0) ** (1/3)                 IEC 61400-12 density-equivalent speed"""
import os, pathlib
import numpy as np, pandas as pd
import fit_curves

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RHO0 = 1.225
NWP_START = pd.Timestamp("2024-03-07", tz="UTC")  # first full day of the ECMWF previous-runs archive
# Which archived run feeds the day-ahead forecast. 1: previous_day1 (lead ~24-30 h), fresher than NESO could use for
# afternoon/evening hours (those runs start after the 10:00 D-1 deadline). 2: previous_day2 (lead ~48-54 h), every run
# available before the deadline - the conservative, strictly fair setting. Caches are kept separately per setting.
# "00z": the exact 00 UTC run of D-1 from ECMWF open data (/Users/matt/energydata/ecmwf_ifs025_00z_sites), the
# like-for-like setting for comparison with NESO: every hour of D comes from a run available before the deadline.
NWP_DAY = os.environ.get("NWP_DAY", "1")
OPEN_00Z = pathlib.Path("/Users/matt/energydata/ecmwf_ifs025_00z_sites/days")

# UNC_TARGET=1: train on curtailed (BOA) hours too, with the estimated unconstrained output (unconstrained.py) as the
# target. Excluding them biases the models low: NESO curtails when output is high, so for heavily curtailed units the
# hours that escape curtailment at a high forecast are the ones where the wind came in low (CF 0.60 vs 0.82 truth).
UNC_TARGET = os.environ.get("UNC_TARGET", "0") == "1"

# TRAIN_END=YYYY-MM-DD: train on data before this date instead of the default 2025-10-01 (monthly retraining runs,
# retrain_monthly.py); outputs get a _t<date> suffix so the default run's caches are untouched.
TRAIN_END_OVERRIDE = os.environ.get("TRAIN_END")

def tag():
    return (("" if NWP_DAY == "1" else f"_d{NWP_DAY}") + ("_u" if UNC_TARGET else "")
            + (f"_t{TRAIN_END_OVERRIDE}" if TRAIN_END_OVERRIDE else ""))

def cache(name):
    """Cache path for outputs that depend on NWP_DAY / UNC_TARGET / TRAIN_END."""
    p = pathlib.Path(name)
    return DATA / "cache" / f"{p.stem}{tag()}{p.suffix}"

def site_meta():
    m = pd.read_parquet(DATA / "site_meta.parquet")
    b = pd.read_parquet(DATA / "bmu_loc.parquet").set_index("bmu")["loc"]
    return m, b

def bm_hourly():
    """Hourly metered output + clean mask per BMU (cached: the half-hour -> hour groupby takes a while)."""
    f = DATA / "cache" / "bm_hourly.parquet"
    if f.exists() and f.stat().st_mtime > (DATA / "bm_hh.parquet").stat().st_mtime:
        return pd.read_parquet(f)
    h = fit_curves.clean(fit_curves.hourly_bm())
    h["cf"] = h.metered / h.cap_eff
    f.parent.mkdir(exist_ok=True)
    h.to_parquet(f, index=False)
    return h

def rho_from_rh(t_c, rh, p_hpa):
    """Moist-air density from 2 m temperature (C), relative humidity (%) and surface pressure (hPa)."""
    e = rh / 100 * 610.94 * np.exp(17.625 * t_c / (t_c + 243.04))
    t = t_c + 273.15
    return (p_hpa * 100 - e) / (287.05 * t) + e / (461.5 * t)

def shear(ws100, ws10):
    return (np.log(np.clip(ws100, 0.5, None) / np.clip(ws10, 0.5, None)) / np.log(10)).clip(-0.1, 0.6)

BOUNDARIES = (10, 22)  # first hour (UTC) of each ERA5 12-h 4D-Var window (09-21, 21-09 UTC)

def smooth_assimilation(e, cols=("ws100", "ws10"), a=1.0, win=12, shrink="wiener"):
    """Remove the step at each ERA5 assimilation-window boundary by spreading it as a linear drift.

    Jump at boundary B (per site, per day): J = [x(B) - x(B-1)] - mean(x(B-1) - x(B-2), x(B+1) - x(B)), i.e.
    the change across B in excess of the neighbouring hourly changes. J also contains real weather, so per
    site and boundary hour it is shrunk toward its mean: J' = mu + s (J - mu), s = 1 - var(J off-boundary) /
    var(J at B) (shrink="wiener": least-squares estimate of the artefact) or s = 1 - sd ratio ("match": leaves
    the boundary hour as variable as other hours). The mean step mu is all artefact and is removed in full. A share `a` of J' is spread back over
    the preceding window as a ramp (the free-running model drifting away from the next analysis), 1 - a
    forward over the following window. a=1 corrects drift only backwards; a=0.5 splits it."""
    e, shrink_mode = e.sort_values(["loc", "time"]).copy(), shrink
    for c in cols:
        w = e.pivot(index="time", columns="loc", values=c)
        w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="h"))
        x = w.values
        corr = np.zeros_like(x)
        hrs = w.index.hour.values
        jall = np.full_like(x, np.nan)  # jump statistic at every hour
        jall[2:-1] = (x[2:-1] - x[1:-2]) - 0.5 * ((x[1:-2] - x[:-3]) + (x[3:] - x[2:-1]))
        far = ~np.isin(hrs, [b + d for b in BOUNDARIES for d in (-1, 0, 1)])
        var_bg = np.nanvar(jall[far], axis=0)
        shrink, mode = {}, shrink_mode
        for b in BOUNDARIES:
            jb = jall[hrs == b]
            mu = np.nanmean(jb, axis=0)
            r = var_bg / np.nanvar(jb, axis=0)
            shrink[b] = (mu, np.clip(1 - (r if mode == "wiener" else np.sqrt(r)), 0, 1))
        for i in np.where(np.isin(hrs, BOUNDARIES))[0]:
            if i < 2 or i + 1 >= len(x):
                continue
            mu, sh = shrink[hrs[i]]
            J = np.nan_to_num(mu + sh * (jall[i] - mu))
            k = np.arange(1, win + 1)  # back ramp: k=win is hour B-1
            lo = max(0, i - win)
            corr[lo:i] += a * J * (k[win - (i - lo):] / win)[:, None]
            j = np.arange(win)  # forward ramp: j=0 is hour B
            hi = min(len(x), i + win)
            corr[i:hi] -= (1 - a) * J * (1 - j[:hi - i] / win)[:, None]
        sm = pd.DataFrame(np.clip(x + corr, 0, None), index=w.index, columns=w.columns).stack()
        sm.index.names = ["time", "loc"]
        e = e.drop(columns=c).merge(sm.rename(c).reset_index(), on=["time", "loc"], how="left")
    return e

def era5(smooth=0.5, shrink="wiener"):
    """smooth: None (raw ERA5) or the backward share `a` passed to smooth_assimilation. Default 0.5 (split):
    it removes the hour-of-day step in curve residuals at no cost to curve skill; a=1 (drift spread backwards
    only) made curves worse, consistent with 4D-Var fitting observations across the whole window."""
    e = pd.read_parquet(DATA / "era5_sites.parquet")
    if smooth is not None:
        e = smooth_assimilation(e, a=smooth, shrink=shrink)
    e["alpha"] = shear(e.ws100, e.ws10)
    return e

def wdir(u, v):
    return np.degrees(np.arctan2(-u, -v)) % 360  # meteorological: direction the wind comes from

def nwp_00z():
    """Exact-run forecasts, hourly. Each delivery day's 11 three-hourly steps (lead 21..51 h of the D-1 00Z run,
    valid D-1 21:00 .. D+1 03:00) are interpolated to hours within that run; neighbouring-hour features (t-3..t+3)
    are built within the same run, so no value comes from a later run. Only hours of the local delivery day D are kept."""
    return _hourly_from_steps(pd.concat(pd.read_parquet(f) for f in sorted(OPEN_00Z.glob("*.parquet"))))

TZ = "Europe/London"

def issue_time(t):
    """Day-ahead issue time for delivery hours t (UTC series): 09:00 London time on D-1, D = local day of t.
    NESO's day-ahead WINDFOR is published at 09:30 local (10:00 deadline), so no input may be later than this."""
    d = t.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize() - pd.Timedelta("15h")
    return d.dt.tz_localize(TZ).dt.tz_convert("UTC")

GRID = pathlib.Path("/Users/matt/energydata/nwp_gb_grid")

def nwp_grid(model):
    """Exact-run forecasts for one model from the gridded store (fetch_nwp_grid.py), at the current site list.
    Speeds are interpolated as speeds (bilinear), direction from interpolated u/v; then hourly within each run and
    neighbouring-hour features within the run, as in nwp_00z. Cached per model; rebuilt when sites or days change."""
    meta = pd.read_parquet(DATA / "site_meta.parquet")
    files = sorted((GRID / model).glob("*.npz"))
    f_cache = DATA / "cache" / f"nwp_grid_{model}.parquet"
    stamp = f"v2:{len(files)}:{files[-1].name if files else ''}:{len(meta)}"  # v2: local delivery days
    if f_cache.exists() and pd.read_parquet(f_cache, columns=["stamp"]).stamp.iloc[0] == stamp:
        return pd.read_parquet(f_cache).drop(columns="stamp")
    z = np.load(files[0])
    lat, lon = z["lat"], z["lon"]
    y = (lat[0] - meta.lat.values) / 0.25; x = (meta.lon.values - lon[0]) / 0.25
    j0, i0 = np.floor(y).astype(int), np.floor(x).astype(int); fy, fx = y - j0, x - i0
    def bil(a):  # a (..., 49, 47) -> (..., n_sites)
        return ((1 - fy) * (1 - fx) * a[..., j0, i0] + (1 - fy) * fx * a[..., j0, i0 + 1]
                + fy * (1 - fx) * a[..., j0 + 1, i0] + fy * fx * a[..., j0 + 1, i0 + 1])
    parts = []
    for f in files:
        z = np.load(f)
        A, P = z["main"].astype(float), z["prev"].astype(float)  # (11, 7, ...), (11, 2, ...)
        ws100, ws10 = bil(np.hypot(A[:, 0], A[:, 1])), bil(np.hypot(A[:, 2], A[:, 3]))
        cols = {"ws100": ws100, "u100": bil(A[:, 0]), "v100": bil(A[:, 1]), "ws10": ws10, "t2m": bil(A[:, 4]) - 273.15,
                "d2m": bil(A[:, 5]) - 273.15, "sp": bil(A[:, 6]) / 100, "ws100_prev": bil(np.hypot(P[:, 0], P[:, 1])),
                "u100_prev": bil(P[:, 0]), "v100_prev": bil(P[:, 1])}
        D = pd.Timestamp(f.stem)
        n = len(meta)
        d = pd.DataFrame({k: v.T.ravel() for k, v in cols.items()})  # site-major, 11 steps each
        d["loc"] = np.repeat(meta.index.values, 11)
        d["lead_h"] = np.tile(z["steps"], n)
        d["delivery_day"] = D
        parts.append(d)
    out = _hourly_from_steps(pd.concat(parts, ignore_index=True))
    out.assign(stamp=stamp).to_parquet(f_cache, index=False)
    return out

def _hourly_from_steps(d):
    """Shared by nwp_00z / nwp_grid: 11 three-hourly steps per (loc, day) -> hourly D, features within the run."""
    cols = ["ws100", "u100", "v100", "ws10", "t2m", "d2m", "sp", "ws100_prev", "u100_prev", "v100_prev"]
    for c in cols:
        if c not in d:
            d[c] = np.nan
    d = d.sort_values(["loc", "delivery_day", "lead_h"])
    g = d.groupby(["loc", "delivery_day"]).size()
    g = g[g == 11]
    d = d.set_index(["loc", "delivery_day"]).loc[g.index].reset_index()
    A = d[cols].to_numpy(float).reshape(len(g), 11, len(cols))
    k = np.arange(31) // 3; f = (np.arange(31) % 3 / 3)[None, :, None]
    H = A[:, np.minimum(k, 10)] * (1 - f) + A[:, np.minimum(k + 1, 10)] * f
    w = H[:, :, 0]
    pos = np.arange(31)
    sh = lambda j: w[:, np.clip(pos + j, 0, 30)]  # neighbouring hours within the same run (clipped at its ends)
    feats = {f"ws_m{j}": sh(-j) for j in (1, 2, 3)} | {f"ws_p{j}": sh(j) for j in (1, 2, 3)}
    win = np.stack([sh(j) for j in range(-3, 4)], axis=2)
    feats["ws_roll7"], feats["ws_std7"] = win.mean(2), win.std(2, ddof=1)
    feats["ws_trend"] = sh(2) - sh(-2)
    n = pd.DataFrame({c: H[:, :, i].ravel() for i, c in enumerate(cols)} | {c: v.ravel() for c, v in feats.items()})
    day0 = pd.to_datetime(g.index.get_level_values("delivery_day"))
    day0 = day0.tz_localize(None) if day0.tz is not None else day0
    n["loc"] = np.repeat(g.index.get_level_values("loc").values, 31)
    n["time"] = pd.to_datetime(np.repeat(day0.values, 31) + np.tile(pos - 3, len(g)) * np.timedelta64(1, "h"), utc=True)
    # keep the hours of the LOCAL delivery day D (NESO's day-ahead unit): in BST that is D-1 23:00 .. D 22:00 UTC,
    # and 23 / 25 hours on clock-change days. All lie within the run's leads 21..51 h.
    local_day = n.time.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    n = n[(local_day == np.repeat(day0.values, 31)).values].reset_index(drop=True)
    n["wd100"], n["wd100_d2"] = wdir(n.u100, n.v100), wdir(n.u100_prev, n.v100_prev)
    n["ws100_d2"] = n.ws100_prev
    rh = 100 * np.exp(17.625 * n.d2m / (n.d2m + 243.04)) / np.exp(17.625 * n.t2m / (n.t2m + 243.04))
    n["rho"] = rho_from_rh(n.t2m, rh, n.sp)
    n["alpha"] = shear(n.ws100, n.ws10)
    return n.drop(columns=["u100", "v100", "u100_prev", "v100_prev", "ws100_prev", "d2m"]).dropna(subset=["ws100", "wd100"])

def nwp_blend():
    """ECMWF as the primary input, GFS alongside as *_gfs columns (speed, direction, shear, density, t+-k speeds)."""
    e, g = nwp_grid("ecmwf"), nwp_grid("gfs")
    keep = ["ws100", "ws10", "wd100", "rho", "alpha", "t2m", "ws_m2", "ws_m3", "ws_p2", "ws_p3"]
    g = g[["loc", "time"] + keep].rename(columns={c: f"{c}_gfs" for c in keep})
    return e.merge(g, on=["loc", "time"], how="left")

def nwp(model="ecmwf_ifs025", day=None):
    """Day-ahead forecast at each site. day=1 is the primary input (lead ~24-30 h); the other day's run is
    kept as ws100_d2/wd100_d2 so run-to-run disagreement can be used as an uncertainty signal."""
    day = day or NWP_DAY
    if str(day) == "00z":
        return nwp_00z()
    if str(day) in ("g_ecmwf", "g_gfs"):
        return nwp_grid(str(day)[2:])
    if str(day) == "g_blend":
        return nwp_blend()
    day = int(day)
    d = pd.concat(pd.read_parquet(f) for f in sorted((DATA / "raw" / "nwp").glob(f"{model}_*.parquet")))
    o = 3 - day
    out = pd.DataFrame({
        "loc": d["loc"], "time": pd.to_datetime(d.time).dt.tz_localize("UTC"),
        "ws100": d[f"wind_speed_100m_previous_day{day}"], "wd100": d[f"wind_direction_100m_previous_day{day}"],
        "ws10": d[f"wind_speed_10m_previous_day{day}"], "t2m": d[f"temperature_2m_previous_day{day}"],
        "sp": d[f"surface_pressure_previous_day{day}"],
        "ws100_d2": d[f"wind_speed_100m_previous_day{o}"], "wd100_d2": d[f"wind_direction_100m_previous_day{o}"]})
    if day == 2:  # the other run (day 1) is the fresher one NESO could not have used: no run-spread feature
        out["ws100_d2"], out["wd100_d2"] = np.nan, np.nan
    out["rho"] = rho_from_rh(out.t2m, d[f"relative_humidity_2m_previous_day{day}"].values, out.sp)
    out["alpha"] = shear(out.ws100, out.ws10)
    return out[out.time >= NWP_START].dropna(subset=["ws100", "wd100"])

def derive(w, meta):
    """Add hub-height and density-equivalent speed, and direction components, to a site-weather frame."""
    m = meta.reindex(w["loc"].values)
    w = w.copy()
    w["ws_hub"] = m.k_gwa_100.values * w.ws100 * (m.hub_est.values / 100) ** w.alpha
    w["ws_eq"] = w.ws_hub * (w.rho / RHO0) ** (1 / 3)
    if "ws100_gfs" in w:  # second model: same hub/GWA/density treatment
        hub = m.k_gwa_100.values * (m.hub_est.values / 100) ** w.alpha_gfs
        w["ws_eq_gfs"] = hub * w.ws100_gfs * (w.rho_gfs.fillna(RHO0) / RHO0) ** (1 / 3)
        w["gfs_lr"] = np.log(np.clip(w.ws_eq_gfs, 0.5, None) / np.clip(w.ws_eq, 0.5, None))  # GFS vs ECMWF speed
        for c in ("ws_m2", "ws_m3", "ws_p2", "ws_p3"):
            w[f"gfs_lr_{c[3:]}"] = np.log(np.clip(w[f"{c}_gfs"], 0.5, None) / np.clip(w[c], 0.5, None))
    th = np.radians(w.wd100)
    w["wd_sin"], w["wd_cos"] = np.sin(th), np.cos(th)
    return w

def table(source="era5", **kw):
    """BMU-hour table: bm_hourly joined to site weather (inner join on loc, time)."""
    meta, bloc = site_meta()
    w = derive(era5(**kw) if source == "era5" else nwp(**kw), meta)
    h = bm_hourly()
    h["loc"] = h.bmu.map(bloc)
    d = h.merge(w, on=["loc", "time"], how="inner")
    m = meta.reindex(d["loc"].values)
    for c in ["offshore", "hub_est", "rotor_est", "k_gwa_100", "lat", "lon"]:
        d[c] = m[c].values
    d["offshore"] = d.offshore.astype(bool)
    d["boa_orig"] = d.boa
    if UNC_TARGET:
        u = pd.read_parquet(DATA / "cache" / "unconstrained_hourly.parquet", columns=["bmu", "time", "cf_unc"])
        d = d.merge(u, on=["bmu", "time"], how="left")
        rep = d.boa & d.cf_unc.notna()
        d.loc[rep, "cf"] = d.cf_unc[rep]
        d.loc[rep, "clean"] = (d.time >= d.commissioned)[rep] & (d.mel.isna() | (d.mel >= 0.95 * d.cap_eff))[rep]
        d.loc[rep, "metered"] = (d.cf * d.cap_eff)[rep]  # so range checks downstream see the estimated output
        d.loc[rep, "boa"] = False
        d = d.drop(columns="cf_unc")
    return d

"""Day-ahead NWP over the GB box, stored as gridded cutouts so new wind sites need no re-download.

For each delivery day D and model, the run NESO could use for its 10:00 D-1 forecast:
  main  00 UTC run of D-1, steps 21..51 h every 3 h (valid D-1 21:00 .. D+1 03:00):
        u100 v100 u10 v10 t2m d2m sp
  prev  12 UTC run of D-2, steps 33..63 h, u100 v100 only (run-to-run spread)
Models:
  ecmwf  ECMWF IFS 0.25 deg open data, s3://ecmwf-forecasts (JSON .index per step)
  gfs    NOAA GFS 0.25 deg, s3://noaa-gfs-bdp-pds (text .idx per step; 00Z run complete ~04:30 UTC)
Only the needed GRIB messages are fetched (byte ranges; nearby messages merged into one request).
Output: /Users/matt/energydata/nwp_gb_grid/<model>/YYYY-MM-DD.npz with arrays
  main (11, 7, 49, 47), prev (11, 2, 49, 47), lat (49,), lon (47,), steps; float32, NaN where a field is missing.
Site values: nwp_grid_sites.py. Usage (needs eccodes, e.g. the pypsa-gb env):
  /Users/matt/anaconda3/envs/pypsa-gb/bin/python scripts/fetch_nwp_grid.py gfs 2024-03-07 2026-10-22"""
import json, pathlib, random, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests, eccodes

ROOT_OUT = pathlib.Path("/Users/matt/energydata/nwp_gb_grid")
LAT = np.arange(61.5, 49.49, -0.25)    # 49 rows, north -> south (as stored in both GRIBs)
LON = np.arange(-8.5, 3.01, 0.25)      # 47 cols
MAIN = ["u100", "v100", "u10", "v10", "t2m", "d2m", "sp"]
PREV = ["u100", "v100"]
STEPS_MAIN, STEPS_PREV = list(range(21, 52, 3)), list(range(33, 64, 3))
MERGE_GAP = 3_000_000
session = requests.Session()

MODELS = {
    "ecmwf": dict(
        base=lambda run, step: f"https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/{run:%Y%m%d}/{run:%H}z/ifs/0p25/oper/"
                               f"{run:%Y%m%d%H}0000-{step}h-oper-fc",
        grib=".grib2", index=".index",
        keys={"u100": "100u", "v100": "100v", "u10": "10u", "v10": "10v", "t2m": "2t", "d2m": "2d", "sp": "sp"}),
    "gfs": dict(
        base=lambda run, step: f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{run:%Y%m%d}/{run:%H}/atmos/"
                               f"gfs.t{run:%H}z.pgrb2.0p25.f{step:03d}",
        grib="", index=".idx",
        keys={"u100": "UGRD:100 m above ground", "v100": "VGRD:100 m above ground", "u10": "UGRD:10 m above ground",
              "v10": "VGRD:10 m above ground", "t2m": "TMP:2 m above ground", "d2m": "DPT:2 m above ground",
              "sp": "PRES:surface"}),
}

def get(url, **kw):
    for k in range(30):
        try:
            r = session.get(url, timeout=120, **kw)
            if r.status_code in (200, 206):
                return r
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(min(20, 0.25 * 2 ** k) * (0.5 + random.random()))  # S3 503 SlowDown is common: retry fast
    raise RuntimeError(f"gave up on {url}")

def index(model, text):
    """name -> (offset, length) for the fields we need."""
    want = {v: k for k, v in MODELS[model]["keys"].items()}
    out = {}
    if model == "ecmwf":
        for r in (json.loads(l) for l in text.splitlines() if l.startswith("{")):
            if r.get("levtype") == "sfc" and r["param"] in want:
                out[want[r["param"]]] = (r["_offset"], r["_length"])
    else:
        rows = [l.split(":") for l in text.splitlines() if l]
        offs = [int(r[1]) for r in rows] + [None]
        for i, r in enumerate(rows):
            key = f"{r[3]}:{r[4]}"
            if key in want and offs[i + 1] is not None:
                out[want[key]] = (offs[i], offs[i + 1] - offs[i])
    return out

def cutout(h):
    """Decode one GRIB message and cut the GB box (works for lon starting at 0 or 180)."""
    v = eccodes.codes_get_values(h).reshape(eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni"))
    lat0, lon0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
    d = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
    rows = np.rint((lat0 - LAT) / d).astype(int)
    cols = np.rint(((LON - lon0) % 360) / d).astype(int)
    return v[np.ix_(rows, cols)].astype(np.float32)

def fields(model, run, step, names):
    m = MODELS[model]
    base = m["base"](run, step)
    idx = get(base + m["index"])
    if idx is None:
        return None
    pos = index(model, idx.text)
    want = sorted((pos[n][0], pos[n][1], n) for n in names if n in pos)
    if not want:
        return None
    groups, cur = [], [want[0]]
    for w in want[1:]:
        if w[0] - (cur[-1][0] + cur[-1][1]) > MERGE_GAP:
            groups.append(cur)
            cur = [w]
        else:
            cur.append(w)
    groups.append(cur)
    out = {}
    for grp in groups:
        lo, hi = grp[0][0], grp[-1][0] + grp[-1][1]
        r = get(base + m["grib"], headers={"Range": f"bytes={lo}-{hi - 1}"})
        if r is None:
            continue
        for off, ln, n in grp:
            h = eccodes.codes_new_from_message(r.content[off - lo: off - lo + ln])
            out[n] = cutout(h)
            eccodes.codes_release(h)
    return out

def day(model, D):
    out = ROOT_OUT / model / f"{D:%Y-%m-%d}.npz"
    if out.exists():
        return
    main_run, prev_run = D - pd.Timedelta("1D"), D - pd.Timedelta("2D") + pd.Timedelta("12h")
    A = np.full((len(STEPS_MAIN), len(MAIN), len(LAT), len(LON)), np.nan, np.float32)
    P = np.full((len(STEPS_PREV), len(PREV), len(LAT), len(LON)), np.nan, np.float32)
    for i, (sm, sp_) in enumerate(zip(STEPS_MAIN, STEPS_PREV)):
        a = fields(model, main_run, sm, MAIN)
        if a is None:
            print("missing", model, D.date(), sm, flush=True)
            return
        for j, n in enumerate(MAIN):
            if n in a:
                A[i, j] = a[n]
        b = fields(model, prev_run, sp_, PREV) or {}
        for j, n in enumerate(PREV):
            if n in b:
                P[i, j] = b[n]
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, main=A, prev=P, lat=LAT, lon=LON, steps=np.array(STEPS_MAIN), run=str(main_run))
    tmp.rename(out)  # atomic
    print(model, D.date(), flush=True)

def safe_day(model, D):
    try:
        day(model, D)
    except Exception as e:  # one stubborn request must not stop the run; rerunning picks the day up again
        print("failed", model, D.date(), repr(e)[:200], flush=True)

if __name__ == "__main__":
    model = sys.argv[1]
    (ROOT_OUT / model).mkdir(parents=True, exist_ok=True)
    days = pd.date_range(sys.argv[2], sys.argv[3], freq="D")
    with ThreadPoolExecutor(int(sys.argv[4]) if len(sys.argv) > 4 else 12) as ex:
        list(ex.map(lambda D: safe_day(model, D), days))

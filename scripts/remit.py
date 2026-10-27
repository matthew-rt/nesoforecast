"""Hourly availability per BMU from REMIT unavailability messages (/Users/matt/SCORESdata/elexon_remit_wind).

Two series, as fraction of normal capacity (1 = fully available):
  avail_final  each event's last revision (ex-post truth; for cleaning curve fits)
  avail_da     the revision current at the day-ahead issue time, D-1 09:00 London (no later information), for forecasting
  avail_now    the revision current at that hour itself (real-time state; at the issue hour = outage already running)
Dismissed revisions withdraw the event. Where a revision carries an outageProfile, its piecewise capacities are used.
Overlapping events on one BMU combine as the largest unavailability within the hour (time-weighted per event)."""
import ast, glob, json
import numpy as np, pandas as pd
import features as F

SRC = "/Users/matt/SCORESdata/elexon_remit_wind/messages"
BMUNITS = "/Users/matt/code/nesoforecast/data/raw/sites/elexon_bmunits.json"

def messages():
    m = pd.concat(pd.read_parquet(f) for f in glob.glob(f"{SRC}/*.parquet")).drop_duplicates("id")
    for c in ["publishTime", "eventStartTime", "eventEndTime"]:
        m[c] = pd.to_datetime(m[c], utc=True, errors="coerce")  # some messages carry typo dates (e.g. year 1020)
    m = m.dropna(subset=["publishTime", "eventStartTime", "eventEndTime"])
    m = m[(m.eventStartTime.dt.year >= 2015) & (m.eventEndTime.dt.year <= 2035)]
    m = m.sort_values(["mrid", "revisionNumber", "publishTime"])
    m["next_publish"] = m.groupby("mrid").publishTime.shift(-1).fillna(pd.Timestamp("2100-01-01", tz="UTC"))
    m["final"] = m.next_publish.dt.year == 2100
    m["bmu"] = bmu_ids(m)
    return m

def bmu_ids(m):
    """Elexon BMU id per message. assetId is usually the BMU (T_HOWAO-1) but some participants give the EIC
    (48W00000FALGW-1T) or a free-text name (RAMP-1); fall back to the affected unit's EIC, then T_<affectedUnit>."""
    b = json.load(open(BMUNITS))
    b = pd.DataFrame(b if isinstance(b, list) else b["data"])
    known, by_eic = set(b.elexonBmUnit), b.dropna(subset=["eic"]).set_index("eic").elexonBmUnit
    t_unit = "T_" + m.affectedUnit.fillna("").str.upper()
    out = m.assetId.where(m.assetId.isin(known))
    out = out.fillna(m.affectedUnitEIC.map(by_eic)).fillna(m.assetId.map(by_eic))
    return out.fillna(t_unit.where(t_unit.isin(known)))

def segments(m):
    """One row per (revision, constant-capacity segment): bmu, start, end, frac, publish, next_publish, final."""
    rows = []
    for r in m.itertuples():
        if r.eventStatus == "Dismissed" or not isinstance(r.bmu, str):
            continue
        norm = r.normalCapacity if r.normalCapacity and r.normalCapacity > 0 else np.nan
        prof = []
        if isinstance(getattr(r, "outageProfile", None), str) and r.outageProfile.startswith("["):
            try:
                prof = ast.literal_eval(r.outageProfile)
            except (ValueError, SyntaxError):
                prof = []
        segs = [(p["startTime"], p["endTime"], p["capacity"]) for p in prof] or \
               [(r.eventStartTime, r.eventEndTime, r.availableCapacity)]
        for s, e, cap in segs:
            rows.append((r.bmu, s, e, cap / norm, r.publishTime, r.next_publish, r.final))
    s = pd.DataFrame(rows, columns=["bmu", "start", "end", "frac", "publish", "next_publish", "final"])
    s["start"], s["end"] = pd.to_datetime(s.start, utc=True, errors="coerce"), pd.to_datetime(s.end, utc=True, errors="coerce")
    s = s.dropna(subset=["start", "end"])
    s = s[(s.start.dt.year >= 2015) & (s.end.dt.year <= 2035)]
    s["frac"] = s.frac.clip(0, 1)
    return s[(s.end > s.start) & s.frac.notna() & (s.frac < 1)]

def hourly(s):
    """Expand segments to hours with the fraction of each hour covered."""
    s = s.reset_index(drop=True)
    h0, h1 = s.start.dt.floor("h"), (s.end - pd.Timedelta("1ns")).dt.floor("h")
    n = ((h1 - h0) / pd.Timedelta("1h")).astype(int) + 1
    i = np.repeat(s.index.values, n)
    off = np.concatenate([np.arange(k) for k in n])
    t = h0.values[i] + off * np.timedelta64(1, "h")
    x = s.loc[i].reset_index(drop=True)
    x["time"] = pd.to_datetime(t, utc=True)
    a = np.maximum(x.start, x.time)
    b = np.minimum(x.end, x.time + pd.Timedelta("1h"))
    x["cover"] = (b - a) / pd.Timedelta("1h")
    x["loss"] = x.cover * (1 - x.frac)
    return x

def availability():
    s = segments(messages())
    x = hourly(s)
    final = x[x.final].groupby(["bmu", "time"]).loss.max().rename("loss_final")
    issue = F.issue_time(x.time)  # D-1 09:00 London
    known = x[(x.publish <= issue) & (issue < x.next_publish)]
    da = known.groupby(["bmu", "time"]).loss.max().rename("loss_da")
    now = x[(x.publish <= x.time) & (x.time < x.next_publish)].groupby(["bmu", "time"]).loss.max().rename("loss_now")
    out = pd.concat([final, da, now], axis=1).fillna(0)
    out["avail_final"], out["avail_da"], out["avail_now"] = 1 - out.loss_final, 1 - out.loss_da, 1 - out.loss_now
    return out[["avail_final", "avail_da", "avail_now"]].reset_index()

if __name__ == "__main__":
    a = availability()
    a.to_parquet("/Users/matt/code/nesoforecast/data/remit_hourly.parquet", index=False)
    print(a.describe().round(3).to_string())
    print("BMU-hours with any outage:", len(a), " BMUs:", a.bmu.nunique())
    print("share of outage hours already known day-ahead (avail_da < 1):", round((a.avail_da < 1).mean(), 3))

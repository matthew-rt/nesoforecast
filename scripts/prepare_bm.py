"""Per-BMU half-hourly table for power-curve fitting: data/bm_hh.parquet.
Columns: bmu, start (UTC), metered (MW), pn (MW), mel (MW), boa (any accepted bid/offer active in the SP),
boa_level (lowest accepted level in the SP, MW)."""
import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw" / "bm", ROOT / "data"
RAW_E = ROOT / "data" / "raw" / "bm_embedded"  # E_ units (fetch_bm.py --embedded)
GLF_B1610 = pathlib.Path("/Users/matt/code/genloadfactor/data/raw/b1610")  # existing per-BMU-year B1610 CSVs
TZ = "Europe/London"
KEY = ["bmu", "date", "sp"]
RUN_RANK = {"II": 0, "SF": 1, "R1": 2, "R2": 3, "R3": 4, "RF": 5, "DF": 6}  # later settlement runs supersede earlier

def load(ds, cols):
    files = sorted((RAW / ds).glob("*.parquet")) + sorted((RAW_E / ds).glob("*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    return d.rename(columns={"bmUnit": "bmu", "settlementDate": "date", "settlementPeriod": "sp"})

def read_csv_or_none(f, cols):
    try:
        return pd.read_csv(f, usecols=cols)
    except pd.errors.EmptyDataError:  # unit-years before a unit existed are empty files
        return None

def b1610():
    """Metered output: our monthly pulls plus genloadfactor's CSVs for the E_ units; keep the latest run per SP."""
    cols = ["bmUnit", "settlementDate", "settlementPeriod", "settlementRunType", "quantity"]
    d = load("B1610", cols)
    g = pd.concat([read_csv_or_none(f, cols) for f in sorted(GLF_B1610.glob("E_*_20*.csv")) if int(f.stem[-4:]) >= 2021])
    g = g.rename(columns={"bmUnit": "bmu", "settlementDate": "date", "settlementPeriod": "sp"})
    d = pd.concat([d, g[g.bmu.isin(set(pd.read_csv(OUT / "sites.csv").elexonBmUnit))]])
    d["rank"] = d.settlementRunType.map(RUN_RANK).fillna(-1)
    return d.sort_values("rank").drop_duplicates(KEY, keep="last")

def segment_mean(d):
    """Time-weighted mean of linear level segments within each SP (PN / MEL are piecewise-linear)."""
    d = d.drop_duplicates(["bmu", "timeFrom", "timeTo"], keep="last")
    dur = (pd.to_datetime(d.timeTo) - pd.to_datetime(d.timeFrom)).dt.total_seconds()
    d = d.assign(w=dur, wl=dur * (d.levelFrom + d.levelTo) / 2)
    g = d[d.w > 0].groupby(KEY)[["w", "wl"]].sum()
    return g.wl / g.w

def main():
    b = b1610().set_index(KEY)
    out = pd.DataFrame({"metered": 2 * b.quantity})  # MWh per half hour -> mean MW

    pn = load("PN", ["bmUnit", "settlementDate", "settlementPeriod", "timeFrom", "timeTo", "levelFrom", "levelTo"])
    out["pn"] = segment_mean(pn)

    mel = load("MELS", ["bmUnit", "settlementDate", "settlementPeriod", "timeFrom", "timeTo", "levelFrom",
                        "levelTo", "notificationSequence"])
    mel = mel.sort_values("notificationSequence")  # keep the latest resubmission for each segment
    out["mel"] = segment_mean(mel)

    out = out.reset_index()
    midnight = pd.to_datetime(out.date).dt.tz_localize(TZ).dt.tz_convert("UTC")
    out["start"] = midnight + pd.to_timedelta((out.sp - 1) * 30, unit="min")

    # acceptances can cross midnight (SP numbers wrap), so map them to half-hours by UTC time
    a = load("BOALF", ["bmUnit", "timeFrom", "timeTo", "levelFrom", "levelTo"]).reset_index(drop=True)
    t0 = pd.to_datetime(a.timeFrom).dt.floor("30min"); t1 = pd.to_datetime(a.timeTo)
    n = np.ceil((t1 - t0) / pd.Timedelta("30min")).clip(lower=1).astype(int)
    a = a.loc[a.index.repeat(n)]
    a["start"] = t0.loc[a.index] + pd.to_timedelta(a.groupby(level=0).cumcount() * 30, unit="min")
    a["lvl"] = a[["levelFrom", "levelTo"]].min(axis=1)
    g = a.groupby(["bmu", "start"]).lvl.min()
    idx = pd.MultiIndex.from_frame(out[["bmu", "start"]])
    out["boa"] = idx.isin(g.index)
    out["boa_level"] = g.reindex(idx).values

    out = out.drop(columns=["date", "sp"]).sort_values(["bmu", "start"])
    out.to_parquet(OUT / "bm_hh.parquet", index=False)
    print(out.describe(include="all").T)
    print("frac SPs with BOA:", out.boa.mean().round(4), " PN coverage:", out.pn.notna().mean().round(4),
          " MEL coverage:", out.mel.notna().mean().round(4))
    return out

if __name__ == "__main__":
    main()

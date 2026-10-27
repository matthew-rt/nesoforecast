"""Build aligned hourly forecast-vs-outturn tables for demand and wind."""
import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data"
TZ = "Europe/London"

def sp_start_utc(date, sp):
    """UTC start of a GB settlement period (handles 46/50-period clock-change days)."""
    midnight = pd.to_datetime(date).dt.tz_localize(TZ).dt.tz_convert("UTC")
    return midnight + pd.to_timedelta((sp - 1) * 30, unit="min")

def demand():
    d = pd.read_csv(RAW / "neso_da_perf.csv").drop_duplicates(["Date", "Settlement_Period"])
    d["start"] = sp_start_utc(d.Date, d.Settlement_Period)
    # TRIAD-corrected outturn is NESO's own basis: the forecast does not try to predict triad avoidance
    d = d.rename(columns={"Demand_Forecast": "fc", "TRIAD_Avoidance_Corrected_Demand_Outturn": "act"})
    d["publish"] = pd.to_datetime(d.Publish_Datetime).dt.tz_localize(TZ, ambiguous="NaT").dt.tz_convert("UTC")
    hh = d[["start", "fc", "act", "publish"]].copy()
    # naive benchmark: same settlement period one week earlier (known at D-1)
    a = hh.set_index("start").act
    hh["naive"] = a.reindex(hh.start - pd.Timedelta("7D")).values
    hh.to_parquet(OUT / "demand_hh.parquet")
    g = hh.assign(start=hh.start.dt.floor("h")).groupby("start")[["fc", "act", "naive"]]
    hr = g.mean()[g.count().min(axis=1) == 2]  # only hours with both half-hours present
    hr.reset_index().to_parquet(OUT / "demand_hourly.parquet")
    return hr

def curtailment_hh():
    """Total accepted BOA volume on wind BMUs per settlement period, MW (negative = curtailed)."""
    b = pd.concat([pd.read_csv(f) for f in sorted(RAW.glob("wind_boa_*.csv"))])
    b["Date"] = pd.to_datetime(b.Date, format="mixed").dt.strftime("%Y-%m-%d")
    b = b.groupby(["Date", "Settlement_Period"]).BOA_Volume.sum().reset_index()
    b["start"] = sp_start_utc(b.Date, b.Settlement_Period)
    return b.groupby("start").BOA_Volume.sum()

def fuelhh_wind():
    """FUELHH wind per settlement period, MW, with feed outages set to NaN. The published series has ~80 periods
    (mostly 10:00-12:00 UTC) at zero or a fraction of the true value, while unit-level B1610 metering shows normal
    output: FUELHH is ~1.05x the sum of our B1610 units (P1 0.96), so a period below 0.8x that sum is a gap."""
    f = pd.read_parquet(RAW / "fuelhh.parquet")
    f["start"] = pd.to_datetime(f.startTime)
    s = f.drop_duplicates("start").set_index("start").generation.sort_index().asfreq("30min").astype(float)
    b = pd.read_parquet(OUT / "bm_hh.parquet", columns=["start", "metered"]).groupby("start").metered.sum()
    b.index = pd.to_datetime(b.index, utc=True)
    b = b.reindex(s.index)
    bad = (b > 300) & (s < 0.8 * b)
    print(f"FUELHH wind: {int(bad.sum())} settlement periods dropped as feed outages, {int(s.isna().sum())} missing")
    return s.mask(bad)

def wind(align, add_back_boa=False):
    w = pd.read_parquet(RAW / "windfor.parquet")
    w["publish"] = pd.to_datetime(w.publishTime); w["start"] = pd.to_datetime(w.startTime)
    w = w.drop_duplicates(["publish", "start"])
    # day-ahead = latest forecast published at or before 10:00 London time on D-1 (local day D)
    # local midnight of D minus 14h = 10:00 on D-1, localised after the arithmetic so DST is right
    cutoff = (w.start.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize() - pd.Timedelta("14h")) \
        .dt.tz_localize(TZ).dt.tz_convert("UTC")
    w = w[w.publish <= cutoff]
    w = w.sort_values("publish").groupby("start").tail(1)[["start", "publish", "generation"]] \
        .rename(columns={"generation": "fc"})

    act_hh = fuelhh_wind()
    if add_back_boa:  # unconstrained outturn: what wind would have generated without BM curtailment
        act_hh = act_hh - curtailment_hh().reindex(act_hh.index).fillna(0)
    if align == "hour":      # mean of [h, h+1h)
        act = act_hh.rolling(2).mean().shift(-1)
    elif align == "centred":  # mean of [h-30m, h+30m)
        act = act_hh.rolling(2).mean()
    else:                     # period starting at h only
        act = act_hh
    w["act"] = act.reindex(w.start).values
    # naive benchmark: persistence of last half-hour outturn known at publish time
    last = act_hh.reindex(w.publish.dt.floor("30min") - pd.Timedelta("30min")).values
    w["naive"] = last
    return w.dropna(subset=["fc", "act"])

if __name__ == "__main__":
    demand()
    for a in ["hour", "centred", "sp"]:
        w = wind(a)
        print(a, len(w), "MAE", round((w.fc - w.act).abs().mean(), 1))
    w = wind("centred")
    w["lead_h"] = (w.start - w.publish).dt.total_seconds() / 3600
    print(w.lead_h.describe())
    w.to_parquet(OUT / "wind_hourly.parquet")
    u = wind("centred", add_back_boa=True)
    print("unconstrained MAE", round((u.fc - u.act).abs().mean(), 1), "bias", round((u.fc - u.act).mean(), 1))
    u.to_parquet(OUT / "wind_unconstrained_hourly.parquet")

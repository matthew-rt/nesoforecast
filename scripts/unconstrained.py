"""Estimated unconstrained output for curtailed (BOA) unit-hours -> data/cache/unconstrained_hourly.parquet.

Curtailed volumes are measured from PN, and PN overstates what wind units produce: in hours with no BOA, metered/PN
is 0.90-0.94 for Scottish units at high PN. So PN is a biased stand-in for output in curtailed hours. Three estimates
were compared on held-out no-BOA hours of the units curtailed >5% of the time (scratch unc_proxy.py, 2026-10-25):
                         bias    MAE    MAE at PN>60%
  raw PN                +0.025  0.100   0.174 (bias +0.112)
  PN x r (unit, PN band) -0.002  0.094   0.152
  ERA5 hindcast curve    0.000  0.090   0.138
  learned PN+ERA5+MEL    0.000  0.079   0.122  <- used here
The learned estimate: LightGBM (L2) on PN/capacity, the unit's ERA5 directional power curve (fitted on its clean
hours), MEL/capacity, ERA5 hub speed, hour, unit and time (months since 2021: PN accuracy has improved), trained on
no-BOA hours; everything out-of-fold by month
(4 interleaved folds), so no hour's estimate uses that month. ERA5 is the reanalysis of the actual weather; it is used
only to reconstruct the target, never as a forecast input."""
import numpy as np, pandas as pd, lightgbm as lgb
import features as F, fit_curves as fc

# t_months: PN accuracy changes over time (uncurtailed overstatement at PN>60% fell from 0.13-0.17 of capacity in
# 2021-23 to ~0.06 in 2025H2-2026H1, after NESO's FPN-accuracy guidance), so the PN -> output mapping may drift
FEATS = ["pn_cf", "era", "mel_cf", "ws_hub", "hour", "bmu_c", "t_months"]

def main():
    e = F.table("era5")
    boa_units = e.groupby("bmu").boa.any()
    e = e[e.bmu.isin(boa_units[boa_units].index) & (e.time >= e.commissioned) & e.pn.notna() & e.metered.notna()].copy()
    e["v"] = e.ws_hub * (e.rho / F.RHO0) ** (1 / 3)
    e["pn_cf"], e["mel_cf"] = e.pn / e.cap_eff, (e.mel / e.cap_eff).clip(0, 1.2)
    e["fold"] = (e.time.dt.year * 12 + e.time.dt.month) % 4
    e["era"] = np.nan
    for b, x in e.groupby("bmu"):
        for k in range(4):
            c = x[x.clean & (x.fold != k)]
            if len(c) < 800:
                continue
            m = fc.fit_site(c.v.values, c.wd100.values, c.cf.values, n_harm=4)
            t = x.index[x.fold == k]
            e.loc[t, "era"] = fc.predict_site(m, x.loc[t, "v"].values, x.loc[t, "wd100"].values)
    e["hour"], e["bmu_c"] = e.time.dt.hour, pd.Categorical(e.bmu)
    e["t_months"] = (e.time.dt.year - 2021) * 12 + e.time.dt.month
    e["cf_unc"] = np.nan
    for k in range(4):
        tr = e[(e.fold != k) & ~e.boa]
        m = lgb.train(dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=8),
                      lgb.Dataset(tr[FEATS], tr.cf), 400)
        t = (e.fold == k) & e.boa
        e.loc[t, "cf_unc"] = m.predict(e.loc[t, FEATS])
    out = e[e.boa][["bmu", "time", "cf_unc", "pn_cf", "era"]].copy()
    half = out.time.dt.year.astype(str) + np.where(out.time.dt.month <= 6, "H1", "H2")
    print("mean CF in curtailed hours by half-year (estimate / ERA5 curve / raw PN):")
    print(out.groupby(half)[["cf_unc", "era", "pn_cf"]].mean().round(3).T.to_string())
    out["cf_unc"] = out.cf_unc.clip(0, 1.1)
    out.to_parquet(F.DATA / "cache" / "unconstrained_hourly.parquet", index=False)
    print(f"{len(out):,} curtailed unit-hours, {out.bmu.nunique()} BMUs; mean CF: estimate {out.cf_unc.mean():.3f}, "
          f"raw PN {out.pn_cf.mean():.3f}, ERA5 curve {out.era.mean():.3f}")

if __name__ == "__main__":
    main()

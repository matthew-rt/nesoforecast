"""Fleet day-ahead benchmark: our bottom-up forecast vs NESO WINDFOR, test year 2025-10 -> 2026-10.

Target (as in prepare.py / analyze.py): FUELHH wind (all BM-registered wind) with BM curtailment added back,
hourly, centred on the hour. WINDFOR = latest forecast published by 10:00 London time on D-1.
Ours = sum over BMUs of forecast CF x capacity:
  - units with a model: per-BMU hybrid network (fleet_nn.py) or LightGBM/curve baselines (nn_baselines.py)
  - units without one (too little training history, e.g. commissioning in the test year): pooled type curve on the
    forecast speed x MEL at issue time (MEL tracks the capacity built so far)
  - x k for FUELHH wind outside our unit set (supplier/other units), k estimated on the training period only."""
import numpy as np, pandas as pd
import features as F, fit_curves as fc, residual as R
from nn_data import frame

TEST_END = pd.Timestamp("2026-10-19", tz="UTC")

def unit_unconstrained():
    """Per-BMU hourly unconstrained actual: metered, or PN in hours with a BOA (generator's pre-curtailment plan)."""
    h = F.bm_hourly()
    u = h.metered.where(~h.boa, h.pn)
    return h.assign(unc_mw=u)[["bmu", "time", "unc_mw"]]

PN_BINS = [-0.01, 0.2, 0.4, 0.6, 0.8, 0.95, 5]

def pn_correction():
    """Hourly MW by which the PN-based curtailment add-back overstates unconstrained output.
    PN overstates what wind units actually produce (in hours with no BOA, metered/PN is 0.90-0.94 for Scottish units
    at high PN), and curtailed volume is measured from PN. Per BMU and PN level, r = sum(metered)/sum(PN) over no-BOA
    hours; in BOA hours the corrected unconstrained output is PN x r, so the add-back shrinks by PN x (1 - r)."""
    h = F.bm_hourly()
    h = h[h.pn.notna() & h.metered.notna() & (h.time >= h.commissioned)].copy()
    h["bin"] = pd.cut(h.pn / h.cap_eff, PN_BINS)
    free = h[~h.boa]
    r = (free.groupby(["bmu", "bin"], observed=True).metered.sum() / free.groupby(["bmu", "bin"], observed=True).pn.sum()).clip(0.5, 1.2)
    c = h[h.boa].join(r.rename("r"), on=["bmu", "bin"])
    c["r"] = c.r.fillna(1.0)
    return (c.pn * (1 - c.r)).groupby(c.time).sum()

def learned_correction():
    """Hourly MW by which the PN-based add-back exceeds the learned unconstrained estimate (unconstrained.py)."""
    u = pd.read_parquet(F.DATA / "cache" / "unconstrained_hourly.parquet", columns=["bmu", "time", "cf_unc"])
    h = F.bm_hourly()[["bmu", "time", "pn", "cap_eff", "boa"]]
    c = h[h.boa].merge(u, on=["bmu", "time"], how="inner")
    return (c.pn - c.cf_unc * c.cap_eff).groupby(c.time).sum()

def generic(a):
    """Pooled curve per type (offshore / onshore) on ECMWF speed, fitted on unconstrained training hours."""
    tr = a[a.unc & (a.time < R.TRAIN_END)]
    out = pd.Series(np.nan, index=a.index)
    for off, x in tr.groupby("offshore"):
        m = fc.fit_site(x.ws_eq.values, x.wd100.values, x.cf.values, n_harm=0)
        sel = a.offshore == off
        out[sel] = fc.predict_site(m, a.ws_eq[sel].values, a.wd100[sel].values)
    return out

def score(err):
    return pd.Series({"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()), "bias": err.mean()})

def boot_diff(e1, e2, n=2000, seed=0):
    """95% interval of MAE(e1) - MAE(e2), resampling whole weeks (errors are autocorrelated)."""
    d = pd.DataFrame({"a": e1.abs(), "b": e2.abs()}).dropna()
    wk = d.index.to_period("W")
    s, c = d.groupby(wk).sum(), d.groupby(wk).size()
    rng = np.random.default_rng(seed)
    r = [(s.a.values[i].sum() - s.b.values[i].sum()) / c.values[i].sum() for i in (rng.choice(len(c), len(c)) for _ in range(n))]
    return np.percentile(r, [2.5, 97.5])

def main():
    a = frame(all_rows=True)
    te = a[(a.time >= R.TRAIN_END) & (a.time < TEST_END)].copy()
    cap = te.cap_eff
    te["generic"] = generic(a).loc[te.index] * te.mel_da * cap  # MW
    nn = pd.read_parquet(F.cache("fleet_nn_preds.parquet"))
    b = pd.read_parquet(F.cache("nn_baselines.parquet"))
    te = te.merge(nn, on=["bmu", "time"], how="left").merge(b.drop(columns=["unc"]), on=["bmu", "time"], how="left")
    models = {"hybrid NN per BMU": "nn", "LightGBM L1 x MEL": "LGBM (L1) x MEL", "curve x MEL": "curve x MEL"}
    if F.cache("fleet_nn_preds_l2.parquet").exists():  # mean-targeting versions (squared-error loss)
        te = te.merge(pd.read_parquet(F.cache("fleet_nn_preds_l2.parquet")).rename(columns={"nn": "nn_l2"}), on=["bmu", "time"], how="left")
        models |= {"hybrid NN per BMU, mean (L2)": "nn_l2", "LightGBM L2 x MEL": "LGBM (L2) x MEL"}
    mw = {}
    for name, col in models.items():
        has = te[col].notna()
        mw[name] = (te[col] * te.cap_eff).where(has, te.generic)
    mw["average(NN, LightGBM L1)"] = (mw["hybrid NN per BMU"] + mw["LightGBM L1 x MEL"]) / 2
    if "NN ECMWF only" in mw:
        mw["average(NN ECMWF only, NN GFS only)"] = (mw["NN ECMWF only"] + mw["NN GFS only"]) / 2
    if F.NWP_DAY == "g_blend":  # simple two-model average: ECMWF-only and GFS-only networks, if both were run
        tag = "_u" if F.UNC_TARGET else ""
        fe, fg = F.DATA / "cache" / f"fleet_nn_preds_dg_ecmwf{tag}.parquet", F.DATA / "cache" / f"fleet_nn_preds_dg_gfs{tag}.parquet"
        if fe.exists() and fg.exists():
            for col, f in (("nn_e", fe), ("nn_g", fg)):
                te = te.merge(pd.read_parquet(f).rename(columns={"nn": col}), on=["bmu", "time"], how="left")
            models |= {"NN ECMWF only": "nn_e", "NN GFS only": "nn_g"}
    if "nn_l2" in te:
        mw["average(NN L2, LightGBM L2)"] = (mw["hybrid NN per BMU, mean (L2)"] + mw["LightGBM L2 x MEL"]) / 2
    # per-unit hybrid forecast (MW, generic curve for units without a network): scored on NESO's metric
    te[["bmu", "time"]].assign(mw=mw["hybrid NN per BMU"].values).to_parquet(F.cache("unit_forecast_mw.parquet"), index=False)
    fleet = pd.DataFrame({k: v.groupby(te.time).sum() for k, v in mw.items()})
    n_units = te.groupby("time").bmu.nunique()
    print(f"units forecast per hour: median {n_units.median():.0f}; with own model: {te.nn.notna().groupby(te.bmu).any().sum()} BMUs,"
          f" generic only: {(~te.nn.notna().groupby(te.bmu).any()).sum()} BMUs ({te[te.nn.isna()].groupby('time').generic.sum().mean():.0f} MW mean)")

    # coverage factor k: FUELHH-unconstrained / sum of our units' unconstrained actual, on the training period
    w = pd.read_parquet(F.DATA / "wind_unconstrained_hourly.parquet").set_index("start")
    wm = pd.read_parquet(F.DATA / "wind_hourly.parquet").set_index("start")
    uu = unit_unconstrained()
    uu = uu[uu.bmu.isin(set(a.bmu))]
    cov = uu.groupby("time").unc_mw.sum()
    trn = w.index[(w.index >= pd.Timestamp("2024-03-07", tz="UTC")) & (w.index < R.TRAIN_END)]
    k = w.act.reindex(trn).sum() / cov.reindex(trn).sum()
    km = (w.act.reindex(trn).groupby(trn.to_period("M")).sum() / cov.reindex(trn).groupby(trn.to_period("M")).sum())
    print(f"coverage factor k = {k:.4f} (training months range {km.min():.3f}-{km.max():.3f}); "
          f"test-period ratio (for reference only) {w.act.reindex(fleet.index).sum() / cov.reindex(fleet.index).sum():.4f}")

    idx = fleet.index.intersection(w.index)
    act, act_m = w.act.reindex(idx), wm.act.reindex(idx)
    res = {"NESO WINDFOR (D-1 10:00)": w.fc.reindex(idx)}
    for c in fleet:
        res[f"ours: {c}"] = fleet[c].reindex(idx) * k
    res["ours: hybrid NN, no coverage factor"] = fleet["hybrid NN per BMU"].reindex(idx)
    res["persistence (last outturn at publish)"] = w.naive.reindex(idx)
    print(f"\nTest hours: {len(idx):,} ({idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}); mean unconstrained outturn {act.mean():.0f} MW")
    t = pd.DataFrame({k_: score(v - act) for k_, v in res.items()}).T
    t["nMAE %"] = 100 * t.MAE / act.mean()
    base = res["NESO WINDFOR (D-1 10:00)"] - act
    t["MAE vs WINDFOR"] = t.MAE - t.loc["NESO WINDFOR (D-1 10:00)", "MAE"]
    t["95% range"] = [("" if k_.startswith("NESO") else "[%+.0f, %+.0f]" % tuple(boot_diff(v - act, base))) for k_, v in res.items()]
    pd.set_option("display.width", 220)
    print("\n== against UNCONSTRAINED outturn (what WINDFOR forecasts)"); print(t.round(1).to_string())
    addback = act - act_m
    clean = addback < 50
    tc = pd.DataFrame({k_: score(v[clean] - act[clean]) for k_, v in res.items()}).T
    tc["MAE vs WINDFOR"] = tc.MAE - tc.loc["NESO WINDFOR (D-1 10:00)", "MAE"]
    tc["95% range"] = [("" if k_.startswith("NESO") else "[%+.0f, %+.0f]" % tuple(boot_diff((v - act)[clean], base[clean]))) for k_, v in res.items()]
    print(f"\n== hours with essentially NO curtailment (add-back < 50 MW): {clean.sum():,} of {len(idx):,} hours, mean outturn {act[clean].mean():.0f} MW")
    print(tc.round(1).to_string())
    act_c = act - pn_correction().reindex(idx).fillna(0)
    tcc = pd.DataFrame({k_: score(v - act_c) for k_, v in res.items()}).T
    tcc["MAE vs WINDFOR"] = tcc.MAE - tcc.loc["NESO WINDFOR (D-1 10:00)", "MAE"]
    base_c = res["NESO WINDFOR (D-1 10:00)"] - act_c
    tcc["95% range"] = [("" if k_.startswith("NESO") else "[%+.0f, %+.0f]" % tuple(boot_diff(v - act_c, base_c))) for k_, v in res.items()]
    print(f"\n== all hours, PN-CORRECTED unconstrained outturn (add-back reduced by {(act - act_c).mean():.0f} MW on average)")
    print(tcc.round(1).to_string())
    act_l = act - learned_correction().reindex(idx).fillna(0)
    tl = pd.DataFrame({k_: score(v - act_l) for k_, v in res.items()}).T
    tl["MAE vs WINDFOR"] = tl.MAE - tl.loc["NESO WINDFOR (D-1 10:00)", "MAE"]
    base_l = res["NESO WINDFOR (D-1 10:00)"] - act_l
    tl["95% range"] = [("" if k_.startswith("NESO") else "[%+.0f, %+.0f]" % tuple(boot_diff(v - act_l, base_l))) for k_, v in res.items()]
    print(f"\n== all hours, LEARNED unconstrained outturn (PN + ERA5 + MEL estimate; add-back reduced by {(act - act_l).mean():.0f} MW on average)")
    print(tl.round(1).to_string())
    print("\n== against METERED outturn (FUELHH, after curtailment) - neither forecast includes curtailment")
    print(pd.DataFrame({k_: score(v - act_m) for k_, v in res.items()}).T.round(1).to_string())
    mo = pd.DataFrame({k_: (v - act).abs().groupby(idx.to_period("M")).mean() for k_, v in res.items()
                       if k_ in ("NESO WINDFOR (D-1 10:00)", "ours: hybrid NN per BMU", "ours: hybrid NN per BMU, mean (L2)",
                                 "ours: average(NN L2, LightGBM L2)")})
    print("\nMAE by month (MW):"); print(mo.round(0).T.to_string())
    print("months where hybrid NN beats WINDFOR:", int((mo["ours: hybrid NN per BMU"] < mo["NESO WINDFOR (D-1 10:00)"]).sum()), "of", len(mo))
    pd.DataFrame(res).assign(act_unconstrained=act, act_metered=act_m, act_pn_corrected=act_c, act_learned=act_l).to_parquet(
        F.DATA.parent / "output" / f"fleet_benchmark_hourly_day{F.NWP_DAY}{'_u' if F.UNC_TARGET else ''}"
        f"{'_t' + F.TRAIN_END_OVERRIDE if F.TRAIN_END_OVERRIDE else ''}.parquet")

if __name__ == "__main__":
    main()

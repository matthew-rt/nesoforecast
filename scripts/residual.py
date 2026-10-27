"""Step 3: ERA5-fitted directional power curve driven by ECMWF day-ahead, plus a LightGBM model of its residual.

Split (test year never touches any fit):
  curve fit   ERA5 clean hours            2021-01 -> 2025-09
  residual    ECMWF day-1 clean hours     2024-03 -> 2025-09  (every 5th week held out for early stopping)
  test        ECMWF day-1 clean hours     2025-10 -> 2026-10
Residual variants: one pooled model (BMU as a categorical), one model per BMU, and the pooled model fine-tuned
per BMU (continued boosting from the pooled model on that BMU's hours)."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
import features as F, fit_curves as fc

TRAIN_END = pd.Timestamp(F.TRAIN_END_OVERRIDE or "2025-10-01", tz="UTC")
TEST_END = pd.Timestamp("2026-11-01", tz="UTC")
GAMMA, N_HARM, LAM = 1 / 3, 4, 0.01  # from curve_cv.py
# day-of-year was tried and dropped: with ~1.5 years of training it memorises that year's weather (test MAE worse)
TEMPORAL = [f"ws_{s}{k}" for k in (1, 2, 3) for s in ("m", "p")] + ["ws_roll7", "ws_std7", "ws_trend"]
WEATHER = ["p_curve", "ws_eq", "ws100", "ws_spread", "wd_sin", "wd_cos", "wd_diff", "alpha", "rho", "t2m", "sp",
           "sst_t2m", "hour"] + TEMPORAL
SITE = ["bmu_c", "offshore", "hub_est", "rotor_est", "k_gwa_100", "lat", "lon", "cap_eff"]
if F.NWP_DAY == "g_blend":
    WEATHER = WEATHER + ["p_gfs", "ws_eq_gfs", "gfs_lr", "gfs_lr_m3", "gfs_lr_p3"]
BASE = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)
SITE_P = dict(BASE, num_leaves=15, min_data_in_leaf=100, lambda_l2=10.0)
TUNE_P = dict(BASE, learning_rate=0.02, num_leaves=7, min_data_in_leaf=200, lambda_l2=10.0)

def eq_speed(d):
    return (d.ws_hub * (d.rho.fillna(F.RHO0) / F.RHO0) ** GAMMA).values

def fit_curves(d):
    out = {}
    for b, x in d.groupby("bmu"):
        if len(x) >= 1000:
            m = fc.fit_site(eq_speed(x), x.wd100.values, x.cf.values, n_harm=N_HARM, lam=LAM)
            if m is not None:
                out[b] = m
    return out

def apply_curves(d, curves):
    p = np.full(len(d), np.nan)
    for b, i in d.groupby("bmu").indices.items():
        if b in curves:
            p[i] = fc.predict_site(curves[b], eq_speed(d.iloc[i]), d.wd100.values[i])
    return p

def sst_climatology():
    """Per-site SST by day of year from ERA5 (SST moves slowly; stands in until a forecast SST is available)."""
    e = pd.read_parquet(F.DATA / "era5_sites.parquet", columns=["loc", "time", "sst"])
    e["doy"] = e.time.dt.dayofyear
    c = e.groupby(["loc", "doy"]).sst.mean().unstack()
    c = pd.concat([c.iloc[:, -15:], c, c.iloc[:, :15]], axis=1).T.rolling(31, center=True, min_periods=1).mean().iloc[15:-15].T
    c.columns = pd.Index(range(1, c.shape[1] + 1), name="doy")
    return c.stack().rename("sst_clim")

def build():
    era = F.table("era5")
    era = era[era.clean & (era.time < TRAIN_END)].dropna(subset=["cf", "ws_hub", "wd100"])
    curves_era = fit_curves(era)
    n = F.table("nwp").reset_index(drop=True)
    n = n[n.time < TEST_END]
    n["p_era"] = apply_curves(n, curves_era)
    tr = n[n.clean & (n.time < TRAIN_END)].dropna(subset=["cf"])
    n["p_nwp"] = apply_curves(n, fit_curves(tr))  # same curve form, fitted on the forecast itself
    if "ws_eq_gfs" in n:  # blend mode: the same curve fitted on the second model's speed
        g = lambda x: x.assign(ws_hub=x.ws_eq_gfs, rho=F.RHO0, wd100=x.wd100_gfs.fillna(x.wd100))
        ok = n.ws_eq_gfs.notna()
        n["p_gfs"] = np.nan
        n.loc[ok, "p_gfs"] = apply_curves(g(n[ok]).reset_index(drop=True), fit_curves(g(tr[tr.ws_eq_gfs.notna()])))
    n["ws_spread"] = n.ws100 - n.ws100_d2
    n["wd_diff"] = np.abs((n.wd100 - n.wd100_d2 + 180) % 360 - 180)
    n["hour"] = n.time.dt.hour
    doy = n.time.dt.dayofyear
    n["doy_sin"], n["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    n["doy"] = doy
    n = n.merge(sst_climatology().reset_index(), on=["loc", "doy"], how="left")
    n["sst_t2m"] = n.sst_clim - n.t2m
    n["bmu_c"] = pd.Categorical(n.bmu)
    return n, curves_era

def temporal(n):
    """Forecast speed at neighbouring hours (t-3..t+3), 7 h rolling mean/sd and trend, per site: lets the model
    hedge against timing (phase) errors in the forecast. Caveat: near midnight the t+k values come from the next
    day's run, slightly later than a 10:00 D-1 forecast would have; a strict reconstruction should use one run."""
    if F.NWP_DAY in ("00z", "g_ecmwf", "g_gfs", "g_blend") and set(TEMPORAL) <= set(n.columns):
        return n  # already built within each run by features.nwp_00z (on 100 m speed)
    w = n.drop_duplicates(["loc", "time"]).set_index(["time", "loc"]).ws_eq.unstack("loc").sort_index()
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="h"))
    f = {}
    for k in (1, 2, 3):
        f[f"ws_m{k}"], f[f"ws_p{k}"] = w.shift(k), w.shift(-k)
    f["ws_roll7"] = w.rolling(7, center=True, min_periods=4).mean()
    f["ws_std7"] = w.rolling(7, center=True, min_periods=4).std()
    f["ws_trend"] = w.shift(-2) - w.shift(2)
    t = pd.concat({k: v.stack() for k, v in f.items()}, axis=1)
    t.index.names = ["time", "loc"]
    return n.drop(columns=list(f), errors="ignore").merge(t.reset_index(), on=["time", "loc"], how="left")

def ds(x, cols, base):
    return lgb.Dataset(x[cols], x.cf - x[base], free_raw_data=False)

def split(n, base):
    c = n[n.clean & n[base].notna() & n.cf.notna()]
    tr, te = c[c.time < TRAIN_END], c[(c.time >= TRAIN_END)]
    wk = (tr.time - tr.time.min()).dt.days // 7
    return tr[wk % 5 != 0], tr[wk % 5 == 0], te

def fit_es(params, tr, va, cols, base, init=None, rounds=3000):
    m = lgb.train(params, ds(tr, cols, base), rounds, valid_sets=[ds(va, cols, base)], init_model=init,
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m

def per_site(b, tr, va, te, cols, base, pooled, pooled_cols):
    t, v, s = tr[tr.bmu == b], va[va.bmu == b], te[te.bmu == b]
    if len(t) < 500 or len(v) < 100 or not len(s):
        return None
    own = fit_es(SITE_P, t, v, cols, base)
    tune = fit_es(TUNE_P, t, v, pooled_cols, base, init=pooled, rounds=1000)
    return pd.DataFrame({"r_site": own.predict(s[cols], num_iteration=own.best_iteration),
                         "r_tune": tune.predict(s[pooled_cols], num_iteration=tune.best_iteration)}, index=s.index)

def score(te, preds, cap):
    rows, mw = {}, te[["time"]].copy()
    for k, p in preds.items():
        e = p - te.cf
        mw[k] = e * te.bmu.map(cap)
        rows[k] = {"cf_mae": e.abs().mean(), "cf_rmse": np.sqrt((e ** 2).mean()), "cf_bias": e.mean()}
    f = mw.groupby("time")[list(preds)].sum()
    for k in preds:
        rows[k]["fleet_mw_mae"] = f[k].abs().mean()
        rows[k]["fleet_mw_rmse"] = np.sqrt((f[k] ** 2).mean())
    return pd.DataFrame(rows).T

def evaluate(n, base="p_nwp", weather=WEATHER, site=SITE, per_site_models=True, show=True):
    """Residual models on top of curve `base`; returns the scored table and the test frame."""
    n = n.assign(p_curve=n[base])
    tr, va, te = split(n, base)
    te = te[te.bmu.isin(set(tr.bmu) & set(va.bmu)) & te.p_era.notna() & te.p_nwp.notna()]
    pooled = fit_es(BASE, tr, va, weather + site, base)
    te = te.assign(r_pool=pooled.predict(te[weather + site], num_iteration=pooled.best_iteration))
    clip = lambda p: np.clip(p, -0.02, 1.05)
    preds = {"ERA5 curve (dir+rho)": te.p_era, "NWP-fitted curve (dir+rho)": te.p_nwp,
             f"{base} + pooled LGBM": clip(te[base] + te.r_pool)}
    if per_site_models:
        parts = Parallel(n_jobs=4)(delayed(per_site)(b, tr, va, te, weather, base, pooled, weather + site)
                                   for b in te.bmu.unique())
        te = te.join(pd.concat([p for p in parts if p is not None])).dropna(subset=["r_site", "r_tune"])
        preds = {k: v.loc[te.index] for k, v in preds.items()}
        preds[f"{base} + per-site LGBM"] = clip(te[base] + te.r_site)
        preds[f"{base} + pooled, fine-tuned"] = clip(te[base] + te.r_tune)
    cap = n.groupby("bmu").cap_eff.first()
    out = {"all": score(te, preds, cap)}
    for lab, sel in [("offshore", te.offshore.values), ("onshore", ~te.offshore.values)]:
        out[lab] = score(te[sel], {k: v[sel] for k, v in preds.items()}, cap)
    if show:
        print(f"\nbase={base}; pooled rounds {pooled.best_iteration}; test {te.bmu.nunique()} BMUs, {len(te):,} unit-hours")
        for k, v in out.items():
            print(f"-- {k}"); print(v.round(4).to_string())
        imp = pd.Series(pooled.feature_importance("gain"), index=weather + site).sort_values(ascending=False)
        print("pooled gain share:", (imp / imp.sum()).round(3).head(10).to_dict())
    return out, te, pooled

if __name__ == "__main__":
    f = F.cache("residual_frame.parquet")
    if "--rebuild" in sys.argv or not f.exists():
        n, _ = build()
        n.to_parquet(f, index=False)
    n = temporal(pd.read_parquet(f))
    n["bmu_c"] = pd.Categorical(n.bmu)
    # --pooled-only: skip the per-site comparison models (slow; the fleet benchmark only needs the pooled model)
    out, te, pooled = evaluate(n, base="p_nwp", per_site_models="--pooled-only" not in sys.argv)
    pooled.save_model(str(F.cache("lgb_pooled.txt")))
    te.to_parquet(F.cache("residual_test.parquet"), index=False)

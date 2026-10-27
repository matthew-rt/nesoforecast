"""Does REMIT help the day-ahead forecast? Test year 2025-10 -> 2026-10, ECMWF day-1.

Evaluation hours: every BMU-hour with metered data, after commissioning and with no BOA, i.e. unconstrained by the
BM but *including* outages and MEL derates (the clean-hour scores elsewhere exclude those).
Forecast = base capacity factor x availability known at D-1 09:00 London:
  none          base only (curve fitted on clean hours + pooled LightGBM residual)
  REMIT         x avail_da (latest REMIT revision published before issue)
  MEL persist   x MEL at issue time / capacity (outages persist for days)
  min(both)     x min of the two
Also: refit the curve excluding REMIT outage hours (avail_final < 0.98) from training, scored on clean hours."""
import numpy as np, pandas as pd, lightgbm as lgb
import features as F, residual as R, remit


def mel_persist(n):
    """MEL / cap_eff in the hour ending at the issue time (D-1 08:00-09:00 London) for each delivery hour, per BMU.
    The hourly MEL is a mean over that hour, so the hour must end by the issue time."""
    h = F.bm_hourly()[["bmu", "time", "mel", "cap_eff"]]
    h = h.assign(mel_frac=(h.mel / h.cap_eff).clip(0, 1)).set_index(["bmu", "time"]).mel_frac
    key = pd.MultiIndex.from_arrays([n.bmu, F.issue_time(n.time) - pd.Timedelta("1h")])
    return h.reindex(key).values

def main():
    n = R.temporal(pd.read_parquet(F.cache("residual_frame.parquet")))
    n["bmu_c"] = pd.Categorical(n.bmu)
    a = remit.availability()
    n = n.merge(a, on=["bmu", "time"], how="left").fillna({"avail_final": 1.0, "avail_da": 1.0, "avail_now": 1.0})
    n["mel_da"] = mel_persist(n)
    booster = lgb.Booster(model_file=str(F.cache("lgb_pooled.txt")))
    cols = booster.feature_name()
    n["p_curve"] = n.p_nwp
    te = n[(n.time >= R.TRAIN_END) & n.p_nwp.notna() & n.cf.notna() & ~n.boa & (n.time >= n.commissioned)
           & n.metered.between(-0.02 * n.cap_eff, 1.1 * n.cap_eff)].copy()
    te["base"] = np.clip(te.p_nwp + booster.predict(te[cols]), -0.02, 1.05)
    av = {"none": 1.0, "REMIT": te.avail_da, "MEL persist": te.mel_da.fillna(1.0),
          "min(both)": np.minimum(te.avail_da, te.mel_da.fillna(1.0))}
    preds = {k: te.base * v for k, v in av.items()}
    cap = n.groupby("bmu").cap_eff.first()
    out = R.score(te, preds, cap)
    affected = (te.avail_final < 0.98) | (te.mel < 0.95 * te.cap_eff)
    print(f"test hours {len(te):,} ({te.bmu.nunique()} BMUs); outage/derate-affected {affected.mean():.1%}; "
          f"REMIT outage known day-ahead in {(te.avail_da < 0.98).mean():.1%}, ex-post {(te.avail_final < 0.98).mean():.1%}")
    print("\nall unconstrained test hours"); print(out.round(4).to_string())
    print("\naffected hours only"); print(R.score(te[affected], {k: v[affected] for k, v in preds.items()}, cap).round(4).to_string())

    # availability as model features: pooled LightGBM trained on all unconstrained hours (not only clean ones)
    n["mel_da"] = n.mel_da.fillna(1.0)
    n["p_avail"] = n.p_nwp * n.mel_da
    now = a.set_index(["bmu", "time"]).avail_now
    n["remit_at_issue"] = now.reindex(pd.MultiIndex.from_arrays([n.bmu, F.issue_time(n.time)])).fillna(1.0).values
    n["remit_ends"] = (n.avail_da > n.remit_at_issue + 0.02).astype(float)  # REMIT expects recovery by delivery
    feats = R.WEATHER + R.SITE + ["mel_da", "avail_da", "p_avail", "remit_at_issue", "remit_ends"]
    unc = n[n.p_nwp.notna() & n.cf.notna() & ~n.boa & (n.time >= n.commissioned)
            & n.metered.between(-0.02 * n.cap_eff, 1.1 * n.cap_eff)]
    te = te.join(unc[["remit_at_issue", "remit_ends", "p_avail"]], how="left")
    # base p_nwp: the model must learn the availability scaling itself; base p_avail: curve x MEL persistence is
    # built in and the model learns corrections to it (e.g. REMIT says the outage ends before delivery)
    for base, label in [("p_nwp", "LGBM + avail features, base curve"), ("p_avail", "LGBM + avail features, base curve x MEL")]:
        u = unc.assign(p_curve=unc[base])
        tr = u[u.time < R.TRAIN_END]
        wk = (tr.time - tr.time.min()).dt.days // 7
        m = R.fit_es(R.BASE, tr[wk % 5 != 0], tr[wk % 5 == 0], feats, base)
        x = te.assign(p_curve=te[base])
        preds[label] = np.clip(te[base] + m.predict(x[feats], num_iteration=m.best_iteration), -0.02, 1.05)
    print("\nwith availability-aware LightGBM (trained on all unconstrained hours)")
    print(R.score(te, preds, cap).round(4).to_string())
    print("affected hours only"); print(R.score(te[affected], {k: v[affected] for k, v in preds.items()}, cap).round(4).to_string())
    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("gain share:", (imp / imp.sum()).round(3).head(8).to_dict())

    # curve refit excluding REMIT outage hours from training (scored on clean test hours, like residual.py)
    era_like = n[n.clean & (n.time < R.TRAIN_END)].dropna(subset=["cf"])
    cur = R.fit_curves(era_like)
    cur_x = R.fit_curves(era_like[era_like.avail_final >= 0.98])
    c = n[n.clean & (n.time >= R.TRAIN_END) & n.cf.notna()].copy()
    c["p_all"], c["p_ex"] = R.apply_curves(c, cur), R.apply_curves(c, cur_x)
    c = c.dropna(subset=["p_all", "p_ex"])
    print(f"\ncurve fitted on clean hours vs clean hours minus REMIT outages "
          f"(training hours removed: {(era_like.avail_final < 0.98).mean():.1%}); clean test hours")
    print(R.score(c, {"curve, all clean": c.p_all, "curve, ex REMIT outages": c.p_ex}, cap).round(4).to_string())

if __name__ == "__main__":
    main()

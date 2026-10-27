"""LightGBM baselines on the hybrid-network test rows (separate process from torch, see nn_data.py).
Writes data/cache/nn_baselines.parquet: curve x MEL, LightGBM L2 x MEL, LightGBM L1 x MEL."""
import numpy as np, pandas as pd, lightgbm as lgb
import features as F, residual as R
from nn_data import frame, split

def lgb_l1(n, te):
    """LightGBM residual on the curve, L1 objective, trained on unconstrained clean hours, predicted for `te`."""
    cats = sorted(set(n.bmu) | set(te.bmu))
    n = n.assign(p_curve=n.p_nwp, bmu_c=pd.Categorical(n.bmu, categories=cats))
    tr, va, _ = split(n[n.clean])
    m = R.fit_es(dict(R.BASE, objective="l1"), tr, va, R.WEATHER + R.SITE, "p_nwp")
    x = te.assign(p_curve=te.p_nwp, bmu_c=pd.Categorical(te.bmu, categories=cats))
    return pd.Series(np.clip(te.p_nwp + m.predict(x[R.WEATHER + R.SITE], num_iteration=m.best_iteration), -0.02, 1.05), index=te.index)

def main():
    n = frame()
    a = frame(all_rows=True)
    te = a[a.time >= R.TRAIN_END]  # every test-year unit-hour, curtailed ones included
    out = pd.DataFrame({"curve x MEL": te.p_nwp * te.mel_da}, index=te.index)
    booster = lgb.Booster(model_file=str(F.cache("lgb_pooled.txt")))
    cats = sorted(pd.read_parquet(F.cache("residual_frame.parquet"), columns=["bmu"]).bmu.unique())
    x = te.assign(p_curve=te.p_nwp, bmu_c=pd.Categorical(te.bmu, categories=cats))
    out["LGBM (L2) x MEL"] = np.clip(te.p_nwp + booster.predict(x[booster.feature_name()]), -0.02, 1.05) * te.mel_da
    out["LGBM (L1) x MEL"] = lgb_l1(n, te).loc[te.index] * te.mel_da
    out.assign(bmu=te.bmu, time=te.time, unc=te.unc).to_parquet(F.cache("nn_baselines.parquet"))
    print("baselines written", out.shape)

if __name__ == "__main__":
    main()

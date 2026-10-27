"""Per-BMU hybrid networks (A = MEL at issue, 3 seeds averaged) predicting every test-year unit-hour, including
curtailed hours, for the fleet benchmark against NESO WINDFOR. Writes data/cache/fleet_nn_preds[_l2].parquet.
--l2: train on squared error (per-unit mean, so the fleet sum is unbiased) instead of absolute error (median)."""
import sys
import numpy as np, pandas as pd, torch
import features as F, residual as R
import hybrid_nn as H
from nn_data import frame

SEEDS = 3
LOSS = "l2" if "--l2" in sys.argv else "l1"

def main():
    a = frame(all_rows=True)
    n = a[a.unc]
    bmus = sorted(n.bmu.unique())
    Y0 = H.init_curves(n, bmus)
    pred_rows = a[a.time >= R.TRAIN_END]
    out = []
    for b in bmus:
        sub = n[n.bmu == b]
        tr, va, _ = H.split(sub)
        if len(tr) < 500 or len(va) < 100:
            continue
        p = [H.fit_group(sub, Y0, emb=0, hidden=16, verbose=False, learn_avail=False, seed=s, pred=pred_rows, loss=LOSS)[0]
             for s in range(SEEDS)]
        out.append(pd.concat(p, axis=1).mean(axis=1))
    s = pd.concat(out)
    res = pred_rows.loc[s.index, ["bmu", "time"]].assign(nn=s.values)
    res.to_parquet(F.cache("fleet_nn_preds_l2.parquet" if LOSS == "l2" else "fleet_nn_preds.parquet"), index=False)
    print(f"{res.bmu.nunique()} BMUs, {len(res):,} unit-hours predicted")

if __name__ == "__main__":
    main()

"""Data assembly for the hybrid network and its baselines (no torch import here: torch and LightGBM in one
process segfault on macOS, so LightGBM baselines run separately via nn_baselines.py)."""
import numpy as np, pandas as pd
import features as F, residual as R, remit_eval as RE

GRID = np.arange(0, 30.01, 0.5)
W_COLS_FULL = ["ws_eq", "ws100", "ws_spread", "wd_diff", "alpha", "rho", "t2m", "sp", "sst_t2m", "hour_sin", "hour_cos"] + R.TEMPORAL
# Input selection (2026-10-25, 40 BMUs, permutation importance + drop-and-retrain on the test year): dropping t+-1,
# 7 h mean/sd/trend, run direction disagreement, raw 100 m speed, T, p, rho, SST-T and hour changed test MAE by
# +0.03% onshore / -0.4% offshore; removing direction harmonics cost +6.0% / +2.4%, shear +0.3% / +0.6%.
W_COLS = ["ws_eq", "ws_spread", "alpha", "ws_m2", "ws_m3", "ws_p2", "ws_p3"]  # + 8 direction harmonics in the network
if F.NWP_DAY == "g_blend":  # second model: GFS/ECMWF speed ratio now and at t+-3 (lets the speed correction lean on GFS)
    W_COLS = W_COLS + ["gfs_lr", "gfs_lr_m3", "gfs_lr_p3"]
S_COLS = ["offshore", "hub_est", "rotor_est", "k_gwa_100", "lat", "lon", "log_cap"]
A_COLS = ["mel_da", "avail_da", "remit_at_issue", "remit_ends", "lead_h", "offshore"]
N_HARM = 4

def frame(all_rows=False):
    """BMU-hour features. Default: only unconstrained training/scoring rows (no BOA, post-commissioning, valid
    metering, unit has a fitted curve). all_rows=True keeps every row with a forecast speed, flagged by `unc`, so a
    fleet forecast can be made for curtailed hours and for units without a fitted curve too."""
    n = R.temporal(pd.read_parquet(F.cache("residual_frame.parquet")))
    a = pd.read_parquet(F.DATA / "remit_hourly.parquet")
    n = n.merge(a, on=["bmu", "time"], how="left").fillna({"avail_final": 1.0, "avail_da": 1.0, "avail_now": 1.0})
    n["mel_da"] = RE.mel_persist(n)
    n["mel_da"] = n.mel_da.fillna(1.0)
    issue = F.issue_time(n.time)
    now = a.set_index(["bmu", "time"]).avail_now
    n["remit_at_issue"] = now.reindex(pd.MultiIndex.from_arrays([n.bmu, issue])).fillna(1.0).values
    n["remit_ends"] = (n.avail_da > n.remit_at_issue + 0.02).astype(float)
    n["lead_h"] = (n.time - issue) / pd.Timedelta("1h")
    n["hour_sin"], n["hour_cos"] = np.sin(2 * np.pi * n.hour / 24), np.cos(2 * np.pi * n.hour / 24)
    n["log_cap"] = np.log(n.cap_eff)
    n["rho"] = n.rho.fillna(F.RHO0)
    for c in R.TEMPORAL:
        n[c] = n[c].fillna(n.ws_eq) if c != "ws_std7" else n[c].fillna(0)
    n["sst_t2m"] = n.sst_t2m.fillna(0)
    for c in [c for c in W_COLS if c.startswith("gfs_lr")]:
        n[c] = n[c].fillna(0)  # missing GFS -> no information
    n["unc"] = (n.p_nwp.notna() & n.cf.notna() & ~n.boa & (n.time >= n.commissioned)
                & n.metered.between(-0.02 * n.cap_eff, 1.1 * n.cap_eff) & n.ws_eq.notna())
    n = (n[n.ws_eq.notna()] if all_rows else n[n.unc]).reset_index(drop=True)
    n["offshore"] = n.offshore.astype(float)
    return n

def split(n):
    tr = n[n.time < R.TRAIN_END]
    wk = (tr.time - tr.time.min()).dt.days // 7
    return tr[wk % 5 != 0], tr[wk % 5 == 0], n[n.time >= R.TRAIN_END]


"""Empirical per-BMU power curves from ERA5 100 m speed and metered output (curtailment/derate-free hours).

Curve = mean capacity factor per 0.5 m/s speed bin, isotonic (non-decreasing) up to the peak bin, free above it
(storm control / cut-out), linearly interpolated. The bin mean (not median) is the MSE-optimal mapping from
*ERA5* speed to output, so ERA5 speed error is folded into the curve's shape."""
import pathlib
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BINS = np.arange(0, 32.5, 0.5)
MIN_BIN = 10  # hours needed for a bin to count

def bmu_loc():
    s = pd.read_csv(DATA / "sites.csv").dropna(subset=["lat"])
    s["loc"] = s.osm_id.fillna("repd" + s.repd_id.astype(str))
    return s.set_index("elexonBmUnit")[["loc", "offshore", "cap"]]

def hourly_bm():
    """Pair each ERA5 hour h with mean output over [h-30m, h+30m)."""
    d = pd.read_parquet(DATA / "bm_hh.parquet")
    d["time"] = (d.start + pd.Timedelta("30min")).dt.floor("h")
    g = d.groupby(["bmu", "time"])
    h = g.agg(metered=("metered", "mean"), pn=("pn", "mean"), mel=("mel", "min"), boa=("boa", "any"),
              n=("metered", "size")).reset_index()
    return h[h.n == 2].drop(columns="n")

def clean(h):
    """Effective capacity, commissioning date and a clean-hour mask per BMU."""
    out = []
    for bmu, x in h.groupby("bmu"):
        x = x.sort_values("time").copy()
        ok = ~x.boa & x.metered.notna()
        if ok.sum() < 500 or x.metered[ok].quantile(.995) < 1:
            continue
        cap = x.metered[ok].quantile(.995)
        roll = x.set_index("time").metered.rolling("30D").max()
        full = roll[roll >= 0.9 * cap]
        if full.empty:
            continue
        x["cap_eff"], x["commissioned"] = cap, full.index[0]
        x["clean"] = (ok & (x.time >= full.index[0]) & (x.mel.isna() | (x.mel >= 0.95 * cap))
                      & x.metered.between(-0.02 * cap, 1.1 * cap))
        out.append(x)
    return pd.concat(out)

def fit_curve(ws, cf, stat="mean"):
    """stat="mean" targets squared error; "median" targets absolute error (MAE)."""
    idx = np.digitize(ws, BINS) - 1
    df = pd.DataFrame({"b": idx, "cf": cf}).groupby("b").cf.agg([stat, "size"])
    df = df[df["size"] >= MIN_BIN]
    if len(df) < 5:
        return None
    x, y, w = BINS[df.index] + 0.25, df[stat].values, df["size"].values
    k = int(np.argmax(y))
    y_up = IsotonicRegression(increasing=True).fit(x[:k + 1], y[:k + 1], sample_weight=w[:k + 1]).predict(x[:k + 1])
    y = np.r_[y_up, y[k + 1:]]
    return np.r_[0.0, x], np.r_[0.0, np.clip(y, 0, None)]

def predict(curve, ws):
    x, y = curve
    return np.interp(ws, x, y, left=0, right=y[-1])

def harmonics(wd, n):
    th = np.radians(np.asarray(wd, float))
    return np.column_stack([f(k * th) for k in range(1, n + 1) for f in (np.cos, np.sin)]) if n else np.zeros((len(th), 0))

def dlog(curve, v, h=0.05):
    """d cf / d ln v, by central difference over +-5% (smooths the piecewise-linear, partly flat isotonic curve)."""
    return (predict(curve, v * (1 + h)) - predict(curve, v * (1 - h))) / (2 * h)

def fit_site(ws, wd, cf, n_harm=0, lam=0.01, iters=3, max_delta=0.25, stat="mean"):
    """Curve on a direction-adjusted speed v = ws * (1 + delta(wd)), delta a Fourier series in direction.

    Wakes, blockage and direction-dependent reanalysis bias act mainly as a fractional speed deficit, so
    the correction is a speed factor (it vanishes at rated, as wake losses do). Alternates: fit the curve on v,
    then a ridge step for delta from the linearisation cf - f(v) ~ f'(ln v) * delta. With few hours in a sector
    the ridge shrinks delta toward 0, i.e. back to the omnidirectional curve."""
    ws, cf = np.asarray(ws, float), np.asarray(cf, float)
    H = harmonics(wd, n_harm)
    a = np.zeros(H.shape[1])
    for _ in range(iters if n_harm else 1):
        v = ws * (1 + np.clip(H @ a, -max_delta, max_delta))
        c = fit_curve(v, cf, stat)
        if c is None or not n_harm:
            break
        X = dlog(c, v)[:, None] * H
        r = cf - predict(c, v)
        L = lam * len(cf) * np.eye(len(a))  # ridge on the total a, not the step
        a = a + np.linalg.solve(X.T @ X + L, X.T @ r - L @ a)
    if n_harm and c is not None:  # final curve on the final speed factor
        c = fit_curve(ws * (1 + np.clip(H @ a, -max_delta, max_delta)), cf, stat)
    return None if c is None else {"curve": c, "a": a, "n_harm": n_harm, "max_delta": max_delta}

def predict_site(m, ws, wd):
    d = np.clip(harmonics(wd, m["n_harm"]) @ m["a"], -m["max_delta"], m["max_delta"])
    return predict(m["curve"], np.asarray(ws, float) * (1 + d))

def build(test_months=None):
    """Fit one curve per BMU. If test_months is given, exclude those months from fitting (block CV)."""
    loc = bmu_loc()
    h = clean(hourly_bm())
    h["loc"] = h.bmu.map(loc["loc"])
    e = pd.read_parquet(DATA / "era5_sites.parquet")
    d = h.merge(e, on=["loc", "time"], how="inner")
    d["cf"] = d.metered / d.cap_eff
    curves = {}
    for bmu, x in d[d.clean].groupby("bmu"):
        if test_months is not None:
            x = x[~x.time.dt.to_period("M").isin(test_months)]
        c = fit_curve(x.ws100.values, x.cf.values)
        if c is not None:
            curves[bmu] = c
    return d, curves

if __name__ == "__main__":
    d, curves = build()
    d = d[d.bmu.isin(curves)]
    d["pred_cf"] = np.nan
    for b, i in d.groupby("bmu").groups.items():
        d.loc[i, "pred_cf"] = predict(curves[b], d.loc[i, "ws100"].values)
    c = d[d.clean]
    err = (c.pred_cf - c.cf)
    print("BMUs with curves:", len(curves), " clean hours:", len(c))
    print("in-sample clean-hour CF MAE: %.3f  bias: %+.3f" % (err.abs().mean(), err.mean()))

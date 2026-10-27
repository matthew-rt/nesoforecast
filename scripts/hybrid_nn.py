"""Step 4: physics-structured hybrid neural network for day-ahead unconstrained output per BMU (PyTorch).

    cf_hat = A * curve_b(v') * cutout_b(v'),     v' = v * exp(0.3 * tanh(g(x)))

  v          ECMWF day-1 density-equivalent hub-height speed (features.py)
  g          MLP (64-64, SiLU) on weather, direction harmonics, site embedding and site descriptors: a speed
             correction (direction-dependent wakes, stability-dependent forecast bias, ...), bounded to e^+-0.3 (-26% .. +35%)
  curve_b    per-BMU monotone power curve on a 0-30 m/s grid (61 knots, cumulative softplus increments),
             initialised from that BMU's empirical curve fitted on the training forecasts
  cutout_b   sigmoid((v_cut_b - v') / 1 m/s), v_cut_b learned per BMU (init 25 m/s)
  A          availability = mel + (1 - mel) u - mel d, with mel = MEL / capacity at issue time (hour to D-1 09:00 London),
             u (derate ends by delivery) and d (outage beyond MEL) from a small MLP on MEL/REMIT features
Loss: L1 (MAE). Ground truth: B1610 settlement-metered output; hours with any BOA (curtailment) are excluded.
Variants: one fleet model, separate onshore/offshore models, one small model per BMU."""
import sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import features as F, residual as R, fit_curves as fc
from nn_data import GRID, W_COLS, S_COLS, A_COLS, N_HARM, frame, split

torch.set_num_threads(6)
def init_curves(n, bmus):
    """Empirical curve per BMU (clean training hours, ECMWF speed) resampled to GRID, made monotone.
    Returns a DataFrame indexed by BMU (rows) x GRID (columns)."""
    tr = n[n.clean & (n.time < R.TRAIN_END)]
    curves = R.fit_curves(tr)
    Y = np.zeros((len(bmus), len(GRID)))
    for i, b in enumerate(bmus):
        if b in curves:
            x, y = curves[b]["curve"]
            Y[i] = np.maximum.accumulate(np.interp(GRID, x, y, left=0, right=y[-1]))
        else:  # generic fallback: fleet-median curve
            Y[i] = np.nan
    Y[np.isnan(Y[:, 0])] = np.nanmedian(Y, axis=0)
    return pd.DataFrame(Y, index=bmus)

def mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))

class Hybrid(nn.Module):
    def __init__(self, Y0, n_w, n_s, n_a, emb=8, hidden=64, learn_avail=True):
        super().__init__()
        self.learn_avail = learn_avail  # False: A = MEL at issue exactly (weather part only is learned)
        n_sites = Y0.shape[0]
        inc = np.clip(np.diff(Y0, axis=1), 1e-4, None)
        theta0 = torch.tensor(np.log(np.expm1(inc)), dtype=torch.float32)
        self.theta = nn.Parameter(theta0.clone())
        self.register_buffer("theta0", theta0)
        self.v_cut = nn.Parameter(torch.full((n_sites,), 25.0))
        self.emb = nn.Embedding(n_sites, emb) if emb else None
        e = emb or 0
        self.g = mlp(n_w + 2 * N_HARM + n_s + e, hidden, 1)
        self.h = mlp(n_a + e, 32, 2, depth=1)
        nn.init.zeros_(self.g[-1].weight); nn.init.zeros_(self.g[-1].bias)        # start: no speed correction
        nn.init.zeros_(self.h[-1].weight); nn.init.constant_(self.h[-1].bias, -6.0)  # start: A ~ MEL (u, d ~ 0.0025)

    def curve(self, site, v):
        y = torch.cat([torch.zeros(self.theta.shape[0], 1), torch.cumsum(nn.functional.softplus(self.theta), 1)], 1)
        k = torch.clamp(v / 0.5, 0, len(GRID) - 1 - 1e-4)
        i = k.long(); f = k - i
        yi = y[site, i] * (1 - f) + y[site, i + 1] * f
        return yi * torch.sigmoid(self.v_cut[site] - v)

    def forward(self, site, xw, wdh, xs, xa, v, mel):
        e = [self.emb(site)] if self.emb is not None else []
        s = 0.3 * torch.tanh(self.g(torch.cat([xw, wdh, xs] + e, 1)).squeeze(1))
        p = self.curve(site, v * torch.exp(s))
        if self.learn_avail:
            u, d = torch.sigmoid(self.h(torch.cat([xa] + e, 1))).unbind(1)
            A = mel + (1 - mel) * u - mel * d
        else:
            A = mel
        return A * p, p, A, s

class Data:
    def __init__(self, n, bmus, stats=None):
        self.site = torch.tensor(n.bmu.map({b: i for i, b in enumerate(bmus)}).values, dtype=torch.long)
        cols = W_COLS + S_COLS + A_COLS
        x = n[cols].astype(float)
        self.stats = stats or (x.mean(), x.std().replace(0, 1))
        z = ((x - self.stats[0]) / self.stats[1]).fillna(0).values.astype(np.float32)
        nw, ns = len(W_COLS), len(S_COLS)
        self.xw, self.xs, self.xa = map(torch.tensor, (z[:, :nw], z[:, nw:nw + ns], z[:, nw + ns:]))
        self.wdh = torch.tensor(fc.harmonics(n.wd100.values, N_HARM).astype(np.float32))
        self.v = torch.tensor(n.ws_eq.values.astype(np.float32))
        self.mel = torch.tensor(n.mel_da.clip(0, 1).values.astype(np.float32))
        self.y = torch.tensor(n.cf.values.astype(np.float32))

    def batch(self, idx):
        return (self.site[idx], self.xw[idx], self.wdh[idx], self.xs[idx], self.xa[idx], self.v[idx], self.mel[idx]), self.y[idx]

def train(model, dtr, dva, epochs=60, bs=8192, lr=2e-3, curve_pen=1e-3, patience=4, verbose=True, loss="l1"):
    """loss "l1" targets the per-unit median (best per-unit MAE); "l2" the mean (unbiased when units are summed)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    best, best_state, bad = np.inf, None, 0
    n = len(dtr.y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for k in range(0, n, bs):
            xb, yb = dtr.batch(perm[k:k + bs])
            pred = model(*xb)[0]
            err = (pred - yb).abs() if loss == "l1" else (pred - yb) ** 2
            obj = err.mean() + curve_pen * ((model.theta - model.theta0) ** 2).mean()
            opt.zero_grad(); obj.backward(); opt.step()
        va = evaluate(model, dva, loss)
        if va < best - 1e-5:
            best, best_state, bad = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if verbose:
            print(f"  epoch {ep:2d} val MAE {va:.4f}", flush=True)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model

@torch.no_grad()
def predict(model, d, bs=65536):
    model.eval()
    out = [model(*d.batch(torch.arange(k, min(k + bs, len(d.y))))[0])[0] for k in range(0, len(d.y), bs)]
    return torch.cat(out).numpy()

def evaluate(model, d, loss="l1"):
    e = predict(model, d) - d.y.numpy()
    return float(np.abs(e).mean() if loss == "l1" else (e ** 2).mean())

def fit_group(n, Y0, emb=8, hidden=64, verbose=True, learn_avail=True, seed=0, pred=None, loss="l1"):
    """Train one Hybrid on the BMUs in n (curves initialised from rows of Y0); return predictions for `pred`
    (default: n's test rows), which may include curtailed hours."""
    bmus = sorted(n.bmu.unique())
    Y0 = Y0.loc[bmus].values
    tr, va, te = split(n)
    te = te if pred is None else pred[pred.bmu.isin(bmus)]
    dtr = Data(tr, bmus); dva, dte = Data(va, bmus, dtr.stats), Data(te, bmus, dtr.stats)
    torch.manual_seed(seed)
    m = Hybrid(Y0, len(W_COLS), len(S_COLS), len(A_COLS), emb=emb, hidden=hidden, learn_avail=learn_avail)
    m = train(m, dtr, dva, verbose=verbose, loss=loss)
    return pd.Series(predict(m, dte), index=te.index), m

def main():
    t0 = time.time()
    n = frame()
    bmus = sorted(n.bmu.unique())
    Y0 = init_curves(n, bmus)
    te = n[n.time >= R.TRAIN_END].copy()
    print(f"{len(n):,} unconstrained BMU-hours ({len(bmus)} BMUs); test {len(te):,}; built in {time.time()-t0:.0f}s", flush=True)
    b = pd.read_parquet(F.cache("nn_baselines.parquet"))  # from nn_baselines.py (run first)
    assert (b.bmu.values == te.bmu.values).all() and (b.time.values == te.time.values).all()
    preds = {k: pd.Series(b[k].values, index=te.index) for k in ["curve x MEL", "LGBM (L2) x MEL", "LGBM (L1) x MEL"]}
    print("== NN: fleet", flush=True)
    preds["NN fleet"], m_fleet = fit_group(n, Y0)
    torch.save(m_fleet.state_dict(), F.cache("nn_fleet.pt"))
    parts = []
    for off in (True, False):
        print(f"== NN: {'offshore' if off else 'onshore'} only", flush=True)
        sub = n[n.offshore == float(off)]
        parts.append(fit_group(sub, Y0)[0])
    preds["NN onshore/offshore split"] = pd.concat(parts)
    print("== NN: per BMU", flush=True)
    per = []
    for i, b in enumerate(bmus):
        sub = n[n.bmu == b]
        tr, va, ts = split(sub)
        if len(tr) < 500 or len(va) < 100 or not len(ts):
            continue
        per.append(fit_group(sub, Y0, emb=0, hidden=16, verbose=False)[0])
    preds["NN per BMU"] = pd.concat(per)
    print(f"all models trained in {(time.time()-t0)/60:.1f} min", flush=True)

    P = pd.DataFrame({k: pd.Series(np.asarray(v), index=te.index) if not isinstance(v, pd.Series) else v for k, v in preds.items()})
    P = P.dropna()
    t = te.loc[P.index]
    cap = n.groupby("bmu").cap_eff.first()
    for lab, sel in [("all no-BOA test hours", np.ones(len(t), bool)), ("clean hours", t.clean.values),
                     ("outage/derate-affected", ((t.avail_final < 0.98) | (t.mel < 0.95 * t.cap_eff)).values),
                     ("offshore", t.offshore.values == 1), ("onshore", t.offshore.values == 0)]:
        print(f"\n-- {lab}: {sel.sum():,} BMU-hours, {t[sel].bmu.nunique()} BMUs")
        print(R.score(t[sel], {k: P[k][sel] for k in P}, cap).round(4).to_string())
    P.assign(bmu=t.bmu, time=t.time, cf=t.cf).to_parquet(F.cache("nn_test_preds.parquet"), index=False)

if __name__ == "__main__":
    main()

"""
COP (Conformal Optimistic Prediction, ICLR 2026, arXiv:2512.07770) baseline on the
EXTREME cross-dataset stream PathoSAM -> NuInsSeg (cal = PanNuke), same setting as
Bang 9a. COP shares the EXACT nonconformity score with PB-JCI Online (PB-sigma scaled
|gt - E[N]|/sigma); the two differ only in the recalibration rule:

  COP update (Algorithm 1):
    1. q_hat_{t+1} = q_hat_t + eta * (1[s_t > q_t] - alpha)          # OGD on pinball loss
    2. F_hat_{t+1}(q_hat) = (1/w) sum_{i in last w} 1[s_i <= q_hat]   # empirical CDF, w=100
    3. q_{t+1}     = q_hat_{t+1} - lambda * (F_hat_{t+1}(q_hat_{t+1}) - (1-alpha))

We give COP its best shot by sweeping the learning rate eta and reporting the run whose
coverage is closest to the nominal 90% (standard "strong baseline" practice). lambda=1,
w=100 per the paper. CPU only, cached pkls.  python pathosam_cop_baseline.py
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kaggle" / "lib"))
from conformal import empirical_quantile, pb_count, pb_variance  # noqa

ALPHA = 0.1


def load():
    d = REPO / "data"
    with open(d / "pathosam_predictions.pkl", "rb") as f:
        dpan = pickle.load(f)
    with open(d / "pathosam_nuinsseg_preds.pkl", "rb") as f:
        dnu = pickle.load(f)
    pan_src = dpan["predictions_by_setting"]["in_dist"]
    pan_gtc = np.asarray(dpan["gt_counts"])
    pan_preds = [{"scores": np.asarray(p["scores"]),
                  "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in pan_src]
    pan_gts = [np.array([float(g.sum())]) for g in pan_gtc]
    return pan_preds, pan_gts, dnu["preds"], dnu["gts"]


def nonconf(p, gt):
    if len(p["scores"]) == 0:
        return float(abs(gt[0]))
    n = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return abs(gt[0] - n) / sg


def interval(p, q):
    if len(p["scores"]) == 0:
        return 0.0, 0.0
    n = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return max(0.0, n - q * sg), n + q * sg


PAN_PREDS, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_SCORES = np.array([nonconf(PAN_PREDS[i], PAN_GTS[i]) for i in range(len(PAN_PREDS))])
print(f"PanNuke cal {len(PAN_SCORES)} | NuInsSeg {len(NU_PREDS)} | "
      f"q0={empirical_quantile(PAN_SCORES, ALPHA):.3f}")


class COP:
    def __init__(self, alpha, eta, lam=1.0, w=100, warm=None):
        self.alpha, self.eta, self.lam, self.w = alpha, eta, lam, w
        self.qhat = empirical_quantile(np.asarray(warm), alpha)
        self.q = self.qhat
        self.win = list(np.asarray(warm)[-w:])

    def get_q(self):
        return max(0.0, self.q)

    def update(self, s, q_used):
        # 1. primary OGD step on the pinball gradient
        self.qhat = max(0.0, self.qhat + self.eta * ((1.0 if s > q_used else 0.0) - self.alpha))
        self.win.append(float(s)); self.win = self.win[-self.w:]
        # 2-3. distribution-informed refinement using the windowed empirical CDF
        Fhat = float(np.mean(np.asarray(self.win) <= self.qhat))
        self.q = max(0.0, self.qhat - self.lam * (Fhat - (1.0 - self.alpha)))


def cop_run(order, eta, lam=1.0):
    m = COP(ALPHA, eta, lam, w=100, warm=PAN_SCORES)
    c, w = [], []
    for i in order:
        q = m.get_q(); lo, hi = interval(NU_PREDS[i], q)
        c.append(lo <= NU_GTS[i][0] <= hi); w.append(hi - lo)
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]), q)
    return c, w


def agg(eta, lam=1.0, seeds=5):
    cs, ws = [], []
    for sd in range(seeds):
        order = np.random.RandomState(sd).permutation(len(NU_PREDS))
        c, w = cop_run(order, eta, lam)
        cs.append(np.mean(c) * 100); ws.append(np.mean(w))
    return np.mean(cs), np.std(cs), np.mean(ws)


# ---- reference methods on the SAME seeds for an exact width comparison ----
from conformal import PBAwareJointConformalOnline  # noqa


def pbo_run(order, window=300):
    m = PBAwareJointConformalOnline(alpha=ALPHA, window=window); m.warmstart(PAN_SCORES)
    c, w = [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q)
        c.append(lo <= NU_GTS[i][0] <= hi); w.append(hi - lo)
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w


def wquantile(scores, weights, level):
    o = np.argsort(scores); s = np.asarray(scores)[o]; wt = np.asarray(weights)[o]
    cw = np.cumsum(wt) / wt.sum()
    return s[min(np.searchsorted(cw, level), len(s) - 1)]


def nexcp_run(order, rho=0.99):
    hist = list(PAN_SCORES); c, w = [], []
    for i in order:
        wts = rho ** (len(hist) - 1 - np.arange(len(hist)))
        q = wquantile(hist, wts, 1 - ALPHA)
        lo, hi = interval(NU_PREDS[i], q)
        c.append(lo <= NU_GTS[i][0] <= hi); w.append(hi - lo)
        hist.append(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w


def agg_fn(fn, seeds=5):
    cs, ws = [], []
    for sd in range(seeds):
        order = np.random.RandomState(sd).permutation(len(NU_PREDS))
        c, w = fn(order)
        cs.append(np.mean(c) * 100); ws.append(np.mean(w))
    return np.mean(cs), np.std(cs), np.mean(ws)


print("\n" + "=" * 64)
print("COP (ICLR 2026) on PathoSAM->NuInsSeg | target 90% | lambda=1, w=100")
print("=" * 64)
print(f"{'eta':>6s} | {'Coverage':>14s} | {'Width':>10s}")
print("-" * 64)
best = None
for eta in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    cm, cs, wm = agg(eta)
    if best is None or abs(cm - 90) < abs(best[1] - 90):
        best = (eta, cm, cs, wm)
    print(f"{eta:6.2f} | {cm:6.1f}+/-{cs:4.1f}% | {wm:8.2f}")
print("-" * 64)
print(f"COP best-coverage eta={best[0]}: {best[1]:.1f}+/-{best[2]:.1f}%  width {best[3]:.2f}")
print("\n-- reference (same 5 seeds) --")
for nm, fn in [("PB-JCI Online (ours)", pbo_run), ("NexCP (Barber23)", nexcp_run)]:
    cm, cs, wm = agg_fn(fn)
    print(f"{nm:22s}: {cm:5.1f}+/-{cs:4.1f}%  width {wm:6.2f}")

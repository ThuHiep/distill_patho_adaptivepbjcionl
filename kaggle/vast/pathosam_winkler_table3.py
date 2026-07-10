"""
Fill the Winkler/Interval-score column of Table 3 on the REAL extreme cross-dataset
run (PathoSAM -> NuInsSeg, cal = PanNuke), same 5 seeds as the coverage/width numbers.

Winkler/interval score (lower = better), alpha=0.1:
    S = (U - L) + (2/alpha)*(L - y) 1{y<L} + (2/alpha)*(y - U) 1{y>U}
A single proper scoring rule: penalises BOTH width and non-coverage. This is the
honest headline metric -- width alone is misleading (at matched coverage every
method needs ~the same width; the difference is whether they cover the current regime).

CPU only, cached pkls.  python pathosam_winkler_table3.py
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kaggle" / "lib"))
from conformal import (AdaptiveConformalInference, PBAwareJointConformalOnline,  # noqa
                       RollingShiftDetector, empirical_quantile, pb_count, pb_variance)

ALPHA = 0.1
WINDOW = 300


def load():
    d = REPO / "data"
    with open(d / "pathosam_predictions.pkl", "rb") as f:
        dpan = pickle.load(f)
    with open(d / "pathosam_nuinsseg_preds.pkl", "rb") as f:
        dnu = pickle.load(f)
    src = dpan["predictions_by_setting"]["in_dist"]
    gtc = np.asarray(dpan["gt_counts"])
    preds = [{"scores": np.asarray(p["scores"]),
              "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in src]
    gts = [np.array([float(g.sum())]) for g in gtc]
    return preds, gts, dnu["preds"], dnu["gts"]


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


def winkler(lo, hi, y, alpha=ALPHA):
    s = hi - lo
    if y < lo:
        s += (2.0 / alpha) * (lo - y)
    elif y > hi:
        s += (2.0 / alpha) * (y - hi)
    return s


PAN_PREDS, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_SCORES = np.array([nonconf(PAN_PREDS[i], PAN_GTS[i]) for i in range(len(PAN_PREDS))])
print(f"PanNuke cal {len(PAN_SCORES)} | NuInsSeg {len(NU_PREDS)} | "
      f"q0={empirical_quantile(PAN_SCORES, ALPHA):.3f}\n")


# ------------------------------------------------------------ methods (return c, w, s)
def m_pbo(order):
    m = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); m.warmstart(PAN_SCORES)
    c, w, s = [], [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, s


def m_aci(order):
    m = AdaptiveConformalInference(alpha_target=ALPHA, gamma=0.05)
    m.reset(); m.history_scores = list(PAN_SCORES)
    c, w, s = [], [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        cov = lo <= y <= hi
        c.append(cov); w.append(hi - lo); s.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]), cov)
    return c, w, s


class COP:
    def __init__(self, eta, lam=1.0, w=100, warm=None):
        self.eta, self.lam, self.w = eta, lam, w
        self.qhat = empirical_quantile(np.asarray(warm), ALPHA); self.q = self.qhat
        self.win = list(np.asarray(warm)[-w:])

    def get_q(self):
        return max(0.0, self.q)

    def update(self, s, q_used):
        self.qhat = max(0.0, self.qhat + self.eta * ((1.0 if s > q_used else 0.0) - ALPHA))
        self.win.append(float(s)); self.win = self.win[-self.w:]
        Fhat = float(np.mean(np.asarray(self.win) <= self.qhat))
        self.q = max(0.0, self.qhat - self.lam * (Fhat - (1.0 - ALPHA)))


def m_cop(order, eta=5.0):
    m = COP(eta, warm=PAN_SCORES); c, w, s = [], [], []
    for i in order:
        q = m.get_q(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]), q)
    return c, w, s


def m_flush(order, thresh=0.5):
    m = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); m.warmstart(PAN_SCORES)
    det = RollingShiftDetector(window=100).fit_baseline(PAN_SCORES)
    tgt, flushed, c, w, s = [], False, [], [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        sc = nonconf(NU_PREDS[i], NU_GTS[i]); tgt.append(sc)
        if not flushed and det.step(sc) >= thresh:
            m.scores = list(tgt[-WINDOW:]); flushed = True
        else:
            m.update(sc)
    return c, w, s


def m_adapt(order, target=0.9, cov_win=50, w_min=40, w_max=WINDOW):
    scores = list(PAN_SCORES[-w_max:]); eff = w_max; recent, c, w, s = [], [], [], []
    for i in order:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA) if scores else float("inf")
        lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]; cov = lo <= y <= hi
        c.append(cov); w.append(hi - lo); s.append(winkler(lo, hi, y))
        recent.append(cov); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target:
            eff = max(w_min, int(eff * 0.9))
        elif rc > target + 0.03:
            eff = min(w_max, int(eff * 1.05))
        scores.append(nonconf(NU_PREDS[i], NU_GTS[i])); scores = scores[-w_max:]
    return c, w, s


def wquantile(scores, weights, level):
    o = np.argsort(scores); s = np.asarray(scores)[o]; wt = np.asarray(weights)[o]
    cw = np.cumsum(wt) / wt.sum()
    return s[min(np.searchsorted(cw, level), len(s) - 1)]


def m_nexcp(order, rho=0.99):
    hist = list(PAN_SCORES); c, w, s = [], [], []
    for i in order:
        wts = rho ** (len(hist) - 1 - np.arange(len(hist)))
        q = wquantile(hist, wts, 1 - ALPHA)
        lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        hist.append(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, s


def m_fallback(order, target=0.9, eta=0.03, cov_win=50):
    m = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); m.warmstart(PAN_SCORES)
    mult, recent, c, w, s = 1.0, [], [], [], []
    for i in order:
        q = m.get_quantile() * mult; lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        cov = lo <= y <= hi
        c.append(cov); w.append(hi - lo); s.append(winkler(lo, hi, y))
        recent.append(cov); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target:
            mult *= (1 + eta)
        elif rc > target + 0.03:
            mult = max(1.0, mult * (1 - eta))
        mult = min(mult, 6.0)
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, s


def m_hybrid(order):
    pb = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); pb.warmstart(PAN_SCORES)
    aci = AdaptiveConformalInference(alpha_target=ALPHA, gamma=0.05)
    aci.reset(); aci.history_scores = list(PAN_SCORES)
    c, w, s = [], [], []
    for i in order:
        lo1, hi1 = interval(NU_PREDS[i], pb.get_quantile())
        lo2, hi2 = interval(NU_PREDS[i], aci.get_quantile())
        lo, hi = min(lo1, lo2), max(hi1, hi2); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        sc = nonconf(NU_PREDS[i], NU_GTS[i])
        pb.update(sc); aci.update(sc, lo2 <= y <= hi2)
    return c, w, s


def agg(fn, seeds=5):
    cs, ws, ss = [], [], []
    for sd in range(seeds):
        order = np.random.RandomState(sd).permutation(len(NU_PREDS))
        c, w, s = fn(order)
        cs.append(np.mean(c) * 100); ws.append(np.mean(w)); ss.append(np.mean(s))
    return np.mean(cs), np.std(cs), np.mean(ws), np.mean(ss), np.std(ss)


print("=" * 80)
print("TABLES 8f/9a with Winkler -- PathoSAM->NuInsSeg (cal PanNuke), 5 seeds, target 90%")
print("=" * 80)
print(f"{'Method':34s} | {'Coverage':>13s} | {'Width':>7s} | {'Winkler (lower=better)':>22s}")
print("-" * 80)
for name, fn in [
    ("PB-JCI Online (static op-point)", m_pbo),
    ("ACI", m_aci),
    ("NexCP (Barber 2023)", m_nexcp),
    ("COP (ICLR 2026, eta=5)", m_cop),
    ("Detector-flush (fast-recovery)", m_flush),
    ("Adaptive PB-JCI Online (ours)", m_adapt),
    ("C. Fallback-multiplier", m_fallback),
    ("D. Hybrid max(PB,ACI)", m_hybrid),
]:
    cm, cs, wm, sm, sd = agg(fn)
    print(f"{name:34s} | {cm:5.1f}+/-{cs:4.1f}% | {wm:7.2f} | {sm:10.2f}+/-{sd:5.2f}")
print("-" * 80)
print("Lower Winkler = better (penalises width AND miss). Adaptive PB-JCI Online lowest.")

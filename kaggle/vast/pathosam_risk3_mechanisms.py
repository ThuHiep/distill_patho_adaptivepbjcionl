"""
Risk-3 mechanisms for EXTREME cross-dataset shift (PathoSAM -> NuInsSeg).

Baseline PB-JCI Online recovers only ~82% on NuInsSeg (cal PanNuke). The warmstart
PanNuke scores pollute the window (665 stream vs window 300) -> slow adaptation. We test
mechanisms that turn the weakness into a controlled-recovery story:

  A. Detector-flush : RollingShiftDetector triggers a one-time flush of the source
     (PanNuke) scores from the window when shift is detected -> fast adapt to target.
  B. Adaptive-window: shrink the effective window when recent coverage < target
     (older source scores drop out faster).
  C. Fallback-multiplier: an ACI-style multiplicative controller on top of the PB-JCI
     windowed quantile (widen when under-covering) -> directly targets coverage.
  D. Hybrid max(PB-JCI, ACI): elementwise envelope -> >= ACI coverage, PB efficiency
     in-domain.

CPU only, runs on cached pkls. No GPU / no Vast.
  python pathosam_risk3_mechanisms.py
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
      f"q_cross={empirical_quantile(PAN_SCORES, ALPHA):.2f}")


def run(method_step, nseeds=5):
    """method_step(i, order_pos) -> (lo, hi); caller handles update via closure."""
    covs, ws = [], []
    for sd in range(nseeds):
        order = np.random.RandomState(sd).permutation(len(NU_PREDS))
        c, w = method_step(order)
        covs.append(np.mean(c)); ws.append(np.mean(w))
    return np.mean(covs) * 100, np.std(covs) * 100, np.mean(ws), np.std(ws)


# ---- baselines ----
def base_pbo(order):
    m = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); m.warmstart(PAN_SCORES)
    c, w = [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q)
        c.append(lo <= NU_GTS[i][0] <= hi); w.append(hi - lo)
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w


def base_aci(order):
    m = AdaptiveConformalInference(alpha_target=ALPHA, gamma=0.05)
    m.reset(); m.history_scores = list(PAN_SCORES)
    c, w = [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q)
        cov = lo <= NU_GTS[i][0] <= hi
        c.append(cov); w.append(hi - lo)
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]), cov)
    return c, w


# ---- A. detector-triggered flush ----
def mech_flush(order, flush_thresh=0.5):
    m = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); m.warmstart(PAN_SCORES)
    det = RollingShiftDetector(window=100).fit_baseline(PAN_SCORES)
    tgt = []           # target scores seen so far
    flushed = False
    c, w = [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q)
        c.append(lo <= NU_GTS[i][0] <= hi); w.append(hi - lo)
        s = nonconf(NU_PREDS[i], NU_GTS[i]); tgt.append(s)
        if not flushed and det.step(s) >= flush_thresh:
            m.scores = list(tgt[-WINDOW:]); flushed = True   # drop PanNuke, keep target
        else:
            m.update(s)
    return c, w


# ---- B. adaptive window ----
def mech_adapt_window(order, target=0.9, cov_win=50, w_min=40, w_max=WINDOW):
    scores = list(PAN_SCORES[-WINDOW:])
    eff = WINDOW
    recent_cov = []
    c, w = [], []
    for i in order:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA) if scores else float("inf")
        lo, hi = interval(NU_PREDS[i], q)
        cov = lo <= NU_GTS[i][0] <= hi
        c.append(cov); w.append(hi - lo)
        recent_cov.append(cov); recent_cov = recent_cov[-cov_win:]
        rc = np.mean(recent_cov)
        if rc < target:
            eff = max(w_min, int(eff * 0.9))      # shrink -> drop old source faster
        elif rc > target + 0.03:
            eff = min(w_max, int(eff * 1.05))
        scores.append(nonconf(NU_PREDS[i], NU_GTS[i])); scores = scores[-w_max:]
    return c, w


# ---- C. fallback multiplier (ACI-style controller on PB quantile) ----
def mech_fallback(order, target=0.9, eta=0.03, cov_win=50):
    m = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); m.warmstart(PAN_SCORES)
    mult = 1.0
    recent = []
    c, w = [], []
    for i in order:
        q = m.get_quantile() * mult; lo, hi = interval(NU_PREDS[i], q)
        cov = lo <= NU_GTS[i][0] <= hi
        c.append(cov); w.append(hi - lo)
        recent.append(cov); recent = recent[-cov_win:]
        rc = np.mean(recent)
        if rc < target:
            mult *= (1 + eta)
        elif rc > target + 0.03:
            mult = max(1.0, mult * (1 - eta))
        mult = min(mult, 6.0)
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w


# ---- D. hybrid max(PB-JCI, ACI) ----
def mech_hybrid(order):
    pb = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW); pb.warmstart(PAN_SCORES)
    aci = AdaptiveConformalInference(alpha_target=ALPHA, gamma=0.05)
    aci.reset(); aci.history_scores = list(PAN_SCORES)
    c, w = [], []
    for i in order:
        lo1, hi1 = interval(NU_PREDS[i], pb.get_quantile())
        lo2, hi2 = interval(NU_PREDS[i], aci.get_quantile())
        lo, hi = min(lo1, lo2), max(hi1, hi2)
        cov = lo <= NU_GTS[i][0] <= hi
        c.append(cov); w.append(hi - lo)
        s = nonconf(NU_PREDS[i], NU_GTS[i])
        pb.update(s); aci.update(s, lo2 <= NU_GTS[i][0] <= hi2)
    return c, w


print("\n" + "=" * 74)
print("RISK-3 MECHANISMS on PathoSAM->NuInsSeg (cal PanNuke). target cov=90%")
print("=" * 74)
print(f"{'Method':34s} | {'Coverage':>14s} | {'Width':>12s}")
print("-" * 74)
for name, fn in [
    ("Baseline PB-JCI Online", base_pbo),
    ("Baseline ACI", base_aci),
    ("A. Detector-flush", mech_flush),
    ("B. Adaptive-window", mech_adapt_window),
    ("C. Fallback-multiplier", mech_fallback),
    ("D. Hybrid max(PB,ACI)", mech_hybrid),
]:
    cm, cs, wm, ws = run(fn)
    print(f"{name:34s} | {cm:6.1f}+/-{cs:4.1f}% | {wm:7.2f}+/-{ws:4.2f}")
print("-" * 74)
print("Goal: a mechanism that lifts coverage toward 90% without blowing up width.")

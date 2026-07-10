from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kaggle" / "lib"))
from conformal import empirical_quantile, pb_count, pb_variance  

ALPHA = 0.1
WINDOW = 300
Z90 = 1.6448536269514722  

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

def nhat_sigma(p):
    n = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return n, sg

def nonconf(p, gt, use_sigma=True):
    if len(p["scores"]) == 0:
        return float(abs(gt[0]))
    n, sg = nhat_sigma(p)
    r = abs(gt[0] - n)
    return r / sg if use_sigma else r

def interval(p, q, use_sigma=True):
    if len(p["scores"]) == 0:
        return 0.0, 0.0
    n, sg = nhat_sigma(p)
    half = q * sg if use_sigma else q
    return max(0.0, n - half), n + half

def winkler(lo, hi, y, alpha=ALPHA):
    s = hi - lo
    if y < lo:
        s += (2.0 / alpha) * (lo - y)
    elif y > hi:
        s += (2.0 / alpha) * (y - hi)
    return s

PAN_PREDS, PAN_GTS, NU_PREDS, NU_GTS = load()

def cal_scores(use_sigma):
    return np.array([nonconf(PAN_PREDS[i], PAN_GTS[i], use_sigma)
                     for i in range(len(PAN_PREDS))])

def _collect(order, lo_hi_fn, update_fn=None, state=None):
    c, w, s = [], [], []
    for i in order:
        lo, hi = lo_hi_fn(i, state)
        y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        if update_fn is not None:
            update_fn(i, state, lo <= y <= hi)
    return c, w, s

def v_naive(order):
    c, w, s = [], [], []
    for i in order:
        if len(NU_PREDS[i]["scores"]) == 0:
            lo, hi = 0.0, 0.0
        else:
            n, sg = nhat_sigma(NU_PREDS[i]); lo, hi = max(0.0, n - Z90 * sg), n + Z90 * sg
        y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
    return c, w, s

def v_static(order, use_sigma=True):
    q = empirical_quantile(cal_scores(use_sigma), ALPHA)
    c, w, s = [], [], []
    for i in order:
        lo, hi = interval(NU_PREDS[i], q, use_sigma); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
    return c, w, s

def v_online_fixed(order, use_sigma=True):
    scores = list(cal_scores(use_sigma)[-WINDOW:])
    c, w, s = [], [], []
    for i in order:
        q = empirical_quantile(np.asarray(scores[-WINDOW:]), ALPHA) if scores else float("inf")
        lo, hi = interval(NU_PREDS[i], q, use_sigma); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        scores.append(nonconf(NU_PREDS[i], NU_GTS[i], use_sigma)); scores = scores[-WINDOW:]
    return c, w, s

def v_full(order, use_sigma=True, target=0.9, cov_win=50, w_min=40, w_max=WINDOW):
    scores = list(cal_scores(use_sigma)[-w_max:]); eff = w_max; recent = []
    c, w, s = [], [], []
    for i in order:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA) if scores else float("inf")
        lo, hi = interval(NU_PREDS[i], q, use_sigma); y = NU_GTS[i][0]; cov = lo <= y <= hi
        c.append(cov); w.append(hi - lo); s.append(winkler(lo, hi, y))
        recent.append(cov); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target:
            eff = max(w_min, int(eff * 0.9))
        elif rc > target + 0.03:
            eff = min(w_max, int(eff * 1.05))
        scores.append(nonconf(NU_PREDS[i], NU_GTS[i], use_sigma)); scores = scores[-w_max:]
    return c, w, s

def agg(fn, seeds=5):
    cs, ws, ss = [], [], []
    for sd in range(seeds):
        order = np.random.RandomState(sd).permutation(len(NU_PREDS))
        c, w, s = fn(order)
        cs.append(np.mean(c) * 100); ws.append(np.mean(w)); ss.append(np.mean(s))
    return np.mean(cs), np.std(cs), np.mean(ws), np.std(ws), np.mean(ss), np.std(ss)

ROWS = [
    ("Naive PB (no conformal)",            lambda o: v_naive(o)),
    ("- PB-sigma (raw residual)",          lambda o: v_full(o, use_sigma=False)),
    ("- online (static split)",            lambda o: v_static(o, True)),
    ("- adaptive (fixed window)",          lambda o: v_online_fixed(o, True)),
    ("Full Adaptive PB-JCI Online",        lambda o: v_full(o, True)),
]

print("=" * 78)
print("ABLATION (leave-one-out) -- PathoSAM->NuInsSeg, cal=PanNuke, 5 seeds, target 90%")
print("=" * 78)
print(f"{'Variant':32s} | {'Coverage':>13s} | {'Width':>15s} | {'Winkler':>15s}")
print("-" * 78)
for name, fn in ROWS:
    cm, csd, wm, wsd, sm, ssd = agg(fn)
    print(f"{name:32s} | {cm:5.1f}+/-{csd:4.1f}% | {wm:6.2f}+/-{wsd:5.2f} | {sm:6.2f}+/-{ssd:5.2f}")
print("-" * 78)

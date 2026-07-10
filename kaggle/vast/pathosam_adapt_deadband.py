"""
Test: does a DEAD-BAND make the adaptive window INERT under stable/mild shift
(so it equals fixed PB-JCI Online there) while still recovering under extreme shift?

Compares 3 controllers on the SAME streams:
  - fixed     : PB-JCI Online, window 300 (no adaptation)
  - adapt-old : shrink when recent-cov < 0.90, grow when > 0.93   (current code)
  - adapt-new : shrink when recent-cov < 0.85, grow when > 0.90   (dead-band 0.85-0.90)

Streams:
  Controlled (joint K=5, PathoSAM): cal=in_dist -> test {in_dist, mild, severe}
  Extreme  (K=1, PathoSAM->NuInsSeg): cal=PanNuke -> test NuInsSeg

Inert if adapt-new ~= fixed on in_dist/mild. Useful if adapt-new still ~90% on extreme.
  python pathosam_adapt_deadband.py    (CPU, cached pkl)
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kaggle" / "lib"))
from conformal import empirical_quantile, pb_count, pb_variance  # noqa

ALPHA, WINDOW, K = 0.1, 300, 5

with open(REPO / "data" / "pathosam_predictions.pkl", "rb") as f:
    D = pickle.load(f)
PBS = D["predictions_by_setting"]
GT = np.asarray(D["gt_counts"])
N = len(GT)
with open(REPO / "data" / "pathosam_nuinsseg_preds.pkl", "rb") as f:
    DNU = pickle.load(f)
NU_PREDS, NU_GTS = DNU["preds"], DNU["gts"]
print(f"PathoSAM Fold-3 {N} | NuInsSeg {len(NU_PREDS)} | settings={list(PBS.keys())}")


# ---------------- joint K=5 ----------------
def jscore(p, gt):
    if len(p["scores"]) == 0:
        return float(np.abs(gt).max())
    n = pb_count(p["scores"], p["probs"]); sg = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
    return max(abs(gt[k] - n[k]) / sg[k] for k in range(K))


def jinterval(p, q):
    if len(p["scores"]) == 0:
        return np.zeros(K), np.zeros(K)
    n = pb_count(p["scores"], p["probs"]); sg = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
    return np.maximum(0, n - q * sg), n + q * sg


# ---------------- total K=1 ----------------
def tscore(p, gt):
    if len(p["scores"]) == 0:
        return float(abs(gt[0]))
    n = pb_count(p["scores"], p["probs"])[0]; sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return abs(gt[0] - n) / sg


def tinterval(p, q):
    if len(p["scores"]) == 0:
        return 0.0, 0.0
    n = pb_count(p["scores"], p["probs"])[0]; sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return max(0.0, n - q * sg), n + q * sg


def winkler1(lo, hi, y):
    s = hi - lo
    if y < lo:
        s += (2.0 / ALPHA) * (lo - y)
    elif y > hi:
        s += (2.0 / ALPHA) * (y - hi)
    return s


def run_stream(warm, items, score_fn, interval_fn, cover_fn, width_fn,
               shrink_thr=None, grow_thr=0.93, cov_win=50, grow_f=1.05, shrink_f=0.9,
               w_min=40, wink_fn=None):
    """shrink_thr=None => fixed window; else adaptive: dead-band + (shrink_f, w_min) khi tụt."""
    scores = list(np.asarray(warm)[-WINDOW:]); eff = WINDOW; recent = []
    cov, wid, wk = [], [], []
    for p, gt in items:
        ref = scores[-eff:] if shrink_thr is not None else scores[-WINDOW:]
        q = empirical_quantile(np.asarray(ref), ALPHA) if ref else float("inf")
        lo, hi = interval_fn(p, q)
        c = cover_fn(lo, hi, gt); cov.append(c); wid.append(width_fn(lo, hi))
        if wink_fn is not None:
            wk.append(wink_fn(lo, hi, gt))
        if shrink_thr is not None:
            recent.append(c); recent = recent[-cov_win:]; rc = np.mean(recent)
            if rc < shrink_thr:
                eff = max(w_min, int(eff * shrink_f))
            elif rc > grow_thr:
                eff = min(WINDOW, int(eff * grow_f))
        scores.append(score_fn(p, gt)); scores = scores[-WINDOW:]
    return np.mean(cov) * 100, np.mean(wid), (np.mean(wk) if wk else float("nan"))


# (name, shrink_thr, grow_thr, cov_win, grow_f, shrink_f, w_min)
CTRL = [
    ("fixed", None, 0.93, 50, 1.05, 0.9, 40),
    ("adapt-old", 0.90, 0.93, 50, 1.05, 0.9, 40),
    ("B-fast cw30 s.7", 0.80, 0.90, 30, 1.15, 0.7, 25),
    ("B-mid cw50 s.75", 0.85, 0.90, 50, 1.10, 0.75, 30),
]


def eval_joint(setting, nseeds=5):
    out = {c[0]: [] for c in CTRL}; outw = {c[0]: [] for c in CTRL}
    for sd in range(nseeds):
        idx = np.random.RandomState(sd).permutation(N)
        cal, test = idx[:N // 2], idx[N // 2:]
        warm = np.array([jscore(PBS["in_dist"][i], GT[i]) for i in cal])
        items = [(PBS[setting][i], GT[i]) for i in test]
        for name, sthr, gthr, cw, gf, sf, wm in CTRL:
            c, w, _ = run_stream(warm, items, jscore, jinterval,
                                 lambda lo, hi, gt: bool(((gt >= lo) & (gt <= hi)).all()),
                                 lambda lo, hi: float((hi - lo).mean()),
                                 shrink_thr=sthr, grow_thr=gthr, cov_win=cw, grow_f=gf,
                                 shrink_f=sf, w_min=wm)
            out[name].append(c); outw[name].append(w)
    return out, outw


def eval_extreme(nseeds=5):
    pan = [{"scores": np.asarray(p["scores"]), "probs": np.ones((len(p["scores"]), 1)), "K": 1}
           for p in PBS["in_dist"]]
    pgt = [np.array([float(g.sum())]) for g in GT]
    warm = np.array([tscore(pan[i], pgt[i]) for i in range(len(pan))])
    out = {c[0]: [] for c in CTRL}; outw = {c[0]: [] for c in CTRL}; outk = {c[0]: [] for c in CTRL}
    for sd in range(nseeds):
        order = np.random.RandomState(sd).permutation(len(NU_PREDS))
        items = [(NU_PREDS[i], NU_GTS[i]) for i in order]
        for name, sthr, gthr, cw, gf, sf, wm in CTRL:
            c, w, k = run_stream(warm, items, tscore, tinterval,
                                 lambda lo, hi, gt: lo <= gt[0] <= hi,
                                 lambda lo, hi: hi - lo,
                                 shrink_thr=sthr, grow_thr=gthr, cov_win=cw, grow_f=gf,
                                 shrink_f=sf, w_min=wm,
                                 wink_fn=lambda lo, hi, gt: winkler1(lo, hi, gt[0]))
            out[name].append(c); outw[name].append(w); outk[name].append(k)
    return out, outw, outk


print("\n" + "=" * 78)
print("DEAD-BAND TEST — Coverage% / Width, target 90%, 5 seeds")
print("=" * 78)
names = [c[0] for c in CTRL]
print(f"{'Stream':20s} | " + " | ".join(f"{n:>20s}" for n in names))
print("-" * 96)
for setting, label in [("in_dist", "in_dist (NO shift)"), ("mild_shift", "mild"), ("severe_shift", "severe")]:
    o, w = eval_joint(setting)
    cells = [f"{np.mean(o[n]):5.1f}% /{np.mean(w[n]):6.2f}" for n in names]
    print(f"{label:20s} | " + " | ".join(f"{c:>20s}" for c in cells))
oe, we, ke = eval_extreme()
cells = [f"{np.mean(oe[n]):5.1f}% /{np.mean(we[n]):6.2f}" for n in names]
print(f"{'NuInsSeg (EXTREME)':20s} | " + " | ".join(f"{c:>20s}" for c in cells))
print("-" * 96)
print("WINKLER trên NuInsSeg (thấp = tốt; so với COP'26 = 113.13):")
for n in names:
    print(f"  {n:22s} : coverage {np.mean(oe[n]):5.1f}% | width {np.mean(we[n]):6.2f} | Winkler {np.mean(ke[n]):7.2f}")
print("-" * 96)
print("WANT: adapt-new ~= fixed on in_dist/mild (inert);  ~90% + Winkler thấp trên NuInsSeg.")

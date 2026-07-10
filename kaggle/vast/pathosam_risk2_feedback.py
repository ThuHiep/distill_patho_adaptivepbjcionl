"""
Risk-2 feedback-robustness ablations for PB-JCI Online (joint K=5, PathoSAM).

PB-JCI Online assumes streaming GT feedback. Reviewers ask: what if feedback is
delayed / sparse / noisy? We stress-test on the SEVERE-shift stream (cal = in_dist),
degrading the feedback that drives the online window update; the prediction intervals
are still produced for EVERY sample.

  - Delayed  : score for step t only updates the window at t+d (d = 10/50/100).
  - Sparse   : only a fraction p of samples give feedback (p = 50/25/10%).
  - Noisy    : GT counts used for the update are perturbed by N(0, sigma) per class
               (sigma = 1/2/3 nuclei), modelling small annotation error.

Coverage/width measured over 5 cal seeds. CPU only, cached pkl. No GPU.
  python pathosam_risk2_feedback.py
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kaggle" / "lib"))
from conformal import PBAwareJointConformalOnline, pb_count, pb_variance  # noqa

ALPHA, WINDOW, K = 0.1, 300, 5

with open(REPO / "data" / "pathosam_predictions.pkl", "rb") as f:
    D = pickle.load(f)
PBS = D["predictions_by_setting"]
GT = np.asarray(D["gt_counts"])
N = len(GT)
print(f"PathoSAM clean Fold-3: {N} imgs | settings={list(PBS.keys())}")


def joint_score(p, gt):
    if len(p["scores"]) == 0:
        return float(np.abs(gt).max())
    n = pb_count(p["scores"], p["probs"])
    sg = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
    return max(abs(gt[k] - n[k]) / sg[k] for k in range(K))


def interval(p, q):
    if len(p["scores"]) == 0:
        return np.zeros(K), np.zeros(K)
    n = pb_count(p["scores"], p["probs"])
    sg = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
    return np.maximum(0, n - q * sg), n + q * sg


def split(seed):
    idx = np.random.RandomState(seed).permutation(N)
    return idx[:N // 2], idx[N // 2:]


def warm_scores(cal_idx):
    return np.array([joint_score(PBS["in_dist"][i], GT[i]) for i in cal_idx])


def stream_eval(test_idx, feedback="full", d=0, p=1.0, sigma=0.0, cal_idx=None, seed=0):
    """Run PB-JCI Online over the severe-shift test stream with degraded feedback."""
    m = PBAwareJointConformalOnline(alpha=ALPHA, window=WINDOW)
    m.warmstart(warm_scores(cal_idx))
    rng = np.random.RandomState(1000 + seed)
    pending = []          # for delayed: (apply_at, score)
    cov, wid = [], []
    for t, i in enumerate(test_idx):
        pr = PBS["severe_shift"][i]
        q = m.get_quantile()
        lo, hi = interval(pr, q)
        cov.append(bool(((GT[i] >= lo) & (GT[i] <= hi)).all()))
        wid.append(float((hi - lo).mean()))

        # compute the (possibly degraded) feedback score
        gt_fb = GT[i].astype(float)
        if sigma > 0:
            gt_fb = np.maximum(0, gt_fb + rng.normal(0, sigma, size=K))
        s = joint_score(pr, gt_fb)

        if feedback == "sparse" and rng.random() > p:
            pass                                   # no feedback this step
        elif feedback == "delayed":
            pending.append((t + d, s))
            for ap, sc in [x for x in pending if x[0] <= t]:
                m.update(sc)
            pending = [x for x in pending if x[0] > t]
        else:
            m.update(s)
    return float(np.mean(cov)) * 100, float(np.mean(wid))


def agg(**kw):
    cs, ws = [], []
    for sd in range(5):
        cal, test = split(sd)
        c, w = stream_eval(test, cal_idx=cal, seed=sd, **kw)
        cs.append(c); ws.append(w)
    return np.mean(cs), np.std(cs), np.mean(ws), np.std(ws)


print("\n" + "=" * 70)
print("RISK-2 FEEDBACK ROBUSTNESS — PB-JCI Online on SEVERE shift (5 seeds)")
print("=" * 70)
print(f"{'Feedback condition':30s} | {'Coverage':>14s} | {'Width':>12s}")
print("-" * 70)
rows = [
    ("Full feedback (baseline)", dict(feedback="full")),
    ("Delayed lag=10", dict(feedback="delayed", d=10)),
    ("Delayed lag=50", dict(feedback="delayed", d=50)),
    ("Delayed lag=100", dict(feedback="delayed", d=100)),
    ("Sparse 50%", dict(feedback="sparse", p=0.50)),
    ("Sparse 25%", dict(feedback="sparse", p=0.25)),
    ("Sparse 10%", dict(feedback="sparse", p=0.10)),
    ("Noisy sigma=1", dict(feedback="noisy", sigma=1.0)),
    ("Noisy sigma=2", dict(feedback="noisy", sigma=2.0)),
    ("Noisy sigma=3", dict(feedback="noisy", sigma=3.0)),
]
for name, kw in rows:
    cm, cs, wm, ws = agg(**kw)
    print(f"{name:30s} | {cm:6.1f}+/-{cs:4.1f}% | {wm:7.2f}+/-{ws:4.2f}")
print("-" * 70)
print("Robust if coverage stays near the full-feedback value as feedback degrades.")

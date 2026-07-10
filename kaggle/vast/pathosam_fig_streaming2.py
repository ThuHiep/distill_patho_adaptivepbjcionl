"""Two-panel streaming figure (RetroAdj-style): rolling LOCAL COVERAGE | rolling LOCAL WIDTH
over an abrupt-shift stream (PanNuke in-dist -> NuInsSeg at change-point), PathoSAM backbone.
Left: coverage vs 90% target; Right: interval width. Shows static collapses while adaptive holds
coverage, and that adaptive's wider intervals are the price of covering the harder domain.
CPU, cached pkls.  python pathosam_fig_streaming2.py
Writes figures/F2_streaming_coverage.png (overwrites the old single-panel version).
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kaggle" / "lib"))
from conformal import empirical_quantile, pb_count, pb_variance, PBAwareJointConformalOnline  # noqa

ALPHA = 0.1
WINDOW = 300
FIGDIR = REPO / "figures"


def load():
    d = REPO / "data"
    with open(d / "pathosam_predictions.pkl", "rb") as f:
        dpan = pickle.load(f)
    with open(d / "pathosam_nuinsseg_preds.pkl", "rb") as f:
        dnu = pickle.load(f)
    src = dpan["predictions_by_setting"]["in_dist"]
    gtc = np.asarray(dpan["gt_counts"])
    pan = [{"scores": np.asarray(p["scores"]),
            "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in src]
    pgt = [np.array([float(g.sum())]) for g in gtc]
    return pan, pgt, dnu["preds"], dnu["gts"]


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


PAN_IN, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_SCORES = np.array([nonconf(PAN_IN[i], PAN_GTS[i]) for i in range(len(PAN_IN))])


def stream_abrupt(seed=0, n_pre=150, n_post=300):
    rng = np.random.RandomState(seed); items = []
    for i in rng.choice(len(PAN_IN), n_pre):
        items.append((PAN_IN[i], PAN_GTS[i]))
    for i in rng.choice(len(NU_PREDS), n_post):
        items.append((NU_PREDS[i], NU_GTS[i]))
    return items, n_pre


def run_static(items):
    q = empirical_quantile(PAN_SCORES, ALPHA)
    c, w = [], []
    for p, gt in items:
        lo, hi = interval(p, q); c.append(int(lo <= gt[0] <= hi)); w.append(hi - lo)
    return c, w


def run_online(items):
    m = PBAwareJointConformalOnline(ALPHA, WINDOW).warmstart(PAN_SCORES); c, w = [], []
    for p, gt in items:
        lo, hi = interval(p, m.get_quantile()); c.append(int(lo <= gt[0] <= hi)); w.append(hi - lo)
        m.update(nonconf(p, gt))
    return c, w


def run_adapt(items, target=0.9, cov_win=50, w_min=40):
    scores = list(PAN_SCORES[-WINDOW:]); eff = WINDOW; recent, c, w = [], [], []
    for p, gt in items:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA)
        lo, hi = interval(p, q); cov = int(lo <= gt[0] <= hi); c.append(cov); w.append(hi - lo)
        recent.append(cov); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target:
            eff = max(w_min, int(eff * 0.9))
        elif rc > target + 0.03:
            eff = min(WINDOW, int(eff * 1.05))
        scores.append(nonconf(p, gt)); scores = scores[-WINDOW:]
    return c, w


def rolling(x, win=50, scale=1.0):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - win + 1):i + 1].mean() for i in range(len(x))]) * scale


METHODS = [
    ("Static conformal (fixed)", run_static, "#d62728", 2.0),
    ("PB-JCI Online-Fixed", run_online, "#1f77b4", 2.0),
    ("Adaptive PB-JCI Online (ours)", run_adapt, "#2ca02c", 2.4),
]


def main():
    items, cp = stream_abrupt(seed=0)
    res = {name: fn(items) for name, fn, _, _ in METHODS}
    fig, (axc, axw) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    for name, _, col, lw in METHODS:
        c, w = res[name]
        axc.plot(rolling(c, scale=100.0), label=name, c=col, lw=lw)
        axw.plot(rolling(w, scale=1.0), label=name, c=col, lw=lw)

    # left: coverage
    axc.axvline(cp, ls="--", c="black", lw=1.1)
    axc.text(cp - 6, 33, "switch to NuInsSeg\n(change-point)", fontsize=8, ha="right")
    axc.axhline(90, ls=":", c="gray")
    axc.text(200, 96, "target 90%", fontsize=8, color="gray")
    axc.set_xlabel("Step in stream"); axc.set_ylabel("Local coverage (%) [window 50]")
    axc.set_title("(a) Local coverage over time")
    axc.set_ylim(20, 100)

    # right: width
    axw.axvline(cp, ls="--", c="black", lw=1.1)
    axw.text(cp - 6, 72, "switch to NuInsSeg\n(change-point)", fontsize=8, ha="right")
    axw.set_xlabel("Step in stream"); axw.set_ylabel("Local width [window 50]")
    axw.set_title("(b) Local interval width over time")

    # single shared legend below both panels (outside the axes, no overlap)
    handles, labels = axc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = FIGDIR / "F2_streaming_coverage.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""
Figures for the report/paper (PathoSAM -> NuInsSeg, cal PanNuke). CPU, cached pkls.
  python kaggle/vast/pathosam_figures.py
Outputs PNG into ./figures/:
  F1_coverage_winkler.png  -- coverage vs Winkler trade-off (Adaptive at best corner)
  F2_streaming_coverage.png-- rolling coverage over an abrupt-shift stream
  F3_conditional_coverage.png- per-regime coverage (static under-covers severe; adaptive holds)
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
from conformal import (PBAwareJointConformalOnline, empirical_quantile,  # noqa
                       pb_count, pb_variance)

ALPHA, WINDOW = 0.1, 300
FIGDIR = REPO / "figures"; FIGDIR.mkdir(exist_ok=True)


def load():
    d = REPO / "data"
    with open(d / "pathosam_predictions.pkl", "rb") as f:
        dpan = pickle.load(f)
    with open(d / "pathosam_nuinsseg_preds.pkl", "rb") as f:
        dnu = pickle.load(f)
    s = dpan["predictions_by_setting"]; gtc = np.asarray(dpan["gt_counts"])
    gts = [np.array([float(g.sum())]) for g in gtc]
    mk = lambda key: [{"scores": np.asarray(p["scores"]),
                       "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in s[key]]
    pan = {"in": mk("in_dist"), "mild": mk("mild_shift"), "sev": mk("severe_shift")}
    return pan, gts, dnu["preds"], dnu["gts"]


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


PAN, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_IN = PAN["in"]
PAN_SCORES = np.array([nonconf(PAN_IN[i], PAN_GTS[i]) for i in range(len(PAN_IN))])


# ---------------------------------------------------------------- F1: coverage-Winkler
def fig1():
    # verified numbers (5 seeds) from pathosam_winkler_table3.py
    pts = [
        ("PB-JCI Online (static op-pt)", 81.8, 125.96, "#888888"),
        ("ACI", 84.0, 129.55, "#d62728"),
        ("NexCP (2023)", 84.7, 119.56, "#9467bd"),
        ("COP (ICLR 2026)", 87.9, 113.13, "#ff7f0e"),
        ("Detector-flush (variant)", 88.7, 110.07, "#1f77b4"),
        ("Adaptive PB-JCI Online (ours)", 90.0, 108.67, "#2ca02c"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.set_xlim(80.8, 92.2); ax.set_ylim(104, 132)
    for name, cov, wk, c in pts:
        best = "ours" in name
        ax.scatter(cov, wk, s=200 if best else 90, c=c, zorder=3,
                   edgecolors="black", linewidths=1.5 if best else 0.6,
                   marker="*" if best else "o")
        if best:
            ax.annotate(name, (cov, wk), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")
        else:
            ax.annotate(name, (cov, wk), textcoords="offset points",
                        xytext=(8, 6), fontsize=9)
    ax.axvline(90, ls="--", c="gray", lw=1)
    ax.text(90.1, 131.5, "target 90%", va="top", fontsize=8, color="gray")
    ax.set_xlabel("Coverage (%)  — gần 90% là tốt")
    ax.set_ylabel("Winkler / Interval score  — THẤP là tốt")
    ax.set_title("Trade-off coverage–efficiency dưới extreme shift\n(PathoSAM → NuInsSeg)")
    ax.annotate("góc tốt nhất\n(coverage 90% +\nWinkler thấp nhất)",
                (90.0, 108.67), xytext=(83.4, 116),
                arrowprops=dict(arrowstyle="->", color="#2ca02c"), color="#2ca02c", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGDIR / "F1_coverage_winkler.png", dpi=150)
    print("wrote F1_coverage_winkler.png")


# ---------------------------------------------------------------- streams + methods
def stream_abrupt(seed=0, n_pre=150, n_post=300):
    rng = np.random.RandomState(seed); items, labels = [], []
    for i in rng.choice(len(PAN_IN), n_pre): items.append((PAN_IN[i], PAN_GTS[i])); labels.append("in-dist")
    for i in rng.choice(len(NU_PREDS), n_post): items.append((NU_PREDS[i], NU_GTS[i])); labels.append("NuInsSeg")
    return items, labels, n_pre


def run_static(items):
    q = empirical_quantile(PAN_SCORES, ALPHA)  # fixed cal quantile, never updates
    return [int(interval(p, q)[0] <= gt[0] <= interval(p, q)[1]) for p, gt in items]


def run_online(items):
    m = PBAwareJointConformalOnline(ALPHA, WINDOW).warmstart(PAN_SCORES); c = []
    for p, gt in items:
        lo, hi = interval(p, m.get_quantile()); c.append(int(lo <= gt[0] <= hi))
        m.update(nonconf(p, gt))
    return c


def run_adapt(items, target=0.9, cov_win=50, w_min=40):
    scores = list(PAN_SCORES[-WINDOW:]); eff = WINDOW; recent, c = [], []
    for p, gt in items:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA)
        lo, hi = interval(p, q); cov = int(lo <= gt[0] <= hi); c.append(cov)
        recent.append(cov); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target: eff = max(w_min, int(eff * 0.9))
        elif rc > target + 0.03: eff = min(WINDOW, int(eff * 1.05))
        scores.append(nonconf(p, gt)); scores = scores[-WINDOW:]
    return c


def rolling(c, w=50):
    c = np.asarray(c, float)
    return np.array([c[max(0, i - w + 1):i + 1].mean() for i in range(len(c))]) * 100


# ---------------------------------------------------------------- F2: streaming trace
def fig2():
    items, _, cp = stream_abrupt(seed=0)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(rolling(run_static(items)), label="Conformal tĩnh (cố định)", c="#d62728", lw=2)
    ax.plot(rolling(run_online(items)), label="PB-JCI Online (cửa sổ cố định)", c="#1f77b4", lw=2)
    ax.plot(rolling(run_adapt(items)), label="Adaptive PB-JCI Online (ours)", c="#2ca02c", lw=2.4)
    ax.axvline(cp, ls="--", c="black", lw=1.2); ax.text(cp + 4, 30, "← đổi sang NuInsSeg\n(change-point)", fontsize=9)
    ax.axhline(90, ls=":", c="gray"); ax.text(2, 91, "target 90%", fontsize=8, color="gray")
    ax.set_xlabel("Bước trong stream"); ax.set_ylabel("Rolling coverage (%) [cửa sổ 50]")
    ax.set_title("Coverage theo thời gian dưới shift đột ngột\n(tĩnh sụp tại change-point; adaptive kéo về 90%)")
    ax.set_ylim(20, 100); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGDIR / "F2_streaming_coverage.png", dpi=150)
    print("wrote F2_streaming_coverage.png")


# ---------------------------------------------------------------- F3: conditional coverage
def fig3():
    rng = np.random.RandomState(0); per = 150
    items, labels = [], []
    for name, key in [("in-dist", "in"), ("mild", "mild"), ("severe", "sev")]:
        for i in rng.choice(len(PAN[key]), per): items.append((PAN[key][i], PAN_GTS[i])); labels.append(name)
    segs = ["in-dist", "mild", "severe"]
    def seg_cov(c):
        c = np.asarray(c, float)
        return [c[[i for i, l in enumerate(labels) if l == s]].mean() * 100 for s in segs]
    cov_static = seg_cov(run_static(items)); cov_adapt = seg_cov(run_adapt(items))
    x = np.arange(len(segs)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.bar(x - w / 2, cov_static, w, label="Conformal tĩnh", color="#d62728")
    ax.bar(x + w / 2, cov_adapt, w, label="Adaptive PB-JCI Online", color="#2ca02c")
    ax.axhline(90, ls="--", c="gray"); ax.text(-0.4, 91, "target 90%", fontsize=8, color="gray")
    for i, v in enumerate(cov_static): ax.text(i - w / 2, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    for i, v in enumerate(cov_adapt): ax.text(i + w / 2, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}\n(shift tăng dần →)" if i == 1 else s for i, s in enumerate(segs)])
    ax.set_ylabel("Coverage theo regime (%)"); ax.set_ylim(0, 105)
    ax.set_title("Conditional coverage: tĩnh under-cover regime shift mạnh,\nadaptive giữ ~90% ở mọi regime")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGDIR / "F3_conditional_coverage.png", dpi=150)
    print("wrote F3_conditional_coverage.png")


fig1(); fig2(); fig3()
print(f"\nAll figures in: {FIGDIR}")

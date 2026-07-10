from __future__ import annotations
from pathlib import Path
import json

HERE = Path(__file__).parent
OUT = HERE / "sam3_pannuke_phaseD_streaming.ipynb"
LIB_DIR = HERE / "lib"

CONFORMAL = "%%writefile conformal.py\n" + (LIB_DIR / "conformal.py").read_text(encoding="utf-8")

def md(*lines: str) -> dict:
    src = [ln if ln.endswith("\n") else ln + "\n" for ln in lines]
    if src:
        src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(body: str) -> dict:
    lines = body.lstrip("\n").splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}

cells: list[dict] = []

cells.append(md(
    "# Phase D — Streaming conformal under temporal drift (CPU)",
    "",
    "**Goal:** Section 4.4 figure — running coverage & interval width **over time** as the",
    "input stream drifts (in-dist → mild → severe → recover). Shows how each online method",
    "reacts at change-points: split methods lose coverage, ACI recovers but width explodes,",
    "**PB-JCI Online recovers coverage with stable width**.",
    "",
    "**No GPU / no model.** Reuses `phase_C_preds_seed42.pkl` (cached inference from Phase C/Vast).",
    "Runs in minutes on CPU.",
    "",
    "**Attach dataset:** `sam3-q1-multiseed-ckpts` (holds `phase_C_preds_seed{42,100,200}.pkl`).",
))

cells.append(md("## 00 — Setup"))
cells.append(code('''
import os, glob, json, pickle, time
import numpy as np
import matplotlib.pyplot as plt
print("numpy", np.__version__)
'''))

cells.append(md("## 01 — conformal.py (baked in)"))
cells.append(code(CONFORMAL))
cells.append(code('''
import sys
if "." not in sys.path:
    sys.path.insert(0, ".")
from conformal import (MarginalSplitConformal, AdaptiveConformalInference, ShiftAwareACI,
                       PBAwareJointConformal, PBAwareJointConformalOnline,
                       RollingShiftDetector, empirical_quantile, pb_count, pb_variance)
print("conformal loaded.")
'''))

cells.append(md("## 02 — Load cached predictions"))
cells.append(code('''
WORK = "/kaggle/working"
# Prefer the multi-seed pkls (phase_C_preds_seed42/100/200.pkl); use seed 42 for the
# streaming figure (variance comes from the 5 stream seeds). Fall back to old single name.
PKL = None
for s in [42, 100, 200]:
    hits = glob.glob(f"/kaggle/input/**/phase_C_preds_seed{s}.pkl", recursive=True)
    if hits:
        PKL = hits[0]; break
if PKL is None:  # backward compat with the old single-seed pkl
    hits = glob.glob("/kaggle/input/**/phase_C_predictions.pkl", recursive=True)
    PKL = hits[0] if hits else None
assert PKL, "No phase_C_preds_seed*.pkl found - attach dataset sam3-q1-multiseed-ckpts"
print("Loading:", PKL)
with open(PKL, "rb") as f:
    d = pickle.load(f)
predictions_by_setting = d["predictions_by_setting"]
gt_counts = np.asarray(d["gt_counts"])
print("settings:", list(predictions_by_setting.keys()), "| N =", len(gt_counts))
'''))

cells.append(md(
    "## 03 — Build a temporal-drift stream",
    "",
    "Stream = in_dist → mild → severe → in_dist (degrade then recover), patches sampled",
    "per segment. Change-points let us see each method's reaction & recovery.",
))
cells.append(code('''
ALPHA = 0.1
SEG_LEN = 350
ORDER = [("in_dist", SEG_LEN), ("mild_shift", SEG_LEN),
         ("severe_shift", SEG_LEN), ("in_dist", SEG_LEN)]
CHANGE_POINTS = np.cumsum([n for _, n in ORDER])[:-1]

def _score(p, gt):
    if len(p["scores"]) == 0:
        return float(abs(gt).max())
    n = pb_count(p["scores"], p["probs"]); sg = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
    return max(abs(gt[k] - n[k]) / sg[k] for k in range(len(gt)))

def _interval(p, q, K):
    if len(p["scores"]) == 0:
        return np.zeros(K), np.zeros(K)
    n = pb_count(p["scores"], p["probs"]); sg = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
    return np.maximum(0, n - q * sg), n + q * sg

def build_stream(seed):
    rng = np.random.RandomState(seed)
    N = len(gt_counts)
    pools = {s: rng.permutation(N) for s in predictions_by_setting}
    ptr = {s: 0 for s in predictions_by_setting}
    sp, sg, seg = [], [], []
    for setting, n in ORDER:
        for _ in range(n):
            i = pools[setting][ptr[setting] % N]; ptr[setting] += 1
            sp.append(predictions_by_setting[setting][i]); sg.append(gt_counts[i]); seg.append(setting)
    # calibration scores from a fresh in_dist draw (disjoint-ish)
    cal_idx = pools["in_dist"][-(N // 3):]
    cal_scores = np.array([_score(predictions_by_setting["in_dist"][i], gt_counts[i]) for i in cal_idx])
    return sp, sg, seg, cal_scores

print(f"Stream length = {sum(n for _, n in ORDER)} | change-points at {CHANGE_POINTS.tolist()}")
'''))

cells.append(md("## 04 — Run online methods over the stream (avg over seeds)"))
cells.append(code('''
ROLL = 100
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["aci", "sa_aci", "pb_jci_online", "pb_jci_split"]
mlabel = {"aci": "ACI", "sa_aci": "SA-ACI", "pb_jci_online": "PB-JCI Online",
          "pb_jci_split": "PB-JCI (split)"}

def run_stream(sp, sg, cal_scores):
    K = len(sg[0]); T = len(sp)
    covered = {m: np.zeros(T) for m in METHODS}
    width = {m: np.zeros(T) for m in METHODS}

    aci = AdaptiveConformalInference(alpha_target=ALPHA, gamma=0.05)
    aci.reset(); aci.history_scores = list(cal_scores)
    saaci = ShiftAwareACI(alpha_target=ALPHA, gamma_0=0.05, lambda_=3.0, gamma_max=0.15)
    saaci.reset(); saaci.history_scores = list(cal_scores)
    det = RollingShiftDetector(window=100).fit_baseline(cal_scores)
    pbo = PBAwareJointConformalOnline(alpha=ALPHA, window=300); pbo.warmstart(cal_scores)
    q_split = empirical_quantile(cal_scores, ALPHA)

    for t in range(T):
        p, gt = sp[t], sg[t]; s = _score(p, gt)
        for m, q in (("aci", aci.get_quantile()), ("sa_aci", saaci.get_quantile()),
                     ("pb_jci_online", pbo.get_quantile()), ("pb_jci_split", q_split)):
            lo, hi = _interval(p, q, K)
            covered[m][t] = float(((gt >= lo) & (gt <= hi)).all())
            width[m][t] = float((hi - lo).mean())
        c_aci = bool(covered["aci"][t]); aci.update(s, c_aci)
        c_sa = bool(covered["sa_aci"][t]); saaci.update(s, c_sa, delta_t=det.step(s))
        pbo.update(s)
    return covered, width

acc_cov = {m: [] for m in METHODS}
acc_wid = {m: [] for m in METHODS}
t0 = time.time()
for sd in SEEDS:
    sp, sg, seg, cal_scores = build_stream(sd)
    cov, wid = run_stream(sp, sg, cal_scores)
    for m in METHODS:
        acc_cov[m].append(cov[m]); acc_wid[m].append(wid[m])
    print(f"  seed {sd} done")
T = len(sp)
roll_cov = {m: np.convolve(np.mean(acc_cov[m], axis=0), np.ones(ROLL)/ROLL, mode="valid") for m in METHODS}
roll_wid = {m: np.convolve(np.mean(acc_wid[m], axis=0), np.ones(ROLL)/ROLL, mode="valid") for m in METHODS}
print(f"Done in {time.time()-t0:.1f}s")
'''))

cells.append(md("## 05 — Figure: running coverage & width over time"))
cells.append(code('''
xs = np.arange(ROLL - 1, T)
colors = {"aci": "tab:blue", "sa_aci": "tab:orange",
          "pb_jci_online": "tab:green", "pb_jci_split": "tab:red"}

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax = axes[0]
for m in METHODS:
    ax.plot(xs, roll_cov[m] * 100, label=mlabel[m], color=colors[m], lw=1.8)
ax.axhline(90, ls="--", color="k", alpha=0.6, label="target 90%")
for cp in CHANGE_POINTS:
    ax.axvline(cp, ls=":", color="gray", alpha=0.7)
ax.set_ylabel("Running joint coverage (%)"); ax.set_ylim(0, 102)
ax.set_title("Streaming behavior under drift (in-dist -> mild -> severe -> recover)")
ax.legend(loc="lower left", ncol=3, fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
for m in METHODS:
    ax.plot(xs, roll_wid[m], label=mlabel[m], color=colors[m], lw=1.8)
for cp in CHANGE_POINTS:
    ax.axvline(cp, ls=":", color="gray", alpha=0.7)
seg_names = ["in-dist", "mild", "severe", "recover (in-dist)"]
mids = np.concatenate([[0], CHANGE_POINTS]) + SEG_LEN / 2
for mid, nm in zip(mids, seg_names):
    ax.text(mid, ax.get_ylim()[1]*0.92, nm, ha="center", fontsize=9, color="gray")
ax.set_ylabel("Running interval width"); ax.set_xlabel("stream step t")
ax.legend(loc="upper left", ncol=2, fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{WORK}/phase_D_streaming.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved: {WORK}/phase_D_streaming.png")
'''))

cells.append(md("## 06 — Per-segment summary + save"))
cells.append(code('''
seg_bounds = np.concatenate([[0], CHANGE_POINTS, [T]])
seg_names = ["in_dist", "mild", "severe", "recover"]
summary = {}
print(f"{'Method':16s} | " + " | ".join(f"{s:>16s}" for s in seg_names))
print("-" * 90)
for m in METHODS:
    cov_mean = np.mean(acc_cov[m], axis=0); wid_mean = np.mean(acc_wid[m], axis=0)
    row = {}
    cells_txt = []
    for a, b, nm in zip(seg_bounds[:-1], seg_bounds[1:], seg_names):
        c = float(cov_mean[a:b].mean()) * 100; w = float(wid_mean[a:b].mean())
        row[nm] = {"coverage": c, "width": w}
        cells_txt.append(f"{c:5.1f}% / {w:6.2f}")
    summary[m] = row
    print(f"{mlabel[m]:16s} | " + " | ".join(f"{c:>16s}" for c in cells_txt))

with open(f"{WORK}/phase_D_streaming_results.json", "w") as f:
    json.dump({"config": {"alpha": ALPHA, "order": [s for s, _ in ORDER],
                          "seg_len": SEG_LEN, "seeds": SEEDS, "roll_window": ROLL},
               "per_segment": summary}, f, indent=2)
print(f"\\nSaved: {WORK}/phase_D_streaming_results.json")
'''))

cells.append(md(
    "## Notes (Section 4.4)",
    "",
    "- Figure shows recovery dynamics: at each change-point, split methods (PB-JCI split)",
    "  drop coverage and stay low; ACI recovers coverage but width spikes; **PB-JCI Online**",
    "  recovers coverage with much smaller, stable width.",
    "- SA-ACI included to show adaptive step-size does not help (ablation, see Phase C).",
    "- For model-seed CI on this figure, run per `phase_C_predictions_seed{s}.pkl` and",
    "  average the curves (CPU, cheap).",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT.name}: {len(cells)} cells")

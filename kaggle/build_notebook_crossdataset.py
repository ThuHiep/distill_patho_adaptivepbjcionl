"""
Builder -> sam3_pannuke_crossdataset.ipynb

CROSS-DATASET shift (the strongest robustness evidence):
  CALIBRATE on PanNuke (source)  ->  TEST on NuInsSeg (target, real shift).

Total-count conformal (K=1). Loads:
  - phase_C_preds_seed42.pkl  (PanNuke per-class -> summed to total count) [sam3-q1-multiseed-ckpts]
  - phase_E_nuinsseg_preds.pkl (NuInsSeg total count) [from Phase E output, attach as dataset]

Story: split conformal calibrated on PanNuke UNDER-covers on NuInsSeg (exchangeability
broken by real shift) -> motivates adaptive; ACI / PB-JCI Online (with streaming feedback)
recover coverage. Also reports in-domain NuInsSeg baseline for side-by-side.

CPU only, seconds.
"""
from __future__ import annotations
from pathlib import Path
import json

HERE = Path(__file__).parent
OUT = HERE / "sam3_pannuke_crossdataset.ipynb"
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
    "# Cross-dataset shift — calibrate PanNuke, test NuInsSeg (CPU)",
    "",
    "Total-count conformal. **Calibrate on PanNuke (source) → test on NuInsSeg (target).**",
    "Real distribution shift (different dataset, 31 organs, model trained only on PanNuke).",
    "",
    "**Attach datasets:**",
    "- `hipinhththu/sam3-q1-multiseed-ckpts` — has `phase_C_preds_seed42.pkl` (PanNuke)",
    "- the dataset / notebook-output holding `phase_E_nuinsseg_preds.pkl` (NuInsSeg, from Phase E)",
))

cells.append(md("## 00 — Setup"))
cells.append(code('''
import os, glob, json, pickle
import numpy as np
print("numpy:", np.__version__)
'''))

cells.append(md("## 01 — conformal.py (baked)"))
cells.append(code(CONFORMAL))
cells.append(code('''
import sys
if "." not in sys.path: sys.path.insert(0, ".")
from conformal import (AdaptiveConformalInference, PBAwareJointConformalOnline,
                       empirical_quantile, pb_count, pb_variance)
print("conformal loaded.")
'''))

cells.append(md("## 02 — Load both pkls; PanNuke -> total count"))
cells.append(code('''
def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None

pan_path = find("phase_C_preds_seed42.pkl")
nu_path  = find("phase_E_nuinsseg_preds.pkl")
assert pan_path, "phase_C_preds_seed42.pkl not found - attach sam3-q1-multiseed-ckpts"
assert nu_path,  "phase_E_nuinsseg_preds.pkl not found - attach the Phase E output dataset"
print("PanNuke:", pan_path)
print("NuInsSeg:", nu_path)

with open(pan_path, "rb") as f: dpan = pickle.load(f)
with open(nu_path, "rb") as f:  dnu  = pickle.load(f)

# PanNuke per-class -> TOTAL count (K=1): total pred = sum_i s_i ; total GT = sum over 5 classes
pan_src = dpan["predictions_by_setting"]["in_dist"]
pan_gtc = np.asarray(dpan["gt_counts"])
def to_total(p):
    s = np.asarray(p["scores"])
    return {"scores": s, "probs": np.ones((len(s), 1)), "K": 1}
pan_preds = [to_total(p) for p in pan_src]
pan_gts   = [np.array([float(g.sum())]) for g in pan_gtc]

# NuInsSeg already total-count
nu_preds = dnu["preds"]
nu_gts   = dnu["gts"]

print(f"\\nPanNuke (source): {len(pan_preds)} patches | GT total mean={np.mean([g[0] for g in pan_gts]):.1f}")
print(f"NuInsSeg (target): {len(nu_preds)} patches | GT total mean={np.mean([g[0] for g in nu_gts]):.1f}")
'''))

cells.append(md("## 03 — Total-count nonconformity / interval (K=1)"))
cells.append(code('''
ALPHA = 0.1

def nonconf(p, gt):
    if len(p["scores"]) == 0: return float(abs(gt[0]))
    n  = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return abs(gt[0] - n) / sg

def interval(p, q):
    if len(p["scores"]) == 0: return 0.0, 0.0
    n  = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return max(0.0, n - q * sg), n + q * sg

def cov_width(preds, gts, q):
    los = np.array([interval(p, q)[0] for p in preds])
    his = np.array([interval(p, q)[1] for p in preds])
    g   = np.array([gg[0] for gg in gts])
    return float(np.mean((g >= los) & (g <= his))), float(np.mean(his - los))
'''))

cells.append(md(
    "## 04 — CROSS-DATASET: calibrate PanNuke -> test NuInsSeg",
    "",
    "Split conformal (no adaptation) shows the honest coverage drop under shift.",
    "Online methods warm-start on PanNuke, then stream NuInsSeg with feedback (5 stream seeds).",
))
cells.append(code('''
# Calibrate quantile on ALL PanNuke
pan_scores = np.array([nonconf(pan_preds[i], pan_gts[i]) for i in range(len(pan_preds))])
q_cross = empirical_quantile(pan_scores, ALPHA)
print(f"q (calibrated on PanNuke) = {q_cross:.3f}")

# (A) Split conformal: fixed PanNuke quantile applied to NuInsSeg (no update)
split_cov, split_w = cov_width(nu_preds, nu_gts, q_cross)

# (B) Online methods: warm-start PanNuke, stream NuInsSeg with feedback
def stream(kind, nseeds=5):
    covs, ws = [], []
    for sd in range(nseeds):
        order = np.random.RandomState(sd).permutation(len(nu_preds))
        if kind == "aci":
            m = AdaptiveConformalInference(alpha_target=ALPHA, gamma=0.05)
            m.reset(); m.history_scores = list(pan_scores)
        else:
            m = PBAwareJointConformalOnline(alpha=ALPHA, window=300)
            m.warmstart(pan_scores)
        c, w = [], []
        for i in order:
            q = m.get_quantile(); lo, hi = interval(nu_preds[i], q)
            covered = lo <= nu_gts[i][0] <= hi
            c.append(covered); w.append(hi - lo)
            s = nonconf(nu_preds[i], nu_gts[i])
            m.update(s, covered) if kind == "aci" else m.update(s)
        covs.append(np.mean(c)); ws.append(np.mean(w))
    return float(np.mean(covs)), float(np.std(covs)), float(np.mean(ws)), float(np.std(ws))

aci_c, aci_cs, aci_w, aci_ws = stream("aci")
pbo_c, pbo_cs, pbo_w, pbo_ws = stream("pbo")

print("\\nCROSS-DATASET (cal PanNuke -> test NuInsSeg):")
print(f"  Split (no adapt) : cov {split_cov*100:.1f}% | width {split_w:.2f}")
print(f"  ACI (stream)     : cov {aci_c*100:.1f}+/-{aci_cs*100:.1f}% | width {aci_w:.2f}+/-{aci_ws:.2f}")
print(f"  PB-JCI Online    : cov {pbo_c*100:.1f}+/-{pbo_cs*100:.1f}% | width {pbo_w:.2f}+/-{pbo_ws:.2f}")
'''))

cells.append(md("## 05 — In-domain NuInsSeg reference (calibrate NuInsSeg, 5 seeds)"))
cells.append(code('''
def indomain_split(nseeds=5):
    covs, ws = [], []
    for sd in [42, 100, 200, 300, 400][:nseeds]:
        idx = np.random.RandomState(sd).permutation(len(nu_preds))
        ncal = len(idx) // 2
        cal, test = idx[:ncal], idx[ncal:]
        cs = np.array([nonconf(nu_preds[i], nu_gts[i]) for i in cal])
        q = empirical_quantile(cs, ALPHA)
        c, w = cov_width([nu_preds[i] for i in test], [nu_gts[i] for i in test], q)
        covs.append(c); ws.append(w)
    return float(np.mean(covs)), float(np.std(covs)), float(np.mean(ws)), float(np.std(ws))

id_c, id_cs, id_w, id_ws = indomain_split()
print(f"In-domain split (cal NuInsSeg): cov {id_c*100:.1f}+/-{id_cs*100:.1f}% | width {id_w:.2f}+/-{id_ws:.2f}")
'''))

cells.append(md("## 06 — Summary table + save"))
cells.append(code('''
print("=" * 78)
print("CROSS-DATASET SHIFT: PanNuke (cal) -> NuInsSeg (test) | total count, alpha=0.1")
print("=" * 78)
print(f"{'Setting / Method':38s} | {'Coverage':>14s} | {'Width':>12s}")
print("-" * 78)
print(f"{'In-domain split (cal NuInsSeg)':38s} | {id_c*100:>6.1f}+/-{id_cs*100:<4.1f}% | {id_w:>7.2f}")
print(f"{'Cross split (cal PanNuke, no adapt)':38s} | {split_cov*100:>11.1f}% | {split_w:>7.2f}")
print(f"{'Cross ACI (stream feedback)':38s} | {aci_c*100:>6.1f}+/-{aci_cs*100:<4.1f}% | {aci_w:>7.2f}")
print(f"{'Cross PB-JCI Online (stream)':38s} | {pbo_c*100:>6.1f}+/-{pbo_cs*100:<4.1f}% | {pbo_w:>7.2f}")
print("-" * 78)
drop = (id_c - split_cov) * 100
print(f"\\nCoverage DROP (in-domain -> cross split): {drop:+.1f} pp  "
      f"-> {'shift hurts split conformal' if drop > 1 else 'mild'}")

out = {
    "in_domain_split":  {"coverage": [id_c, id_cs], "width": [id_w, id_ws]},
    "cross_split":      {"coverage": split_cov, "width": split_w},
    "cross_aci":        {"coverage": [aci_c, aci_cs], "width": [aci_w, aci_ws]},
    "cross_pbjci_online": {"coverage": [pbo_c, pbo_cs], "width": [pbo_w, pbo_ws]},
    "q_cross": float(q_cross), "alpha": ALPHA,
}
with open("/kaggle/working/cross_dataset_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\\nSaved: /kaggle/working/cross_dataset_results.json")
'''))

cells.append(md(
    "## Notes",
    "",
    "- **Split (cal PanNuke, no adapt)** = honest coverage under real shift; expect < 90%.",
    "- **ACI / PB-JCI Online** assume streaming GT feedback on NuInsSeg → recover coverage.",
    "- Send `cross_dataset_results.json` to fill the cross-dataset row in PAPER_TABLES.",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT.name}: {len(cells)} cells | conformal {len(CONFORMAL)} chars")

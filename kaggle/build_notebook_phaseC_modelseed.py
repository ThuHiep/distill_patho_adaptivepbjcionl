"""
Builder -> sam3_pannuke_phaseC_modelseed.ipynb

Self-contained Kaggle CPU notebook. Attach the dataset holding the 3 pkls
(phase_C_preds_seed{42,100,200}.pkl, from run_eval_combined.py on Vast ->
backed up to hipinhththu/sam3-q1-multiseed-ckpts). Runs the CORRECTED conformal
benchmark (lambda=3, PB-JCI Online, temporal_drift) for 3 model x 5 cal seeds,
aggregates mean+/-std -> phase_C_modelseed_results.json (paper Table 4c).

No GPU. Same conformal logic as build_notebook_phaseC.py.
"""
from __future__ import annotations
from pathlib import Path
import json

HERE = Path(__file__).parent
OUT = HERE / "sam3_pannuke_phaseC_modelseed.ipynb"
LIB_DIR = HERE / "lib"
MAIN_BUILDER = HERE / "build_notebook_phaseC.py"

CONFORMAL = "%%writefile conformal.py\n" + (LIB_DIR / "conformal.py").read_text(encoding="utf-8")

def extract_block(src: str, anchor: str) -> str:
    i = src.find(anchor)
    if i < 0:
        raise ValueError(f"anchor not found: {anchor}")
    start = src.rfind("code('''", 0, i)
    start += len("code('''")
    end = src.find("'''))", start)
    return src[start:end].strip("\n")

# Reuse the EXACT benchmark block (functions + run_benchmark) from the main builder,
# but drop its trailing single-seed driver lines so we control the loop ourselves.
main_src = MAIN_BUILDER.read_text(encoding="utf-8")
BENCH_BLOCK = extract_block(main_src, "ALPHA = 0.1")
# strip the 2 driver lines at the end ("results, n_cal, ... = run_benchmark(cal_seed=42...")
_lines = BENCH_BLOCK.splitlines()
while _lines and ("run_benchmark(cal_seed=42" in _lines[-1] or _lines[-1].strip().startswith("print(")
                  or _lines[-1].strip() == ""):
    _lines.pop()
BENCH_BLOCK = "\n".join(_lines)

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
    "# Phase C — MODEL-SEED CI (CPU, Kaggle)",
    "",
    "Loads `phase_C_preds_seed{42,100,200}.pkl` and runs the CORRECTED conformal",
    "benchmark (lambda=3, gamma_max=0.15, PB-JCI Online, temporal_drift) for",
    "3 model x 5 cal = 15 runs. Aggregates mean+/-std -> Table 4c.",
    "",
    "**Attach dataset:** `hipinhththu/sam3-q1-multiseed-ckpts` (holds the 3 pkls).",
    "No GPU. Runs in seconds.",
))

cells.append(md("## 00 — Setup"))
cells.append(code('''
import os, json, pickle, glob, time
import numpy as np
print("numpy:", np.__version__)
'''))

cells.append(md("## 01 — Write conformal.py (fixed version, baked in)"))
cells.append(code(CONFORMAL))

cells.append(code('''
import sys
if "." not in sys.path:
    sys.path.insert(0, ".")
from conformal import (
    MarginalSplitConformal, AdaptiveConformalInference, ShiftAwareACI,
    PBAwareJointConformal, PBAwareJointConformalOnline, ClassStratifiedConformal,
    RollingShiftDetector, local_coverage_stats,
    coverage_per_class, joint_coverage, avg_width_per_class, macro_width,
    pb_count, pb_variance,
)
print("conformal helpers loaded.")
'''))

cells.append(md("## 02 — Benchmark functions (same as main Phase C)"))
cells.append(code(BENCH_BLOCK))

cells.append(md(
    "## 03 — Loop 3 model seeds x 5 cal seeds -> model-seed CI",
    "",
    "`run_benchmark` reads globals `predictions_by_setting`, `gt_counts`, `SETTINGS`;",
    "we reassign them per model-seed pkl, then aggregate across all 15 runs.",
))
cells.append(code('''
WORK = "/kaggle/working"
MODEL_SEEDS = [42, 100, 200]
CAL_SEEDS   = [42, 100, 200, 300, 400]

def find_pkl(seed):
    for pat in [f"/kaggle/input/**/phase_C_preds_seed{seed}.pkl",
                f"/kaggle/input/**/phase_C_preds_seed{seed}_*.pkl"]:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None

method_names = {
    "marginal_split": "Marginal Split", "aci": "ACI (Gibbs-Candes)",
    "sa_aci": "SA-ACI (Ours)", "pb_jci": "PB-Aware JCI (Ours)",
    "pb_jci_online": "PB-JCI Online (Ours)", "class_strat": "Class-Strat Bonf",
}

raw = {s: {m: {"marginal_coverage": [], "joint_coverage": [], "macro_width": [],
               "min_local_cov": [], "max_miss_run": []}
           for m in METHODS} for s in EVAL_SETTINGS}
per_run = {}
n_test_last = None

for model_seed in MODEL_SEEDS:
    pkl = find_pkl(model_seed)
    assert pkl, f"pkl for model seed {model_seed} not found under /kaggle/input"
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    # rebind globals that run_benchmark reads
    predictions_by_setting = d["predictions_by_setting"]
    gt_counts = np.asarray(d["gt_counts"])
    SETTINGS = list(predictions_by_setting.keys())
    globals().update(predictions_by_setting=predictions_by_setting,
                     gt_counts=gt_counts, SETTINGS=SETTINGS)
    print(f"model seed {model_seed}: {pkl}  N={len(gt_counts)}  settings={SETTINGS}")

    for cal_seed in CAL_SEEDS:
        res, _, n_test = run_benchmark(cal_seed, verbose=False)
        n_test_last = n_test
        per_run[f"model{model_seed}_cal{cal_seed}"] = res
        for s in EVAL_SETTINGS:
            for m in METHODS:
                for key in raw[s][m]:
                    raw[s][m][key].append(res[s][m][key])
    print(f"  done 5 cal seeds")

def ms(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())

n_runs = len(MODEL_SEEDS) * len(CAL_SEEDS)
print("\\n" + "=" * 120)
print(f"PHASE C MODEL-SEED CI | {len(MODEL_SEEDS)} model x {len(CAL_SEEDS)} cal = {n_runs} runs | N_test~{n_test_last}")
print("=" * 120)
print(f"\\n{'Setting':<15s} | {'Method':<21s} | {'JointCov':>14s} | {'Width':>15s} | {'MinLocal':>14s}")
print("-" * 120)
agg = {s: {} for s in EVAL_SETTINGS}
for s in EVAL_SETTINGS:
    for m in METHODS:
        mc_m, mc_s = ms(raw[s][m]["marginal_coverage"])
        jc_m, jc_s = ms(raw[s][m]["joint_coverage"])
        w_m, w_s   = ms(raw[s][m]["macro_width"])
        ml_m, ml_s = ms(raw[s][m]["min_local_cov"])
        agg[s][m] = {"marg": [mc_m, mc_s], "joint": [jc_m, jc_s],
                     "width": [w_m, w_s], "min_local": [ml_m, ml_s]}
        print(f"{s:<15s} | {method_names[m]:<21s} | "
              f"{jc_m*100:>6.1f}+/-{jc_s*100:>4.1f}% | "
              f"{w_m:>8.2f}+/-{w_s:>5.2f} | "
              f"{ml_m*100:>6.1f}+/-{ml_s*100:>4.1f}%")
    print("-" * 120)

with open(f"{WORK}/phase_C_modelseed_results.json", "w") as f:
    json.dump({"config": {"model_seeds": MODEL_SEEDS, "cal_seeds": CAL_SEEDS,
                          "alpha": ALPHA, "lambda": LAMBDA, "gamma_max": GAMMA_MAX,
                          "n_runs": n_runs, "eval_settings": EVAL_SETTINGS,
                          "methods": METHODS},
               "per_run": per_run, "raw": raw, "aggregate": agg}, f, indent=2)
print(f"\\nSaved: {WORK}/phase_C_modelseed_results.json")
'''))

cells.append(md(
    "## Notes",
    "",
    "- Output `phase_C_modelseed_results.json` -> send to fill paper Table 4c.",
    "- Same conformal logic as the main Phase C notebook; only difference is the",
    "  loop over 3 model-seed pkls (model-seed + cal-seed CI combined).",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT.name}: {len(cells)} cells")
print(f"  conformal: {len(CONFORMAL)} chars | bench: {len(BENCH_BLOCK)} chars")

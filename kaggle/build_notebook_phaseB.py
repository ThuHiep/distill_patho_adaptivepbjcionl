from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "sam3_pannuke_phaseB_shift.ipynb"
LIB_DIR = Path(__file__).parent / "lib"

def _read(name: str) -> str:
    return (LIB_DIR / name).read_text(encoding="utf-8")

PANNUKE_LOADER  = "%%writefile pannuke_loader.py\n" + _read("pannuke_loader.py")
METRICS         = "%%writefile metrics.py\n"        + _read("metrics.py")
SHIFT_DETECTOR  = "%%writefile shift_detector.py\n" + _read("shift_detector.py")

def md(*lines: str) -> dict:
    src = [ln if ln.endswith("\n") else ln + "\n" for ln in lines]
    if src:
        src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(body: str) -> dict:
    lines = body.lstrip("\n").splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": lines,
    }

cells: list[dict] = []

cells.append(md(
    "# Phase B: Distribution Shift Detection trên SAM3 embeddings (multi-seed ×5)",
    "",
    "**Goal:** Calibrate shift detector $\\delta_t$ cho SA-ACI (Phase C/D), với CI mean±std.",
    "",
    "**Outputs:**",
    "- $\\delta_t$ values cho 3 detectors × 11 conditions, mean ± std qua 5 seeds",
    "- Calibrated $\\lambda$ for SA-ACI: $\\gamma_t = \\gamma_0(1 + \\lambda \\delta_t)$",
    "",
    "**Setup:**",
    "- Reference: PanNuke Fold 1 (in-domain reference)",
    "- Test conditions: Fold 2/3, simulated HED/blur/HSV shifts",
    "- Independent of Phase A2 — only needs frozen SAM3 backbone",
    "- Seeds [0,1,2,3,4]: re-sample subset + re-randomize augmentation per seed",
    "",
    "**Compute:** ~5h trên T4 (embedding extraction ×5 là bottleneck). Per-seed cache → resume.",
))

cells.append(md("## 00 — Setup"))

cells.append(code('''
import subprocess, sys, os, platform, time, json
import numpy as np
import torch
from PIL import Image
print("Python:", sys.version.split()[0])
print("Torch :", torch.__version__, "| CUDA:", torch.cuda.is_available())

WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/sam3_research"
SAM3_DIR = f"{REPO_DIR}/sam3"
CHECKPOINT_PATH = "/kaggle/input/datasets/hipinhththu/sam3-native-pt/sam3.pt"
DATA_ROOT = "/kaggle/input/datasets/hipinhththu/pannuke"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone",
                    "https://github.com/duonguwu/sam3_research.git", REPO_DIR],
                   check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=False)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", SAM3_DIR], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scikit-image", "scikit-learn", "matplotlib", "opencv-python",
                "pycocotools", "einops"], check=True)

result = subprocess.run(
    [sys.executable, "-c", "import numpy; print(numpy.__version__)"],
    capture_output=True, text=True,
)
disk_np = result.stdout.strip()
print(f"OK setup.")
print(f"  Numpy on disk (after install): {disk_np}")
print(f"  Numpy loaded in this kernel  : {np.__version__}")

if disk_np.split(".")[0] != np.__version__.split(".")[0]:
    print()
    print("=" * 65)
    print("WARNING: numpy on disk != loaded version (major mismatch).")
    print("Cell tiep theo se crash 'numpy.dtype size changed'.")
    print(">>> ACTION: Menu -> Run -> Restart & Run All (or Factory Reset)")
    print("=" * 65)
'''))

cells.append(md("## Helper modules"))
cells.append(code(PANNUKE_LOADER))
cells.append(code(METRICS))
cells.append(code(SHIFT_DETECTOR))

cells.append(code('''
import sys
for p in [".", "/kaggle/working", SAM3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from shift_detector import (ShiftDetector, extract_sam3_features,
                             apply_hed_shift, apply_blur_shift, apply_hsv_jitter,
                             mmd_squared, wasserstein_1d_mean, energy_distance_mean)

print("Helpers loaded")
'''))

cells.append(md("## 01 — Build SAM3 (frozen backbone only)"))

cells.append(code('''
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam3_image_model(
    device=device, eval_mode=True,
    checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
)
model.eval()

processor = Sam3Processor(model, device=device, resolution=1008)
transform = processor.transform
print(f"SAM3 ready. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
'''))

cells.append(md("## 02 — Multi-seed config + per-seed condition builder"))

cells.append(code('''
SEEDS      = [0, 1, 2, 3, 4]
REF_SIZE   = 200
TEST_SIZE  = 200
SEVERITIES = ("mild", "moderate", "severe")
DETECTORS  = ["MMD2", "Wasserstein", "Energy"]

CACHE_DIR = f"{WORK}/phase_B_seeds"
os.makedirs(CACHE_DIR, exist_ok=True)

folds = [PanNukeFold(DEFAULT_ROOT, k) for k in (1, 2, 3)]
for i, f in enumerate(folds):
    print(f"Fold {i+1}: {len(f)} patches")

def build_conditions(seed):
    rng = np.random.RandomState(seed)
    ref_idx = rng.choice(len(folds[0]), size=REF_SIZE, replace=False)
    ref_imgs = [Image.fromarray(folds[0][int(i)]["image"]).convert("RGB") for i in ref_idx]

    conds = {}
    f2 = rng.choice(len(folds[1]), size=TEST_SIZE, replace=False)
    conds["fold2"] = [Image.fromarray(folds[1][int(i)]["image"]).convert("RGB") for i in f2]
    f3 = rng.choice(len(folds[2]), size=TEST_SIZE, replace=False)
    conds["fold3"] = [Image.fromarray(folds[2][int(i)]["image"]).convert("RGB") for i in f3]

    np.random.seed(seed)
    for sev in SEVERITIES:
        conds[f"hed_{sev}"]  = [Image.fromarray(apply_hed_shift(np.array(im), sev))  for im in ref_imgs]
    for sev in SEVERITIES:
        conds[f"blur_{sev}"] = [Image.fromarray(apply_blur_shift(np.array(im), sev)) for im in ref_imgs]
    for sev in SEVERITIES:
        conds[f"hsv_{sev}"]  = [Image.fromarray(apply_hsv_jitter(np.array(im), sev)) for im in ref_imgs]
    return ref_imgs, conds

n_conds = 2 + 3 * len(SEVERITIES)
per_seed_imgs = REF_SIZE + TEST_SIZE * 2 + REF_SIZE * 3 * len(SEVERITIES)
print(f"\\nPer seed: ref={REF_SIZE} + {n_conds} conditions = {per_seed_imgs} images")
print(f"Total ({len(SEEDS)} seeds): {per_seed_imgs*len(SEEDS)} images "
      f"(~{per_seed_imgs*len(SEEDS)*1.3/60:.0f} min on T4)")
'''))

cells.append(md("## 03 — Extract features + compute shift δ, per seed (cached)"))

cells.append(code('''
from tqdm import tqdm

def run_seed(seed):
    cache = f"{CACHE_DIR}/feats_seed{seed}.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        ref_feats = torch.from_numpy(d["ref"])
        test_feats = {k[5:]: torch.from_numpy(d[k]) for k in d.keys() if k.startswith("test_")}
        print(f"[seed {seed}] cache hit ({len(test_feats)} conds)")
    else:
        ref_imgs, conds = build_conditions(seed)
        print(f"[seed {seed}] extract ref({len(ref_imgs)}) + {len(conds)} conditions...")
        ref_feats = extract_sam3_features(model, transform, ref_imgs,
                                          device=device, desc=f"s{seed}-ref")
        test_feats = {}
        for name, imgs in conds.items():
            test_feats[name] = extract_sam3_features(model, transform, imgs,
                                                     device=device, desc=f"s{seed}-{name}")
        np.savez(cache, ref=ref_feats.numpy(),
                 **{f"test_{k}": v.numpy() for k, v in test_feats.items()})
        print(f"[seed {seed}] saved {cache}")

    ref_gpu = ref_feats.to(device).float()
    ref_np  = ref_feats.numpy()
    res = {}
    for name, feats in test_feats.items():
        res[name] = {
            "MMD2":        mmd_squared(ref_gpu, feats.to(device).float()),
            "Wasserstein": wasserstein_1d_mean(ref_np, feats.numpy()),
            "Energy":      energy_distance_mean(ref_np, feats.numpy()),
        }
    return res

t0 = time.time()
per_seed_results = {}
for sd in SEEDS:
    per_seed_results[sd] = run_seed(sd)
    with open(f"{CACHE_DIR}/shift_seed{sd}.json", "w") as f:
        json.dump(per_seed_results[sd], f, indent=2)
    print(f"  seed {sd} done | {(time.time()-t0)/60:.1f} min elapsed\\n")

print(f"All {len(SEEDS)} seeds done in {(time.time()-t0)/60:.1f} min")
CONDITIONS = list(per_seed_results[SEEDS[0]].keys())
'''))

cells.append(md("## 04 — Aggregate: mean ± std across seeds"))

cells.append(code('''
agg = {}
for c in CONDITIONS:
    agg[c] = {}
    for det in DETECTORS:
        vals = np.array([per_seed_results[sd][c][det] for sd in SEEDS], dtype=float)
        agg[c][det] = {"mean": float(vals.mean()), "std": float(vals.std())}

print(f"PHASE B — shift detection (mean +/- std over {len(SEEDS)} seeds)")
print("=" * 86)
print(f"{'Condition':16s} | " + " | ".join(f"{d:>20s}" for d in DETECTORS))
print("-" * 86)
for c in CONDITIONS:
    row = f"{c:16s} | "
    row += " | ".join(
        f"{agg[c][d]['mean']:>9.4f}+/-{agg[c][d]['std']:<8.4f}" for d in DETECTORS)
    print(row)
'''))

cells.append(md("## 05 — Visualize shift hierarchy (mean ± std)"))

cells.append(code('''
import matplotlib.pyplot as plt

groups = {
    "Within-PanNuke": ["fold2", "fold3"],
    "HED stain":     ["hed_mild", "hed_moderate", "hed_severe"],
    "Blur":          ["blur_mild", "blur_moderate", "blur_severe"],
    "HSV jitter":    ["hsv_mild", "hsv_moderate", "hsv_severe"],
}
gcolor = {"Within-PanNuke": "steelblue", "HED stain": "indianred",
          "Blur": "seagreen", "HSV jitter": "goldenrod"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric in zip(axes, DETECTORS):
    y_pos = 0
    yticks, ytlabels = [], []
    for grp_name, conds in groups.items():
        for c in conds:
            m = agg[c][metric]["mean"]
            s = agg[c][metric]["std"]
            ax.barh(y_pos, m, xerr=s, color=gcolor[grp_name],
                    label=grp_name if c == conds[0] else None,
                    error_kw={"ecolor": "black", "capsize": 3})
            yticks.append(y_pos); ytlabels.append(c)
            y_pos += 1
        y_pos += 0.5
    ax.set_yticks(yticks); ax.set_yticklabels(ytlabels)
    ax.set_xlabel(metric); ax.set_title(f"delta_t via {metric}")
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{WORK}/phase_B_shift_hierarchy.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"Saved: {WORK}/phase_B_shift_hierarchy.png")
'''))

cells.append(md("## 06 — Sanity check: expected shift order (on means)"))

cells.append(code('''
print("Sanity (expect mild < moderate < severe within each augmentation):")
for aug in ("hed", "blur", "hsv"):
    vals = [agg[f"{aug}_{s}"]["MMD2"]["mean"] for s in SEVERITIES]
    ok = vals[0] <= vals[1] <= vals[2]
    print(f"  {aug:6s}: {vals[0]:.4f} -> {vals[1]:.4f} -> {vals[2]:.4f}  {'OK' if ok else 'FAIL'}")

print("\\nExpect severe shifts > within-PanNuke (fold2/fold3):")
within = max(agg["fold2"]["MMD2"]["mean"], agg["fold3"]["MMD2"]["mean"])
for aug in ("hed", "blur", "hsv"):
    sev = agg[f"{aug}_severe"]["MMD2"]["mean"]
    print(f"  {aug:6s} severe ({sev:.4f}) > within ({within:.4f}): {'OK' if sev > within else 'CHECK'}")
'''))

cells.append(code('''
out = {
    "config": {
        "seeds": SEEDS, "ref_size": REF_SIZE, "test_size": TEST_SIZE,
        "ref_source": "Fold 1",
        "detectors": DETECTORS,
        "shift_types": ["HED", "blur", "HSV"],
        "severities": list(SEVERITIES),
    },
    "shift_scores_aggregate": agg,
    "shift_scores_per_seed": per_seed_results,
    "groups": groups,
}
with open(f"{WORK}/phase_B_multiseed_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved: {WORK}/phase_B_multiseed_results.json")
'''))

cells.append(md(
    "## 07 — Calibrate λ for SA-ACI (Phase C/D)",
    "",
    "Target: $\\gamma_t = \\gamma_0(1 + \\lambda \\delta_t)$ với $\\gamma_t \\in [\\gamma_0, \\gamma_{max}]$.",
    "",
    "Constraint: $\\gamma_0(1 + \\lambda \\delta_{\\max}) \\le \\gamma_{max}$",
    "→ $\\lambda \\le (\\gamma_{max}/\\gamma_0 - 1) / \\delta_{\\max}$",
))

cells.append(code('''
gamma_0 = 0.05
target_gamma_max = 0.15

mmd_means = {c: agg[c]["MMD2"]["mean"] for c in CONDITIONS}
delta_max = max(mmd_means.values())
delta_typical_severe = float(np.mean([mmd_means[f"{a}_severe"] for a in ("hed", "blur", "hsv")]))

lambda_max_safe = (target_gamma_max / gamma_0 - 1) / delta_max if delta_max > 0 else 1.0

print(f"Observed delta_max (worst shift): {delta_max:.4f}")
print(f"Typical severe shift delta      : {delta_typical_severe:.4f}")
print(f"\\nSuggested SA-ACI hyperparameters:")
print(f"  gamma_0       = {gamma_0}")
print(f"  gamma_max     = {target_gamma_max}")
print(f"  lambda (safe) = {lambda_max_safe:.2f}")

calib = {
    "gamma_0": gamma_0, "gamma_max": target_gamma_max,
    "delta_max_observed": float(delta_max),
    "delta_typical_severe": float(delta_typical_severe),
    "lambda_max_safe": float(lambda_max_safe),
}
with open(f"{WORK}/phase_B_lambda_calibration.json", "w") as f:
    json.dump(calib, f, indent=2)
print(f"\\nSaved: {WORK}/phase_B_lambda_calibration.json")
'''))

cells.append(md(
    "### Phase B PASS criteria",
    "",
    "- **Sanity hierarchy**: severity ordering correct (mild < moderate < severe) cho cả 3 augmentations",
    "- **Cross-condition order**: severe shifts > within-PanNuke (intuition)",
    "- **Low std**: std nhỏ so với mean → detector ổn định qua seeds",
    "- **3 detectors agree** (qualitatively) trên rank order",
    "",
    "### Outputs lưu vào /kaggle/working/",
    "- `phase_B_seeds/feats_seed*.npz` — SAM3 features per seed (cache, resume)",
    "- `phase_B_multiseed_results.json` — δ mean±std + per-seed raw",
    "- `phase_B_shift_hierarchy.png` — visualization với error bars",
    "- `phase_B_lambda_calibration.json` — suggested SA-ACI hyperparams",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT.name}: {len(cells)} cells")

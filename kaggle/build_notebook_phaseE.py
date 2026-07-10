from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "sam3_pannuke_phaseE_nuinsseg.ipynb"
LIB_DIR = Path(__file__).parent / "lib"

def _read(name: str) -> str:
    return (LIB_DIR / name).read_text(encoding="utf-8")

METRICS    = "%%writefile metrics.py\n"    + _read("metrics.py")
LORA_SAM3  = "%%writefile lora_sam3.py\n"  + _read("lora_sam3.py")
SAM3_TRAIN = "%%writefile sam3_train.py\n" + _read("sam3_train.py")
CONFORMAL  = "%%writefile conformal.py\n"  + _read("conformal.py")

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
    "# Phase E — Cross-dataset total-count conformal on NuInsSeg",
    "",
    "**Goal:** Validate the conformal counting pipeline on a 2nd dataset (NuInsSeg,",
    "31 organs) as **total nucleus count** (K=1). Shows the method generalizes to a",
    "real, different dataset — and (later, on CPU) tests coverage under real cross-",
    "dataset shift when calibrated on PanNuke.",
    "",
    "**Why total-count (not per-class):** NuInsSeg has instance masks but NO cell-type",
    "labels. Per-class stays on PanNuke (main results); here we collapse to one class",
    "= 'nucleus' and run total-count conformal. NuInsSeg's 31-organ diversity is a",
    "strong real-shift testbed for PB-JCI Online.",
    "",
    "**Model:** SAM3 + A2 LoRA seed42 (same as Phase C/D; no TypeHead needed).",
    "",
    "**Attach datasets:** `ipateam/nuinsseg`, `hipinhththu/sam3-native-pt`,",
    "`hipinhththu/sam3-q1-multiseed-ckpts`.",
    "",
    "**Compute:** GPU T4. Inference is the bottleneck; cached for cheap re-eval.",
))

cells.append(md("## 00 — Setup"))
cells.append(code('''
import subprocess, sys, os, glob, time, json
import numpy as np
import torch
from PIL import Image
print("Python:", sys.version.split()[0])
print("Torch :", torch.__version__, "| CUDA:", torch.cuda.is_available())

WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/sam3_research"
SAM3_DIR = f"{REPO_DIR}/sam3"
CHECKPOINT_PATH = "/kaggle/input/datasets/hipinhththu/sam3-native-pt/sam3.pt"

LORA_CANDIDATES = [
    # prefer the documented multi-seed model (seed 42, same as Phase C/D)
    "/kaggle/input/datasets/hipinhththu/sam3-q1-multiseed-ckpts/sam3_lora_seed42_final.pt",
    "/kaggle/input/sam3-q1-multiseed-ckpts/sam3_lora_seed42_final.pt",
    "/kaggle/input/datasets/hipinhththu/phase-a2-lora-weights/sam3_lora_rank16_final.pt",
    "/kaggle/input/phase-a2-lora-weights/sam3_lora_rank16_final.pt",
]
LORA_PATH = next((p for p in LORA_CANDIDATES if os.path.exists(p)), None)
if LORA_PATH is None:  # robust fallback: any seed42 lora under /kaggle/input
    hits = glob.glob("/kaggle/input/**/sam3_lora_seed42_final.pt", recursive=True) \
        or glob.glob("/kaggle/input/**/sam3_lora_rank16_final.pt", recursive=True)
    LORA_PATH = hits[0] if hits else None

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone",
                    "https://github.com/duonguwu/sam3_research.git", REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=False)

assert os.path.exists(CHECKPOINT_PATH), "Attach hipinhththu/sam3-native-pt"
assert LORA_PATH, "Attach hipinhththu/sam3-q1-multiseed-ckpts (has sam3_lora_seed42_final.pt)"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", SAM3_DIR], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scikit-image", "scikit-learn", "matplotlib", "opencv-python",
                "pycocotools", "einops", "tqdm", "tifffile"], check=True)
print("OK setup | LoRA:", LORA_PATH)
'''))

cells.append(md("## Helper modules"))
cells.append(code(METRICS))
cells.append(code(LORA_SAM3))
cells.append(code(SAM3_TRAIN))
cells.append(code(CONFORMAL))

cells.append(code('''
import sys
for p in [".", "/kaggle/working", SAM3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
from lora_sam3 import inject_lora, freeze_non_lora, load_lora_state, DEFAULT_LORA_TARGETS
from sam3_train import make_transform, encode_image_frozen, encode_text, forward_decoder_with_grad
from conformal import (MarginalSplitConformal, AdaptiveConformalInference,
                       PBAwareJointConformal, PBAwareJointConformalOnline,
                       empirical_quantile, pb_count, pb_variance,
                       coverage_per_class, joint_coverage, avg_width_per_class)
print("Helpers loaded.")
'''))

cells.append(md("## 01 — Explore NuInsSeg structure (fail-fast if loader wrong)"))
cells.append(code('''
NUINSSEG_CANDS = [
    "/kaggle/input/datasets/ipateam/nuinsseg",
    "/kaggle/input/nuinsseg", "/kaggle/input/nuinsseg/NuInsSeg",
    "/kaggle/input/nu-insseg",
]
ROOT = next((c for c in NUINSSEG_CANDS if os.path.isdir(c)), None)
if ROOT is None:
    # robust fallback: search anywhere under /kaggle/input for a "tissue images" dir
    td = glob.glob("/kaggle/input/**/tissue images", recursive=True)
    if td:
        # ROOT = parent of the organ folder that holds "tissue images"
        ROOT = os.path.dirname(os.path.dirname(td[0]))
    else:
        print("Top of /kaggle/input:", glob.glob("/kaggle/input/*"))
assert ROOT, "NuInsSeg not found - attach ipateam/nuinsseg"
print("ROOT =", ROOT)

print("\\n--- Top-level entries ---")
for d in sorted(os.listdir(ROOT))[:40]:
    full = os.path.join(ROOT, d)
    print(("[DIR] " if os.path.isdir(full) else "      ") + d)

tissue_dirs = glob.glob(os.path.join(ROOT, "**", "tissue images"), recursive=True)
if not tissue_dirs:
    tissue_dirs = [p for p in glob.glob(os.path.join(ROOT, "**"), recursive=True)
                   if os.path.isdir(p) and "tissue" in os.path.basename(p).lower()]
print(f"\\nFound {len(tissue_dirs)} 'tissue images' dirs (showing 5):")
for t in tissue_dirs[:5]:
    print("  ", t)
    sib = os.listdir(os.path.dirname(t))
    print("      siblings:", sib)
    imgs = os.listdir(t)[:3]
    print("      sample images:", imgs)
'''))

cells.append(md("## 02 — NuInsSeg loader (auto-discover image/mask pairs)"))
cells.append(code('''
IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")

def _find_mask_dir(organ_dir):
    for name in os.listdir(organ_dir):
        full = os.path.join(organ_dir, name)
        low = name.lower()
        if os.path.isdir(full) and "label" in low and "mask" in low and "modif" not in low:
            return full
    for name in os.listdir(organ_dir):
        full = os.path.join(organ_dir, name)
        if os.path.isdir(full) and "label" in name.lower():
            return full
    return None

def _load_mask(path):
    try:
        import tifffile
        if path.lower().endswith((".tif", ".tiff")):
            return np.asarray(tifffile.imread(path))
    except Exception:
        pass
    return np.asarray(Image.open(path))

def build_nuinsseg_index(root):
    tissue_dirs = glob.glob(os.path.join(root, "**", "tissue images"), recursive=True)
    if not tissue_dirs:
        tissue_dirs = [p for p in glob.glob(os.path.join(root, "**"), recursive=True)
                       if os.path.isdir(p) and "tissue" in os.path.basename(p).lower()]
    samples = []
    for tdir in tissue_dirs:
        organ_dir = os.path.dirname(tdir)
        organ = os.path.basename(organ_dir)
        mdir = _find_mask_dir(organ_dir)
        if mdir is None:
            continue
        masks = {os.path.splitext(f)[0]: os.path.join(mdir, f) for f in os.listdir(mdir)}
        for f in sorted(os.listdir(tdir)):
            if not f.lower().endswith(IMG_EXT):
                continue
            stem = os.path.splitext(f)[0]
            if stem in masks:
                samples.append({"organ": organ, "image": os.path.join(tdir, f),
                                "mask": masks[stem]})
    return samples

samples = build_nuinsseg_index(ROOT)
print(f"Indexed {len(samples)} (image, mask) pairs across "
      f"{len(set(s['organ'] for s in samples))} organs")
assert len(samples) > 0, "No pairs found - paste the explore output so loader can be fixed"

s0 = samples[0]
img0 = np.asarray(Image.open(s0["image"]).convert("RGB"))
m0 = _load_mask(s0["mask"])
gt0 = int(len(np.unique(m0)) - (1 if (m0 == 0).any() else 0))
print(f"\\nSample 0: organ={s0['organ']}")
print(f"  image shape={img0.shape}  mask shape={m0.shape} dtype={m0.dtype}")
print(f"  GT nuclei (unique nonzero)={gt0}")
'''))

cells.append(md("## 03 — Build SAM3 + load A2 LoRA"))
cells.append(code('''
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam3_image_model(device=device, eval_mode=True,
                               checkpoint_path=CHECKPOINT_PATH, load_from_HF=False)
model.eval()
inject_lora(model, target_module_names=DEFAULT_LORA_TARGETS, r=16, alpha=32, dropout=0.0)
load_lora_state(model, LORA_PATH)
for p in model.parameters():
    p.requires_grad = False
print("SAM3 + A2 LoRA ready.")

transform = make_transform(resolution=1008)
find_stage = FindStage(
    img_ids=torch.tensor([0], device=device, dtype=torch.long),
    text_ids=torch.tensor([0], device=device, dtype=torch.long),
    input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
    input_points=None, input_points_mask=None)
INFER_PROMPT = "cell"
SCORE_THRESH = 0.3

@torch.no_grad()
def predict_total(pil_img):
    bb = encode_image_frozen(model, transform, pil_img, device=device)
    tout = encode_text(model, INFER_PROMPT, device=device)
    bb.update(tout)
    outputs = forward_decoder_with_grad(model, bb, find_stage, model._get_dummy_prompt())
    cls_prob = outputs["pred_logits"].float().sigmoid()
    pres = outputs["presence_logit_dec"].float().sigmoid().unsqueeze(1)
    prob = (cls_prob * pres).squeeze(-1).squeeze(0)
    keep = prob > SCORE_THRESH
    scores = prob[keep].cpu().numpy() if keep.sum() > 0 else np.zeros(0)
    return scores
print("Inference fn ready.")
'''))

cells.append(md("## 04 — Smoke: run on 3 patches"))
cells.append(code('''
for s in samples[:3]:
    pil = Image.open(s["image"]).convert("RGB")
    m = _load_mask(s["mask"])
    gt = int(len(np.unique(m)) - (1 if (m == 0).any() else 0))
    sc = predict_total(pil)
    print(f"{s['organ']:24s} | GT={gt:4d} | n_det={len(sc):4d} | "
          f"pred_count={sc.sum():7.1f}")
print("\\nIf pred_count is in a sane ballpark vs GT -> proceed to full run.")
'''))

cells.append(md("## 05 — Full inference (cached pred scores + GT counts)"))
cells.append(code('''
import pickle
from tqdm import tqdm
CACHE = f"{WORK}/phase_E_nuinsseg_preds.pkl"

if os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        data = pickle.load(f)
    preds, gts, organs = data["preds"], data["gts"], data["organs"]
    print(f"Loaded cache: {len(preds)} patches")
else:
    preds, gts, organs = [], [], []
    t0 = time.time()
    for s in tqdm(samples, desc="NuInsSeg infer"):
        pil = Image.open(s["image"]).convert("RGB")
        m = _load_mask(s["mask"])
        gt = int(len(np.unique(m)) - (1 if (m == 0).any() else 0))
        sc = predict_total(pil)
        preds.append({"scores": sc, "probs": np.ones((len(sc), 1)), "K": 1})
        gts.append(np.array([gt], dtype=float))
        organs.append(s["organ"])
    with open(CACHE, "wb") as f:
        pickle.dump({"preds": preds, "gts": gts, "organs": organs}, f)
    print(f"Done {len(preds)} patches in {(time.time()-t0)/60:.1f} min -> {CACHE}")

gts_arr = np.array(gts)
print(f"GT count: mean={gts_arr.mean():.1f} min={gts_arr.min():.0f} max={gts_arr.max():.0f}")
pred_counts = np.array([p["scores"].sum() for p in preds])
print(f"Pred count: mean={pred_counts.mean():.1f}")
mae = np.abs(gts_arr.ravel() - pred_counts).mean()
print(f"Total-count MAE = {mae:.2f}")
'''))

cells.append(md("## 06 — In-domain total-count conformal (5 cal seeds)"))
cells.append(code('''
ALPHA = 0.1
CAL_SEEDS = [42, 100, 200, 300, 400]
METHODS = ["marginal_split", "aci", "pb_jci", "pb_jci_online"]
mnames = {"marginal_split": "Marginal Split", "aci": "ACI",
          "pb_jci": "PB-JCI (split)", "pb_jci_online": "PB-JCI Online"}

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

def eval_run(cal_seed):
    rng = np.random.RandomState(cal_seed)
    idx = rng.permutation(len(preds))
    ncal = len(idx) // 2
    cal, test = idx[:ncal], idx[ncal:]
    cal_scores = np.array([nonconf(preds[i], gts[i]) for i in cal])
    out = {}

    q_split = empirical_quantile(cal_scores, ALPHA)
    for m in METHODS:
        los, his, cov = [], [], []
        if m in ("marginal_split", "pb_jci"):
            for i in test:
                lo, hi = interval(preds[i], q_split)
                los.append(lo); his.append(hi)
                cov.append(lo <= gts[i][0] <= hi)
        elif m == "aci":
            aci = AdaptiveConformalInference(alpha_target=ALPHA, gamma=0.05)
            aci.reset(); aci.history_scores = list(cal_scores)
            for i in test:
                q = aci.get_quantile(); lo, hi = interval(preds[i], q)
                los.append(lo); his.append(hi); c = lo <= gts[i][0] <= hi
                cov.append(c); aci.update(nonconf(preds[i], gts[i]), c)
        else:
            pbo = PBAwareJointConformalOnline(alpha=ALPHA, window=300)
            pbo.warmstart(cal_scores)
            for i in test:
                q = pbo.get_quantile(); lo, hi = interval(preds[i], q)
                los.append(lo); his.append(hi); cov.append(lo <= gts[i][0] <= hi)
                pbo.update(nonconf(preds[i], gts[i]))
        los = np.array(los); his = np.array(his)
        out[m] = {"coverage": float(np.mean(cov)), "width": float((his - los).mean())}
    return out

agg = {m: {"coverage": [], "width": []} for m in METHODS}
for sd in CAL_SEEDS:
    r = eval_run(sd)
    for m in METHODS:
        agg[m]["coverage"].append(r[m]["coverage"])
        agg[m]["width"].append(r[m]["width"])
    print(f"  seed {sd} done")

print("\\n" + "=" * 72)
print(f"PHASE E — NuInsSeg total-count conformal (in-domain) | alpha={ALPHA}, target cov={1-ALPHA:.0%}")
print("=" * 72)
print(f"{'Method':18s} | {'Coverage':>16s} | {'Width':>16s}")
print("-" * 72)
res_json = {}
for m in METHODS:
    cm, cs = np.mean(agg[m]["coverage"]), np.std(agg[m]["coverage"])
    wm, ws = np.mean(agg[m]["width"]), np.std(agg[m]["width"])
    res_json[m] = {"coverage": [float(cm), float(cs)], "width": [float(wm), float(ws)]}
    print(f"{mnames[m]:18s} | {cm*100:>6.1f}+/-{cs*100:<4.1f}% | {wm:>7.2f}+/-{ws:<6.2f}")

with open(f"{WORK}/phase_E_results.json", "w") as f:
    json.dump({"config": {"alpha": ALPHA, "cal_seeds": CAL_SEEDS,
                          "n_patches": len(preds), "total_count_MAE": float(mae)},
               "results": res_json}, f, indent=2)
print(f"\\nSaved: {WORK}/phase_E_results.json")
'''))

cells.append(md(
    "## Notes / next",
    "",
    "- `phase_E_nuinsseg_preds.pkl` caches (pred scores, GT counts, organ) per patch.",
    "- **Cross-dataset shift experiment (CPU, later):** calibrate on PanNuke total-count",
    "  (derive from `phase_C_predictions.pkl`, sum over classes), test on NuInsSeg here.",
    "  Resize/normalize for patch-size comparability. Shows PB-JCI Online recovering",
    "  coverage under REAL shift.",
    "- Per-organ breakdown possible from the cached `organs` list (31-organ robustness).",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT.name}: {len(cells)} cells")

from __future__ import annotations
from pathlib import Path

OUT = Path(__file__).parent / "sam3_pannuke_phaseA1_kong_subset.ipynb"
LIB_DIR = Path(__file__).parent / "lib"

def _read(name: str) -> str:
    return (LIB_DIR / name).read_text(encoding="utf-8")

PANNUKE_LOADER = "%%writefile pannuke_loader.py\n" + _read("pannuke_loader.py")
METRICS        = "%%writefile metrics.py\n"        + _read("metrics.py")
SAM3_TRAIN     = "%%writefile sam3_train.py\n"     + _read("sam3_train.py")

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
    "# Phase A1 — Kong-protocol subset eval (Table A1 appendix)",
    "",
    "**Goal:** Direct numerical comparison vs Kong et al. 2025 Table 1 dùng",
    "Kong-protocol matched subset (~907 random Fold 3 patches, ~33% per Kong's 5:1:3 split).",
    "",
    "**Khác A1 extended full Fold 3:**",
    "- N = 907 random patches (vs 2722 full)",
    "- Seed cố định cho reproducibility",
    "- Output cho Table A1 appendix paper",
    "",
    "**Compute:** ~30 min trên T4.",
))

cells.append(md("## 00 — Setup"))
cells.append(code('''
import subprocess, sys, os, platform, time, json
print("Python:", sys.version.split()[0])
import torch
print("Torch :", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name(0))
'''))

cells.append(code('''
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

assert os.path.exists(CHECKPOINT_PATH), "Attach hipinhththu/sam3-native-pt"
assert os.path.exists(DATA_ROOT), "Attach hipinhththu/pannuke"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", SAM3_DIR], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scikit-image", "scikit-learn", "matplotlib", "opencv-python",
                "pycocotools", "einops", "tqdm"], check=True)
print("OK setup")
'''))

cells.append(md("## Helper modules"))
cells.append(code(PANNUKE_LOADER))
cells.append(code(METRICS))
cells.append(code(SAM3_TRAIN))

cells.append(code('''
import sys
for p in [".", "/kaggle/working", SAM3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from PIL import Image
from tqdm import tqdm

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from metrics import ClassWiseAccumulator, PerPromptClassAccumulator
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                        forward_decoder_with_grad, inference_to_binary)
print("Helpers loaded.")
'''))

cells.append(md("## 01 — Build SAM3 zero-shot (NO LoRA)"))
cells.append(code('''
from sam3.model_builder import build_sam3_image_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam3_image_model(
    device=device, eval_mode=True,
    checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"SAM3 params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print("Zero-shot baseline ready.")
'''))

cells.append(md("## 02 — Kong-protocol matched subset (~33% of Fold 3)"))
cells.append(code('''

KONG_SUBSET_RATIO = 3/9
SEED = 42

fold3 = PanNukeFold(DEFAULT_ROOT, 3)
np.random.seed(SEED)
n_subset = int(len(fold3) * KONG_SUBSET_RATIO)
subset_indices = np.random.choice(len(fold3), n_subset, replace=False)

print(f"Fold 3 total: {len(fold3)}")
print(f"Kong-subset (33%): {n_subset} patches  (seed={SEED})")
'''))

cells.append(md("## 03 — Prompt strategies (paper protocol)"))
cells.append(code('''
PROMPTS_MEDICAL = {
    "Neoplastic":   ["histopathology image of neoplastic tissue"],
    "Inflammatory": ["histopathology image of inflammatory tissue"],
    "Connective":   ["histopathology image of connective tissue"],
    "Dead":         ["histopathology image of dead tissue"],
    "Epithelial":   ["histopathology image of epithelial tissue"],
}
PROMPTS_LLM = {
    "Neoplastic":   ["Neoplastic cell", "Tumor cell", "Cancer cell", "Malignant cell"],
    "Inflammatory": ["Inflammatory cell", "Lymphocyte", "Immune cell", "Leukocyte"],
    "Connective":   ["Connective tissue cell", "Fibroblast", "Stromal cell"],
    "Dead":         ["Dead cell", "Apoptotic cell", "Necrotic cell"],
    "Epithelial":   ["Epithelial cell", "Epithelium", "Lining cell",
                     "Surface cell", "Mucosal cell nucleus"],
}
PROMPT_GENERIC = "cell"
SCORE_THRESH = 0.3
print("Prompts ready.")
'''))

cells.append(md("## 04 — Inference helpers"))
cells.append(code('''
from sam3.model.data_misc import FindStage

transform = make_transform(resolution=1008)
find_stage = FindStage(
    img_ids=torch.tensor([0], device=device, dtype=torch.long),
    text_ids=torch.tensor([0], device=device, dtype=torch.long),
    input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
    input_points=None, input_points_mask=None,
)

@torch.no_grad()
def encode_cached(pil): return encode_image_frozen(model, transform, pil, device=device)

@torch.no_grad()
def predict(state, prompt):
    st = dict(state)
    text_out = encode_text(model, prompt, device=device)
    st.update(text_out)
    outputs = forward_decoder_with_grad(model, st, find_stage, model._get_dummy_prompt())
    pm = inference_to_binary(outputs, target_hw=(256, 256), score_threshold=SCORE_THRESH)
    return pm.cpu().numpy().astype(bool)
'''))

cells.append(md("## 05 — Eval on Kong subset"))
cells.append(code('''
acc_med = ClassWiseAccumulator(CELL_TYPES)
acc_llm = PerPromptClassAccumulator(CELL_TYPES, PROMPTS_LLM)
acc_gen = ClassWiseAccumulator(CELL_TYPES)

t0 = time.time()
for idx in tqdm(subset_indices, desc="Kong subset eval"):
    sample = fold3[int(idx)]
    pil = Image.fromarray(sample["image"]).convert("RGB")
    gt = {c: (sample["masks"][CELL_TYPES.index(c)] > 0) for c in CELL_TYPES}
    state = encode_cached(pil)

    pred_gen = predict(state, PROMPT_GENERIC)
    for c in CELL_TYPES: acc_gen.update(pred_gen, gt[c], c)
    for c in CELL_TYPES:
        pred_m = predict(state, PROMPTS_MEDICAL[c][0])
        acc_med.update(pred_m, gt[c], c)
    for c, prompts in PROMPTS_LLM.items():
        for p in prompts:
            pred_l = predict(state, p)
            acc_llm.update(pred_l, gt[c], c, p)

elapsed = time.time() - t0
print(f"\\nDone. {elapsed/60:.1f}min ({elapsed/n_subset:.1f}s/image)")
'''))

cells.append(md("## 06 — Compare vs Kong Table 1"))
cells.append(code('''
KONG_TABLE1 = {
    "Medical": {"mIoU": 0.26, "Dice": 0.37},
    "LLM":     {"mIoU": 4.08, "Dice": 5.16},
    "Generic": {"mIoU": 6.22, "Dice": 8.13},
}

results = {
    "Medical": acc_med.summary(),
    "LLM":     acc_llm.summary(),
    "Generic": acc_gen.summary(),
}

print("=" * 90)
print(f"PHASE A1 — Kong-protocol subset (N={n_subset}, seed={SEED})")
print("=" * 90)
print(f"\\n{'Strategy':12s} | {'Kong paper':>12s} | {'A1 Kong-subset':>18s} | {'Δ':>8s}")
print("-" * 90)
for name, kong in KONG_TABLE1.items():
    ours = results[name]["mIoU"] * 100
    delta = ours - kong["mIoU"]
    sign = "+" if delta >= 0 else ""
    print(f"{name:12s} | {kong['mIoU']:>11.2f}% | {ours:>17.2f}% | {sign}{delta:>6.2f}pp")

print("\\nPer-class mIoU (Kong-subset):")
print(f"{'Class':14s} | {'Medical':>9s} | {'LLM-avg':>9s} | {'Generic':>9s}")
print("-" * 60)
for c in CELL_TYPES:
    m = results["Medical"]["per_class"][c]["IoU"] * 100
    l = results["LLM"]["per_class"][c]["IoU"] * 100
    g = results["Generic"]["per_class"][c]["IoU"] * 100
    print(f"  {c:12s} | {m:>8.2f}% | {l:>8.2f}% | {g:>8.2f}%")
'''))

cells.append(code('''
out_path = f"{WORK}/phase_A1_kong_subset_results.json"
with open(out_path, "w") as f:
    json.dump({
        "config": {"n_subset": int(n_subset), "seed": SEED,
                    "ratio": KONG_SUBSET_RATIO, "elapsed_minutes": elapsed/60},
        "kong_reference": KONG_TABLE1,
        "ours_kong_subset": results,
    }, f, indent=2)
print(f"Saved: {out_path}")
print("\\nUse for paper Appendix Table A1 — direct Kong comparison.")
'''))

cells.append(md(
    "## Notes for paper Section 4.1",
    "",
    "Declare protocol khác nhau:",
    "",
    "```",
    "Main results (Table 1): Gamper standard PanNuke protocol",
    "  Train Folds 1+2 (N=5,181), test full Fold 3 (N=2,722).",
    "",
    "Appendix (Table A1): Kong-protocol matched subset",
    "  Random 33% subset of Fold 3 (N=907, seed=42),",
    "  matching Kong et al. 2025 internal 5:1:3 split for direct numerical compare.",
    "```",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

import json
with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT.name}: {len(cells)} cells")

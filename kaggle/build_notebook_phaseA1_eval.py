from __future__ import annotations
from pathlib import Path

OUT = Path(__file__).parent / "sam3_pannuke_phaseA1_eval.ipynb"
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
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }

cells: list[dict] = []

cells.append(md(
    "# Phase A1 — Eval Extended (full Fold 3, paper-grade)",
    "",
    "**Goal:** Eval zero-shot SAM3 trên **full Fold 3** (2722 patches) với 3 prompting",
    "strategies (Medical/LLM/Generic) theo paper protocol — đồng bộ scope với A2 eval.",
    "",
    "**Khác Phase A1 cũ (N=1000):**",
    "- Full Fold 3 (2722 patches) thay vì 1000 random",
    "- Same scope as A2 eval → fair direct comparison",
    "- Paper-grade statistical power (CI ±1.3% thay vì ±2.5%)",
    "",
    "**KHÔNG dùng LoRA:**",
    "- A1 = zero-shot baseline (SAM3 native)",
    "- So sánh với A2 (LoRA) sẽ cho thấy gain của fine-tuning",
    "",
    "**Prerequisites Kaggle:**",
    "1. GPU: T4",
    "2. Datasets:",
    "   - `hipinhththu/pannuke`",
    "   - `hipinhththu/sam3-native-pt`",
    "",
    "**Compute budget:** ~1.5h trên T4 (no LoRA overhead).",
))

cells.append(md("## 00 — Setup"))

cells.append(code('''
import subprocess, sys, os, platform, time, json
print("Python  :", sys.version.split()[0])
print("Platform:", platform.platform())
import torch
print("Torch   :", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
    print("VRAM    :", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
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

cells.append(md("## Helper modules (writefile)"))
cells.append(code(PANNUKE_LOADER))
cells.append(code(METRICS))
cells.append(code(SAM3_TRAIN))

cells.append(code('''
import sys
for p in [".", "/kaggle/working", SAM3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from metrics import (ClassWiseAccumulator, PerPromptClassAccumulator, union_masks)
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                        forward_decoder_with_grad, inference_to_binary)

print("Helpers loaded.")
'''))

cells.append(md("## 01 — Build SAM3 (zero-shot, NO LoRA)"))

cells.append(code('''
from sam3.model_builder import build_sam3_image_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Build SAM3 (zero-shot, no LoRA)...")
model = build_sam3_image_model(
    device=device, eval_mode=True,
    checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"SAM3 params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print("Model frozen. Zero-shot baseline ready.")
'''))

cells.append(md("## 02 — Load full Fold 3"))

cells.append(code('''
from PIL import Image
import numpy as np
from tqdm import tqdm

fold3 = PanNukeFold(DEFAULT_ROOT, 3)
print(f"Fold 3: {len(fold3)} patches (FULL eval — paper-grade)")
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
NUM_SAMPLES = len(fold3)

n_med = sum(len(v) for v in PROMPTS_MEDICAL.values())
n_llm = sum(len(v) for v in PROMPTS_LLM.values())
print(f"Eval {NUM_SAMPLES} images x {n_med + n_llm + 1} prompts/image")
print(f"  Medical: {n_med} prompts (1/class)")
print(f"  LLM-gen: {n_llm} prompts (avg over synonyms)")
print(f"  Generic: 1 prompt ('cell')")
'''))

cells.append(md("## 04 — Inference helpers (image cache pattern)"))

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
def encode_image_cached(pil_img):
    """Backbone forward 1 LAN -> return state. Re-use cho nhieu prompts."""
    return encode_image_frozen(model, transform, pil_img, device=device)

@torch.no_grad()
def predict_from_state(backbone_out_cached, prompt):
    """Run 1 prompt tren backbone_out da encode. Trả binary mask (256, 256)."""
    state = dict(backbone_out_cached)
    text_out = encode_text(model, prompt, device=device)
    state.update(text_out)
    geometric_prompt = model._get_dummy_prompt()
    outputs = forward_decoder_with_grad(
        model, state, find_stage, geometric_prompt
    )
    pm = inference_to_binary(outputs, target_hw=(256, 256),
                             score_threshold=SCORE_THRESH)
    return pm.cpu().numpy().astype(bool)

print("Inference helpers ready.")
'''))

cells.append(md("## 05 — Full Fold 3 eval (3 strategies, zero-shot)"))

cells.append(code('''
acc_medical = ClassWiseAccumulator(CELL_TYPES)
acc_llm     = PerPromptClassAccumulator(CELL_TYPES, PROMPTS_LLM)
acc_generic = ClassWiseAccumulator(CELL_TYPES)

t0 = time.time()
for i in tqdm(range(NUM_SAMPLES), desc="Phase A1 eval extended"):
    sample = fold3[i]
    pil_img = Image.fromarray(sample["image"]).convert("RGB")
    gt = {c: (sample["masks"][CELL_TYPES.index(c)] > 0) for c in CELL_TYPES}

    state = encode_image_cached(pil_img)

    pred_gen = predict_from_state(state, PROMPT_GENERIC)
    for c in CELL_TYPES:
        acc_generic.update(pred_gen, gt[c], c)

    for c in CELL_TYPES:
        pred_m = predict_from_state(state, PROMPTS_MEDICAL[c][0])
        acc_medical.update(pred_m, gt[c], c)

    for c, prompts in PROMPTS_LLM.items():
        for p in prompts:
            pred_l = predict_from_state(state, p)
            acc_llm.update(pred_l, gt[c], c, p)

elapsed = time.time() - t0
print(f"\\nDone. {elapsed/60:.1f} min ({elapsed/NUM_SAMPLES:.1f}s/image)")
'''))

cells.append(md("## 06 — Report + Compare vs Phase A1 (N=1000) + Paper"))

cells.append(code('''
PAPER_TABLE1 = {
    "Medical terminology"      : {"mIoU": 0.26, "Dice": 0.37},
    "LLM-generated vocabulary" : {"mIoU": 4.08, "Dice": 5.16},
    "General medical ('cell')" : {"mIoU": 6.22, "Dice": 8.13},
}

PHASE_A1_N1000_RESULTS = {
    "Medical terminology"      : {"mIoU": 5.26, "Dice": 9.61},
    "LLM-generated vocabulary" : {"mIoU": 7.51, "Dice": 11.39},
    "General medical ('cell')" : {"mIoU": 13.99, "Dice": 22.21},
}

results_a1_ext = {
    "Medical terminology"      : acc_medical.summary(),
    "LLM-generated vocabulary" : acc_llm.summary(),
    "General medical ('cell')" : acc_generic.summary(),
}

print("=" * 110)
print(f"PHASE A1 EVAL EXTENDED — Zero-shot SAM3 on PanNuke FULL Fold 3 | N={NUM_SAMPLES}")
print("=" * 110)
print(f"\\n{'Strategy':35s} | {'Paper':>10s} | {'A1 (N=1000)':>14s} | {'A1 ext (full)':>15s} | {'delta':>8s}")
print("-" * 110)
for name, paper in PAPER_TABLE1.items():
    a1_ext_miou = results_a1_ext[name]["mIoU"] * 100
    a1_1k_miou = PHASE_A1_N1000_RESULTS[name]["mIoU"]
    delta = a1_ext_miou - a1_1k_miou
    arrow = "+" if delta >= 0 else ""
    print(f"{name:35s} | {paper['mIoU']:>9.2f}% | {a1_1k_miou:>13.2f}% | "
          f"{a1_ext_miou:>14.2f}% | {arrow}{delta:>6.2f}pp")

print("\\n" + "-" * 80)
print("Per-class mIoU (%) — full Fold 3:")
print(f"{'Class':14s} | {'Medical':>9s} | {'LLM-avg':>9s} | {'Generic':>9s}")
print("-" * 80)
for c in CELL_TYPES:
    m = results_a1_ext["Medical terminology"]["per_class"][c]["IoU"] * 100
    l = results_a1_ext["LLM-generated vocabulary"]["per_class"][c]["IoU"] * 100
    g = results_a1_ext["General medical ('cell')"]["per_class"][c]["IoU"] * 100
    print(f"  {c:12s} | {m:>8.2f}% | {l:>8.2f}% | {g:>8.2f}%")

print("\\n" + "=" * 80)
print("PASS criteria check (vs paper):")
gen_miou = results_a1_ext["General medical ('cell')"]["mIoU"] * 100
med_miou = results_a1_ext["Medical terminology"]["mIoU"] * 100
llm_miou = results_a1_ext["LLM-generated vocabulary"]["mIoU"] * 100

order_ok = med_miou < llm_miou < gen_miou
print(f"  Paper order (Med < LLM < Gen)? Med={med_miou:.2f} LLM={llm_miou:.2f} Gen={gen_miou:.2f}")
print(f"  -> {'PASS' if order_ok else 'WARN'}")
print(f"  Generic <= 2x paper baseline? {gen_miou:.2f}% vs paper 6.22% -> "
      f"{'PASS' if gen_miou <= 14 else 'WARN (higher than paper - might be protocol diff)'}")
'''))

cells.append(code('''
final_out = {
    "config": {
        "num_samples": NUM_SAMPLES,
        "fold": 3,
        "score_thresh": SCORE_THRESH,
        "no_lora": True,
        "elapsed_minutes": elapsed / 60,
        "prompts_medical": PROMPTS_MEDICAL,
        "prompts_llm": PROMPTS_LLM,
        "prompt_generic": PROMPT_GENERIC,
    },
    "paper_reference": PAPER_TABLE1,
    "phase_a1_n1000_results": PHASE_A1_N1000_RESULTS,
    "phase_a1_extended_results": results_a1_ext,
}
out_path = f"{WORK}/phase_A1_extended_results.json"
with open(out_path, "w") as f:
    json.dump(final_out, f, indent=2)
print(f"Saved: {out_path}")

print("\\n" + "=" * 80)
print("PHASE A1 EXTENDED DONE — paper-grade zero-shot baseline on full Fold 3")
print("=" * 80)
print(f"  Results JSON : {out_path}")
print(f"  Use for fair compare vs A2 (LoRA, same scope N={NUM_SAMPLES})")
'''))

cells.append(md(
    "## Phase A1 Extended — PASS criteria",
    "",
    "- **Paper order**: Medical < LLM < Generic (giữ ranking như Kong)",
    "- **Generic mIoU < 18%** (close to paper 6.22% nhưng higher do protocol diff)",
    "- **Per-class** Dead > 0 (full Fold 3 có Dead instances)",
    "",
    "**Khác Phase A1 N=1000:**",
    "- Số liệu có thể giảm nhẹ (regression to mean với N lớn hơn)",
    "- CI tight hơn",
    "- Đồng bộ scope với A2 → fair compare",
    "",
    "**Khác paper Kong 2025:**",
    "- Ta dùng Gamper protocol (full Fold 3) — Kong dùng internal 5:1:3 split mean cross-fold",
    "- Khai báo rõ trong Section 4.1 paper draft",
    "- Subset Kong-protocol (~907 random Fold 3) optional cho Table A1 appendix",
    "",
    "**Output:**",
    "- `/kaggle/working/phase_A1_extended_results.json` — paper Section 4.2 zero-shot baseline",
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

import json
with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT.name}: {len(cells)} cells")

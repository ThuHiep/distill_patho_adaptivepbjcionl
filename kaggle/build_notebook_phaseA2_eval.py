from __future__ import annotations
from pathlib import Path

OUT = Path(__file__).parent / "sam3_pannuke_phaseA2_eval.ipynb"
LIB_DIR = Path(__file__).parent / "lib"

def _read(name: str) -> str:
    return (LIB_DIR / name).read_text(encoding="utf-8")

PANNUKE_LOADER = "%%writefile pannuke_loader.py\n" + _read("pannuke_loader.py")
METRICS        = "%%writefile metrics.py\n"        + _read("metrics.py")
LORA_SAM3      = "%%writefile lora_sam3.py\n"      + _read("lora_sam3.py")
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
    "# Phase A2 — Eval (post-training)",
    "",
    "**Goal:** Eval SAM3 + LoRA-fine-tuned weights tren full Fold 3 (2722 patches)",
    "với 3 prompting strategies (Medical/LLM/Generic) theo Kong et al. 2025 protocol.",
    "",
    "**Input:** LoRA checkpoint `sam3_lora_rank16_final.pt` (~10MB) từ training notebook.",
    "",
    "**Prerequisites Kaggle:**",
    "1. GPU: T4 (12h session)",
    "2. Datasets:",
    "   - `hipinhththu/pannuke`",
    "   - `hipinhththu/sam3-native-pt`",
    "   - **LoRA checkpoint** — upload thành Kaggle Dataset riêng, hoặc copy từ Phase A2",
    "",
    "**Compute budget:** ~4-5h trên T4 (image cache: 1 backbone forward/image + 25 decoder).",
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

LORA_CKPT_CANDIDATES = [
    "/kaggle/input/datasets/hipinhththu/phase-a2-lora-weights/sam3_lora_rank16_final.pt",
    "/kaggle/input/datasets/hipinhththu/phase-a2-lora-weights/sam3_lora_rank16_epoch2.pt",
    "/kaggle/input/datasets/hipinhththu/phase-a2-lora-weights/sam3_lora_rank16_epoch1.pt",
    "/kaggle/input/phase-a2-lora-weights/sam3_lora_rank16_final.pt",
    f"{WORK}/sam3_lora_rank16_final.pt",
]
LORA_CKPT_PATH = None
for cand in LORA_CKPT_CANDIDATES:
    if os.path.exists(cand):
        LORA_CKPT_PATH = cand
        break

if LORA_CKPT_PATH is None:
    print("ERROR: KHONG tim thay LoRA checkpoint. Da check:")
    for c in LORA_CKPT_CANDIDATES:
        print(f"  - {c}")
    print("\\nFix: Upload sam3_lora_rank16_final.pt thanh Kaggle Dataset")
    print("     ten 'phase-a2-lora-weights' va attach vao notebook nay.")
    raise FileNotFoundError("LoRA checkpoint not found")
print(f"LoRA checkpoint: {LORA_CKPT_PATH}")

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
cells.append(code(LORA_SAM3))
cells.append(code(SAM3_TRAIN))

cells.append(code('''
import sys
for p in [".", "/kaggle/working", SAM3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from metrics import (ClassWiseAccumulator, PerPromptClassAccumulator, union_masks)
from lora_sam3 import (LoRALinear, inject_lora, freeze_non_lora,
                       load_lora_state, DEFAULT_LORA_TARGETS)
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                        forward_decoder_with_grad, semantic_union_mask,
                        inference_to_binary)

print("Helpers loaded.")
'''))

cells.append(md("## 01 — Build SAM3 + Inject LoRA + Load checkpoint"))

cells.append(code('''
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Build SAM3...")
model = build_sam3_image_model(
    device=device, eval_mode=True,
    checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
)
model.eval()
print(f"SAM3 params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

LORA_R = 16
LORA_ALPHA = 32
replaced, n_lora = inject_lora(
    model, target_module_names=DEFAULT_LORA_TARGETS,
    r=LORA_R, alpha=LORA_ALPHA, dropout=0.0,
)
print(f"LoRA inject: {len(replaced)} modules, {n_lora:,} params")

n_loaded = load_lora_state(model, LORA_CKPT_PATH)
print(f"Loaded LoRA weights: {n_loaded} tensors from {LORA_CKPT_PATH}")

freeze_non_lora(model)
model.eval()
'''))

cells.append(md("## 02 — Load Fold 3"))

cells.append(code('''
from PIL import Image
import numpy as np
from tqdm import tqdm

fold3 = PanNukeFold(DEFAULT_ROOT, 3)
print(f"Fold 3: {len(fold3)} patches")
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
    """Backbone forward 1 LAN -> return state. Re-use cho nhieu prompts.

    Phase A1 pattern: 25 prompts/image gom 1 backbone + 25 text+decoder ~5x speedup.
    """
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

cells.append(md("## 05 — Full Fold 3 eval (3 strategies)"))

cells.append(code('''
acc_medical = ClassWiseAccumulator(CELL_TYPES)
acc_llm     = PerPromptClassAccumulator(CELL_TYPES, PROMPTS_LLM)
acc_generic = ClassWiseAccumulator(CELL_TYPES)

t0 = time.time()
for i in tqdm(range(NUM_SAMPLES), desc="Phase A2 eval"):
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

cells.append(md("## 06 — Report + Compare vs Phase A1 + paper"))

cells.append(code('''
PAPER_TABLE1 = {
    "Medical terminology"      : {"mIoU": 0.26, "Dice": 0.37},
    "LLM-generated vocabulary" : {"mIoU": 4.08, "Dice": 5.16},
    "General medical ('cell')" : {"mIoU": 6.22, "Dice": 8.13},
}

PHASE_A1_RESULTS = {
    "Medical terminology"      : {"mIoU": 5.26, "Dice": 9.61},
    "LLM-generated vocabulary" : {"mIoU": 7.51, "Dice": 11.39},
    "General medical ('cell')" : {"mIoU": 13.99, "Dice": 22.21},
}

results_a2 = {
    "Medical terminology"      : acc_medical.summary(),
    "LLM-generated vocabulary" : acc_llm.summary(),
    "General medical ('cell')" : acc_generic.summary(),
}

print("=" * 100)
print(f"PHASE A2 EVAL — LoRA-fine-tuned SAM3 on PanNuke Fold 3 | N={NUM_SAMPLES}")
print("=" * 100)
print(f"\\n{'Strategy':35s} | {'Paper':>8s} | {'A1':>8s} | {'A2':>8s} | {'gain A2-A1':>12s}")
print("-" * 100)
for name, paper in PAPER_TABLE1.items():
    a2_miou = results_a2[name]["mIoU"] * 100
    a1_miou = PHASE_A1_RESULTS[name]["mIoU"]
    gain = a2_miou - a1_miou
    arrow = "+" if gain > 0 else ""
    print(f"{name:35s} | {paper['mIoU']:>7.2f}% | {a1_miou:>7.2f}% | "
          f"{a2_miou:>7.2f}% | {arrow}{gain:>10.2f}pp")

print("\\n" + "-" * 100)
print("Per-class breakdown (mIoU %):")
print(f"{'Class':14s} | {'Medical':>8s} | {'LLM-avg':>8s} | {'Generic':>8s}")
print("-" * 50)
for c in CELL_TYPES:
    m = results_a2["Medical terminology"]["per_class"][c]["IoU"] * 100
    l = results_a2["LLM-generated vocabulary"]["per_class"][c]["IoU"] * 100
    g = results_a2["General medical ('cell')"]["per_class"][c]["IoU"] * 100
    print(f"  {c:12s} | {m:>7.2f}% | {l:>7.2f}% | {g:>7.2f}%")

print("\\n" + "=" * 100)
print("PASS criteria check:")
gen_miou = results_a2["General medical ('cell')"]["mIoU"] * 100
med_miou = results_a2["Medical terminology"]["mIoU"] * 100
dead_iou = results_a2["General medical ('cell')"]["per_class"]["Dead"]["IoU"] * 100

print(f"  Generic mIoU >= 25%?    {gen_miou:.2f}%  ->  "
      f"{'PASS' if gen_miou >= 25 else 'FAIL'}")
print(f"  Medical mIoU >= 15%?    {med_miou:.2f}%  ->  "
      f"{'PASS' if med_miou >= 15 else 'FAIL'}")
print(f"  Dead class >= 5%?       {dead_iou:.2f}%  ->  "
      f"{'PASS' if dead_iou >= 5 else 'WARN (Dead rare in PanNuke)'}")
'''))

cells.append(code('''
import json

final_out = {
    "config": {
        "num_samples": NUM_SAMPLES,
        "lora_ckpt": LORA_CKPT_PATH,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "elapsed_minutes": elapsed / 60,
    },
    "paper_reference": PAPER_TABLE1,
    "phase_a1_baseline": PHASE_A1_RESULTS,
    "phase_a2_results": results_a2,
}
out_path = f"{WORK}/phase_A2_final_results.json"
with open(out_path, "w") as f:
    json.dump(final_out, f, indent=2)
print(f"Saved: {out_path}")
'''))

cells.append(md(
    "## Phase A2 Eval — PASS criteria",
    "",
    "- **Generic mIoU ≥ 25%** (gain ≥ 11pp vs Phase A1 baseline 13.99%)",
    "- **Medical mIoU ≥ 15%** (gain ≥ 10pp vs Phase A1 baseline 5.26%)",
    "- **Per-class** Dead ≥ 5% (cải thiện class hiếm)",
    "- **Order vs paper**: Medical < LLM < Generic (giữ ranking)",
    "",
    "**Nếu Generic < 20%:**",
    "- LoRA training chưa đủ → cần thêm epoch (Colab Pro A100)",
    "- Hoặc rank LoRA quá nhỏ → thử r=32 với weights load lại",
    "",
    "**Output saved:**",
    "- `/kaggle/working/phase_A2_final_results.json` — full table for paper Section 4.2",
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

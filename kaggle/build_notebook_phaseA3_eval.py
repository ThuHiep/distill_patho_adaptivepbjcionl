from __future__ import annotations
from pathlib import Path

OUT = Path(__file__).parent / "sam3_pannuke_phaseA3_eval.ipynb"
LIB_DIR = Path(__file__).parent / "lib"

def _read(name: str) -> str:
    return (LIB_DIR / name).read_text(encoding="utf-8")

PANNUKE_LOADER = "%%writefile pannuke_loader.py\n" + _read("pannuke_loader.py")
METRICS        = "%%writefile metrics.py\n"        + _read("metrics.py")
LORA_SAM3      = "%%writefile lora_sam3.py\n"      + _read("lora_sam3.py")
SAM3_TRAIN     = "%%writefile sam3_train.py\n"     + _read("sam3_train.py")
TYPE_HEAD      = "%%writefile type_head.py\n"      + _read("type_head.py")

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
    "# Phase A3 — Eval (post-training, full Fold 3)",
    "",
    "**Goal:** Eval A3 TypeHead trên **full Fold 3** (2722 patches) —",
    "đồng bộ scope với A2 eval, paper-grade evidence cho Section 4.3.",
    "",
    "**Khác A3 training notebook (eval block 500 Fold 1):**",
    "- Full Fold 3 (2722 patches) thay vì 500 Fold 1",
    "- Test fold (Fold 3) — không leak với train fold (Fold 1+2)",
    "- Dead class có thật instances thay vì 0",
    "",
    "**Input:**",
    "- LoRA checkpoint `sam3_lora_rank16_final.pt` (Phase A2)",
    "- TypeHead checkpoint `type_head_final.pt` (Phase A3)",
    "- PanNuke Fold 3",
    "",
    "**Prerequisites Kaggle:**",
    "1. GPU: T4 (12h session, sẽ dùng ~2h)",
    "2. Datasets:",
    "   - `hipinhththu/pannuke`",
    "   - `hipinhththu/sam3-native-pt`",
    "   - `hipinhththu/phase-a2-lora-weights`",
    "   - `hipinhththu/phase-a3-typehead-weights`",
    "",
    "**Compute budget:** ~2h trên T4 (1 backbone + 1 prompt + N type forwards / image).",
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
    "/kaggle/input/phase-a2-lora-weights/sam3_lora_rank16_final.pt",
    f"{WORK}/sam3_lora_rank16_final.pt",
]
LORA_CKPT_PATH = next((p for p in LORA_CKPT_CANDIDATES if os.path.exists(p)), None)
assert LORA_CKPT_PATH, f"Khong tim thay LoRA. Da check: {LORA_CKPT_CANDIDATES}"
print(f"LoRA   : {LORA_CKPT_PATH}")

TYPEHEAD_CKPT_CANDIDATES = [
    "/kaggle/input/datasets/hipinhththu/phase-a3-typehead-weights/type_head_final.pt",
    "/kaggle/input/datasets/hipinhththu/phase-a3-typehead-weights/type_head_epoch3.pt",
    "/kaggle/input/datasets/hipinhththu/phase-a3-typehead-weights/type_head_epoch2.pt",
    "/kaggle/input/phase-a3-typehead-weights/type_head_final.pt",
    f"{WORK}/type_head_final.pt",
]
TYPEHEAD_CKPT_PATH = next((p for p in TYPEHEAD_CKPT_CANDIDATES if os.path.exists(p)), None)
assert TYPEHEAD_CKPT_PATH, f"Khong tim thay TypeHead. Da check: {TYPEHEAD_CKPT_CANDIDATES}"
print(f"TypeHead: {TYPEHEAD_CKPT_PATH}")

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
cells.append(code(TYPE_HEAD))

cells.append(code('''
import sys
for p in [".", "/kaggle/working", SAM3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from lora_sam3 import (inject_lora, freeze_non_lora, load_lora_state,
                       DEFAULT_LORA_TARGETS)
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                        forward_decoder_with_grad, inference_to_binary)
from type_head import (TypeHead, roi_pool_feature, compute_iou_matrix,
                       hungarian_match, extract_gt_instances,
                       per_class_counts, per_class_variance)
print("Helpers loaded.")
'''))

cells.append(md("## 01 — Build SAM3 + Load LoRA + Load TypeHead"))

cells.append(code('''
from sam3.model_builder import build_sam3_image_model

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
n_loaded = load_lora_state(model, LORA_CKPT_PATH)
print(f"LoRA inject: {len(replaced)} modules, {n_loaded} tensors loaded")

for p in model.parameters():
    p.requires_grad = False
model.eval()

type_head = TypeHead(in_dim=256, hidden_dim=128, num_classes=5, dropout=0.0).to(device)
state = torch.load(TYPEHEAD_CKPT_PATH, map_location=device)
type_head.load_state_dict(state)
type_head.eval()
n_th = sum(p.numel() for p in type_head.parameters())
print(f"TypeHead loaded: {n_th:,} params ({n_th/1e3:.1f}K)")
print(f"  Source: {TYPEHEAD_CKPT_PATH}")
print("Model fully ready (frozen, eval mode).")
'''))

cells.append(md("## 02 — Load full Fold 3"))

cells.append(code('''
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)

fold3 = PanNukeFold(DEFAULT_ROOT, 3)
print(f"Fold 3: {len(fold3)} patches (FULL eval — paper-grade)")

EVAL_PROMPT = "cell"
IOU_THRESH = 0.3
SCORE_THRESH = 0.3
NUM_EVAL = len(fold3)
print(f"Prompt: '{EVAL_PROMPT}'  |  IoU thresh: {IOU_THRESH}  |  N={NUM_EVAL}")
'''))

cells.append(md("## 03 — Inference pipeline"))

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
def run_sam3_inference(pil_img, prompt):
    """Run SAM3+LoRA, return (pred_masks_list, scores, backbone_feat)."""
    backbone_out = encode_image_frozen(model, transform, pil_img, device=device)

    feat = None
    if "vision_features" in backbone_out:
        feat = backbone_out["vision_features"]
    elif "backbone_fpn" in backbone_out:
        feat = backbone_out["backbone_fpn"][-1]
    else:
        for k, v in backbone_out.items():
            if isinstance(v, torch.Tensor) and v.dim() == 4:
                feat = v
                break
    assert feat is not None, "Cannot find backbone feature for ROI pooling"
    if feat.dim() == 4:
        feat = feat[0]

    text_out = encode_text(model, prompt, device=device)
    backbone_out.update(text_out)
    geom = model._get_dummy_prompt()
    outputs = forward_decoder_with_grad(model, backbone_out, find_stage, geom)

    pred_logits = outputs["pred_logits"].float()
    pred_masks  = outputs["pred_masks"].float()
    pres_logit  = outputs["presence_logit_dec"].float()
    cls_prob = pred_logits.sigmoid()
    pres = pres_logit.sigmoid().unsqueeze(1)
    prob = (cls_prob * pres).squeeze(-1).squeeze(0)

    masks_up = F.interpolate(
        pred_masks, size=(256, 256), mode="bilinear", align_corners=False
    ).sigmoid().squeeze(0)
    masks_bin = (masks_up > 0.5)

    keep = prob > SCORE_THRESH
    if keep.sum() == 0:
        return [], [], feat
    pred_masks_list = [masks_bin[i].cpu().numpy().astype(bool)
                       for i in range(len(masks_bin)) if keep[i]]
    scores_list = prob[keep].cpu().tolist()
    return pred_masks_list, scores_list, feat

@torch.no_grad()
def predict_types_for_image(pil_img, prompt=EVAL_PROMPT):
    """Forward SAM3+TypeHead. Returns (masks, scores, type_probs, type_logits)."""
    pred_masks_list, scores_list, backbone_feat = run_sam3_inference(pil_img, prompt)
    N = len(pred_masks_list)
    if N == 0:
        return [], [], np.zeros((0, 5)), np.zeros((0, 5))

    features = torch.zeros(N, 256, device=device)
    for i, pm in enumerate(pred_masks_list):
        mask_t = torch.from_numpy(pm).to(device)
        features[i] = roi_pool_feature(backbone_feat, mask_t)

    type_logits = type_head(features)
    type_probs = type_logits.softmax(dim=-1)
    return pred_masks_list, scores_list, type_probs.cpu().numpy(), type_logits.cpu().numpy()

print("Inference pipeline ready.")
'''))

cells.append(md("## 04 — Full Fold 3 eval"))

cells.append(code('''
correct = 0
total = 0
per_class_correct = {c: 0 for c in CELL_TYPES}
per_class_total = {c: 0 for c in CELL_TYPES}
confusion = np.zeros((5, 5), dtype=np.int64)

counting_errors = {c: [] for c in CELL_TYPES}
pred_counts_log = {c: [] for c in CELL_TYPES}
gt_counts_log = {c: [] for c in CELL_TYPES}

n_detections_per_image = []
n_gt_per_image = []
n_matched_per_image = []
n_images_no_detection = 0
n_images_no_gt = 0

t0 = time.time()
for idx in tqdm(range(NUM_EVAL), desc="A3 eval full Fold 3"):
    sample = fold3[idx]
    pil_img = Image.fromarray(sample["image"]).convert("RGB")

    gt_masks, gt_classes = extract_gt_instances(sample, CELL_TYPES)
    gt_counts = {c: 0 for c in CELL_TYPES}
    for ci in gt_classes:
        gt_counts[CELL_TYPES[ci]] += 1
    n_gt_per_image.append(len(gt_masks))
    for c in CELL_TYPES:
        gt_counts_log[c].append(gt_counts[c])

    if len(gt_masks) == 0:
        n_images_no_gt += 1

    pred_masks_list, scores_list, type_probs, _ = predict_types_for_image(
        pil_img, prompt=EVAL_PROMPT
    )
    n_detections_per_image.append(len(pred_masks_list))

    if len(pred_masks_list) == 0:
        n_images_no_detection += 1
        for c in CELL_TYPES:
            counting_errors[c].append(gt_counts[c])
            pred_counts_log[c].append(0)
        n_matched_per_image.append(0)
        continue

    scores_arr = np.array(scores_list)
    pred_counts = per_class_counts(scores_arr, type_probs)
    for ci in range(5):
        c = CELL_TYPES[ci]
        counting_errors[c].append(abs(pred_counts[ci] - gt_counts[c]))
        pred_counts_log[c].append(float(pred_counts[ci]))

    if len(gt_masks) > 0:
        iou_matrix = compute_iou_matrix(pred_masks_list, gt_masks)
        matches = hungarian_match(iou_matrix, iou_thresh=IOU_THRESH)
        n_matched_per_image.append(len(matches))
        for pred_i, gt_j in matches:
            true_class = gt_classes[gt_j]
            pred_class = int(type_probs[pred_i].argmax())
            confusion[true_class, pred_class] += 1
            per_class_total[CELL_TYPES[true_class]] += 1
            if true_class == pred_class:
                correct += 1
                per_class_correct[CELL_TYPES[true_class]] += 1
            total += 1
    else:
        n_matched_per_image.append(0)

elapsed = time.time() - t0
print(f"\\nDone {NUM_EVAL} patches in {elapsed/60:.1f} min ({elapsed/NUM_EVAL:.2f}s/patch)")
'''))

cells.append(md("## 05 — Report"))

cells.append(code('''
A3_TRAIN_EVAL_RESULTS = {
    "Neoplastic":   89.11,
    "Inflammatory": 55.22,
    "Connective":   79.98,
    "Dead":         0.00,
    "Epithelial":   85.04,
    "Macro":        83.17,
}

macro_acc = correct / max(total, 1)

print("=" * 80)
print(f"PHASE A3 EVAL — Full Fold 3 (N={NUM_EVAL})")
print("=" * 80)
print(f"\\nMacro accuracy: {macro_acc*100:.2f}%  ({correct}/{total} matched detections)")
print(f"Total matched detections: {total:,}")
print(f"Images with no detection: {n_images_no_detection}/{NUM_EVAL}")
print(f"Images with no GT:        {n_images_no_gt}/{NUM_EVAL}")
print(f"Avg detections/image:     {np.mean(n_detections_per_image):.1f}")
print(f"Avg GT/image:             {np.mean(n_gt_per_image):.1f}")
print(f"Avg matched/image:        {np.mean(n_matched_per_image):.1f}")

print("\\n" + "-" * 80)
print(f"{'Class':14s} | {'A3 train (500 F1)':>20s} | {'A3 eval (full F3)':>20s} | {'delta':>8s}")
print("-" * 80)
for c in CELL_TYPES:
    if per_class_total[c] > 0:
        acc_c = per_class_correct[c] / per_class_total[c] * 100
    else:
        acc_c = 0.0
    train_acc = A3_TRAIN_EVAL_RESULTS[c]
    delta = acc_c - train_acc
    arrow = "+" if delta >= 0 else ""
    print(f"  {c:12s} | {train_acc:>17.2f}%  | {acc_c:>17.2f}%  | {arrow}{delta:>6.2f}pp")
print("-" * 80)
delta_macro = macro_acc*100 - A3_TRAIN_EVAL_RESULTS["Macro"]
arrow_m = "+" if delta_macro >= 0 else ""
print(f"  {'Macro':12s} | {A3_TRAIN_EVAL_RESULTS['Macro']:>17.2f}%  | "
      f"{macro_acc*100:>17.2f}%  | {arrow_m}{delta_macro:>6.2f}pp")

print("\\n" + "-" * 80)
print("Confusion matrix (rows=true, cols=pred):")
print(f"  {'':14s} " + " ".join(f"{c[:6]:>8s}" for c in CELL_TYPES))
for i, c_true in enumerate(CELL_TYPES):
    row = " ".join(f"{confusion[i, j]:>8d}" for j in range(5))
    total_i = confusion[i].sum()
    print(f"  {c_true:14s} {row}   ({total_i} total)")

print("\\n" + "-" * 80)
print("Counting MAE per class (|N_pred - N_gt| averaged over images):")
print(f"  {'Class':14s} | {'MAE':>8s} | {'Mean GT':>8s} | {'Mean Pred':>10s} | {'rel error':>10s}")
print("-" * 80)
for c in CELL_TYPES:
    if counting_errors[c]:
        mae = float(np.mean(counting_errors[c]))
        mean_gt = float(np.mean(gt_counts_log[c]))
        mean_pred = float(np.mean(pred_counts_log[c]))
        rel = mae / max(mean_gt, 1e-6) * 100
        print(f"  {c:12s} | {mae:>7.3f}  | {mean_gt:>7.2f}  | {mean_pred:>9.2f}  | {rel:>8.1f}%")
'''))

cells.append(code('''
print("\\n" + "=" * 80)
print("PASS criteria check (paper-grade eval):")
print("=" * 80)

macro = macro_acc * 100
neo   = per_class_correct["Neoplastic"]   / max(per_class_total["Neoplastic"], 1)   * 100
inf   = per_class_correct["Inflammatory"] / max(per_class_total["Inflammatory"], 1) * 100
con   = per_class_correct["Connective"]   / max(per_class_total["Connective"], 1)   * 100
dead  = per_class_correct["Dead"]         / max(per_class_total["Dead"], 1)         * 100
epi   = per_class_correct["Epithelial"]   / max(per_class_total["Epithelial"], 1)   * 100

dead_count = per_class_total["Dead"]
neo_mae = float(np.mean(counting_errors["Neoplastic"]))

print(f"  Macro accuracy >= 60%?       {macro:.2f}%  ->  {'PASS' if macro >= 60 else 'FAIL'}")
print(f"  Neoplastic   accuracy >= 75%? {neo:.2f}%  ->  {'PASS' if neo >= 75 else 'FAIL'}")
print(f"  Epithelial   accuracy >= 70%? {epi:.2f}%  ->  {'PASS' if epi >= 70 else 'FAIL'}")
print(f"  Connective   accuracy >= 60%? {con:.2f}%  ->  {'PASS' if con >= 60 else 'FAIL'}")
print(f"  Inflammatory accuracy >= 40%? {inf:.2f}%  ->  {'PASS' if inf >= 40 else 'FAIL'}")
print(f"  Dead present (N >= 5)?        {dead_count} instances  ->  "
      f"{'PASS' if dead_count >= 5 else 'WARN (Dead rare in PanNuke Fold 3)'}")
print(f"  Counting MAE Neoplastic <= 5? {neo_mae:.2f}  ->  {'PASS' if neo_mae <= 5 else 'FAIL'}")
'''))

cells.append(code('''
final_results = {
    "config": {
        "num_eval": NUM_EVAL,
        "fold": 3,
        "prompt": EVAL_PROMPT,
        "iou_thresh": IOU_THRESH,
        "score_thresh": SCORE_THRESH,
        "lora_ckpt": LORA_CKPT_PATH,
        "typehead_ckpt": TYPEHEAD_CKPT_PATH,
        "elapsed_minutes": elapsed / 60,
    },
    "a3_train_eval_reference": A3_TRAIN_EVAL_RESULTS,
    "eval": {
        "macro_type_accuracy": macro_acc,
        "per_class_accuracy": {
            c: (per_class_correct[c] / max(per_class_total[c], 1))
            for c in CELL_TYPES
        },
        "per_class_total_matched": {c: per_class_total[c] for c in CELL_TYPES},
        "per_class_correct": {c: per_class_correct[c] for c in CELL_TYPES},
        "per_class_counting_mae": {
            c: float(np.mean(counting_errors[c]))
            for c in CELL_TYPES if counting_errors[c]
        },
        "per_class_mean_gt_count": {
            c: float(np.mean(gt_counts_log[c]))
            for c in CELL_TYPES if gt_counts_log[c]
        },
        "per_class_mean_pred_count": {
            c: float(np.mean(pred_counts_log[c]))
            for c in CELL_TYPES if pred_counts_log[c]
        },
        "confusion_matrix": confusion.tolist(),
        "total_matched_detections": total,
        "n_images_no_detection": n_images_no_detection,
        "n_images_no_gt": n_images_no_gt,
        "mean_detections_per_image": float(np.mean(n_detections_per_image)),
        "mean_gt_per_image": float(np.mean(n_gt_per_image)),
        "mean_matched_per_image": float(np.mean(n_matched_per_image)),
    },
}
out_path = f"{WORK}/phase_A3_final_results.json"
with open(out_path, "w") as f:
    json.dump(final_results, f, indent=2)
print(f"Saved: {out_path}")

print("\\n" + "=" * 80)
print("PHASE A3 EVAL DONE — paper-grade results on full Fold 3")
print("=" * 80)
print(f"  Results JSON: {out_path}")
print(f"  Next step   : Phase C (conformal counting) uses (s_i, p_ik) ready")
'''))

cells.append(md(
    "## Phase A3 Eval — PASS criteria",
    "",
    "- **Macro accuracy ≥ 60%** (relaxed from train eval 83% vì test fold)",
    "- **Neoplastic ≥ 75%** (most common class, should be robust)",
    "- **Epithelial ≥ 70%**",
    "- **Connective ≥ 60%**",
    "- **Inflammatory ≥ 40%** (challenging — small cells, similar to others)",
    "- **Dead present** (N ≥ 5 matched) — even rare class should appear",
    "- **Counting MAE Neoplastic ≤ 5** cells/image",
    "",
    "**Nếu macro < 50%:** A3 train trên Fold 1+2 không transfer tốt sang Fold 3.",
    "Cần consider:",
    "- Cross-fold train (rotate folds)",
    "- Feature normalization",
    "- Test ablation: train on Fold 1 only + eval Fold 2 + 3",
    "",
    "**Output:**",
    "- `/kaggle/working/phase_A3_final_results.json` — paper Section 4.3 table",
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

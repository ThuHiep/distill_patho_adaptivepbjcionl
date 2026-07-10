from __future__ import annotations
from pathlib import Path

OUT = Path(__file__).parent / "sam3_pannuke_phaseA3_typehead.ipynb"
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
    "# Phase A3: Type Head Training (Linear 256→5)",
    "",
    "**Goal:** Train lightweight MLP type head on top of SAM3+LoRA (Phase A2) để output",
    "per-detection per-class probability $p_{ik}$ — input cho Phase C/D conformal counting.",
    "",
    "**Architecture:**",
    "- SAM3 backbone (frozen) → multi-scale FPN features",
    "- SAM3 decoder + LoRA (frozen, từ A2) → N detections {mask_i, score_i}",
    "- For each detection: ROI-pool backbone features under mask_i → 256-d feature",
    "- **TypeHead** Linear(256 → 128 → 5) → cross-entropy loss",
    "",
    "**Training:**",
    "- Generic prompt 'cell' (cho phép adapt vào nhiều cell types)",
    "- Hungarian matching predicted masks ↔ GT instances theo IoU",
    "- Cross-entropy loss trên matched detections",
    "",
    "**Prerequisites:**",
    "- `hipinhththu/pannuke`",
    "- `hipinhththu/sam3-native-pt`",
    "- `phase-a2-lora-weights` (sam3_lora_rank16_final.pt)",
    "",
    "**Compute budget:** ~4-5h trên Kaggle T4.",
))

cells.append(md("## 00 — Setup"))

cells.append(code('''
import subprocess, sys, os, platform, time, json
print("Python  :", sys.version.split()[0])
import torch
print("Torch   :", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
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
]
LORA_CKPT_PATH = next((p for p in LORA_CKPT_CANDIDATES if os.path.exists(p)), None)
assert LORA_CKPT_PATH, f"Khong tim thay LoRA. Da check: {LORA_CKPT_CANDIDATES}"
print(f"LoRA: {LORA_CKPT_PATH}")

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
from metrics import ClassWiseAccumulator
from lora_sam3 import (inject_lora, freeze_non_lora, load_lora_state,
                       DEFAULT_LORA_TARGETS)
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                        forward_decoder_with_grad, inference_to_binary)
from type_head import (TypeHead, roi_pool_feature, compute_iou_matrix,
                       hungarian_match, extract_gt_instances,
                       per_class_counts, per_class_variance)
print("Helpers loaded.")
'''))

cells.append(md("## 01 — Build SAM3 + Inject LoRA + Load A2 weights"))

cells.append(code('''
from sam3.model_builder import build_sam3_image_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Build SAM3...")
model = build_sam3_image_model(
    device=device, eval_mode=True,
    checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
)
model.eval()

LORA_R = 16
LORA_ALPHA = 32
replaced, n_lora = inject_lora(
    model, target_module_names=DEFAULT_LORA_TARGETS,
    r=LORA_R, alpha=LORA_ALPHA, dropout=0.0,
)
n_loaded = load_lora_state(model, LORA_CKPT_PATH)
print(f"LoRA injected {len(replaced)} modules, {n_loaded} tensors loaded")

for p in model.parameters():
    p.requires_grad = False
model.eval()
print("Model fully frozen (SAM3 + LoRA both frozen). Train ONLY TypeHead.")
'''))

cells.append(md("## 02 — Initialize TypeHead"))

cells.append(code('''
type_head = TypeHead(in_dim=256, hidden_dim=128, num_classes=5, dropout=0.1).to(device)
n_type_params = sum(p.numel() for p in type_head.parameters())
print(f"TypeHead params: {n_type_params:,} ({n_type_params/1e3:.1f}K)")
print(f"  Architecture: Linear(256, 128) -> LayerNorm -> ReLU -> Dropout -> Linear(128, 5)")
'''))

cells.append(md("## 03 — Dataloader"))

cells.append(code('''
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

fold1 = PanNukeFold(DEFAULT_ROOT, 1)
fold2 = PanNukeFold(DEFAULT_ROOT, 2)
print(f"Train: Fold 1+2 = {len(fold1)+len(fold2)} patches (Fold 3 NOT loaded)")

TRAIN_PROMPT = "cell"
print(f"Train prompt: '{TRAIN_PROMPT}' (Generic, lay tat ca cells)")
'''))

cells.append(md("## 04 — Inference pipeline for type head"))

cells.append(code('''
from sam3.model.data_misc import FindStage
import torch.nn.functional as F

transform = make_transform(resolution=1008)
find_stage = FindStage(
    img_ids=torch.tensor([0], device=device, dtype=torch.long),
    text_ids=torch.tensor([0], device=device, dtype=torch.long),
    input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
    input_points=None, input_points_mask=None,
)
SCORE_THRESH = 0.3

@torch.no_grad()
def run_sam3_inference(pil_img, prompt):
    """Run SAM3+LoRA va return: (pred_masks, scores, backbone_feat).

    pred_masks: list of (256, 256) bool
    scores: list of float
    backbone_feat: (D, H', W') tensor de roi-pool features
    """
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
    outputs = forward_decoder_with_grad(
        model, backbone_out, find_stage, geom
    )

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

def predict_types_for_image(pil_img, prompt=TRAIN_PROMPT):
    """Run inference + type head. Return per-detection type probs.

    Returns:
        pred_masks: list of (256,256) bool
        scores: list of float
        type_probs: (N, 5) softmax probs
        features: (N, 256) pooled features
    """
    pred_masks_list, scores_list, backbone_feat = run_sam3_inference(pil_img, prompt)
    N = len(pred_masks_list)
    if N == 0:
        return [], [], np.zeros((0, 5)), np.zeros((0, 256))

    features = torch.zeros(N, 256, device=device)
    for i, pm in enumerate(pred_masks_list):
        mask_t = torch.from_numpy(pm).to(device)
        features[i] = roi_pool_feature(backbone_feat, mask_t)

    with torch.no_grad():
        type_logits = type_head(features)
        type_probs = type_logits.softmax(dim=-1)

    return pred_masks_list, scores_list, type_probs.cpu().numpy(), features.cpu().numpy()

print("Inference pipeline ready.")
'''))

cells.append(md("## 05 — Training Loop"))

cells.append(code('''
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 300
MAX_TRAIN_PER_EPOCH = 2500
IOU_THRESH = 0.3
LOG_EVERY = 100

optimizer = AdamW(type_head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * MAX_TRAIN_PER_EPOCH)

train_indices = list(range(len(fold1))) + [(len(fold1) + i) for i in range(len(fold2))]

def train_step(sample):
    """1 image. Forward + Hungarian match + cross-entropy."""
    pil_img = Image.fromarray(sample["image"]).convert("RGB")
    pred_masks_list, scores_list, backbone_feat = run_sam3_inference(
        pil_img, TRAIN_PROMPT
    )
    if len(pred_masks_list) == 0:
        return None

    gt_masks, gt_classes = extract_gt_instances(sample, CELL_TYPES)
    if len(gt_masks) == 0:
        return None

    iou_matrix = compute_iou_matrix(pred_masks_list, gt_masks)
    matches = hungarian_match(iou_matrix, iou_thresh=IOU_THRESH)
    if len(matches) == 0:
        return None

    matched_features = []
    matched_labels = []
    for pred_i, gt_j in matches:
        mask_t = torch.from_numpy(pred_masks_list[pred_i]).to(device)
        feat = roi_pool_feature(backbone_feat, mask_t)
        matched_features.append(feat)
        matched_labels.append(gt_classes[gt_j])

    features = torch.stack(matched_features)
    labels = torch.tensor(matched_labels, dtype=torch.long, device=device)

    type_logits = type_head(features)
    loss = F.cross_entropy(type_logits, labels)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(type_head.parameters(), max_norm=1.0)
    optimizer.step()

    pred_labels = type_logits.argmax(dim=-1)
    accuracy = (pred_labels == labels).float().mean().item()

    return {"loss": loss.item(), "acc": accuracy, "n_matched": len(matches)}

training_log = {"config": {
    "num_epochs": NUM_EPOCHS, "lr": LEARNING_RATE,
    "max_train_per_epoch": MAX_TRAIN_PER_EPOCH,
    "iou_thresh": IOU_THRESH,
    "lora_ckpt": LORA_CKPT_PATH,
}, "epochs": []}

global_step = 0
t0 = time.time()

all_folds = [fold1, fold2]

for epoch in range(NUM_EPOCHS):
    print(f"\\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")
    type_head.train()
    losses, accs = [], []

    indices = list(range(len(fold1) + len(fold2)))
    random.shuffle(indices)
    indices = indices[:MAX_TRAIN_PER_EPOCH]

    pbar = tqdm(indices, desc=f"Epoch {epoch+1}")
    for step, idx in enumerate(pbar):
        if idx < len(fold1):
            sample = fold1[idx]
        else:
            sample = fold2[idx - len(fold1)]

        if global_step < WARMUP_STEPS:
            lr_now = LEARNING_RATE * (global_step + 1) / WARMUP_STEPS
            for g in optimizer.param_groups:
                g["lr"] = lr_now
        else:
            scheduler.step()

        result = train_step(sample)
        if result is None:
            continue

        losses.append(result["loss"])
        accs.append(result["acc"])
        global_step += 1

        if global_step % LOG_EVERY == 0:
            recent_loss = np.mean(losses[-LOG_EVERY:])
            recent_acc = np.mean(accs[-LOG_EVERY:])
            pbar.set_postfix({
                "loss": f"{recent_loss:.4f}",
                "acc": f"{recent_acc:.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
            })

    epoch_loss = float(np.mean(losses)) if losses else 0.0
    epoch_acc = float(np.mean(accs)) if accs else 0.0
    elapsed = time.time() - t0
    print(f"  Epoch {epoch+1} done. Avg loss: {epoch_loss:.4f}, Acc: {epoch_acc*100:.2f}%")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    training_log["epochs"].append({
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "acc": epoch_acc,
        "n_steps": len(losses),
        "elapsed_seconds": elapsed,
    })

    ckpt_path = f"{WORK}/type_head_epoch{epoch+1}.pt"
    torch.save(type_head.state_dict(), ckpt_path)
    print(f"  Saved: {ckpt_path}")

final_path = f"{WORK}/type_head_final.pt"
torch.save(type_head.state_dict(), final_path)
print(f"\\n===== Training done. Final: {final_path} =====")
'''))

cells.append(md(
    "## 06 — Save Training Artifacts (NO eval here)",
    "",
    "Eval đã tách riêng → `sam3_pannuke_phaseA3_eval.ipynb` trên **full Fold 3**.",
    "Notebook này chỉ train + save TypeHead checkpoints, không leak Fold 3.",
))

cells.append(code('''
final_results = {
    "config": training_log["config"],
    "training_log": training_log,
    "data_split": "Fold 1+2 train ONLY, Fold 3 held out for eval notebook",
    "checkpoints": {
        "final": f"{WORK}/type_head_final.pt",
        "per_epoch": [f"{WORK}/type_head_epoch{e+1}.pt" for e in range(NUM_EPOCHS)],
    },
}
with open(f"{WORK}/phase_A3_train_log.json", "w") as f:
    json.dump(final_results, f, indent=2)
print(f"Saved: {WORK}/phase_A3_train_log.json")

print("\\n" + "=" * 70)
print("PHASE A3 TRAIN DONE — no eval here (strict Fold 3 holdout)")
print("=" * 70)
print(f"  TypeHead weights : {WORK}/type_head_final.pt")
print(f"  Train log JSON   : {WORK}/phase_A3_train_log.json")
print(f"\\n  Next steps:")
print(f"    1. Upload type_head_final.pt as Kaggle Dataset")
print(f"    2. Run sam3_pannuke_phaseA3_eval.ipynb on full Fold 3")
print(f"    3. Then Phase C conformal prediction")
'''))

cells.append(md(
    "## Phase A3 PASS criteria",
    "",
    "- **Macro type accuracy ≥ 45%** (vs random 20%)",
    "- **Neoplastic accuracy ≥ 60%** (most common class)",
    "- **Counting MAE per class ≤ 10** cells per image",
    "- **Inflammatory/Connective accuracy ≥ 40%**",
    "",
    "**Nếu macro acc < 35%:**",
    "- Tăng hidden_dim TypeHead 128 → 256",
    "- Train thêm epoch (3-5)",
    "- Adjust IOU_THRESH (try 0.2 hoặc 0.5)",
    "",
    "**Output cho Phase C:**",
    "- `type_head_final.pt` — TypeHead weights",
    "- `phase_A3_results.json` — per-class accuracy + MAE",
    "- Per-detection $p_{ik}$ ready for Poisson-Binomial conformal counting",
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

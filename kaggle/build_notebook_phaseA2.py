from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "sam3_pannuke_phaseA2_lora.ipynb"

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
    "# Phase A2: SAM3 + LoRA Fine-tune on PanNuke (TRAIN ONLY)",
    "",
    "**Goal:** Match SAM3-Adapter baseline (Kong et al. 2025 Fig 5, ~30-40% macro mIoU).",
    "",
    "**Setup:**",
    "- Backbone SAM3 frozen, LoRA rank=16 on decoder attention (~5-8M trainable params)",
    "- **Train**: PanNuke Fold 1 + Fold 2 ONLY (~5179 patches)",
    "- **Val/Eval**: SKIP here — Fold 3 strictly held out for `sam3_pannuke_phaseA2_eval.ipynb`",
    "- Gamper protocol: no Fold 3 leak into training",
    "",
    "**Prerequisites Kaggle:**",
    "1. GPU: T4 x2 hoặc P100",
    "2. Internet: ON",
    "3. Datasets:",
    "   - `hipinhththu/pannuke`",
    "   - `hipinhththu/sam3-native-pt`",
    "",
    "**Compute budget (2 epoch, LR 3e-4, full Fold1+2):**",
    "- Kaggle T4: ~9-10h (margin 2-3h trong 12h session)",
    "- Colab A100: ~3-4h",
    "",
    "*Note:* config đã tối ưu cho Kaggle T4. Trên A100 có thể tăng `NUM_EPOCHS=3` và",
    "hạ `LEARNING_RATE=2e-4` để gain thêm ~2-3pp mIoU.",
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
cells.append(code(LORA_SAM3))
cells.append(code(SAM3_TRAIN))

cells.append(code('''
import sys
for p in [".", "/kaggle/working", SAM3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from metrics import ClassWiseAccumulator, union_masks
from lora_sam3 import (LoRALinear, inject_lora, freeze_non_lora,
                       save_lora_state, load_lora_state, DEFAULT_LORA_TARGETS)
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                        forward_decoder_with_grad, semantic_union_mask,
                        semantic_seg_loss, inference_to_binary)

print("Helpers loaded.")
'''))

cells.append(md("## 01 — Build SAM3 + Inject LoRA"))

cells.append(code('''
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Build SAM3 (fp32, frozen backbone)...")
model = build_sam3_image_model(
    device=device, eval_mode=True,
    checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
)
model.eval()
n_total = sum(p.numel() for p in model.parameters())
print(f"SAM3 params: {n_total/1e6:.1f}M")
'''))

cells.append(code('''
print("Decoder Linear submodules (verify LoRA targets):")
count = 0
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear) and "decoder" in name.lower():
        attr_name = name.split(".")[-1]
        in_lora_targets = attr_name in DEFAULT_LORA_TARGETS
        marker = " <- LoRA target" if in_lora_targets else ""
        print(f"  {name}: {mod.in_features}->{mod.out_features}{marker}")
        count += 1
        if count >= 30:
            print(f"  ... (showing first 30 of many)")
            break
print(f"\\nDefault LoRA targets: {sorted(DEFAULT_LORA_TARGETS)}")
'''))

cells.append(code('''
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

replaced, n_lora = inject_lora(
    model,
    target_module_names=DEFAULT_LORA_TARGETS,
    r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT,
)

n_train, n_total = freeze_non_lora(model)
print(f"\\nTrainable: {n_train/1e6:.2f}M / Total: {n_total/1e6:.1f}M  "
      f"({100*n_train/n_total:.3f}%)")

if len(replaced) == 0:
    print("WARN: KHONG LoRA module nao duoc inject. Check decoder.py structure.")
    print("      Possible fix: extend DEFAULT_LORA_TARGETS trong lora_sam3.py")
'''))

cells.append(md("## 02 — Dataloader (PanNuke Fold 1+2 train ONLY, no Fold 3 here)"))

cells.append(code('''
fold1 = PanNukeFold(DEFAULT_ROOT, 1)
fold2 = PanNukeFold(DEFAULT_ROOT, 2)

train_size = len(fold1) + len(fold2)
print(f"Train: Fold 1+2 = {train_size} patches")
print("Fold 3 NOT loaded here — strictly held out for eval (Gamper protocol).")

PROMPTS_LLM = {
    "Neoplastic":   ["Neoplastic cell", "Tumor cell", "Cancer cell", "Malignant cell"],
    "Inflammatory": ["Inflammatory cell", "Lymphocyte", "Immune cell", "Leukocyte"],
    "Connective":   ["Connective tissue cell", "Fibroblast", "Stromal cell"],
    "Dead":         ["Dead cell", "Apoptotic cell", "Necrotic cell"],
    "Epithelial":   ["Epithelial cell", "Epithelium", "Lining cell",
                     "Surface cell", "Mucosal cell nucleus"],
}
'''))

cells.append(code('''
import random
import numpy as np
from PIL import Image

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class PanNukeTrainStream:
    """Stream (image, class_idx, gt_mask, prompt) samples cho training.

    Sample (img_i, class_k) random, lay random synonym lam prompt.
    """
    def __init__(self, folds, prompts_per_class, shuffle=True):
        self.folds = folds
        self.prompts = prompts_per_class
        self.cls_names = list(prompts_per_class.keys())
        self.index = []
        for fi, f in enumerate(folds):
            for i in range(len(f)):
                self.index.append((fi, i))
        if shuffle:
            random.shuffle(self.index)

    def __len__(self):
        return len(self.index) * len(self.cls_names)

    def iter_epoch(self):
        for fi, ii in self.index:
            sample = self.folds[fi][ii]
            img = Image.fromarray(sample["image"]).convert("RGB")
            for ck, cname in enumerate(self.cls_names):
                gt_mask = (sample["masks"][ck] > 0)
                if gt_mask.sum() < 10:
                    continue
                prompt = random.choice(self.prompts[cname])
                yield {
                    "image": img,
                    "class_name": cname,
                    "class_idx": ck,
                    "gt_mask": gt_mask,
                    "prompt": prompt,
                }

train_stream = PanNukeTrainStream([fold1, fold2], PROMPTS_LLM, shuffle=True)
print(f"Train stream: ~{len(train_stream)} (img,class) pairs/epoch (Fold 1+2 only)")
print("NOTE: NO val_stream — eval runs on separate eval notebook on Fold 3.")
'''))

cells.append(md("## 03 — Training step (train-only, no eval here)"))

cells.append(code('''
from sam3.model.data_misc import FindStage

transform = make_transform(resolution=1008)

find_stage = FindStage(
    img_ids=torch.tensor([0], device=device, dtype=torch.long),
    text_ids=torch.tensor([0], device=device, dtype=torch.long),
    input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
    input_points=None, input_points_mask=None,
)

def train_step(sample, optimizer):
    """1 (image, class) sample. Backprop qua LoRA params."""
    pil_img = sample["image"]
    prompt  = sample["prompt"]
    gt_mask = torch.from_numpy(sample["gt_mask"].astype(np.float32)).to(device)

    backbone_out = encode_image_frozen(model, transform, pil_img, device=device)
    text_out = encode_text(model, prompt, device=device)
    backbone_out.update(text_out)

    geometric_prompt = model._get_dummy_prompt()
    outputs = forward_decoder_with_grad(model, backbone_out,
                                         find_stage, geometric_prompt)

    pred = semantic_union_mask(outputs, target_hw=gt_mask.shape)

    loss, loss_dict = semantic_seg_loss(pred, gt_mask)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()

    return loss_dict

print("Training step ready. Eval is done in separate notebook (no val on Fold 3 here).")
'''))

cells.append(md("## 04 — Training Loop (2 epochs, Kaggle-friendly, NO Fold 3 eval)"))

cells.append(code('''
from tqdm import tqdm

NUM_EPOCHS = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 500
LOG_EVERY = 200
EVAL_EVERY = 1000
SAVE_EVERY = 500
MAX_TRAIN_PER_EPOCH = 5179

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS * MAX_TRAIN_PER_EPOCH
)

training_log = {
    "config": {
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "max_train_per_epoch": MAX_TRAIN_PER_EPOCH,
        "data_split": "Fold 1+2 train, NO Fold 3 val (eval on separate notebook)",
    },
    "epochs": [],
}

global_step = 0
t0 = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")
    model.train()
    epoch_losses = []

    pbar = tqdm(train_stream.iter_epoch(), total=MAX_TRAIN_PER_EPOCH,
                desc=f"Epoch {epoch+1}")
    for step, sample in enumerate(pbar):
        if step >= MAX_TRAIN_PER_EPOCH:
            break
        try:
            loss_dict = train_step(sample, optimizer)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"OOM step {step}, skip")
                continue
            raise
        if not np.isfinite(loss_dict["loss"]):
            optimizer.zero_grad()
            if step < 50 or step % 200 == 0:
                print(f"  WARN: NaN/Inf loss at step {step}, skipped")
            continue
        if global_step < WARMUP_STEPS:
            warmup_lr = LEARNING_RATE * (global_step + 1) / WARMUP_STEPS
            for g in optimizer.param_groups:
                g["lr"] = warmup_lr
        else:
            scheduler.step()
        epoch_losses.append(loss_dict["loss"])
        global_step += 1

        if global_step % LOG_EVERY == 0:
            recent = epoch_losses[-LOG_EVERY:]
            pbar.set_postfix({"loss": f"{np.mean(recent):.4f}",
                              "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        if global_step % SAVE_EVERY == 0 and global_step > 0:
            from lora_sam3 import save_lora_state
            snap_path = f"{WORK}/lora_snapshot_step{global_step}.pt"
            save_lora_state(model, snap_path)
            import glob
            snaps = sorted(glob.glob(f"{WORK}/lora_snapshot_step*.pt"))
            for old in snaps[:-2]:
                os.remove(old)

    print(f"  Epoch {epoch+1} done. Avg loss: {np.mean(epoch_losses):.4f}")
    elapsed_epoch = time.time() - t0
    print(f"  Elapsed total: {elapsed_epoch/60:.1f} min")

    training_log["epochs"].append({
        "epoch": epoch + 1,
        "avg_loss": float(np.mean(epoch_losses)),
        "elapsed_seconds": elapsed_epoch,
        "note": "eval skipped — run sam3_pannuke_phaseA2_eval.ipynb on Fold 3 after training",
    })

    ckpt_path = f"{WORK}/sam3_lora_rank{LORA_R}_epoch{epoch+1}.pt"
    n_saved = save_lora_state(model, ckpt_path)
    print(f"  Saved LoRA state ({n_saved} tensors): {ckpt_path}")

print(f"\\n===== Training done. Total: {(time.time()-t0)/60:.1f} min =====")
'''))

cells.append(md(
    "## 05 — Save Training Artifacts (no eval here)",
    "",
    "**Eval đã tách sang notebook riêng** `sam3_pannuke_phaseA2_eval.ipynb` (Fold 3 holdout).",
    "",
    "Notebook này CHỈ training + save LoRA weights. Fold 3 chưa bao giờ được load trong notebook này."
))

cells.append(code('''
import json

final_ckpt = f"{WORK}/sam3_lora_rank{LORA_R}_final.pt"
save_lora_state(model, final_ckpt)
print(f"Saved final LoRA: {final_ckpt}")

training_out = {
    "config": {
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "lora_targets": sorted(DEFAULT_LORA_TARGETS),
        "n_modules": len(replaced),
        "num_epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "max_train_per_epoch": MAX_TRAIN_PER_EPOCH,
    },
    "training_log": training_log,
    "elapsed_total_minutes": (time.time() - t0) / 60,
}
with open(f"{WORK}/phase_A2_training_log.json", "w") as f:
    json.dump(training_out, f, indent=2)
print(f"Saved log: {WORK}/phase_A2_training_log.json")

import glob
saved = sorted(glob.glob(f"{WORK}/sam3_lora_*.pt"))
print("\\nAll LoRA checkpoints saved:")
for p in saved:
    size_mb = os.path.getsize(p) / 1e6
    print(f"  {p}  ({size_mb:.2f} MB)")

print("\\n" + "=" * 70)
print("PHASE A2 TRAINING DONE")
print("=" * 70)
print(f"  Final LoRA weights: {final_ckpt}")
print(f"  Training log     : {WORK}/phase_A2_training_log.json")
print()
print("  Next step:")
print("  1. Download {final_ckpt} (or save vao Kaggle Dataset)")
print("  2. Open notebook `sam3_pannuke_phaseA2_eval.ipynb`")
print("  3. Upload LoRA checkpoint + Run All -> 3 strategies x 5 classes eval")
print("=" * 70)
'''))

cells.append(md(
    "## Phase A2 PASS criteria",
    "",
    "- **Train loss giảm đều** qua 2 epochs (no NaN)",
    "- **Val mIoU tăng** so với pre-training (Phase A1 baseline)",
    "- **Best strategy match SAM3-Adapter target**: macro mIoU 25-40%",
    "  (Kong et al. 2025 Fig 5 reported ~30% on PanNuke for SAM3-Adapter)",
    "- **Per-class IoU**: Neoplastic > 30%, các classes khác > 15%, Dead có thể thấp (rare)",
    "",
    "**Nếu macro mIoU < 15% sau training:**",
    "- Check LoRA injection (cell INSPECT in §01) — có đúng module names không",
    "- Tăng LORA_R lên 32, hoặc tăng LR lên 5e-4",
    "- Verify GT masks load đúng (visualize 1 sample)",
    "",
    "**Outputs saved to /kaggle/working/:**",
    "- `sam3_lora_rank16_final.pt` — LoRA weights (~10 MB)",
    "- `sam3_lora_rank16_epoch{1,2}.pt` — checkpoints sau mỗi epoch",
    "- `phase_A2_training_log.json` — loss curve + pre-train eval",
    "",
    "**Next steps:**",
    "- **Notebook eval riêng**: `sam3_pannuke_phaseA2_eval.ipynb` — full Fold 3 với 3 strategies",
    "- Phase A3: Type head (Linear 256→5) on top of fine-tuned model",
    "- Phase C: JCC main table (5 conformal methods × 3 settings)",
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

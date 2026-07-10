import os, sys, time, json, random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config_vast import (
    REPO_DIR, SAM3_DIR, LIB_DIR, WORK, DATA_ROOT,
    CHECKPOINT_PATH, CHECKPOINTS_OUT, verify_env
)
print("[Vast.ai] A2 LoRA Multi-seed Training")
print("=" * 70)
verify_env()
print("=" * 70)

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from lora_sam3 import (LoRALinear, inject_lora, freeze_non_lora,
                       save_lora_state, DEFAULT_LORA_TARGETS)
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                        forward_decoder_with_grad, semantic_union_mask,
                        compute_loss)
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage

SEEDS         = [42, 100, 200]   
NUM_EPOCHS    = 2
LR            = 3e-4
WARMUP_STEPS  = 500
WEIGHT_DECAY  = 1e-4
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
SAVE_EVERY    = 500
LOG_EVERY     = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

TRAIN_PROMPTS = [
    "Neoplastic cell", "Tumor cell", "Cancer cell", "Malignant cell",
    "Inflammatory cell", "Lymphocyte", "Immune cell", "Leukocyte",
    "Connective tissue cell", "Fibroblast", "Stromal cell",
    "Dead cell", "Apoptotic cell", "Necrotic cell",
    "Epithelial cell", "Epithelium", "Lining cell",
    "Surface cell", "Mucosal cell nucleus",
]

def train_one_seed(seed: int):
    print("\n" + "=" * 70)
    print(f"SEED {seed} — start")
    print("=" * 70)

    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
    model = build_sam3_image_model(
        device=device, eval_mode=True,
        checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
    )
    model.eval()

    
    replaced, n_lora = inject_lora(
        model, target_module_names=DEFAULT_LORA_TARGETS,
        r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT,
        path_must_contain="decoder",
    )
    freeze_non_lora(model)
    print(f"LoRA injected: {len(replaced)} modules, {n_lora:,} trainable params")

    
    fold1 = PanNukeFold(DATA_ROOT, 1)
    fold2 = PanNukeFold(DATA_ROOT, 2)
    train_data = []
    for i in range(len(fold1)): train_data.append((fold1, i))
    for i in range(len(fold2)): train_data.append((fold2, i))
    random.shuffle(train_data)
    print(f"Train: {len(train_data)} patches (Fold 1+2)")

    
    transform = make_transform(resolution=1008)
    find_stage = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )

    
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = NUM_EPOCHS * len(train_data)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - WARMUP_STEPS, eta_min=1e-6
    )

    
    log = {"seed": seed, "config": {
        "num_epochs": NUM_EPOCHS, "lr": LR, "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA, "n_trainable": n_lora,
    }, "epochs": []}
    global_step = 0
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        random.shuffle(train_data)
        losses = []
        pbar = tqdm(train_data, desc=f"Seed{seed} E{epoch+1}")
        for fold, idx in pbar:
            sample = fold[idx]
            pil = Image.fromarray(sample["image"]).convert("RGB")
            gt_union = sum((sample["masks"][CELL_TYPES.index(c)] > 0).astype(float)
                            for c in CELL_TYPES).clip(0, 1)
            gt_t = torch.from_numpy(gt_union).to(device).float()
            prompt = random.choice(TRAIN_PROMPTS)

            
            if global_step < WARMUP_STEPS:
                for g in optimizer.param_groups:
                    g["lr"] = LR * (global_step + 1) / WARMUP_STEPS
            else:
                scheduler.step()

            try:
                state = encode_image_frozen(model, transform, pil, device=device)
                text_out = encode_text(model, prompt, device=device)
                state.update(text_out)
                geom = model._get_dummy_prompt()
                outputs = forward_decoder_with_grad(model, state, find_stage, geom)
                pred_mask = semantic_union_mask(outputs, target_hw=(256, 256))
                loss = compute_loss(pred_mask, gt_t)

                if torch.isfinite(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    optimizer.step()
                    losses.append(loss.item())
                    global_step += 1

                    if global_step % LOG_EVERY == 0:
                        pbar.set_postfix({
                            "loss": f"{np.mean(losses[-LOG_EVERY:]):.4f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                        })

                    if global_step % SAVE_EVERY == 0:
                        ck = f"{CHECKPOINTS_OUT}/sam3_lora_seed{seed}_step{global_step}.pt"
                        save_lora_state(model, ck)
            except Exception as e:
                print(f"\nWARN step {global_step}: {e}")
                continue

        epoch_loss = float(np.mean(losses)) if losses else 0.0
        elapsed = time.time() - t0
        print(f"  Seed {seed} Epoch {epoch+1} done. Loss: {epoch_loss:.4f}, time: {elapsed/60:.1f}min")
        log["epochs"].append({
            "epoch": epoch + 1, "loss": epoch_loss,
            "n_steps": len(losses), "elapsed_seconds": elapsed,
        })

    
    final_ck = f"{CHECKPOINTS_OUT}/sam3_lora_seed{seed}_final.pt"
    save_lora_state(model, final_ck)
    print(f"\nSeed {seed} DONE. Final ckpt: {final_ck}")

    
    log["final_ckpt"] = final_ck
    log["total_elapsed_minutes"] = (time.time() - t0) / 60
    log_path = f"{WORK}/phase_A2_seed{seed}_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log: {log_path}")

    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    t_start = time.time()
    for seed in SEEDS:
        train_one_seed(seed)
    t_total = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"ALL {len(SEEDS)} SEEDS DONE in {t_total/3600:.2f}h")
    print(f"Checkpoints saved at: {CHECKPOINTS_OUT}")
    print("=" * 70)
    print("\nNext step: python run_a3_multiseed.py")

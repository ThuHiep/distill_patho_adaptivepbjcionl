import os, sys, time, json, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config_vast import (
    REPO_DIR, SAM3_DIR, LIB_DIR, WORK, DATA_ROOT,
    CHECKPOINT_PATH, CHECKPOINTS_OUT, verify_env
)
print("[Vast.ai] A3 TypeHead Multi-seed Training")
print("=" * 70)
verify_env()

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from lora_sam3 import (inject_lora, freeze_non_lora, load_lora_state,
                       DEFAULT_LORA_TARGETS)
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                        forward_decoder_with_grad, inference_to_binary)
from type_head import (TypeHead, roi_pool_feature, compute_iou_matrix,
                       hungarian_match, extract_gt_instances)
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage

SEEDS                = [42, 100, 200]
NUM_EPOCHS           = 3
LR                   = 1e-3
WEIGHT_DECAY         = 1e-4
WARMUP_STEPS         = 300
MAX_TRAIN_PER_EPOCH  = 2500
IOU_THRESH           = 0.3
TRAIN_PROMPT         = "cell"
SCORE_THRESH         = 0.3
LOG_EVERY            = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

def train_one_seed(seed: int):
    print("\n" + "=" * 70)
    print(f"A3 SEED {seed} — start")
    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
    a2_lora = f"{CHECKPOINTS_OUT}/sam3_lora_seed{seed}_final.pt"
    if not os.path.exists(a2_lora):
        print(f"FAIL: A2 LoRA for seed {seed} not found: {a2_lora}")
        print(f"      Run python run_a2_multiseed.py first!")
        return False

    model = build_sam3_image_model(
        device=device, eval_mode=True,
        checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
    )
    model.eval()
    inject_lora(model, target_module_names=DEFAULT_LORA_TARGETS,
                r=16, alpha=32, dropout=0.0, path_must_contain="decoder")
    load_lora_state(model, a2_lora)
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded A2 LoRA: {a2_lora}")

    
    type_head = TypeHead(in_dim=256, hidden_dim=128, num_classes=5,
                          dropout=0.1).to(device)
    n_params = sum(p.numel() for p in type_head.parameters())
    print(f"TypeHead: {n_params:,} params ({n_params/1e3:.1f}K)")

    
    fold1 = PanNukeFold(DATA_ROOT, 1)
    fold2 = PanNukeFold(DATA_ROOT, 2)
    print(f"Train: Fold 1+2 = {len(fold1)+len(fold2)} patches")

    
    transform = make_transform(resolution=1008)
    find_stage = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )

    @torch.no_grad()
    def run_inference(pil_img):
        backbone_out = encode_image_frozen(model, transform, pil_img, device=device)
        feat = backbone_out.get("vision_features")
        if feat is None and "backbone_fpn" in backbone_out:
            feat = backbone_out["backbone_fpn"][-1]
        if feat is None:
            for k, v in backbone_out.items():
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    feat = v; break
        if feat.dim() == 4:
            feat = feat[0]
        text_out = encode_text(model, TRAIN_PROMPT, device=device)
        backbone_out.update(text_out)
        outputs = forward_decoder_with_grad(model, backbone_out, find_stage,
                                              model._get_dummy_prompt())
        pred_logits = outputs["pred_logits"].float()
        pred_masks  = outputs["pred_masks"].float()
        pres_logit  = outputs["presence_logit_dec"].float()
        prob = (pred_logits.sigmoid() * pres_logit.sigmoid().unsqueeze(1)).squeeze(-1).squeeze(0)
        masks_up = F.interpolate(pred_masks, size=(256, 256), mode="bilinear",
                                   align_corners=False).sigmoid().squeeze(0)
        masks_bin = (masks_up > 0.5)
        keep = prob > SCORE_THRESH
        if keep.sum() == 0:
            return [], feat
        return [masks_bin[i].cpu().numpy().astype(bool)
                for i in range(len(masks_bin)) if keep[i]], feat

    
    optimizer = torch.optim.AdamW(type_head.parameters(), lr=LR,
                                    weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS * MAX_TRAIN_PER_EPOCH
    )

    log = {"seed": seed, "epochs": []}
    global_step = 0
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        type_head.train()
        indices = list(range(len(fold1) + len(fold2)))
        random.shuffle(indices)
        indices = indices[:MAX_TRAIN_PER_EPOCH]
        losses, accs = [], []
        pbar = tqdm(indices, desc=f"Seed{seed} E{epoch+1}")

        for idx in pbar:
            if idx < len(fold1):
                sample = fold1[idx]
            else:
                sample = fold2[idx - len(fold1)]

            if global_step < WARMUP_STEPS:
                for g in optimizer.param_groups:
                    g["lr"] = LR * (global_step + 1) / WARMUP_STEPS
            else:
                scheduler.step()

            pil = Image.fromarray(sample["image"]).convert("RGB")
            pred_masks, feat = run_inference(pil)
            if len(pred_masks) == 0:
                continue
            gt_masks, gt_classes = extract_gt_instances(sample, CELL_TYPES)
            if len(gt_masks) == 0:
                continue
            iou_matrix = compute_iou_matrix(pred_masks, gt_masks)
            matches = hungarian_match(iou_matrix, iou_thresh=IOU_THRESH)
            if len(matches) == 0:
                continue

            features, labels = [], []
            for pred_i, gt_j in matches:
                mt = torch.from_numpy(pred_masks[pred_i]).to(device)
                features.append(roi_pool_feature(feat, mt))
                labels.append(gt_classes[gt_j])
            features = torch.stack(features)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            logits = type_head(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(type_head.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
            accs.append((logits.argmax(-1) == labels).float().mean().item())
            global_step += 1

            if global_step % LOG_EVERY == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(losses[-LOG_EVERY:]):.4f}",
                    "acc": f"{np.mean(accs[-LOG_EVERY:]):.3f}",
                })

        ep_loss = float(np.mean(losses)) if losses else 0.0
        ep_acc = float(np.mean(accs)) if accs else 0.0
        elapsed = time.time() - t0
        print(f"  Seed{seed} E{epoch+1}: loss={ep_loss:.4f}, acc={ep_acc*100:.2f}%, {elapsed/60:.1f}min")
        log["epochs"].append({"epoch": epoch+1, "loss": ep_loss,
                               "acc": ep_acc, "elapsed_seconds": elapsed})
        torch.save(type_head.state_dict(),
                   f"{CHECKPOINTS_OUT}/type_head_seed{seed}_epoch{epoch+1}.pt")

    
    final_ck = f"{CHECKPOINTS_OUT}/type_head_seed{seed}_final.pt"
    torch.save(type_head.state_dict(), final_ck)
    print(f"Seed {seed} DONE. Final: {final_ck}")

    log["final_ckpt"] = final_ck
    log["total_elapsed_minutes"] = (time.time() - t0) / 60
    with open(f"{WORK}/phase_A3_seed{seed}_log.json", "w") as f:
        json.dump(log, f, indent=2)

    del model, type_head
    torch.cuda.empty_cache()
    return True

if __name__ == "__main__":
    t_start = time.time()
    success = []
    for seed in SEEDS:
        ok = train_one_seed(seed)
        success.append((seed, ok))
    print("\n" + "=" * 70)
    print(f"ALL DONE in {(time.time()-t_start)/3600:.2f}h")
    for s, ok in success:
        print(f"  Seed {s}: {'OK' if ok else 'FAIL'}")
    print(f"Checkpoints: {CHECKPOINTS_OUT}")
    print("Next: python run_phaseC_multiseed.py")

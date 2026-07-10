import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config_vast import WORK, DATA_ROOT, CHECKPOINT_PATH, CHECKPOINTS_OUT, verify_env
print("[Vast.ai] A3 Eval Multi-seed (full Fold 3)")
print("=" * 70)
verify_env()

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from lora_sam3 import inject_lora, freeze_non_lora, load_lora_state, DEFAULT_LORA_TARGETS
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                         forward_decoder_with_grad)
from type_head import (TypeHead, roi_pool_feature, compute_iou_matrix,
                       hungarian_match, extract_gt_instances, per_class_counts)
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage

SEEDS = [42, 100, 200]
EVAL_PROMPT = "cell"
IOU_THRESH = 0.3
SCORE_THRESH = 0.3

device = "cuda" if torch.cuda.is_available() else "cpu"
fold3 = PanNukeFold(DATA_ROOT, 3)
print(f"Fold 3: {len(fold3)} patches\n")

all_results = {}
for seed in SEEDS:
    print("=" * 70)
    print(f"SEED {seed} — A3 eval start")
    print("=" * 70)

    lora_ck = f"{CHECKPOINTS_OUT}/sam3_lora_seed{seed}_final.pt"
    th_ck   = f"{CHECKPOINTS_OUT}/type_head_seed{seed}_final.pt"
    assert os.path.exists(lora_ck) and os.path.exists(th_ck), f"Missing ckpts for seed {seed}"

    model = build_sam3_image_model(
        device=device, eval_mode=True,
        checkpoint_path=CHECKPOINT_PATH, load_from_HF=False,
    )
    model.eval()
    inject_lora(model, target_module_names=DEFAULT_LORA_TARGETS,
                r=16, alpha=32, dropout=0.0, path_must_contain="decoder")
    load_lora_state(model, lora_ck)
    for p in model.parameters():
        p.requires_grad = False

    type_head = TypeHead(in_dim=256, hidden_dim=128, num_classes=5, dropout=0.0).to(device)
    type_head.load_state_dict(torch.load(th_ck, map_location=device))
    type_head.eval()

    transform = make_transform(resolution=1008)
    find_stage = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )

    @torch.no_grad()
    def predict_image(pil_img):
        backbone_out = encode_image_frozen(model, transform, pil_img, device=device)
        feat = backbone_out.get("vision_features")
        if feat is None and "backbone_fpn" in backbone_out:
            feat = backbone_out["backbone_fpn"][-1]
        if feat is None:
            for k, v in backbone_out.items():
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    feat = v; break
        if feat.dim() == 4: feat = feat[0]
        text_out = encode_text(model, EVAL_PROMPT, device=device)
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
            return [], [], np.zeros((0, 5))
        pmk = [masks_bin[i].cpu().numpy().astype(bool)
               for i in range(len(masks_bin)) if keep[i]]
        sc = prob[keep].cpu().numpy()
        features = torch.zeros(len(pmk), 256, device=device)
        for i, pm in enumerate(pmk):
            mt = torch.from_numpy(pm).to(device)
            features[i] = roi_pool_feature(feat, mt)
        type_probs = type_head(features).softmax(dim=-1).cpu().numpy()
        return pmk, sc.tolist(), type_probs

    correct = 0; total = 0
    per_cls_corr = {c: 0 for c in CELL_TYPES}
    per_cls_tot  = {c: 0 for c in CELL_TYPES}
    confusion = np.zeros((5, 5), dtype=np.int64)
    counting_err = {c: [] for c in CELL_TYPES}

    t0 = time.time()
    for idx in tqdm(range(len(fold3)), desc=f"Seed{seed} eval"):
        sample = fold3[idx]
        pil = Image.fromarray(sample["image"]).convert("RGB")
        gt_masks, gt_classes = extract_gt_instances(sample, CELL_TYPES)
        gt_counts = {c: 0 for c in CELL_TYPES}
        for ci in gt_classes:
            gt_counts[CELL_TYPES[ci]] += 1

        pmk, sc, tp = predict_image(pil)
        if len(pmk) == 0:
            for c in CELL_TYPES:
                counting_err[c].append(gt_counts[c])
            continue

        
        pred_counts = per_class_counts(np.array(sc), tp)
        for ci in range(5):
            counting_err[CELL_TYPES[ci]].append(abs(pred_counts[ci] - gt_counts[CELL_TYPES[ci]]))

        
        if len(gt_masks) > 0:
            iou_m = compute_iou_matrix(pmk, gt_masks)
            mt = hungarian_match(iou_m, iou_thresh=IOU_THRESH)
            for pi, gj in mt:
                tc = gt_classes[gj]
                pc = int(tp[pi].argmax())
                confusion[tc, pc] += 1
                per_cls_tot[CELL_TYPES[tc]] += 1
                if tc == pc:
                    correct += 1
                    per_cls_corr[CELL_TYPES[tc]] += 1
                total += 1

    elapsed = time.time() - t0
    macro_acc = correct / max(total, 1)
    print(f"\nSeed {seed} done in {elapsed/60:.1f}min")
    print(f"  Macro acc: {macro_acc*100:.2f}%  ({correct}/{total})")
    for c in CELL_TYPES:
        acc_c = per_cls_corr[c] / max(per_cls_tot[c], 1) * 100
        mae = float(np.mean(counting_err[c])) if counting_err[c] else 0.0
        print(f"  {c:14s}: acc={acc_c:6.2f}%  MAE={mae:6.3f}")

    all_results[seed] = {
        "macro_accuracy": macro_acc,
        "per_class_accuracy": {c: per_cls_corr[c] / max(per_cls_tot[c], 1) for c in CELL_TYPES},
        "per_class_counting_mae": {c: float(np.mean(counting_err[c])) if counting_err[c] else 0.0
                                     for c in CELL_TYPES},
        "confusion_matrix": confusion.tolist(),
        "total_matched": total,
    }

    del model, type_head
    torch.cuda.empty_cache()

print(f"\n{'='*70}\nAGGREGATE (mean ± std)\n{'='*70}")
agg = {"per_class_acc": {}, "per_class_mae": {}}
macros = [all_results[s]["macro_accuracy"]*100 for s in SEEDS]
agg["macro_acc_mean"] = float(np.mean(macros))
agg["macro_acc_std"]  = float(np.std(macros))
print(f"  Macro acc: {agg['macro_acc_mean']:.2f}% ± {agg['macro_acc_std']:.2f}%")

for c in CELL_TYPES:
    accs = [all_results[s]["per_class_accuracy"][c]*100 for s in SEEDS]
    maes = [all_results[s]["per_class_counting_mae"][c] for s in SEEDS]
    agg["per_class_acc"][c] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
    agg["per_class_mae"][c] = {"mean": float(np.mean(maes)), "std": float(np.std(maes))}
    print(f"  {c:14s}: acc={agg['per_class_acc'][c]['mean']:6.2f}%±{agg['per_class_acc'][c]['std']:.2f}  "
          f"MAE={agg['per_class_mae'][c]['mean']:.3f}±{agg['per_class_mae'][c]['std']:.3f}")

out = f"{WORK}/phase_A3_eval_multiseed.json"
with open(out, "w") as f:
    json.dump({"per_seed": all_results, "aggregate": agg}, f, indent=2)
print(f"\nSaved: {out}")

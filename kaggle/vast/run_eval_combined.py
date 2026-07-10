"""
Combined eval — ONE Fold 3 pass per model seed produces:
  - work/phase_A2_eval_multiseed.json       (seg mIoU: Medical/LLM/Generic)
  - work/phase_A3_eval_multiseed.json       (type acc + counting MAE + confusion)
  - checkpoints_multiseed/phase_C_preds_seed{seed}.pkl  (scores+type_probs, 3 settings)

Shares the in-dist backbone encode across A2 (25 text prompts) and A3/PhaseC
(cell-prompt detection). Phase C mild/severe re-encode the augmented image.

IMPORTANT: this script does NOT run the conformal benchmark. The pkl is
method-agnostic; run the CORRECTED conformal (lambda=3, PB-JCI Online,
temporal_drift) on CPU via the Kaggle conformal-only notebook.

Usage:
  python run_eval_combined.py            # full: 3 seeds x full Fold 3
  python run_eval_combined.py --smoke    # quick: 1 seed x 20 images (validate first)
"""
import os, sys, time, json, pickle, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config_vast import WORK, DATA_ROOT, CHECKPOINT_PATH, CHECKPOINTS_OUT, verify_env

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="1 seed, 20 images, validate pipeline")
args = ap.parse_args()

print("[Vast.ai] COMBINED eval (A2 + A3 + Phase C preds), one pass/seed")
print("=" * 70)
verify_env()

from pannuke_loader import PanNukeFold, CELL_TYPES, DEFAULT_ROOT
from metrics import ClassWiseAccumulator, PerPromptClassAccumulator
from lora_sam3 import inject_lora, load_lora_state, DEFAULT_LORA_TARGETS
from sam3_train import (make_transform, encode_image_frozen, encode_text,
                         forward_decoder_with_grad, inference_to_binary)
from type_head import (TypeHead, roi_pool_feature, compute_iou_matrix,
                       hungarian_match, extract_gt_instances, per_class_counts)
from shift_detector import apply_blur_shift, apply_hsv_jitter
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage

SEEDS = [42, 100, 200] if not args.smoke else [42]
SCORE_THRESH = 0.3
IOU_THRESH   = 0.3

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
INFER_PROMPT   = "cell"

# Phase C settings — MUST match run_phaseC pkl format (corrected conformal derives temporal_drift)
PHASE_C_SETTINGS = {
    "in_dist":      None,
    "mild_shift":   ("hsv", "moderate"),
    "severe_shift": ("blur", "severe"),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

fold3 = PanNukeFold(DATA_ROOT, 3)
N = len(fold3) if not args.smoke else min(20, len(fold3))
print(f"Fold 3: {len(fold3)} patches (using {N})\n")


def build_model(seed):
    lora_ck = f"{CHECKPOINTS_OUT}/sam3_lora_seed{seed}_final.pt"
    th_ck   = f"{CHECKPOINTS_OUT}/type_head_seed{seed}_final.pt"
    assert os.path.exists(lora_ck), f"Missing {lora_ck}"
    assert os.path.exists(th_ck),   f"Missing {th_ck}"
    model = build_sam3_image_model(device=device, eval_mode=True,
                                   checkpoint_path=CHECKPOINT_PATH, load_from_HF=False)
    model.eval()
    inject_lora(model, target_module_names=DEFAULT_LORA_TARGETS,
                r=16, alpha=32, dropout=0.0, path_must_contain="decoder")
    load_lora_state(model, lora_ck)
    for p in model.parameters():
        p.requires_grad = False
    type_head = TypeHead(in_dim=256, hidden_dim=128, num_classes=5, dropout=0.0).to(device)
    type_head.load_state_dict(torch.load(th_ck, map_location=device))
    type_head.eval()
    return model, type_head


def get_feat(backbone_out):
    feat = backbone_out.get("vision_features")
    if feat is None and "backbone_fpn" in backbone_out:
        feat = backbone_out["backbone_fpn"][-1]
    if feat is None:
        for k, v in backbone_out.items():
            if isinstance(v, torch.Tensor) and v.dim() == 4:
                feat = v; break
    if feat.dim() == 4:
        feat = feat[0]
    return feat


all_a2, all_a3 = {}, {}
t_start = time.time()

for seed in SEEDS:
    print("=" * 70)
    print(f"SEED {seed} — combined eval start")
    print("=" * 70)
    model, type_head = build_model(seed)

    transform = make_transform(resolution=1008)
    find_stage = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )

    @torch.no_grad()
    def encode(pil):
        return encode_image_frozen(model, transform, pil, device=device)

    @torch.no_grad()
    def predict_bin(state, prompt):
        """A2 path: binary mask via inference_to_binary."""
        st = dict(state)
        st.update(encode_text(model, prompt, device=device))
        outputs = forward_decoder_with_grad(model, st, find_stage, model._get_dummy_prompt())
        pm = inference_to_binary(outputs, target_hw=(256, 256), score_threshold=SCORE_THRESH)
        return pm.cpu().numpy().astype(bool)

    @torch.no_grad()
    def predict_detect(state, feat):
        """A3 / Phase C path: per-detection scores + type_probs + masks."""
        st = dict(state)
        st.update(encode_text(model, INFER_PROMPT, device=device))
        outputs = forward_decoder_with_grad(model, st, find_stage, model._get_dummy_prompt())
        pred_logits = outputs["pred_logits"].float()
        pred_masks  = outputs["pred_masks"].float()
        pres_logit  = outputs["presence_logit_dec"].float()
        prob = (pred_logits.sigmoid() * pres_logit.sigmoid().unsqueeze(1)).squeeze(-1).squeeze(0)
        masks_up = F.interpolate(pred_masks, size=(256, 256), mode="bilinear",
                                  align_corners=False).sigmoid().squeeze(0)
        masks_bin = (masks_up > 0.5)
        keep = prob > SCORE_THRESH
        if keep.sum() == 0:
            return [], np.zeros(0), np.zeros((0, 5))
        pmk = masks_bin[keep]
        scores = prob[keep].cpu().numpy()
        features = torch.zeros(len(pmk), 256, device=device)
        for i in range(len(pmk)):
            features[i] = roi_pool_feature(feat, pmk[i].float())
        type_probs = type_head(features).softmax(dim=-1).cpu().numpy()
        pmk_np = [pmk[i].cpu().numpy().astype(bool) for i in range(len(pmk))]
        return pmk_np, scores, type_probs

    # ---- accumulators ----
    acc_med = ClassWiseAccumulator(CELL_TYPES)
    acc_llm = PerPromptClassAccumulator(CELL_TYPES, PROMPTS_LLM)
    acc_gen = ClassWiseAccumulator(CELL_TYPES)

    correct = 0; total = 0
    per_cls_corr = {c: 0 for c in CELL_TYPES}
    per_cls_tot  = {c: 0 for c in CELL_TYPES}
    confusion = np.zeros((5, 5), dtype=np.int64)
    counting_err = {c: [] for c in CELL_TYPES}

    preds_by_setting = {s: [] for s in PHASE_C_SETTINGS}
    gt_counts_list = []

    t0 = time.time()
    for i in tqdm(range(N), desc=f"Seed{seed} combined"):
        sample = fold3[i]
        img_np = sample["image"]
        pil_in = Image.fromarray(img_np).convert("RGB")
        gt = {c: (sample["masks"][CELL_TYPES.index(c)] > 0) for c in CELL_TYPES}

        # in-dist backbone (shared by A2 + A3 + PhaseC in_dist)
        state_in = encode(pil_in)
        feat_in = get_feat(state_in)

        # ---------- A2 segmentation (in_dist, 25 prompts) ----------
        pred_gen = predict_bin(state_in, PROMPT_GENERIC)
        for c in CELL_TYPES:
            acc_gen.update(pred_gen, gt[c], c)
        for c in CELL_TYPES:
            acc_med.update(predict_bin(state_in, PROMPTS_MEDICAL[c][0]), gt[c], c)
        for c, prompts in PROMPTS_LLM.items():
            for p in prompts:
                acc_llm.update(predict_bin(state_in, p), gt[c], c, p)

        # ---------- A3 type + counting (in_dist cell detection) ----------
        gt_masks, gt_classes = extract_gt_instances(sample, CELL_TYPES)
        gt_counts = {c: 0 for c in CELL_TYPES}
        for ci in gt_classes:
            gt_counts[CELL_TYPES[ci]] += 1
        gtc_vec = np.zeros(5)
        for ci in gt_classes:
            gtc_vec[ci] += 1
        gt_counts_list.append(gtc_vec)

        pmk, sc, tp = predict_detect(state_in, feat_in)
        if len(pmk) == 0:
            for c in CELL_TYPES:
                counting_err[c].append(gt_counts[c])
        else:
            pred_counts = per_class_counts(np.array(sc), tp)
            for ci in range(5):
                counting_err[CELL_TYPES[ci]].append(abs(pred_counts[ci] - gt_counts[CELL_TYPES[ci]]))
            if len(gt_masks) > 0:
                iou_m = compute_iou_matrix(pmk, gt_masks)
                for pi, gj in hungarian_match(iou_m, iou_thresh=IOU_THRESH):
                    tc = gt_classes[gj]; pc = int(tp[pi].argmax())
                    confusion[tc, pc] += 1
                    per_cls_tot[CELL_TYPES[tc]] += 1
                    if tc == pc:
                        correct += 1; per_cls_corr[CELL_TYPES[tc]] += 1
                    total += 1

        # ---------- Phase C predictions (3 settings) ----------
        # in_dist reuses the cell detection already computed
        preds_by_setting["in_dist"].append({"scores": sc, "probs": tp, "K": 5})
        for setting, aug in PHASE_C_SETTINGS.items():
            if setting == "in_dist":
                continue
            t, sev = aug
            if t == "blur":  aug_np = apply_blur_shift(img_np, sev)
            elif t == "hsv": aug_np = apply_hsv_jitter(img_np, sev)
            else:            aug_np = img_np
            pil_aug = Image.fromarray(aug_np).convert("RGB")
            st_aug = encode(pil_aug)
            ft_aug = get_feat(st_aug)
            _, sc_a, tp_a = predict_detect(st_aug, ft_aug)
            preds_by_setting[setting].append({"scores": sc_a, "probs": tp_a, "K": 5})

    elapsed = time.time() - t0

    # ---- A2 summary ----
    a2 = {"Medical": acc_med.summary(), "LLM": acc_llm.summary(), "Generic": acc_gen.summary()}
    all_a2[seed] = a2
    print(f"\nSeed {seed} done in {elapsed/60:.1f}min")
    for name, r in a2.items():
        print(f"  A2 {name:8s}: mIoU={r['mIoU']*100:.2f}%")

    # ---- A3 summary ----
    macro_acc = correct / max(total, 1)
    all_a3[seed] = {
        "macro_accuracy": macro_acc,
        "per_class_accuracy": {c: per_cls_corr[c] / max(per_cls_tot[c], 1) for c in CELL_TYPES},
        "per_class_counting_mae": {c: float(np.mean(counting_err[c])) if counting_err[c] else 0.0
                                    for c in CELL_TYPES},
        "confusion_matrix": confusion.tolist(),
        "total_matched": total,
    }
    print(f"  A3 macro acc: {macro_acc*100:.2f}%  ({correct}/{total})")

    # ---- Phase C pkl ----
    cache_path = f"{CHECKPOINTS_OUT}/phase_C_preds_seed{seed}.pkl"
    if args.smoke:
        cache_path = cache_path.replace(".pkl", "_SMOKE.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump({"predictions_by_setting": preds_by_setting,
                     "gt_counts": np.array(gt_counts_list)}, f)
    print(f"  Phase C preds -> {cache_path}")

    del model, type_head
    torch.cuda.empty_cache()

# ---- aggregate + save (skip JSON in smoke) ----
if args.smoke:
    print("\n[SMOKE OK] pipeline ran end-to-end. Inspect numbers above, then run full:")
    print("  python run_eval_combined.py")
    sys.exit(0)

# A2 aggregate
agg_a2 = {}
for strat in ["Medical", "LLM", "Generic"]:
    miou = [all_a2[s][strat]["mIoU"] * 100 for s in SEEDS]
    agg_a2[strat] = {"mIoU_mean": float(np.mean(miou)), "mIoU_std": float(np.std(miou))}
with open(f"{WORK}/phase_A2_eval_multiseed.json", "w") as f:
    json.dump({"per_seed": all_a2, "aggregate": agg_a2}, f, indent=2)

# A3 aggregate
agg_a3 = {"per_class_acc": {}, "per_class_mae": {}}
macros = [all_a3[s]["macro_accuracy"] * 100 for s in SEEDS]
agg_a3["macro_acc_mean"] = float(np.mean(macros))
agg_a3["macro_acc_std"]  = float(np.std(macros))
for c in CELL_TYPES:
    accs = [all_a3[s]["per_class_accuracy"][c] * 100 for s in SEEDS]
    maes = [all_a3[s]["per_class_counting_mae"][c] for s in SEEDS]
    agg_a3["per_class_acc"][c] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
    agg_a3["per_class_mae"][c] = {"mean": float(np.mean(maes)), "std": float(np.std(maes))}
with open(f"{WORK}/phase_A3_eval_multiseed.json", "w") as f:
    json.dump({"per_seed": all_a3, "aggregate": agg_a3}, f, indent=2)

print(f"\n{'='*70}")
print(f"DONE in {(time.time()-t_start)/3600:.2f}h")
print(f"  A2: {WORK}/phase_A2_eval_multiseed.json")
print(f"  A3: {WORK}/phase_A3_eval_multiseed.json")
print(f"  Phase C preds: {CHECKPOINTS_OUT}/phase_C_preds_seed{{42,100,200}}.pkl")
print("Next: backup, then run CORRECTED conformal-only on CPU (Kaggle) per seed.")
print(f"{'='*70}")

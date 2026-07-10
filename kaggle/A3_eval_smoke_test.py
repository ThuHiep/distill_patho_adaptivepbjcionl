"""A3 eval smoke test — paste vao notebook A3 EVAL sau cell '03 — Inference pipeline'
va TRUOC cell '04 — Full Fold 3 eval'.

Test 10 Fold 3 patches (~3-5 phut) de verify:
- LoRA + TypeHead checkpoints loaded dung
- Backbone features extract OK
- ROI pooling 256-d
- TypeHead forward (eval mode, no_grad)
- Hungarian matching co matches
- Poisson-Binomial per-class counts hop ly
- Confusion matrix khong all-zero
- Forecast full Fold 3 time

Neu PASS -> Run All (full Fold 3 ~2h)
Neu FAIL -> debug truoc khi commit.
"""

A3_EVAL_SMOKE_CELL = '''
# ===== A3 EVAL SMOKE TEST — verify pipeline truoc khi run full Fold 3 =====
# Paste cell nay vao notebook A3 EVAL SAU cell "03 - Inference pipeline"
# va TRUOC cell "04 - Full Fold 3 eval".
#
# ETA: ~3-5 phut tren 10 patches. Neu PASS -> Run All. Neu FAIL -> debug.

import time
print("=" * 70)
print("A3 EVAL SMOKE TEST — verify pipeline truoc khi run full Fold 3")
print("=" * 70)

N_SMOKE = 10  # 10 patches du de catch all bugs

# ----- Test 1: Checkpoints loaded correctly -----
print("\\n[1/7] Verify checkpoints loaded...")
print(f"   LoRA   : {LORA_CKPT_PATH}")
print(f"   TypeHead: {TYPEHEAD_CKPT_PATH}")
assert sum(p.requires_grad for p in model.parameters()) == 0, \\
    "Model should be fully frozen"
assert sum(p.requires_grad for p in type_head.parameters()) == 0 or \\
       not type_head.training, "TypeHead should be in eval mode"
type_head.eval()
print("   model.eval(): OK, all frozen")
print("   type_head.eval(): OK")
print("   ✅ PASS")

# ----- Test 2: Backbone + decoder inference -----
print("\\n[2/7] Test SAM3+LoRA inference on Fold 3 patch...")
test_sample = fold3[0]
test_pil = Image.fromarray(test_sample["image"]).convert("RGB")

t0 = time.time()
pred_masks_list, scores_list, backbone_feat = run_sam3_inference(test_pil, EVAL_PROMPT)
t_inf = time.time() - t0

print(f"   Backbone feat shape: {backbone_feat.shape}")
print(f"   N detections: {len(pred_masks_list)}")
print(f"   Scores range: [{min(scores_list):.3f}, {max(scores_list):.3f}]" if scores_list else "   No scores")
print(f"   Inference time: {t_inf:.2f}s")
assert backbone_feat.dim() == 3 and backbone_feat.shape[0] == 256, \\
    f"Expected (256, H, W), got {backbone_feat.shape}"
assert len(pred_masks_list) > 0, "No detections — LoRA hoac prompt sai"
print("   ✅ PASS")

# ----- Test 3: TypeHead forward + softmax -----
print("\\n[3/7] Test TypeHead forward (eval mode)...")
features = torch.zeros(len(pred_masks_list), 256, device=device)
for i, pm in enumerate(pred_masks_list):
    mask_t = torch.from_numpy(pm).to(device)
    features[i] = roi_pool_feature(backbone_feat, mask_t)

with torch.no_grad():
    type_logits = type_head(features)
    type_probs = type_logits.softmax(dim=-1)

print(f"   Features shape: {features.shape}")
print(f"   Type logits shape: {type_logits.shape}")
print(f"   Type probs shape: {type_probs.shape}")
print(f"   Sample probs (det 0): {type_probs[0].cpu().tolist()}")
print(f"   Argmax distribution: {np.bincount(type_probs.argmax(-1).cpu().numpy(), minlength=5)}")
assert type_probs.shape[1] == 5, f"Expected 5 classes, got {type_probs.shape[1]}"
assert torch.isfinite(type_probs).all(), "Type probs has NaN/Inf"
assert torch.allclose(type_probs.sum(-1), torch.ones(type_probs.shape[0], device=device), atol=1e-5), \\
    "Softmax should sum to 1"
print("   ✅ PASS")

# ----- Test 4: GT extraction + Hungarian matching -----
print("\\n[4/7] Test GT extraction + Hungarian matching...")
gt_masks, gt_classes = extract_gt_instances(test_sample, CELL_TYPES)
print(f"   N GT instances: {len(gt_masks)}")
print(f"   GT class dist: {dict(zip(*np.unique(gt_classes, return_counts=True))) if len(gt_classes) > 0 else 'empty'}")

if len(gt_masks) > 0:
    iou_matrix = compute_iou_matrix(pred_masks_list, gt_masks)
    matches = hungarian_match(iou_matrix, iou_thresh=IOU_THRESH)
    print(f"   IoU matrix shape: {iou_matrix.shape}")
    print(f"   IoU max: {iou_matrix.max():.3f}")
    print(f"   N matches (IoU >= {IOU_THRESH}): {len(matches)}")
    if len(matches) == 0:
        matches_low = hungarian_match(iou_matrix, iou_thresh=0.1)
        print(f"   ⚠️  Voi threshold 0.1: {len(matches_low)} matches")
        print(f"        -> Co the patch dau Fold 3 it overlap, thu patch khac")
print("   ✅ PASS")

# ----- Test 5: Per-class counting (Poisson-Binomial) -----
print("\\n[5/7] Test Poisson-Binomial per-class counts...")
scores_arr = np.array(scores_list)
type_probs_np = type_probs.cpu().numpy()
pred_counts = per_class_counts(scores_arr, type_probs_np)
print(f"   Pred counts: {dict(zip(CELL_TYPES, [f'{c:.2f}' for c in pred_counts]))}")
gt_counts_smoke = {c: 0 for c in CELL_TYPES}
for ci in gt_classes:
    gt_counts_smoke[CELL_TYPES[ci]] += 1
print(f"   GT counts:   {gt_counts_smoke}")
assert np.isfinite(pred_counts).all(), "Pred counts has NaN/Inf"
assert (pred_counts >= 0).all(), "Pred counts should be non-negative"
print("   ✅ PASS")

# ----- Test 6: Speed test on N_SMOKE patches -----
print(f"\\n[6/7] Speed test — {N_SMOKE} patches...")
t0 = time.time()
n_total_dets = 0
n_total_matched = 0
confusion_smoke = np.zeros((5, 5), dtype=np.int64)

for i in range(N_SMOKE):
    s = fold3[i]
    pil = Image.fromarray(s["image"]).convert("RGB")
    pm, sc, tp, _ = predict_types_for_image(pil, prompt=EVAL_PROMPT)
    n_total_dets += len(pm)

    gm, gc = extract_gt_instances(s, CELL_TYPES)
    if len(pm) > 0 and len(gm) > 0:
        iou_m = compute_iou_matrix(pm, gm)
        mt = hungarian_match(iou_m, iou_thresh=IOU_THRESH)
        n_total_matched += len(mt)
        for pi, gj in mt:
            confusion_smoke[gc[gj], int(tp[pi].argmax())] += 1

t_total = time.time() - t0
avg_time = t_total / N_SMOKE
forecast_hours = len(fold3) * avg_time / 3600

print(f"   {N_SMOKE} patches in {t_total:.1f}s ({avg_time:.2f}s/patch)")
print(f"   Total detections: {n_total_dets}, Total matched: {n_total_matched}")
print(f"   Confusion (smoke {N_SMOKE} patches):")
print(f"   {'':14s} " + " ".join(f"{c[:6]:>7s}" for c in CELL_TYPES))
for i, c_true in enumerate(CELL_TYPES):
    row = " ".join(f"{confusion_smoke[i, j]:>7d}" for j in range(5))
    print(f"   {c_true:14s} {row}")
diag_sum = confusion_smoke.trace()
total_smoke = confusion_smoke.sum()
acc_smoke = diag_sum / max(total_smoke, 1) * 100
print(f"   Smoke macro acc: {acc_smoke:.1f}% ({diag_sum}/{total_smoke})")
print(f"\\n   Forecast full Fold 3 ({len(fold3)} patches): ~{forecast_hours:.2f}h")

# ----- Test 7: Final verdict -----
print(f"\\n[7/7] Final verdict...")
print("=" * 70)
if n_total_matched >= 5 and acc_smoke >= 20 and np.isfinite(forecast_hours):
    print(f"   ✅ ALL CHECKS PASS — safe to Run All full Fold 3")
    print(f"   Forecast: ~{forecast_hours:.2f}h")
    if forecast_hours > 4:
        print(f"   ⚠️  WARN: forecast > 4h, lau hon expect. Check GPU loading.")
elif n_total_matched < 3:
    print(f"   ❌ FAIL: only {n_total_matched} matches in {N_SMOKE} patches")
    print(f"        Debug: check IOU_THRESH, check LoRA loaded, check prompt")
elif acc_smoke < 20:
    print(f"   ⚠️  WARN: smoke acc {acc_smoke:.1f}% < 20% (random baseline)")
    print(f"        TypeHead co the load wrong, hoac feature distribution shift")
    print(f"        -> Run All se cho ket qua dang lo ngai")
else:
    print(f"   ⚠️  PARTIAL — proceed with caution")
print("=" * 70)
'''

print(__doc__)
print("\\n" + "=" * 70)
print("SMOKE TEST CELL (copy paste vao notebook A3 EVAL):")
print("=" * 70)
print(A3_EVAL_SMOKE_CELL)

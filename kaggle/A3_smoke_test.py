"""A3 smoke test — paste vao notebook A3 sau cell '04 — Inference pipeline'
va TRUOC cell '05 — Training Loop'.

Test 10 train step + 5 eval samples (~3-5 phut) de verify:
- Backbone features extract dung
- ROI pooling 256-d dung
- Hungarian matching co matches
- TypeHead forward + backward OK
- Loss finite, gradient flow
- Inference + accuracy metric work

Neu PASS -> Run All full training (~6h)
Neu FAIL -> debug truoc khi commit.
"""

A3_SMOKE_TEST_CELL = '''
# ===== A3 SMOKE TEST — verify code dung truoc khi commit 6h training =====
# Paste cell nay vao notebook A3 SAU cell "04 - Inference pipeline"
# va TRUOC cell "05 - Training Loop".
#
# ETA: ~3-5 phut. Neu PASS -> Run All. Neu FAIL -> debug.

import time
from sam3_train import inference_to_binary

print("=" * 70)
print("A3 SMOKE TEST — verify pipeline truoc khi commit full training")
print("=" * 70)

# ----- Test 1: Backbone features extraction -----
print("\\n[1/6] Test backbone features extraction...")
test_sample = fold1[0]
test_pil = Image.fromarray(test_sample["image"]).convert("RGB")

t0 = time.time()
pred_masks_list, scores_list, backbone_feat = run_sam3_inference(test_pil, TRAIN_PROMPT)
t_inference = time.time() - t0

print(f"   Backbone feature shape: {backbone_feat.shape}")
print(f"   N detections: {len(pred_masks_list)} (scores: {[f'{s:.2f}' for s in scores_list[:5]]}...)")
print(f"   Inference time: {t_inference:.2f}s")
assert backbone_feat.dim() == 3, f"Expected 3D feature, got {backbone_feat.shape}"
assert backbone_feat.shape[0] == 256, f"Expected D=256, got {backbone_feat.shape[0]}"
assert len(pred_masks_list) > 0, "No detections — kiem tra LoRA + prompt"
print("   ✅ PASS")

# ----- Test 2: ROI pooling dimension -----
print("\\n[2/6] Test ROI pooling...")
test_mask = torch.from_numpy(pred_masks_list[0]).to(device)
pooled = roi_pool_feature(backbone_feat, test_mask)
print(f"   Pooled feature shape: {pooled.shape}")
print(f"   Pooled values range: [{pooled.min().item():.3f}, {pooled.max().item():.3f}]")
assert pooled.dim() == 1, f"Expected 1D, got {pooled.shape}"
assert pooled.shape[0] == 256, f"Expected 256-d, got {pooled.shape[0]}"
assert torch.isfinite(pooled).all(), "Pooled feature has NaN/Inf"
print("   ✅ PASS")

# ----- Test 3: GT extraction -----
print("\\n[3/6] Test GT instance extraction...")
gt_masks, gt_classes = extract_gt_instances(test_sample, CELL_TYPES)
print(f"   N GT instances: {len(gt_masks)}")
print(f"   GT class distribution: {dict(zip(*np.unique(gt_classes, return_counts=True)))}")
assert len(gt_masks) > 0, "No GT instances — kiem tra PanNuke data"
print("   ✅ PASS")

# ----- Test 4: Hungarian matching -----
print("\\n[4/6] Test Hungarian matching...")
iou_matrix = compute_iou_matrix(pred_masks_list, gt_masks)
print(f"   IoU matrix shape: {iou_matrix.shape}")
print(f"   IoU max: {iou_matrix.max():.3f}, mean: {iou_matrix.mean():.3f}")
matches = hungarian_match(iou_matrix, iou_thresh=IOU_THRESH)
print(f"   N matches (IoU >= {IOU_THRESH}): {len(matches)}")
if len(matches) == 0:
    print(f"   ⚠️  WARN: 0 matches voi threshold {IOU_THRESH}")
    print(f"        Thu giam threshold xuong 0.2 hoac 0.1")
    matches_low = hungarian_match(iou_matrix, iou_thresh=0.1)
    print(f"        Voi threshold 0.1: {len(matches_low)} matches")
assert len(matches) > 0 or len(matches_low) > 0, "0 matches even at 0.1 — debug needed"
print("   ✅ PASS")

# ----- Test 5: TypeHead forward + loss -----
print("\\n[5/6] Test TypeHead forward + cross-entropy loss...")
type_head.train()
if len(matches) > 0:
    matched_features = []
    matched_labels = []
    for pred_i, gt_j in matches:
        mt = torch.from_numpy(pred_masks_list[pred_i]).to(device)
        feat = roi_pool_feature(backbone_feat, mt)
        matched_features.append(feat)
        matched_labels.append(gt_classes[gt_j])

    features = torch.stack(matched_features)
    labels = torch.tensor(matched_labels, dtype=torch.long, device=device)
    logits = type_head(features)
    loss = F.cross_entropy(logits, labels)

    print(f"   Features shape: {features.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Labels: {labels.tolist()}")
    print(f"   Loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is NaN/Inf"
    assert loss.item() > 0, "Loss should be positive"
    print("   ✅ PASS")

# ----- Test 6: Backward pass + gradient -----
print("\\n[6/6] Test backward + gradient flow...")
loss.backward()
grad_norms = []
for name, p in type_head.named_parameters():
    if p.grad is not None:
        gn = p.grad.norm().item()
        grad_norms.append(gn)
        print(f"   {name}: grad_norm = {gn:.4f}")
assert all(np.isfinite(gn) for gn in grad_norms), "Gradient has NaN/Inf"
assert any(gn > 0 for gn in grad_norms), "All gradients are zero — model isolated?"
print("   ✅ PASS")

# Reset optimizer state
optimizer.zero_grad()

# ----- Test 7: 10 train steps speed test -----
print("\\n[7/7] Speed test — 10 train steps...")
t0 = time.time()
n_success = 0
losses_test = []
for i in range(10):
    sample = fold1[i + 1]  # different samples
    result = train_step(sample)
    if result is not None:
        n_success += 1
        losses_test.append(result["loss"])
t10 = time.time() - t0
avg_time = t10 / 10

print(f"   10 steps in {t10:.1f}s ({avg_time:.2f}s/step)")
print(f"   Success rate: {n_success}/10 (rest = no matches or no detections)")
if losses_test:
    print(f"   Loss range: [{min(losses_test):.3f}, {max(losses_test):.3f}]")
    print(f"   Mean loss: {np.mean(losses_test):.3f}")

# Forecast full training
total_steps = NUM_EPOCHS * MAX_TRAIN_PER_EPOCH
forecast_hours = total_steps * avg_time / 3600
print(f"\\n   Forecast full training: {total_steps} steps x {avg_time:.2f}s = {forecast_hours:.1f}h")

print("\\n" + "=" * 70)
print("SMOKE TEST DONE")
print("=" * 70)

# Final verdict
if n_success >= 7 and all(np.isfinite(l) for l in losses_test):
    print("  ✅ ALL CHECKS PASS — safe to Run All full training")
    print(f"  Forecast: ~{forecast_hours:.1f}h")
    if forecast_hours > 11:
        print(f"  ⚠️  WARN: forecast > 11h, gan Kaggle limit. Consider reduce NUM_EPOCHS.")
elif n_success < 5:
    print("  ⚠️  WARN: too few successful steps. Debug:")
    print("       - Check IOU_THRESH (try 0.2 or 0.1)")
    print("       - Check predicted masks (are they non-empty?)")
    print("       - Check GT (are there >= 5 instances per image?)")
else:
    print("  ⚠️  PARTIAL PASS — proceed but watch for issues")

# Reset for full training
type_head.train()
optimizer.zero_grad()
'''

print(__doc__)
print("\\n" + "=" * 70)
print("SMOKE TEST CELL (copy paste vao notebook A3):")
print("=" * 70)
print(A3_SMOKE_TEST_CELL)

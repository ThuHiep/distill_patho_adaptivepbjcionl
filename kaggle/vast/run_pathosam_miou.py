"""
Measure PathoSAM SEGMENTATION quality on PanNuke Fold-3 — TWO metrics:

  (1) PER-CLASS semantic mIoU  <-- the FAIR comparison vs SAM3's ~15% (same metric).
      PathoSAM is class-agnostic, so we assign each instance a type via OUR TypeHead
      (argmax p_ik), paint a 5-class semantic map, and compute dataset-level per-class
      IoU then macro-average — exactly the metric SAM3 was scored on.
  (2) BINARY foreground IoU/Dice  <-- raw nucleus-vs-background separation. Almost
      always much higher than per-class (no type penalty, no near-empty Dead class).
      DO NOT compare this number to SAM3's per-class 15% — different, easier metric.

Honest expectation: per-class mIoU may be only comparable to SAM3 (TypeHead ~72% acc,
Dead class ~0%). If so, PathoSAM is a *different* backbone, not a *stronger* one — the
value is predictor-agnostic conformal, not fixing mIoU. Measure to find out.

Clean Fold-3 (exclude colon) by default.
Run:
  micromamba run -p /workspace/penv python run_pathosam_miou.py            # clean 2228
  micromamba run -p /workspace/penv python run_pathosam_miou.py --all      # full 2722
"""
from __future__ import annotations
import os, sys, time, argparse, json
import numpy as np
import torch

REPO = "/workspace/sam3_research"
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from pannuke_loader import PanNukeFold, CELL_TYPES                    # noqa: E402
from type_head import TypeHead                                        # noqa: E402
from pathosam_lib import load_pathosam, pathosam_instances, pool_features  # noqa: E402

DATA_ROOT = f"{REPO}/data/pannuke"
TH_PATH = f"{REPO}/checkpoints/type_head_pathosam.pt"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="use full Fold 3 (incl colon)")
    ap.add_argument("--n", type=int, default=0, help="cap #images (0=all)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    fold3 = PanNukeFold(DATA_ROOT, 3)
    idx = list(range(len(fold3)))
    if not args.all:
        idx = [i for i in idx if "colon" not in str(fold3[i]["tissue"]).lower()]
    if args.n:
        idx = idx[:args.n]
    print(f"Fold 3: {len(fold3)} | evaluating {len(idx)} "
          f"({'full' if args.all else 'clean no-colon'})")

    predictor, segmenter = load_pathosam(device)
    head = TypeHead(in_dim=256, hidden_dim=128, num_classes=5).to(device)
    head.load_state_dict(torch.load(TH_PATH, map_location=device))
    head.eval()

    # dataset-level accumulators
    pc_inter = np.zeros(5); pc_union = np.zeros(5)        # per-class semantic
    bin_ious, bin_dices = [], []                          # binary foreground
    t0 = time.time()
    for k, i in enumerate(idx):
        s = fold3[i]
        masks, scores, feat = pathosam_instances(s["image"], predictor, segmenter)
        H, W = s["image"].shape[:2]

        # ---- per-class semantic map via TypeHead ----
        sem_pred = np.zeros((5, H, W), dtype=bool)
        if len(masks):
            pooled = pool_features(feat, masks, device)
            types = head(pooled).argmax(1).cpu().numpy()
            for j, m in enumerate(masks):
                sem_pred[types[j]] |= m
        gt = np.asarray(s["masks"])                       # (5,H,W) instance ids
        sem_gt = gt > 0
        for c in range(5):
            pc_inter[c] += np.logical_and(sem_pred[c], sem_gt[c]).sum()
            pc_union[c] += np.logical_or(sem_pred[c], sem_gt[c]).sum()

        # ---- binary foreground ----
        pred_fg = sem_pred.any(0); gt_fg = sem_gt.any(0)
        inter = np.logical_and(pred_fg, gt_fg).sum()
        union = np.logical_or(pred_fg, gt_fg).sum()
        denom = pred_fg.sum() + gt_fg.sum()
        bin_ious.append(inter / union if union > 0 else 1.0)
        bin_dices.append(2 * inter / denom if denom > 0 else 1.0)
        if (k + 1) % 200 == 0:
            cur = (pc_inter / np.maximum(pc_union, 1)).mean()
            print(f"  {k+1}/{len(idx)} | per-class mIoU={cur*100:.1f}% "
                  f"binFG IoU={np.mean(bin_ious)*100:.1f}% | {(time.time()-t0)/(k+1):.2f}s/img")

    iou_pc = pc_inter / np.maximum(pc_union, 1)
    macro = float(iou_pc.mean())
    bin_iou, bin_dice = float(np.mean(bin_ious)), float(np.mean(bin_dices))

    print("\n" + "=" * 66)
    print(f"PathoSAM segmentation on Fold-3 ({len(idx)} imgs)")
    print("=" * 66)
    print("PER-CLASS semantic mIoU (FAIR vs SAM3 ~15%):")
    for c in range(5):
        print(f"  {CELL_TYPES[c]:13s}: {iou_pc[c]*100:6.2f}%")
    print(f"  {'MACRO':13s}: {macro*100:6.2f}%   <-- compare to SAM3 ~15% per-class")
    print("-" * 66)
    print(f"BINARY foreground (different/easier metric, do NOT vs SAM3 per-class):")
    print(f"  fg IoU = {bin_iou*100:.2f}% | fg Dice = {bin_dice*100:.2f}%")
    print("=" * 66)

    with open(f"{REPO}/work/pathosam_miou.json", "w") as f:
        json.dump({"n": len(idx), "subset": "full" if args.all else "clean_no_colon",
                   "per_class_miou": {CELL_TYPES[c]: float(iou_pc[c]) for c in range(5)},
                   "macro_per_class_miou": macro,
                   "binary_fg_iou": bin_iou, "binary_fg_dice": bin_dice}, f, indent=2)
    print(f"Saved {REPO}/work/pathosam_miou.json")


if __name__ == "__main__":
    main()

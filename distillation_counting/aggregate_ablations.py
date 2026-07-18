#!/usr/bin/env python3
"""Gom ablations §4.8 leak-free (NuInsSeg 5-seed cross-fit) -> worst-org/Winkler/MAE (cluster n=5).

So MAIN (ch32+detach+teacher, pkl có sẵn) với 4 ablation:
  detachoff (bỏ --detach_mu) | ch16 | ch64 | GT-supervised (--use_gt_density).
Mỗi pkl = distill_student_r2.py --dataset nuinsseg --kfold 5 (leak-free), eval hàng R2-cluster.
"""
import numpy as np
import eval_r2_grouped as e

REPO, SEEDS = "/workspace/sam3_research", [42, 43, 44, 45, 46]
# nhãn -> tag trong tên file student_r2_nuinsseg_cv5_{tag}_s{S}.pkl
TAGS = {
    "MAIN ch32+detach+teacher": "poisson",
    "ablation: detach OFF":     "detachoff",
    "ablation: ch16 (~0.5M)":   "ch16",
    "ablation: ch64 (~7.7M)":   "ch64",
    "ablation: GT-supervised":  "supervised",
}
for label, tag in TAGS.items():
    wo, wk, mae = [], [], []
    for s in SEEDS:
        try:
            r = e.run(f"{REPO}/work/student_r2_nuinsseg_cv5_{tag}_s{s}.pkl",
                      None, 0.1, 20, 0.5, 10, 15, 5)["rows"]["R2-cluster"]
        except Exception as ex:
            print(f"  !! {label} s{s}: {ex}"); continue
        c = r["conditional"]["worst_organ_coverage"]
        if c is not None:
            wo.append(c)
        wk.append(r["winkler"]["mean"]); mae.append(r["mae"]["mean"])
    wo, wk, mae = np.array(wo), np.array(wk), np.array(mae)
    if len(wo):
        print(f"{label:26}: worst-org {wo.mean():.3f}±{wo.std():.3f} | "
              f"Winkler {wk.mean():.1f} | MAE {mae.mean():.2f}")

#!/usr/bin/env python3
"""Gom KD 5-seed NuInsSeg -> worst-org/Winkler/MAE (global + cluster) mean±sd.

KD student pkl = PB-σ (σ=√Σsᵢ(1−sᵢ) tự suy trong load_mu_sigma). Chạy eval_r2_grouped.run()
trên từng pkl, lấy hàng R2-global/R2-cluster (thực chất là KD-PB dưới scheme đó).
Dùng để dựng bảng N4 SAME-SCHEME (§4.1) — KHÔNG so cross-scheme.
"""
import numpy as np
import eval_r2_grouped as e

REPO, SEEDS = "/workspace/sam3_research", [42, 43, 44, 45, 46]
for scheme, label in [("global", "R2-global"), ("cluster", "R2-cluster")]:
    wo, wk, mae = [], [], []
    for s in SEEDS:
        r = e.run(f"{REPO}/work/student_kd_nuinsseg_cv5_s{s}.pkl",
                  None, 0.1, 20, 0.5, 10, 15, 5)["rows"][label]
        c = r["conditional"]["worst_organ_coverage"]
        if c is not None:
            wo.append(c)
        wk.append(r["winkler"]["mean"]); mae.append(r["mae"]["mean"])
    wo, wk, mae = np.array(wo), np.array(wk), np.array(mae)
    print(f"KD {scheme:8}: worst-org {wo.mean():.3f}±{wo.std():.3f} | "
          f"Winkler {wk.mean():.2f}±{wk.std():.2f} | MAE {mae.mean():.2f}  "
          f"seeds_wo={[round(float(x),3) for x in wo]}")

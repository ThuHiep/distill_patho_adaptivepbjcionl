#!/usr/bin/env python3
"""Label-efficiency frontier (piece 1: COUNT-ONLY student) — đóng lỗ "tại sao distill".

Đo worst-org(cluster)+MAE vs ngân sách nhãn: train count-only student (teacher density + GT count)
trên {10,25,50,100}% ảnh train, test trên split cố định. KHÔNG PathoSAM (dùng teacher_density cache).
Piece 2 (mask-supervised, cần gt_density) chạy sau để so cùng ngân sách.

Chạy Kaggle GPU: clone repo + attach sam3-paper2-uqkd (teacher_density_nuinsseg.pkl).
"""
import os, sys, glob, pickle, tempfile
import numpy as np
import torch

REPO = os.environ.get("REPO", "/kaggle/working/repo")
sys.path.insert(0, os.path.join(REPO, "distillation_counting"))
from distill_student_r2 import train, predict_r2   # noqa: E402
import eval_r2_grouped as E                          # noqa: E402

cache = glob.glob("/kaggle/input/**/teacher_density_nuinsseg.pkl", recursive=True)
cache = cache[0] if cache else f"{REPO}/work/teacher_density_nuinsseg.pkl"
data = pickle.load(open(cache, "rb"))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[cache] {cache} | N={len(data)}")

BUDGETS = [0.10, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2]


def worst_mae(model, test_data):
    out = predict_r2(model, test_data, device)
    tmp = tempfile.mktemp(suffix=".pkl")
    pickle.dump(out, open(tmp, "wb"))
    r = E.run(tmp, None, 0.1, 20, 0.5, 10, 15, 5)["rows"]["R2-cluster"]
    return r["conditional"]["worst_organ_coverage"], r["mae"]["mean"]


rows = []   # (frac, nlab, wo_mean, wo_std, mae_mean, mae_std)
for frac in BUDGETS:
    wos, maes, nlab = [], [], 0
    for s in SEEDS:
        rng = np.random.RandomState(s)
        perm = rng.permutation(len(data))
        n_test = len(data) // 5
        test_idx = perm[:n_test].tolist()
        pool = perm[n_test:]
        n_use = max(8, int(round(frac * len(pool))))
        nlab = n_use
        model = train(data, device, 60, 32, 1e-3, pool[:n_use].tolist(),
                      1.0, 0.01, 0.01, 0.5, 16, detach_mu=True, sigma_mode="poisson")
        wo, mae = worst_mae(model, [data[i] for i in test_idx])
        if wo is not None:
            wos.append(wo)
        maes.append(mae)
    wos, maes = np.array(wos), np.array(maes)
    rows.append((frac, nlab, wos.mean(), wos.std(), maes.mean(), maes.std()))
    print(f"[done] frac={frac:.2f} n={nlab}")   # progress only

# ---- FINAL TABLE (in 1 lần ở cuối, tail an toàn) + ghi file ----
lines = [f"{'frac':>6} {'n_lab':>6} | {'worst-org (up)':>16} | {'MAE (down)':>12}",
         "-" * 46]
for frac, nlab, wm, ws, mm, ms in rows:
    lines.append(f"{frac:6.2f} {nlab:6d} | {wm:.3f}+-{ws:.3f}      | {mm:5.2f}+-{ms:.2f}")
table = "\n".join(lines)
out = os.path.join(REPO, "distillation_counting", "label_efficiency_p1_result.txt")
open(out, "w").write(table + "\n")
print("\n===== LABEL-EFFICIENCY PIECE 1 (count-only) =====")
print(table)
print(f"\n[saved] {out}")

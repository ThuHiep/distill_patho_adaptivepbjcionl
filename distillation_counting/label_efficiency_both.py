#!/usr/bin/env python3
"""Label-efficiency HEAD-TO-HEAD (piece 2): DISTILLED (teacher-density) vs SUPERVISED (GT-density).

Đóng lỗ reviewer "tại sao distill?": tại MỖI ngân sách ẢNH giống hệt (cùng split/seed/subset),
train 2 student:
  - DISTILLED : target = PathoSAM teacher density  (nhãn/ảnh = 1 SỐ ĐẾM, rẻ; teacher free/unsup)
  - SUPERVISED: target = GT instance density        (nhãn/ảnh = FULL MASK, đắt ~K instance × 2.4s)
So worst-org(cluster)+MAE. Nếu distilled ≈ supervised ở CÙNG số ảnh -> distilled thắng về CHI PHÍ nhãn
(count vs mask); nếu supervised nhỉnh -> đổi trục x sang annotation-cost (mask ~K×) distilled vẫn trội.

Chạy Kaggle GPU: clone repo + attach BOTH:
  - hipinhththu/sam3-paper2-uqkd   (teacher_density_nuinsseg.pkl)
  - ipateam/nuinsseg               (raw ảnh+mask -> dựng GT density, KHÔNG cần PathoSAM)
GT cache lưu {REPO}/work/gt_density_nuinsseg.pkl -> NHỚ backup kaggle sau khi chạy.
"""
import os, sys, glob, pickle, tempfile
import numpy as np
import torch

REPO = os.environ.get("REPO", "/kaggle/working/repo")
sys.path.insert(0, os.path.join(REPO, "distillation_counting"))
from distill_student_r2 import train, predict_r2, build_teacher_density   # noqa: E402
from distill_student_nuinsseg import build_index, find_root               # noqa: E402
import eval_r2_grouped as E                                               # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- teacher-density cache (distilled target) ----
tc = glob.glob("/kaggle/input/**/teacher_density_nuinsseg.pkl", recursive=True)
tc = tc[0] if tc else f"{REPO}/work/teacher_density_nuinsseg.pkl"
data_T = pickle.load(open(tc, "rb"))
print(f"[teacher] {tc} | N={len(data_T)}")

# ---- GT-density cache (supervised target); dựng nếu chưa có (cần ipateam/nuinsseg) ----
gt_cache = f"{REPO}/work/gt_density_nuinsseg.pkl"
os.makedirs(os.path.dirname(gt_cache), exist_ok=True)
if not os.path.exists(gt_cache):
    print("[gt] building GT-density từ raw NuInsSeg (use_gt=True, KHÔNG PathoSAM)...")
    samples = build_index(find_root())
    print(f"[gt] indexed {len(samples)} pairs")
data_G = build_teacher_density(None if os.path.exists(gt_cache) else build_index(find_root()),
                               device, gt_cache, use_gt=True)
print(f"[gt] {gt_cache} | N={len(data_G)}")

# ---- ALIGN by IMG CONTENT: teacher cache & GT cache khác thứ tự (build_index khác lần dựng).
#      Cả 2 resize cùng code -> img byte-identical cho cùng ảnh gốc -> map GT->teacher qua hash ảnh. ----
assert len(data_T) == len(data_G), f"len mismatch {len(data_T)} vs {len(data_G)}"
from collections import defaultdict
buckets = defaultdict(list)
for j, d in enumerate(data_G):
    buckets[d["img"].tobytes()].append(j)
order, used = [], set()
for i, dT in enumerate(data_T):
    cands = buckets.get(dT["img"].tobytes(), [])
    pick = next((j for j in cands if j not in used
                 and data_G[j]["gt"] == dT["gt"] and data_G[j]["organ"] == dT["organ"]), None)
    if pick is None:  # fallback: cùng ảnh, chưa dùng
        pick = next((j for j in cands if j not in used), None)
    assert pick is not None, f"no img-match for teacher idx {i} ({dT['organ']},{dT['gt']})"
    used.add(pick); order.append(pick)
data_G = [data_G[j] for j in order]
for i in range(len(data_T)):   # verify sau khi reorder
    assert data_T[i]["gt"] == data_G[i]["gt"] and data_T[i]["organ"] == data_G[i]["organ"], \
        f"MISALIGN @ {i} sau reorder"
print(f"[align] OK — {len(data_T)} ảnh khớp img+gt+organ (map qua hash ảnh)")

BUDGETS = [0.10, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2]


def worst_mae(model, test_data):
    out = predict_r2(model, test_data, device)
    tmp = tempfile.mktemp(suffix=".pkl")
    pickle.dump(out, open(tmp, "wb"))
    r = E.run(tmp, None, 0.1, 20, 0.5, 10, 15, 5)["rows"]["R2-cluster"]
    return r["conditional"]["worst_organ_coverage"], r["mae"]["mean"]


def fit_eval(data_train, train_idx, test_data):
    m = train(data_train, device, 60, 32, 1e-3, train_idx,
              1.0, 0.01, 0.01, 0.5, 16, detach_mu=True, sigma_mode="poisson")
    return worst_mae(m, test_data)


rows = []   # (frac, nlab, D_wm, D_ws, D_mm, D_ms, S_wm, S_ws, S_mm, S_ms)
for frac in BUDGETS:
    Dw, Dm, Sw, Sm, nlab = [], [], [], [], 0
    for s in SEEDS:
        rng = np.random.RandomState(s)
        perm = rng.permutation(len(data_T))
        n_test = len(data_T) // 5
        test_idx = perm[:n_test].tolist()
        pool = perm[n_test:]
        n_use = max(8, int(round(frac * len(pool))))
        nlab = n_use
        tr = pool[:n_use].tolist()
        test_T = [data_T[i] for i in test_idx]        # eval imgs (giống nhau cả 2)
        dwo, dmae = fit_eval(data_T, tr, test_T)      # DISTILLED
        swo, smae = fit_eval(data_G, tr, test_T)      # SUPERVISED
        if dwo is not None:
            Dw.append(dwo)
        if swo is not None:
            Sw.append(swo)
        Dm.append(dmae); Sm.append(smae)
    Dw, Dm, Sw, Sm = map(np.array, (Dw, Dm, Sw, Sm))
    rows.append((frac, nlab, Dw.mean(), Dw.std(), Dm.mean(), Dm.std(),
                 Sw.mean(), Sw.std(), Sm.mean(), Sm.std()))
    print(f"[done] frac={frac:.2f} n={nlab}")

# ---- FINAL TABLE (in 1 khối cuối + ghi file) ----
hdr = (f"{'frac':>6} {'n_lab':>6} | {'DISTILL worst':>13} {'DISTILL MAE':>12} "
       f"| {'SUPERV worst':>13} {'SUPERV MAE':>12}")
lines = [hdr, "-" * len(hdr)]
for f, n, dwm, dws, dmm, dms, swm, sws, smm, sms in rows:
    lines.append(f"{f:6.2f} {n:6d} | {dwm:.3f}+-{dws:.3f} {dmm:6.2f}+-{dms:5.2f} "
                 f"| {swm:.3f}+-{sws:.3f} {smm:6.2f}+-{sms:5.2f}")
table = "\n".join(lines)
out = os.path.join(REPO, "distillation_counting", "label_efficiency_p2_result.txt")
open(out, "w").write(table + "\n")
print("\n===== LABEL-EFFICIENCY PIECE 2: DISTILLED vs SUPERVISED (same image budget) =====")
print(table)
print(f"\n[saved] {out}")
print("[!] NHỚ backup gt_density_nuinsseg.pkl lên kaggle (chưa backup lần nào).")

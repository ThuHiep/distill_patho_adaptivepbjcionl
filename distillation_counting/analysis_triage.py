#!/usr/bin/env python3
"""Selective prediction / triage cho count — risk-coverage + AURC + E-AURC.

So chất lượng uncertainty-ranking của R2-σ vs UQ-floor (ensemble/mcdropout/cqr/chdqr),
dùng pkl 5-seed NuInsSeg sẵn có. KHÔNG train (0-compute). Chạy Mac / vast / Kaggle notebook.
  AURC↓   = mean risk khi defer dần ảnh unc cao (end-to-end triage system).
  E-AURC↓ = AURC − oracle(theo lỗi thật) → tách khỏi base-MAE, đo THUẦN σ-ranking.
  MAE@80% = defer 20% unc cao nhất -> MAE nhóm giữ lại (câu bán trực quan).
Winner E-AURC thấp nhất = σ lọc lỗi tốt nhất = "uncertainty CÓ ÍCH" (hook top-tier nếu R2 thắng).

Tìm pkl tự động: biến WORK (nếu set) + mọi dataset Kaggle (/kaggle/input/*) + work/ + '.'.
"""
import os, glob, pickle
import numpy as np

ROOTS = [d for d in [os.environ.get("WORK")] if d] \
    + glob.glob("/kaggle/input/*") \
    + ["/workspace/sam3_research/work", "work", "."]
SEEDS = [42, 43, 44, 45, 46]
METHODS = {
    "R2 (ours)":  "student_r2_nuinsseg_cv5_poisson_s{S}.pkl",
    "Ensemble":   "uq_ensemble_nuinsseg_s{S}.pkl",
    "MC-Dropout": "uq_mcdropout_nuinsseg_s{S}.pkl",
    "CQR":        "uq_cqr_nuinsseg_s{S}.pkl",
    "CHDQR":      "uq_chdqr_nuinsseg_s{S}.pkl",
}


def find(fname):
    for r in ROOTS:
        hits = glob.glob(os.path.join(r, "**", fname), recursive=True)
        if hits:
            return hits[0]
    return None


def load(path):
    d = pickle.load(open(path, "rb"))
    preds, gts = d["preds"], d["gts"]
    mu = np.array([p["mu"] for p in preds], float)
    gt = np.array([float(np.asarray(g).reshape(-1)[0]) for g in gts], float)
    p0 = preds[0]
    if "sigma" in p0:
        unc = np.array([p["sigma"] for p in preds], float)
    elif "q_lo" in p0 and "q_hi" in p0:
        unc = np.array([p["q_hi"] - p["q_lo"] for p in preds], float)
    else:
        raise KeyError(f"pkl thiếu sigma/q_lo: {list(p0.keys())}")
    return mu, gt, unc


def metrics(unc, err):
    n = len(err); ks = np.arange(1, n + 1)
    order = np.argsort(unc, kind="mergesort")          # confident (unc thấp) trước
    risk = np.cumsum(err[order]) / ks
    oracle = (np.cumsum(np.sort(err)) / ks).mean()
    k80 = max(1, int(round(0.8 * n)))
    return risk.mean(), risk.mean() - oracle, risk[k80 - 1], err.mean()


print(f"{'method':12} | {'AURC↓':>7} | {'E-AURC↓':>8} | {'MAE@80%↓':>9} | {'full-MAE':>8}")
print("-" * 58)
for label, pat in METHODS.items():
    A, E, M, F = [], [], [], []
    for s in SEEDS:
        p = find(pat.format(S=s))
        if not p:
            print(f"  !! không tìm thấy {pat.format(S=s)}"); continue
        mu, gt, unc = load(p)
        a, e, m, f = metrics(unc, np.abs(mu - gt))
        A.append(a); E.append(e); M.append(m); F.append(f)
    if A:
        print(f"{label:12} | {np.mean(A):7.2f} | {np.mean(E):8.2f} | "
              f"{np.mean(M):9.2f} | {np.mean(F):8.2f}")

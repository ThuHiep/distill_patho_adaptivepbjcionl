"""
analysis_sigma.py — A5 (phân tích sigma) + P2.9 (low-count failure), hậu kỳ trên pkl R2 (không train lại).
  - Spearman corr(sigma, |y-mu|): sigma có THÔNG TIN về lỗi không (nếu ~0 -> sigma vô dụng).
  - z = (y-mu)/sigma: mean≈0, std≈1 nếu calibrated; báo std theo count-bin.
  - Bin theo count (0-10 / 11-30 / 31-100 / >100): MAE, MAPE, mean sigma, corr trong bin.
  - Gaussian NLL trung bình (thấp = phân phối dự đoán khớp hơn).
Chạy: python analysis_sigma.py --preds work/student_r2_nuinsseg_cv5_poisson_feat.pkl --name NuInsSeg
"""
from __future__ import annotations
import argparse, pickle
import numpy as np


def load(path):
    o = pickle.load(open(path, "rb"))
    mu = np.array([float(p["mu"]) for p in o["preds"]])
    sg = np.maximum(np.array([float(p["sigma"]) for p in o["preds"]]), 1e-6)
    gt = np.array([float(np.asarray(g).reshape(-1)[0]) for g in o["gts"]])
    return mu, sg, gt


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def analyze(mu, sg, gt, name):
    ae = np.abs(gt - mu); z = (gt - mu) / sg
    nll = 0.5 * np.log(2 * np.pi * sg ** 2) + (gt - mu) ** 2 / (2 * sg ** 2)
    print("\n" + "=" * 84)
    print(f"A5 — SIGMA ANALYSIS: {name}  (N={len(mu)})")
    print("=" * 84)
    print(f"corr_Spearman(sigma, |y-mu|) = {spearman(sg, ae):+.3f}   (>0 = sigma phản ánh lỗi -> heteroscedastic có ích)")
    print(f"z=(y-mu)/sigma:  mean={z.mean():+.3f} (≈0 tốt)   std={z.std():.3f} (≈1 = calibrated; <1 over-, >1 under-conf)")
    print(f"|z|<=1.64 (≈90%) thực tế = {(np.abs(z)<=1.64).mean():.3f}   Gaussian NLL trung bình = {nll.mean():.3f}")
    print("-" * 84)
    print(f"{'count-bin':>10} | {'n':>4} | {'MAE':>7} | {'MAPE%':>6} | {'mean σ':>7} | {'z-std':>6} | {'corr(σ,|e|)':>11}")
    print("-" * 84)
    bins = [(0, 10), (11, 30), (31, 100), (101, 10**9)]
    for lo, hi in bins:
        m = (gt >= lo) & (gt <= hi)
        if m.sum() == 0:
            continue
        lab = f"{lo}-{hi if hi<10**9 else '+'}"
        mape = float((ae[m] / np.maximum(gt[m], 1)).mean() * 100)
        cc = spearman(sg[m], ae[m]) if m.sum() >= 5 else float('nan')
        print(f"{lab:>10} | {int(m.sum()):>4} | {ae[m].mean():>7.2f} | {mape:>6.1f} | "
              f"{sg[m].mean():>7.2f} | {z[m].std():>6.2f} | {cc:>11.3f}")
    print("-" * 84)
    print("[ĐỌC] MAPE cao ở bin count thấp = lỗi tương đối lớn khi ít nhân (P2.9). z-std xa 1 = sigma sai scale ở bin đó.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="1 pkl hoặc 'a,b,c' (gộp)")
    ap.add_argument("--name", default="dataset")
    args = ap.parse_args()
    mus, sgs, gts = [], [], []
    for p in args.preds.split(","):
        mu, sg, gt = load(p.strip()); mus.append(mu); sgs.append(sg); gts.append(gt)
    analyze(np.concatenate(mus), np.concatenate(sgs), np.concatenate(gts), args.name)


if __name__ == "__main__":
    main()

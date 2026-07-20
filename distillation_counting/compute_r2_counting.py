#!/usr/bin/env python3
"""compute_r2_counting.py — tính R²/MAE/RMSE/MAPE cho BẢNG CHÍNH (Bảng 1) trên NuInsSeg.

Dùng cho 2 loại nguồn per-ảnh:
  (A) PACT student pkl  : {"preds":[{"mu","sigma"}], "gts":[[gt]], "organs":[...]}
      (do distill_student_r2.py sinh; 5-seed = 5 pkl -> báo mean±sd qua các seed).
  (B) heavy-net csv     : cột (image, pred_count) + gt csv (image, gt[, organ])
      (do dump_cellvit_counts.py / eval_heavy_count.py sinh).

R² đếm = 1 - Σ(gt-pred)² / Σ(gt-mean_gt)²  (coefficient of determination, chuẩn H-Optimus).

VÍ DỤ KAGGLE (thêm dataset: hipinhththu/sam3-r2-nuinsseg-seeds):
  # PACT 5-seed:
  python compute_r2_counting.py --pkl_glob "/kaggle/input/sam3-r2-nuinsseg-seeds/*.pkl"
  # heavy net (nếu có csv):
  python compute_r2_counting.py --csv nulite_preds.csv --gt_csv work/nuinsseg_png/gt_counts.csv --name NuLite-T
"""
import argparse, csv, glob, os, pickle
import numpy as np


def metrics(gt, pred):
    gt = np.asarray(gt, float); pred = np.asarray(pred, float)
    ss_res = ((gt - pred) ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.abs(pred - gt).mean()
    rmse = float(np.sqrt(((pred - gt) ** 2).mean()))
    mape = (np.abs(pred - gt) / np.clip(gt, 1, None)).mean() * 100
    return dict(R2=r2, MAE=mae, RMSE=rmse, MAPE=mape, N=len(gt))


def from_pkl(path):
    """Trả (gt[], mu[]) từ 1 PACT preds pkl."""
    d = pickle.load(open(path, "rb"))
    mu = np.array([p["mu"] for p in d["preds"]], float)
    gt = np.array([float(np.ravel(g)[0]) for g in d["gts"]], float)
    return gt, mu


def read_csv_counts(path, key_pred=("pred_count", "count", "pred", "mu")):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            img = r.get("image") or r.get("name")
            val = next((r[k] for k in key_pred if k in r and r[k] != ""), None)
            if img is not None and val is not None:
                out[img] = float(val)
    return out


def read_gt_csv(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            img = r.get("image") or r.get("name")
            gt = r.get("gt") or r.get("gt_count") or r.get("count")
            if img is not None and gt is not None:
                out[img] = float(gt)
    return out


def fmt(m, extra=""):
    return (f"R2={m['R2']:+.3f} | MAE={m['MAE']:.2f} | RMSE={m['RMSE']:.2f} "
            f"| MAPE={m['MAPE']:.1f}% | N={m['N']}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_glob", help="glob các PACT seed pkl, vd /kaggle/input/sam3-r2-nuinsseg-seeds/*.pkl")
    ap.add_argument("--csv", help="heavy-net preds csv (image,pred_count)")
    ap.add_argument("--gt_csv", help="gt csv (image,gt[,organ]) — bắt buộc khi dùng --csv")
    ap.add_argument("--name", default="model", help="tên hàng để in")
    args = ap.parse_args()

    if args.pkl_glob:
        files = sorted(glob.glob(args.pkl_glob, recursive=True))
        assert files, f"không thấy pkl khớp: {args.pkl_glob}"
        rows = []
        print(f"=== PACT (per-seed, {len(files)} file) ===")
        for fp in files:
            gt, mu = from_pkl(fp)
            m = metrics(gt, mu); rows.append(m)
            print(f"  {os.path.basename(fp):40s} {fmt(m)}")
        arr = {k: np.array([r[k] for r in rows], float) for k in ("R2", "MAE", "RMSE", "MAPE")}
        print("\n=== PACT — mean ± sd qua seed (điền Bảng 1) ===")
        for k in ("R2", "MAE", "RMSE", "MAPE"):
            unit = "%" if k == "MAPE" else ""
            print(f"  {k:5s} = {arr[k].mean():.3f} ± {arr[k].std():.3f}{unit}")

    if args.csv:
        assert args.gt_csv, "cần --gt_csv khi dùng --csv"
        pred = read_csv_counts(args.csv); gtm = read_gt_csv(args.gt_csv)
        common = [im for im in gtm if im in pred]
        assert common, "không khớp được ảnh nào giữa csv preds và gt_csv"
        gt = [gtm[im] for im in common]; pr = [pred[im] for im in common]
        m = metrics(gt, pr)
        print(f"\n=== {args.name} (csv) ===\n  {fmt(m, extra=f' | khớp {len(common)} ảnh')}")


if __name__ == "__main__":
    main()

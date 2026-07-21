#!/usr/bin/env python3
"""stratified_error.py — phân tích lỗi ĐẾM theo TẦNG mật độ (đóng critique C1).

Reviewer C1: "MAPE cao (47.6%) vì cộng-tổng-mật-độ khuếch đại sai số TƯƠNG ĐỐI ở ảnh ÍT nhân
(mẫu số nhỏ)". Đây là tính chất toán của MAPE, không phải model tệ. Script này CHỨNG MINH bằng số:
bin theo GT count -> báo MAE/RMSE/MAPE/R² mỗi bin. Kỳ vọng: MAPE dồn ở bin THẤP, còn MAE/R² ổn.

3 nguồn per-ảnh (giống compute_r2_counting.py):
  (A) PACT pkl   : {"preds":[{"mu",...}], "gts":[[gt]], "organs":[...]}   -> --pkl_glob (5-seed: gộp)
  (B) teacher pkl: {"preds":[{"scores",...}], "gts":[[gt]]}  count=len(scores) -> --teacher_pkl
  (C) heavy csv  : (image,pred_count) + gt csv                -> --csv --gt_csv

Bin mặc định (edges theo count): Thấp 1-20 | TB 21-50 | Cao >50  (≈ tertile NuInsSeg 27/53).
Đổi bằng --edges "20,50".

VD LOCAL (teacher, chạy ngay):
  python stratified_error.py --teacher_pkl ../data/pathosam_nuinsseg_preds.pkl --name PathoSAM
VD KAGGLE (PACT 5-seed):
  python stratified_error.py --pkl_glob "/kaggle/input/sam3-r2-nuinsseg-seeds/*.pkl" --name PACT
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


def from_pact_pkl(path):
    d = pickle.load(open(path, "rb"))
    mu = np.array([p["mu"] for p in d["preds"]], float)
    gt = np.array([float(np.ravel(g)[0]) for g in d["gts"]], float)
    return gt, mu


def from_teacher_pkl(path):
    d = pickle.load(open(path, "rb"))
    pred = np.array([len(p["scores"]) for p in d["preds"]], float)  # count = len(scores)
    gt = np.array([float(np.ravel(g)[0]) for g in d["gts"]], float)
    return gt, pred


def from_csv(csv_path, gt_csv):
    def rd(path, keys):
        out = {}
        with open(path) as f:
            for r in csv.DictReader(f):
                img = r.get("image") or r.get("name")
                val = next((r[k] for k in keys if k in r and r[k] != ""), None)
                if img is not None and val is not None:
                    out[img] = float(val)
        return out
    pred = rd(csv_path, ("pred_count", "count", "pred", "mu"))
    gtm = rd(gt_csv, ("gt", "gt_count", "count"))
    common = [im for im in gtm if im in pred]
    assert common, "không khớp ảnh nào giữa csv preds và gt_csv"
    gt = np.array([gtm[im] for im in common], float)
    pr = np.array([pred[im] for im in common], float)
    return gt, pr


def load_pooled_pact(glob_pat):
    """5-seed: mỗi seed cùng thứ tự ảnh -> pool bằng cách stack rồi bin theo GT (GT giống nhau).
    Báo per-bin metric = mean qua seed, kèm ±sd."""
    files = sorted(glob.glob(glob_pat, recursive=True))
    assert files, f"không thấy pkl: {glob_pat}"
    gts, mus = [], []
    for fp in files:
        g, m = from_pact_pkl(fp)
        gts.append(g); mus.append(m)
    g0 = gts[0]
    for g in gts[1:]:
        assert np.allclose(g, g0), "GT giữa các seed KHÔNG khớp thứ tự -> không pool được"
    return g0, mus, [os.path.basename(f) for f in files]


def print_table(name, gt, pred, edges):
    lo, hi = edges
    bins = [("Thấp (1-%d)" % lo, gt <= lo),
            ("TB (%d-%d)" % (lo + 1, hi), (gt > lo) & (gt <= hi)),
            ("Cao (>%d)" % hi, gt > hi)]
    print(f"\n=== {name} — lỗi theo TẦNG mật độ (N={len(gt)}) ===")
    print(f"{'Bin':14s} {'N':>4s} {'GT̄':>6s} {'MAE':>7s} {'RMSE':>7s} {'MAPE':>7s} {'R²':>7s}")
    for label, mask in bins:
        if mask.sum() == 0:
            continue
        m = metrics(gt[mask], pred[mask])
        print(f"{label:14s} {m['N']:>4d} {gt[mask].mean():>6.1f} "
              f"{m['MAE']:>7.2f} {m['RMSE']:>7.2f} {m['MAPE']:>6.1f}% {m['R2']:>+7.3f}")
    mo = metrics(gt, pred)
    print(f"{'TỔNG':14s} {mo['N']:>4d} {gt.mean():>6.1f} "
          f"{mo['MAE']:>7.2f} {mo['RMSE']:>7.2f} {mo['MAPE']:>6.1f}% {mo['R2']:>+7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_glob", help="PACT 5-seed pkl glob (pool qua seed)")
    ap.add_argument("--teacher_pkl", help="teacher pkl (count=len(scores))")
    ap.add_argument("--csv", help="heavy-net csv")
    ap.add_argument("--gt_csv", help="gt csv (bắt buộc khi --csv)")
    ap.add_argument("--name", default="model")
    ap.add_argument("--edges", default="20,50", help="ngưỡng bin 'lo,hi' (mặc định 20,50)")
    args = ap.parse_args()
    edges = tuple(int(x) for x in args.edges.split(","))

    if args.pkl_glob:
        gt, mus, names = load_pooled_pact(args.pkl_glob)
        # per-seed rồi lấy mean±sd cho MAPE mỗi bin -> chứng minh robust
        lo, hi = edges
        binmasks = [("Thấp (1-%d)" % lo, gt <= lo),
                    ("TB (%d-%d)" % (lo + 1, hi), (gt > lo) & (gt <= hi)),
                    ("Cao (>%d)" % hi, gt > hi)]
        print(f"\n=== {args.name} 5-seed ({len(names)}) — lỗi theo TẦNG (mean±sd qua seed) ===")
        print(f"{'Bin':14s} {'N':>4s} {'GT̄':>6s} {'MAE':>13s} {'MAPE':>15s} {'R²':>15s}")
        for label, mask in binmasks:
            if mask.sum() == 0:
                continue
            per = [metrics(gt[mask], m[mask]) for m in mus]
            mae = np.array([p["MAE"] for p in per]); mape = np.array([p["MAPE"] for p in per])
            r2 = np.array([p["R2"] for p in per])
            print(f"{label:14s} {mask.sum():>4d} {gt[mask].mean():>6.1f} "
                  f"{mae.mean():>6.2f}±{mae.std():<5.2f} "
                  f"{mape.mean():>6.1f}±{mape.std():<5.1f}% "
                  f"{r2.mean():>+6.3f}±{r2.std():<5.3f}")
        per = [metrics(gt, m) for m in mus]
        mae = np.array([p["MAE"] for p in per]); mape = np.array([p["MAPE"] for p in per])
        r2 = np.array([p["R2"] for p in per])
        print(f"{'TỔNG':14s} {len(gt):>4d} {gt.mean():>6.1f} "
              f"{mae.mean():>6.2f}±{mae.std():<5.2f} "
              f"{mape.mean():>6.1f}±{mape.std():<5.1f}% "
              f"{r2.mean():>+6.3f}±{r2.std():<5.3f}")
        return

    if args.teacher_pkl:
        gt, pred = from_teacher_pkl(args.teacher_pkl)
    elif args.csv:
        assert args.gt_csv, "cần --gt_csv khi --csv"
        gt, pred = from_csv(args.csv, args.gt_csv)
    else:
        ap.error("cần 1 trong: --pkl_glob | --teacher_pkl | --csv")
    print_table(args.name, gt, pred, edges)


if __name__ == "__main__":
    main()

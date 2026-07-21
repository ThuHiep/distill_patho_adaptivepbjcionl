#!/usr/bin/env python3
"""significance_counting.py — kiểm định ý nghĩa cho BẢNG 1 (đóng critique B).

Reviewer B: "báo mean±sd + p-value (Wilcoxon/paired-t) cho MAE/RMSE để khẳng định cải thiện
KHÔNG do ngẫu nhiên". Bảng 1 so PACT vs teacher/heavy-net -> đơn vị lặp ĐÚNG = ẢNH (665), test
paired trên |lỗi| per-ảnh (KHÔNG phải per-seed = pseudoreplication, giống eval_r2_grouped critique 3.5).

PACT = mean-over-seed của mu mỗi ảnh (5 pkl cùng thứ tự ảnh). Đối thủ:
  --teacher_pkl : count = len(scores)
  --csv --gt_csv: heavy-net csv (khớp theo tên ảnh)

In: Δ MAE (ours-đối_thủ, âm=ours tốt hơn) + Wilcoxon p + bootstrap 95% CI của ΔMAE.

VD KAGGLE:
  python significance_counting.py --pkl_glob "/kaggle/input/sam3-r2-nuinsseg-seeds/*.pkl" \
      --teacher_pkl /kaggle/input/.../pathosam_nuinsseg_preds.pkl --name_vs teacher
  python significance_counting.py --pkl_glob ".../*.pkl" \
      --csv nulite_preds.csv --gt_csv gt_counts.csv --name_vs NuLite-T
"""
import argparse, csv, glob, os, pickle
import numpy as np


def from_pact_pkl(path):
    d = pickle.load(open(path, "rb"))
    mu = np.array([p["mu"] for p in d["preds"]], float)
    gt = np.array([float(np.ravel(g)[0]) for g in d["gts"]], float)
    return gt, mu


def pact_mean_over_seeds(glob_pat):
    files = sorted(glob.glob(glob_pat, recursive=True))
    assert files, f"không thấy pkl: {glob_pat}"
    gts, mus = [], []
    for fp in files:
        g, m = from_pact_pkl(fp); gts.append(g); mus.append(m)
    g0 = gts[0]
    for g in gts[1:]:
        assert np.allclose(g, g0), "GT các seed KHÔNG khớp thứ tự"
    mu_mean = np.mean(np.stack(mus), axis=0)   # mean-over-seed per ảnh
    return g0, mu_mean, len(files)


def teacher_from_pkl(path):
    d = pickle.load(open(path, "rb"))
    pred = np.array([len(p["scores"]) for p in d["preds"]], float)
    gt = np.array([float(np.ravel(g)[0]) for g in d["gts"]], float)
    return gt, pred


def csv_pred(csv_path, gt_csv):
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
    return pred, gtm


def paired_report(ae_ours, ae_base, name_vs, n_boot=10000, seed=0):
    """paired Wilcoxon + bootstrap CI của ΔMAE = mean(|err_ours|) - mean(|err_base|)."""
    from scipy.stats import wilcoxon
    d = ae_ours - ae_base            # âm = ours ít lỗi hơn
    dmae = float(d.mean())
    try:
        stat, p = wilcoxon(ae_ours, ae_base)
    except ValueError as e:          # all-zero diff
        p = float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sig = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
    print(f"\n=== PACT vs {name_vs} — paired per-ảnh (N={len(d)}) ===")
    print(f"  MAE ours = {np.abs(ae_ours).mean():.2f} | MAE {name_vs} = {np.abs(ae_base).mean():.2f}")
    print(f"  ΔMAE (ours-{name_vs}) = {dmae:+.2f}  95%CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  paired-Wilcoxon p = {p:.3g}  -> {sig}  ({'ours tốt hơn' if dmae<0 else 'ĐỐI THỦ tốt hơn'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_glob", required=True, help="PACT 5-seed pkl (ours)")
    ap.add_argument("--teacher_pkl")
    ap.add_argument("--csv"); ap.add_argument("--gt_csv")
    ap.add_argument("--name_vs", default="baseline")
    args = ap.parse_args()

    gt, mu, nseed = pact_mean_over_seeds(args.pkl_glob)
    ae_ours = np.abs(mu - gt)
    print(f"[ours] PACT mean-over-{nseed}-seed | N={len(gt)} | MAE={ae_ours.mean():.2f}")

    if args.teacher_pkl:
        gtt, pt = teacher_from_pkl(args.teacher_pkl)
        assert np.allclose(gtt, gt), "GT teacher KHÔNG align PACT theo index"
        paired_report(ae_ours, np.abs(pt - gt), args.name_vs)
    elif args.csv:
        assert args.gt_csv, "cần --gt_csv"
        # heavy csv khớp theo TÊN ảnh; PACT pkl không có tên -> cần gt_csv có cùng thứ tự pkl.
        # An toàn: căn theo GT trùng khớp không đủ (trùng số). Yêu cầu pkl có 'names' nếu muốn khớp tên.
        d0 = pickle.load(open(sorted(glob.glob(args.pkl_glob))[0], "rb"))
        names = d0.get("names") or d0.get("images")
        assert names, ("pkl PACT không có key 'names'/'images' -> không khớp tên với heavy csv được. "
                       "Chạy đối chiếu heavy-net bằng compute_r2_counting.py (aggregate) thay vì paired.")
        pred, gtm = csv_pred(args.csv, args.gt_csv)
        idx = [i for i, nm in enumerate(names) if nm in pred]
        ae_b = np.abs(np.array([pred[names[i]] for i in idx]) - gt[idx])
        paired_report(ae_ours[idx], ae_b, args.name_vs)
    else:
        ap.error("cần --teacher_pkl hoặc --csv")


if __name__ == "__main__":
    main()

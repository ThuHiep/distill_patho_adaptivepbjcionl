"""
eval_r2ccp.py — Baseline R2CCP (Regression-as-Classification Conformal Prediction, Guha et al.,
ICLR 2024, arXiv 2404.08168), dùng ĐÚNG code official (github.com/EtashGuha/R2CCP) — KHÔNG tự chế.
R2CCP chia output thành bin → phân loại → CP cho classification → tập dự đoán (có thể ĐA ĐOẠN,
xử lý phân phối đa đỉnh/lệch). **Đếm tế bào rời rạc → binning rất tự nhiên** (domain-fit mạnh).

Áp lên CÙNG biểu diễn distilled: X = đặc trưng sâu/ảnh (pooled bottleneck của student R2, lưu bằng
`distill_student_r2.py --dump_feat`, leak-free), y = count. R2CCP fit mạng riêng trên feature +
conformal nội bộ (tách cal từ chính data fit → leak-free), rồi trả tập/ảnh test.

Vì R2CCP trả TẬP (union đoạn), ta đo:
  - set-coverage : y ∈ bất kỳ đoạn nào
  - set-length   : Σ độ dài đoạn
  - set-Winkler  : Σlen + (2/α)·dist(y, tập gần nhất) nếu miss (tổng quát hoá winkler_score cho tập)
  - worst-org    : set-coverage theo organ (organ_conditional_stats)  ← đấu trục worst-org
  - MAE          : |y − R2CCP.predict| (điểm của chính R2CCP)
Cùng seeds/cal_ratio/organ_conditional_stats → so bảng mục 8. R2CCP RETRAIN mỗi seed (đắt) → mặc
định seeds ÍT hơn (5); tăng nếu cần.

Cài (khỏi pip vì dep crlibm thừa lỗi — dùng repo trực tiếp):
  git clone https://github.com/EtashGuha/R2CCP.git
  pip install pytorch_lightning configargparse torchvision   # (torch/sklearn đã có)
Chạy (pkl phải có 'feat'):
  python eval_r2ccp.py --preds work/student_r2_pannuke_f3_nocolon_poisson_feat.pkl \
      --r2ccp_dir ./R2CCP --seeds 5 --alpha 0.1 --max_epochs 100 --min_organ_imgs 10
"""
from __future__ import annotations
import argparse, os, pickle, sys, warnings, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coverage_transfer import organ_conditional_stats  # noqa: E402

warnings.filterwarnings("ignore")


def load_feat(path):
    obj = pickle.load(open(path, "rb"))
    preds = obj["preds"]
    if "feat" not in preds[0]:
        raise SystemExit("pkl KHÔNG có 'feat' — chạy distill_student_r2.py --dump_feat để tạo lại.")
    gt = np.array([float(np.asarray(g).reshape(-1)[0]) for g in obj["gts"]])
    organs = list(obj.get("organs", ["_all_"] * len(preds)))
    if len(organs) != len(preds):
        organs = ["_all_"] * len(preds)
    feat = np.stack([np.asarray(p["feat"], np.float32) for p in preds])
    mu = np.array([float(p["mu"]) for p in preds])
    return feat, mu, gt, organs


def set_winkler(segments, y, alpha):
    """Winkler tổng quát cho TẬP (union đoạn). segments: list (lo,hi). Σlen + (2/α)·dist nếu miss."""
    length = sum(max(hi - lo, 0.0) for lo, hi in segments)
    inside = any(lo <= y <= hi for lo, hi in segments)
    if inside:
        return length, True
    dist = min(min(abs(y - lo), abs(y - hi)) for lo, hi in segments) if segments else abs(y)
    return length + (2.0 / alpha) * dist, False


def eval_r2ccp(feat, mu, gt, organs, alpha, seeds, cal_ratio, min_organ_imgs, r2ccp_dir, max_epochs):
    sys.path.insert(0, os.path.abspath(r2ccp_dir))
    from R2CCP.main import R2CCP  # code official
    N = len(mu); target = 1 - alpha
    organs = np.asarray(organs, dtype=object)
    per_seed, pooled = [], []
    tmpdir = tempfile.mkdtemp(prefix="r2ccp_")
    for s in range(seeds):
        rng = np.random.RandomState(1000 + s)
        perm = rng.permutation(N); ncal = int(N * cal_ratio)
        cal, tst = perm[:ncal], perm[ncal:]
        model = R2CCP({"model_path": os.path.join(tmpdir, f"m{s}.pth"),
                       "max_epochs": max_epochs, "alpha": alpha})
        model.fit(feat[cal], gt[cal].reshape(-1, 1).astype(np.float32))   # train + conformal nội bộ (leak-free)
        segs_list = model.get_intervals(feat[tst])                        # list/ảnh: [(lo,hi),...]
        pred_pt = np.asarray(model.predict(feat[tst])).reshape(-1)        # điểm R2CCP
        wk, lens, covs = [], [], []
        for j in range(len(tst)):
            segs = [(float(a), float(b)) for a, b in segs_list[j]]
            w, ins = set_winkler(segs, float(gt[tst][j]), alpha)
            wk.append(w); lens.append(sum(b - a for a, b in segs)); covs.append(ins)
            pooled.append((organs[tst[j]], int(tst[j]), bool(ins), float(ins)))
        ae = np.abs(gt[tst] - pred_pt)
        per_seed.append({"coverage": float(np.mean(covs)), "width": float(np.mean(lens)),
                         "winkler": float(np.mean(wk)), "mae": float(np.mean(ae))})
        print(f"  seed {s+1}/{seeds}: cov={np.mean(covs):.3f} len={np.mean(lens):.2f} winkler={np.mean(wk):.2f}")
    def ms(k): v = np.array([d[k] for d in per_seed]); return {"mean": float(v.mean()), "std": float(v.std())}
    cond = organ_conditional_stats(pooled, target, min_organ_imgs)
    return {"coverage": ms("coverage"), "width": ms("width"), "winkler": ms("winkler"),
            "mae": ms("mae"), "conditional": cond, "winkler_seeds": [d["winkler"] for d in per_seed]}


def pretty(d, cfg):
    print("\n" + "=" * 92)
    print(f"R2CCP — Regression-as-Classification CP (Guha et al., ICLR 2024) | alpha={cfg['alpha']} "
          f"target={1-cfg['alpha']:.3f} seeds={cfg['seeds']} (X=deep feat student)")
    print("=" * 92)
    cd = d["conditional"]; wo = cd["worst_organ_coverage"]; gap = cd["organ_coverage_gap"]
    wo_s = f"{wo:.3f}" if wo is not None else "n/a"; gap_s = f"{gap:.3f}" if gap is not None else "n/a"
    print(f"marg.cov={d['coverage']['mean']:.3f}  set-len={d['width']['mean']:.2f}  "
          f"set-Winkler={d['winkler']['mean']:.2f}±{d['winkler']['std']:.2f}  MAE={d['mae']['mean']:.2f}")
    print(f"worst-org={wo_s}  org-gap={gap_s}  #under={cd['n_organs_undercovered']}/{cd['n_organs_eval']}")
    print("[GHI CHÚ] Tập ĐA ĐOẠN (đa đỉnh) → set-len/set-Winkler; so worst-org với R2/CondConf/PCP (mục 8).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="pkl R2 CÓ 'feat' (dump_feat)")
    ap.add_argument("--r2ccp_dir", default="./R2CCP", help="repo EtashGuha/R2CCP (chứa R2CCP/main.py)")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5, help="ít vì R2CCP retrain mỗi seed (đắt)")
    ap.add_argument("--cal_ratio", type=float, default=0.5)
    ap.add_argument("--min_organ_imgs", type=int, default=10)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    feat, mu, gt, organs = load_feat(args.preds)
    print(f"loaded {args.preds} N={len(mu)} feat_dim={feat.shape[1]} organs={len(set(organs))}")
    d = eval_r2ccp(feat, mu, gt, organs, args.alpha, args.seeds, args.cal_ratio,
                   args.min_organ_imgs, args.r2ccp_dir, args.max_epochs)
    cfg = {"alpha": args.alpha, "seeds": args.seeds, "cal_ratio": args.cal_ratio}
    pretty(d, cfg)
    if args.out:
        import json
        json.dump({"config": cfg, "R2CCP": d}, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\nĐã lưu {args.out}")


if __name__ == "__main__":
    main()

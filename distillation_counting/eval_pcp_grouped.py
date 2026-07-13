"""
eval_pcp_grouped.py — Baseline PCP (Posterior Conformal Prediction, Zhang & Candès 2024,
arXiv 2409.19712), dùng ĐÚNG code official (github.com/yaozhang24/pcp, class PCP trong utils.py) —
KHÔNG tự chế. PCP mô hình phân phối score điều kiện = mixture theo CỤM (tự phát hiện từ feature) →
coverage marginal + xấp xỉ CONDITIONAL theo subgroup. Đấu TRỰC TIẾP trục worst-org của R2.

Áp lên CÙNG (μ,σ) student R2 leak-free đã lưu:
  residual R = |GT − μ| / σ    (score chuẩn hoá, y hệt R2/CondConf/KD)
  feature X  = [μ, σ]          (đặc trưng ĐỘ KHÓ liên tục — đúng thiết kế PCP: TỰ phát hiện cụm khó,
                                KHÔNG cần cho nhãn organ). Organ chỉ dùng lúc EVAL → test worst-org
                                công bằng HƠN (PCP không được biết mô mà vẫn phải phủ đều từng mô).
                                (one-hot organ thuần làm PCP overflow nội bộ; --features organ nếu cần.)
  khoảng     = [μ − r·σ, μ + r·σ]   (r = quantile PCP/điểm, clip ≥0)
PCP.calibrate: quantile r CHỈ phụ thuộc X_test + residual cal (R_test chỉ để PCP báo coverage nội
bộ — ta TỰ tính lại cov/Winkler/worst-org) → leak-free. Không train lại student, CPU-only.

Cùng seeds/cal_ratio/organ_conditional_stats/Winkler như eval_r2_grouped.py → so trực tiếp bảng
mục 8. MAE = |GT−μ| (cùng μ). Cần repo pcp trên máy:
  git clone https://github.com/yaozhang24/pcp.git
  pip install numpy scipy scikit-learn statsmodels tqdm
Chạy:
  python eval_pcp_grouped.py --preds work/student_r2_pannuke_f3_nocolon_poisson.pkl \
      --pcp_dir ./pcp --seeds 10 --alpha 0.1 --min_organ_imgs 10
"""
from __future__ import annotations
import argparse, os, pickle, sys, warnings
import random as _random
import numpy as np

# numpy-2 trả np.int64; python-3.12 random.seed CHỈ nhận int/float/str/bytes -> ép int cho seed numpy.
# (PCP nội bộ gọi random.seed(np_int) -> TypeError. Vá tương thích, không đổi thuật toán.)
_orig_random_seed = _random.seed
def _safe_seed(a=None, *args, **kw):
    if isinstance(a, np.integer):
        a = int(a)
    elif isinstance(a, np.floating):
        a = float(a)
    return _orig_random_seed(a, *args, **kw)
_random.seed = _safe_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coverage_transfer import winkler_score, organ_conditional_stats  # noqa: E402

EPS = 1e-6


def load_musigma(path):
    obj = pickle.load(open(path, "rb"))
    preds = obj["preds"]
    gt = np.array([float(np.asarray(g).reshape(-1)[0]) for g in obj["gts"]])
    organs = list(obj.get("organs", ["_all_"] * len(preds)))
    if len(organs) != len(preds):
        organs = ["_all_"] * len(preds)
    mu = np.array([float(p["mu"]) for p in preds])
    sg = np.maximum(np.array([float(p["sigma"]) for p in preds]), EPS)
    return mu, sg, gt, organs


def eval_pcp(mu, sg, gt, organs, alpha, seeds, cal_ratio, min_organ_imgs, pcp_dir, features="musigma"):
    sys.path.insert(0, os.path.abspath(pcp_dir))
    from utils import PCP  # code official (repo pcp)
    N = len(mu); target = 1 - alpha
    organs = np.asarray(organs, dtype=object)
    uniq = sorted(set(organs.tolist())); oid = {o: i for i, o in enumerate(uniq)}; K = len(uniq)
    org_id = np.array([oid[o] for o in organs])
    if features == "organ":
        X_all = np.zeros((N, K), np.float64); X_all[np.arange(N), org_id] = 1.0
    else:  # 'musigma': đặc trưng độ khó liên tục (đúng thiết kế PCP, well-conditioned)
        X_all = np.column_stack([mu, sg]).astype(np.float64)
    R_all = np.abs(gt - mu) / sg                         # score chuẩn hoá (như mọi baseline)

    per_seed, pooled = [], []
    for s in range(seeds):
        rng = np.random.RandomState(1000 + s)
        perm = rng.permutation(N); nc = int(N * cal_ratio)
        cal, tst = perm[:nc], perm[nc:]
        Xc, Rc = X_all[cal], R_all[cal]
        Xt, Rt = X_all[tst], R_all[tst]
        pcp = PCP()
        with np.errstate(all="ignore"), warnings.catch_warnings():   # nén warning NỘI BỘ PCP (không đổi tính toán)
            warnings.simplefilter("ignore")
            pcp.train(Xc, Rc)                                # chọn hyperparam (CV nội bộ trên cal)
            r_list, _ = pcp.calibrate(Xc, Rc, Xt, Rt, alpha)  # quantile/điểm (leak-free)
        r_arr = np.array([float(r) for r in r_list])
        smax = float(np.quantile(Rc, 0.999) + 1.0)
        r_arr = np.where(np.isfinite(r_arr), r_arr, smax)  # thay inf bằng cận an toàn
        lo = np.maximum(0.0, mu[tst] - r_arr * sg[tst]); hi = mu[tst] + r_arr * sg[tst]
        cov = (gt[tst] >= lo) & (gt[tst] <= hi)
        wink = np.array([winkler_score(lo[j], hi[j], gt[tst][j], alpha) for j in range(len(tst))])
        ae = np.abs(gt[tst] - mu[tst])
        per_seed.append({"coverage": float(cov.mean()), "width": float((hi - lo).mean()),
                         "winkler": float(wink.mean()), "mae": float(ae.mean())})
        for j, ti in enumerate(tst):
            pooled.append((organs[ti], int(ti), bool(cov[j]), float(cov[j])))
        print(f"  seed {s+1}/{seeds}: cov={cov.mean():.3f} winkler={wink.mean():.2f}")
    def ms(k): v = np.array([d[k] for d in per_seed]); return {"mean": float(v.mean()), "std": float(v.std())}
    cond = organ_conditional_stats(pooled, target, min_organ_imgs)
    return {"coverage": ms("coverage"), "width": ms("width"), "winkler": ms("winkler"),
            "mae": ms("mae"), "conditional": cond, "winkler_seeds": [d["winkler"] for d in per_seed]}


def pretty(d, cfg):
    print("\n" + "=" * 92)
    print(f"PCP — Posterior Conformal Prediction (Zhang & Candès 2024) | alpha={cfg['alpha']} "
          f"target={1-cfg['alpha']:.3f} seeds={cfg['seeds']} (feature={cfg.get('features','musigma')})")
    print("=" * 92)
    cd = d["conditional"]; wo = cd["worst_organ_coverage"]; gap = cd["organ_coverage_gap"]
    wo_s = f"{wo:.3f}" if wo is not None else "n/a"; gap_s = f"{gap:.3f}" if gap is not None else "n/a"
    print(f"marg.cov={d['coverage']['mean']:.3f}  width={d['width']['mean']:.2f}  "
          f"Winkler={d['winkler']['mean']:.2f}±{d['winkler']['std']:.2f}  MAE={d['mae']['mean']:.2f}")
    print(f"worst-org={wo_s}  org-gap={gap_s}  #under={cd['n_organs_undercovered']}/{cd['n_organs_eval']}")
    print("[GHI CHÚ] So worst-org/Winkler với R2-cluster/mondrian (mục 8) + CondConf-group.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="pkl R2 leak-free (mu,sigma,organs)")
    ap.add_argument("--pcp_dir", default="./pcp", help="đường dẫn repo yaozhang24/pcp (chứa utils.py)")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cal_ratio", type=float, default=0.5)
    ap.add_argument("--min_organ_imgs", type=int, default=10)
    ap.add_argument("--features", choices=["musigma", "organ"], default="musigma",
                    help="musigma: X=[μ,σ] (đúng thiết kế PCP, mặc định); organ: X=onehot (có thể overflow)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    mu, sg, gt, organs = load_musigma(args.preds)
    print(f"loaded {args.preds} N={len(mu)} organs={len(set(organs))} features={args.features}")
    d = eval_pcp(mu, sg, gt, organs, args.alpha, args.seeds, args.cal_ratio,
                 args.min_organ_imgs, args.pcp_dir, args.features)
    cfg = {"alpha": args.alpha, "seeds": args.seeds, "cal_ratio": args.cal_ratio, "features": args.features}
    pretty(d, cfg)
    if args.out:
        import json
        json.dump({"config": cfg, "PCP": d}, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\nĐã lưu {args.out}")


if __name__ == "__main__":
    main()

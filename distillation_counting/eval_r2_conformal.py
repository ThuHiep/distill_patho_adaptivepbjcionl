"""
eval_r2_conformal.py — Đo CỔNG cho R2: split conformal trên score chuẩn hoá r=|gt-mu|/sigma.

Khác eval_coverage_transfer.py (đo transfer T->S/T->T): file này đo chế độ TRIỂN KHAI THẬT
(S->S: recalibrate trên chính student) cho tối đa 2 model để SO TRỰC TIẾP KD vs R2 và in VERDICT
cổng. Cùng một conformal + Winkler cho cả hai nên so công bằng.

Suy (mu, sigma) mỗi ảnh:
  - pred có 'mu'/'sigma' (R2, distill_student_r2.py)          -> dùng trực tiếp.
  - pred có 'scores'/'probs' (KD, distill_student_nuinsseg.py) -> mu = pb_count, sigma = sqrt(pb_var).
  => KD và R2 đi qua ĐÚNG một conformal on score |gt-mu|/sigma. (K=1 cho NuInsSeg.)

Conformal (split, marginal guarantee — Vovk): trên cal, r_i=|gt_i-mu_i|/sigma_i;
q = quantile mức ceil((n+1)(1-alpha))/n. Interval test = [mu-q*sigma, mu+q*sigma] (clip >=0).
sigma heteroscedastic => khoảng RỘNG đúng ở ảnh/organ khó => kỳ vọng cải thiện conditional coverage
& Winkler mà width tổng không nổ. Đây là cơ chế R2 (khác PBUD/CCAD).

Dùng:
  python eval_r2_conformal.py --preds work/student_r2.pkl --seeds 20
  python eval_r2_conformal.py --preds work/student_kd.pkl work/student_r2.pkl \
         --labels KD R2 --seeds 20 --out gate_r2.json
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle", "lib"))
from conformal import pb_count, pb_variance  # noqa: E402
from eval_coverage_transfer import winkler_score, organ_conditional_stats  # noqa: E402

EPS = 1e-6


def load_mu_sigma(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Trả (mu[N], sigma[N], gt[N], organs[N]) — K=1. Suy từ mu/sigma hoặc scores/probs."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    preds = obj["preds"]
    gts = [float(np.asarray(g).reshape(-1)[0]) for g in obj["gts"]]
    organs = list(obj.get("organs", ["_all_"] * len(preds)))
    if len(organs) != len(preds):
        organs = ["_all_"] * len(preds)
    mu, sigma = [], []
    for p in preds:
        if "mu" in p and "sigma" in p:
            mu.append(float(p["mu"])); sigma.append(float(p["sigma"]))
        else:
            s = np.asarray(p["scores"], float); pr = np.asarray(p["probs"], float)
            if len(s) == 0:
                mu.append(0.0); sigma.append(1.0)
            else:
                mu.append(float(pb_count(s, pr).reshape(-1)[0]))
                sigma.append(float(np.sqrt(pb_variance(s, pr).reshape(-1)[0] + EPS)))
    return (np.asarray(mu), np.maximum(np.asarray(sigma), EPS),
            np.asarray(gts), organs)


def conformal_q(r_cal: np.ndarray, alpha: float) -> float:
    """Quantile conformal hữu hạn mẫu: mức ceil((n+1)(1-alpha))/n, clip<=1."""
    n = len(r_cal)
    if n == 0:
        return 0.0
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(r_cal, level, method="higher"))


def eval_one(mu, sigma, gt, organs, alpha, seeds, cal_ratio, min_organ_imgs) -> Dict:
    N = len(mu)
    target = 1 - alpha
    per_seed = []
    pooled = []  # (organ, idx, covered_bool, covered_frac) cho organ_conditional_stats
    for seed in range(seeds):
        rng = np.random.RandomState(1000 + seed)
        idx = rng.permutation(N)
        n_cal = int(N * cal_ratio)
        cal, tst = idx[:n_cal], idx[n_cal:]
        r_cal = np.abs(gt[cal] - mu[cal]) / sigma[cal]
        q = conformal_q(r_cal, alpha)
        lo = np.maximum(0.0, mu[tst] - q * sigma[tst])
        hi = mu[tst] + q * sigma[tst]
        cov = (gt[tst] >= lo) & (gt[tst] <= hi)
        width = hi - lo
        wink = np.array([winkler_score(lo[j], hi[j], gt[tst][j], alpha) for j in range(len(tst))])
        ae = np.abs(gt[tst] - mu[tst])
        per_seed.append({"coverage": float(cov.mean()), "width": float(width.mean()),
                         "winkler": float(wink.mean()), "mae": float(ae.mean()), "q": q})
        for j, ti in enumerate(tst):
            pooled.append((organs[ti], int(ti), bool(cov[j]), float(cov[j])))
    def ms(key):
        v = np.array([d[key] for d in per_seed]); return {"mean": float(v.mean()), "std": float(v.std())}
    cond = organ_conditional_stats(pooled, target, min_organ_imgs)
    return {"coverage": ms("coverage"), "width": ms("width"), "winkler": ms("winkler"),
            "mae": ms("mae"), "q": ms("q"), "conditional": cond}


def run(paths: List[str], labels: List[str], alpha, seeds, cal_ratio, min_organ_imgs) -> Dict:
    out = {}
    ref_gt = None
    for path, lab in zip(paths, labels):
        mu, sigma, gt, organs = load_mu_sigma(path)
        if ref_gt is None:
            ref_gt = gt
        elif len(gt) != len(ref_gt) or not np.allclose(gt, ref_gt):
            print(f"[CẢNH BÁO] GT của {lab} khác model đầu — kiểm tra cùng tập/thứ tự ảnh.")
        out[lab] = {"path": path, **eval_one(mu, sigma, gt, organs, alpha, seeds,
                                              cal_ratio, min_organ_imgs)}
    return {"config": {"alpha": alpha, "seeds": seeds, "cal_ratio": cal_ratio,
                       "min_organ_imgs": min_organ_imgs, "target": 1 - alpha}, "models": out}


def pretty(res: Dict):
    a = res["config"]["alpha"]; tgt = res["config"]["target"]
    print("\n" + "=" * 88)
    print(f"R2 CONFORMAL (S->S self-recalibrate) | alpha={a} target={tgt:.3f} "
          f"seeds={res['config']['seeds']}")
    print("=" * 88)
    hdr = (f"{'model':10} | {'marg.cov':>8} | {'width':>7} | {'Winkler':>8} | {'MAE':>6} | "
           f"{'worst-org':>9} | {'org-gap':>7} | {'#under':>7}")
    print(hdr); print("-" * len(hdr))
    for lab, d in res["models"].items():
        c = d["conditional"]
        wo = c["worst_organ_coverage"]; gap = c["organ_coverage_gap"]
        wo_s = f"{wo:9.3f}" if wo is not None else f"{'n/a':>9}"
        gap_s = f"{gap:7.3f}" if gap is not None else f"{'n/a':>7}"
        und = f"{c['n_organs_undercovered']}/{c['n_organs_eval']}"
        print(f"{lab:10} | {d['coverage']['mean']:8.3f} | {d['width']['mean']:7.2f} | "
              f"{d['winkler']['mean']:8.2f} | {d['mae']['mean']:6.2f} | {wo_s} | {gap_s} | {und:>7}")
    print("-" * len(hdr))

    labs = list(res["models"].keys())
    if len(labs) == 2:
        base, ours = labs  # giả định thứ tự [baseline, ours]
        b, o = res["models"][base], res["models"][ours]
        wb = b["winkler"]["mean"]; wo_ = o["winkler"]["mean"]
        cb = b["conditional"]["worst_organ_coverage"]; co = o["conditional"]["worst_organ_coverage"]
        print(f"\n[CỔNG] ours='{ours}' vs baseline='{base}'")
        print(f"  Winkler:      {ours}={wo_:.2f}  {base}={wb:.2f}   -> ours {'≤' if wo_ <= wb else '>'} baseline "
              f"({'ĐẠT' if wo_ <= wb else 'TRƯỢT'})")
        if cb is not None and co is not None:
            print(f"  worst-org:    {ours}={co:.3f}  {base}={cb:.3f}   -> ours {'≥' if co >= cb else '<'} baseline "
                  f"({'ĐẠT' if co >= cb else 'TRƯỢT'})")
            passed = (wo_ <= wb) and (co >= cb)
        else:
            print("  worst-org:    n/a (không đủ organ đạt min_organ_imgs) — chỉ xét Winkler")
            passed = wo_ <= wb
        print(f"\n  >>> CỔNG R2: {'ĐẠT ✅ — có method-win, đi tiếp mở rộng' if passed else 'TRƯỢT ❌ — hướng này không phải Q1, đổi chủ đề'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="1 hoặc 2 file (baseline trước, ours sau)")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cal_ratio", type=float, default=0.5)
    ap.add_argument("--min_organ_imgs", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.preds]
    assert len(labels) == len(args.preds), "số --labels phải khớp số --preds"
    res = run(args.preds, labels, args.alpha, args.seeds, args.cal_ratio, args.min_organ_imgs)
    pretty(res)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\nĐã lưu {args.out}")


if __name__ == "__main__":
    main()

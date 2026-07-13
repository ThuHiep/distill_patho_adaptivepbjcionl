"""
eval_condconf_grouped.py — Baseline RECENT: Conditional Conformal (Gibbs, Cherian, Candès,
JRSS-B 2025, arXiv 2305.12616), dùng ĐÚNG package chính thức `conditionalconformal` (CondConf) —
KHÔNG tự chế lại thuật toán (giữ trung thực tuyệt đối). Áp lên CÙNG score/khoảng như R2:
  score S(x,y) = |y − μ(x)| / σ(x)   (μ,σ từ student R2 leak-free đã lưu trong pkl)
  khoảng      = {y : S ≤ Ŝ*} = [μ − Ŝ*·σ,  μ + Ŝ*·σ]   (clip ≥0)   -> score_inv_fn
CondConf giải quantile-regression của score trên basis Φ(x) ở mức (1−α) + hiệu chỉnh hữu hạn mẫu
(imputation điểm test) -> đảm bảo coverage ĐIỀU KIỆN theo mọi shift trong span(Φ).

Đây là SOTA 2025 cho conditional coverage -> đấu TRỰC TIẾP trục worst-org của R2 (clustered
conformal). So sánh 2 basis:
  - marginal : Φ(x)=[1]            -> chỉ marginal (mốc, ≈ split conformal thường).
  - group    : Φ(x)=[1, onehot(organ)] -> đảm bảo coverage TỪNG organ (mode group-conditional
               chuẩn của paper). Đây là baseline recent CHÍNH để so worst-org với R2.

Cùng seeds/cal_ratio/organ_conditional_stats/Winkler như eval_r2_grouped.py -> so được với bảng
mục 8. MAE = |GT−μ| (cùng μ=Σdensity của R2) -> công bằng. exact=True (không RKHS).

Cài (vast hoặc Mac, chỉ cần CPU + numpy/scipy/cvxpy, KHÔNG cần GPU/PathoSAM):
  pip install conditionalconformal
Chạy (trên pkl R2 leak-free đã có):
  python eval_condconf_grouped.py --preds work/student_r2_pannuke_f3_nocolon_poisson.pkl \
      --seeds 10 --alpha 0.1 --min_organ_imgs 10
"""
from __future__ import annotations
import argparse, os, pickle, sys
import numpy as np

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


def eval_condconf(mu, sg, gt, organs, alpha, seeds, cal_ratio, min_organ_imgs, basis):
    """basis: 'marginal' -> Φ=[1]; 'group' -> Φ=[1, onehot(organ)]. Trả dict metrics + per-seed."""
    from conditionalconformal import CondConf  # import trễ (chỉ cần khi chạy method này)
    N = len(mu); target = 1 - alpha
    organs = np.asarray(organs, dtype=object)
    uniq = sorted(set(organs.tolist())); oid = {o: i for i, o in enumerate(uniq)}; K = len(uniq)
    org_id = np.array([oid[o] for o in organs])

    def score_fn(x, y):
        i = np.asarray(x).astype(int).reshape(-1)
        return np.abs(np.asarray(y).reshape(-1) - mu[i]) / sg[i]

    def Phi_fn(x):
        i = np.asarray(x).astype(int).reshape(-1)
        ones = np.ones((len(i), 1))
        if basis == "marginal":
            return ones
        oh = np.zeros((len(i), K)); oh[np.arange(len(i)), org_id[i]] = 1.0
        return np.concatenate([ones, oh], axis=1)  # collinear -> setup_problem tự SVD bỏ hạng thừa

    per_seed, pooled = [], []
    for s in range(seeds):
        rng = np.random.RandomState(1000 + s)
        perm = rng.permutation(N); nc = int(N * cal_ratio)
        cal, tst = perm[:nc], perm[nc:]
        cc = CondConf(score_fn, Phi_fn, seed=s)
        cc.setup_problem(cal.reshape(-1, 1).astype(float), gt[cal])

        def score_inv(Sstar, x, _mu=mu, _sg=sg):
            i = int(np.asarray(x).reshape(-1)[0]); t = float(np.asarray(Sstar).reshape(-1)[0])
            lo = max(0.0, _mu[i] - t * _sg[i]); hi = _mu[i] + t * _sg[i]
            return np.array([lo, hi])

        los, his = [], []
        smax = float(np.abs(gt[cal] - mu[cal]).max() / sg[cal].mean() + 1.0)  # cận an toàn nếu inf
        for i in tst:
            iv = cc.predict(1 - alpha, np.array([[float(i)]]), score_inv, exact=True)
            lo, hi = float(iv[0]), float(iv[1])
            if not np.isfinite(hi):
                hi = mu[i] + smax * sg[i]
            los.append(lo); his.append(hi)
        los, his = np.array(los), np.array(his)
        cov = (gt[tst] >= los) & (gt[tst] <= his)
        wink = np.array([winkler_score(los[j], his[j], gt[tst][j], alpha) for j in range(len(tst))])
        ae = np.abs(gt[tst] - mu[tst])
        per_seed.append({"coverage": float(cov.mean()), "width": float((his - los).mean()),
                         "winkler": float(wink.mean()), "mae": float(ae.mean())})
        for j, ti in enumerate(tst):
            pooled.append((organs[ti], int(ti), bool(cov[j]), float(cov[j])))
    def ms(k): v = np.array([d[k] for d in per_seed]); return {"mean": float(v.mean()), "std": float(v.std())}
    cond = organ_conditional_stats(pooled, target, min_organ_imgs)
    return {"basis": basis, "coverage": ms("coverage"), "width": ms("width"),
            "winkler": ms("winkler"), "mae": ms("mae"), "conditional": cond,
            "winkler_seeds": [d["winkler"] for d in per_seed]}


def pretty(rows, cfg):
    print("\n" + "=" * 92)
    print(f"CONDITIONAL CONFORMAL (Gibbs et al. 2025) | alpha={cfg['alpha']} "
          f"target={1-cfg['alpha']:.3f} seeds={cfg['seeds']} cal_ratio={cfg['cal_ratio']}")
    print("=" * 92)
    hdr = (f"{'basis':18} | {'marg.cov':>8} | {'width':>7} | {'Winkler':>13} | {'MAE':>6} | "
           f"{'worst-org':>9} | {'org-gap':>7} | {'#under':>7}")
    print(hdr); print("-" * len(hdr))
    for lab, d in rows.items():
        cd = d["conditional"]; wo = cd["worst_organ_coverage"]; gap = cd["organ_coverage_gap"]
        wo_s = f"{wo:9.3f}" if wo is not None else f"{'n/a':>9}"
        gap_s = f"{gap:7.3f}" if gap is not None else f"{'n/a':>7}"
        und = f"{cd['n_organs_undercovered']}/{cd['n_organs_eval']}"
        print(f"{lab:18} | {d['coverage']['mean']:8.3f} | {d['width']['mean']:7.2f} | "
              f"{d['winkler']['mean']:6.2f}±{d['winkler']['std']:5.2f} | {d['mae']['mean']:6.2f} | "
              f"{wo_s} | {gap_s} | {und:>7}")
    print("-" * len(hdr))
    print("[GHI CHÚ] CondConf-group = SOTA 2025 conditional coverage. So worst-org/Winkler với "
          "R2-cluster/mondrian (bảng mục 8): kỳ vọng R2 ngang/hơn worst-org Ở CÙNG compute, thắng "
          "Winkler+MAE (R2 có σ học được + đếm chính xác hơn).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="pkl R2 leak-free (có mu,sigma,organs)")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cal_ratio", type=float, default=0.5)
    ap.add_argument("--min_organ_imgs", type=int, default=10)
    ap.add_argument("--basis", choices=["group", "marginal", "both"], default="both")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    mu, sg, gt, organs = load_musigma(args.preds)
    print(f"loaded {args.preds} N={len(mu)} organs={len(set(organs))}")
    bases = ["marginal", "group"] if args.basis == "both" else [args.basis]
    rows = {}
    for b in bases:
        print(f"[condconf] basis={b} ...")
        rows[f"CondConf-{b}"] = eval_condconf(mu, sg, gt, organs, args.alpha, args.seeds,
                                              args.cal_ratio, args.min_organ_imgs, b)
    cfg = {"alpha": args.alpha, "seeds": args.seeds, "cal_ratio": args.cal_ratio}
    pretty(rows, cfg)
    if args.out:
        import json
        json.dump({"config": cfg, "rows": rows}, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\nĐã lưu {args.out}")


if __name__ == "__main__":
    main()

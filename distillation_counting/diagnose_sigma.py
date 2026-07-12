"""
diagnose_sigma.py — Soi vì sao Winkler của R2 cao+bất ổn trên NuInsSeg leak-free.

Câu hỏi:
  (1) σ học được có calibrated theo dải count rộng không? (|err| tăng theo count mà σ phẳng => under-scale)
  (2) r=|gt-μ|/σ có đuôi nặng (vài outlier) làm conformal q phình => interval over-wide?
  (3) HẬU KỲ (không train lại): thay σ bằng các dạng scale-theo-count, conformal y hệt, Winkler đổi sao?
      - σ_learned (R2 hiện tại)
      - σ = sqrt(max(μ,1))         (Poisson: var=mean)
      - σ = const (=1)             (homoscedastic; conformal cho width CỐ ĐỊNH)
      -> nếu Poisson-σ cho Winkler thấp hơn => bõ công retrain với poisson_nll. Nếu không => bất ổn do khác.

Dùng:
  python diagnose_sigma.py --preds work/student_r2_nuinsseg_cv5.pkl --seeds 20 --alpha 0.1
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_r2_conformal import load_mu_sigma, conformal_q  # noqa: E402
from eval_coverage_transfer import winkler_score, organ_conditional_stats  # noqa: E402

EPS = 1e-6


def pctl(x, ps=(0, 5, 25, 50, 75, 95, 99, 100)):
    return {p: float(np.percentile(x, p)) for p in ps}


def conformal_eval(mu, sigma, gt, organs, alpha, seeds, cal_ratio=0.5, min_organ_imgs=10):
    """Split-conformal trên r=|gt-μ|/σ, seeds lần; trả winkler/width/cov + worst-org."""
    N = len(mu); target = 1 - alpha
    W, WID, COV, per_seed_w = [], [], [], []
    pooled = []
    for s in range(seeds):
        rng = np.random.RandomState(1000 + s)
        idx = rng.permutation(N); nc = int(N * cal_ratio)
        cal, tst = idx[:nc], idx[nc:]
        r = np.abs(gt[cal] - mu[cal]) / sigma[cal]
        q = conformal_q(r, alpha)
        lo = np.maximum(0.0, mu[tst] - q * sigma[tst]); hi = mu[tst] + q * sigma[tst]
        cov = (gt[tst] >= lo) & (gt[tst] <= hi)
        wk = np.array([winkler_score(lo[j], hi[j], gt[tst][j], alpha) for j in range(len(tst))])
        W.append(wk.mean()); WID.append((hi - lo).mean()); COV.append(cov.mean())
        per_seed_w.append(wk.mean())
        for j, ti in enumerate(tst):
            pooled.append((organs[ti], int(ti), bool(cov[j]), float(cov[j])))
    cond = organ_conditional_stats(pooled, target, min_organ_imgs)
    return {"winkler": (np.mean(W), np.std(W)), "width": np.mean(WID), "cov": np.mean(COV),
            "worst_org": cond["worst_organ_coverage"], "n_under": cond["n_organs_undercovered"],
            "n_eval": cond["n_organs_eval"], "w_seeds": per_seed_w}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    mu, sigma, gt, organs = load_mu_sigma(args.preds)
    organs = np.asarray(organs, dtype=object)
    err = np.abs(gt - mu)
    r = err / sigma

    print("=" * 78)
    print(f"CHẨN ĐOÁN σ | {args.preds} | N={len(mu)}")
    print("=" * 78)
    print(f"count gt   : min={gt.min():.0f} med={np.median(gt):.0f} max={gt.max():.0f} "
          f"mean={gt.mean():.1f} std={gt.std():.1f}  (dải rộng => cần σ scale theo count)")
    print(f"|err|      : med={np.median(err):.1f} p95={np.percentile(err,95):.1f} max={err.max():.1f}")
    print(f"σ học được : med={np.median(sigma):.1f} p95={np.percentile(sigma,95):.1f} "
          f"min={sigma.min():.2f} max={sigma.max():.1f}")
    print(f"r=|err|/σ  : " + " ".join(f"p{p}={v:.1f}" for p, v in pctl(r).items()))

    # (1) σ có calibrated theo count / theo lỗi không?
    def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
    print("\n[1] CALIBRATION σ:")
    print(f"  corr(count, |err|) = {corr(gt, err):+.2f}  (lỗi có tăng theo count? => heteroscedastic theo count)")
    print(f"  corr(count, σ)     = {corr(gt, sigma):+.2f}  (σ có tăng theo count? nếu thấp => σ KHÔNG scale)")
    print(f"  corr(|err|, σ)     = {corr(err, sigma):+.2f}  (σ có bám lỗi thật? càng cao càng calibrated)")

    # (2) outlier r kéo conformal q
    print("\n[2] OUTLIER kéo conformal q (top-8 r cao nhất):")
    top = np.argsort(r)[::-1][:8]
    for i in top:
        print(f"  organ={str(organs[i])[:16]:16} gt={gt[i]:6.0f} μ={mu[i]:6.1f} "
              f"|err|={err[i]:6.1f} σ={sigma[i]:6.2f} r={r[i]:6.1f}")
    q_full = conformal_q(r, args.alpha)
    q_drop2 = conformal_q(np.sort(r)[:-2], args.alpha)
    print(f"  conformal q (alpha={args.alpha}): full={q_full:.2f} | bỏ 2 outlier={q_drop2:.2f} "
          f"({'ỔN ĐỊNH' if q_drop2 > 0.8*q_full else 'NHẠY OUTLIER -> q phình'})")

    # (3) HẬU KỲ: đổi dạng σ, conformal y hệt, so Winkler
    print("\n[3] HẬU KỲ (không train lại) — đổi DẠNG σ, conformal + Winkler y hệt:")
    variants = {
        "σ_learned (R2)": sigma,
        "σ=sqrt(μ)  Poisson": np.sqrt(np.maximum(mu, 1.0)),
        "σ=const=1  homosced": np.ones_like(sigma),
        "σ=|err|+.. oracle": None,  # tham chiếu trên: dùng chính lỗi (không khả thi thực tế, chỉ để soi trần)
    }
    hdr = f"  {'variant':22} | {'Winkler↓':>14} | {'width':>7} | {'marg.cov':>8} | {'worst-org↑':>10} | under"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, sg in variants.items():
        if sg is None:
            sg = np.maximum(err, EPS)  # oracle σ = |err| (chỉ tham chiếu)
        sg = np.maximum(sg, EPS)
        res = conformal_eval(mu, sg, gt, organs, args.alpha, args.seeds)
        wmean, wstd = res["winkler"]
        print(f"  {name:22} | {wmean:7.1f}±{wstd:5.1f} | {res['width']:7.1f} | "
              f"{res['cov']:8.3f} | {res['worst_org'] if res['worst_org'] is not None else float('nan'):10.3f} | "
              f"{res['n_under']}/{res['n_eval']}")
    print("\n  -> Nếu 'σ=sqrt(μ)' Winkler THẤP hơn σ_learned rõ => retrain với poisson_nll đáng làm.")
    print("  -> Nếu mọi variant vẫn cao/bất ổn => vấn đề ở μ (điểm) hoặc dải count, không chỉ σ.")


if __name__ == "__main__":
    main()

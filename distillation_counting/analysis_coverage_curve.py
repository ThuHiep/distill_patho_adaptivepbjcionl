"""
analysis_coverage_curve.py — Phân tích hậu kỳ cho phản biện Q1 (chạy trên pkl R2 đã có, KHÔNG train lại):
  A1) COVERAGE CURVE nhiều alpha (chống cherry-picking 1 alpha): nominal vs empirical (marginal + worst-org),
      width, Winkler ở alpha ∈ {0.20,0.15,0.10,0.05}.
  A3) PER-ORGAN CI: mỗi organ báo n ảnh, coverage, Wilson 95% CI, width — để biết worst-org 0.773 là
      SYSTEMATIC hay chỉ NHIỄU do organ ít mẫu.

Tái dùng eval_scheme/assign_q của eval_r2_grouped.py → khớp 100% conformal đang dùng (split cal/test per-seed,
cluster fit CHỈ trên cal). Multi-pkl (PanNuke 3 fold) -> trung bình.

Chạy:
  python analysis_coverage_curve.py --preds work/student_r2_nuinsseg_cv5_poisson_feat.pkl --scheme cluster --n_clusters 5
  python analysis_coverage_curve.py --preds "f1.pkl,f2.pkl,f3.pkl" --scheme mondrian
"""
from __future__ import annotations
import argparse, os, pickle, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_r2_grouped import eval_scheme, assign_q  # noqa: E402

EPS = 1e-6


def load(path):
    o = pickle.load(open(path, "rb"))
    mu = np.array([float(p["mu"]) for p in o["preds"]])
    sg = np.maximum(np.array([float(p["sigma"]) for p in o["preds"]]), EPS)
    gt = np.array([float(np.asarray(g).reshape(-1)[0]) for g in o["gts"]])
    org = np.asarray(o.get("organs", ["_all_"] * len(mu)), dtype=object)
    return mu, sg, gt, org


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------- A1: coverage curve ----------
def coverage_curve(pkls, scheme, seeds, cal_ratio, min_group, n_clusters, min_organ_imgs):
    alphas = [0.20, 0.15, 0.10, 0.05]
    print("\n" + "=" * 96)
    print(f"A1 — COVERAGE CURVE ({scheme}, {len(pkls)} pkl avg, seeds={seeds}) | scheme + global baseline")
    print("=" * 96)
    hdr = f"{'nominal':>8} | {'scheme':>9} | {'marg.cov':>9} | {'worst-org':>10} | {'width':>8} | {'Winkler':>9} | {'#under':>7}"
    print(hdr); print("-" * len(hdr))
    for a in alphas:
        for sc in (scheme, "global"):
            covs, wos, wds, wks, unders, neval = [], [], [], [], [], []
            for pk in pkls:
                mu, sg, gt, org = load(pk)
                d = eval_scheme(mu, sg, gt, org, a, seeds, cal_ratio, min_organ_imgs, sc, min_group, n_clusters)
                covs.append(d["coverage"]["mean"]); wds.append(d["width"]["mean"])
                wks.append(d["winkler"]["mean"]); cd = d["conditional"]
                if cd["worst_organ_coverage"] is not None:
                    wos.append(cd["worst_organ_coverage"])
                unders.append(cd["n_organs_undercovered"]); neval.append(cd["n_organs_eval"])
            wo = f"{np.mean(wos):.3f}" if wos else "n/a"
            print(f"{1-a:>8.2f} | {sc:>9} | {np.mean(covs):>9.3f} | {wo:>10} | "
                  f"{np.mean(wds):>8.2f} | {np.mean(wks):>9.2f} | {int(round(np.mean(unders)))}/{int(round(np.mean(neval)))}")
        print("-" * len(hdr))
    print("[GHI CHÚ] target=nominal. marg.cov nên ~nominal mọi alpha (không chỉ 0.90). worst-org < nominal = "
          "undercoverage subgroup. So scheme vs global qua CẢ đường cong, không chỉ 1 điểm.")


# ---------- A3: per-organ coverage + Wilson CI ----------
def per_organ_ci(pkls, scheme, alpha, seeds, cal_ratio, min_group, n_clusters):
    """Point estimate = coverage trung bình qua mọi lần ảnh vào test (seeds). CI Wilson dùng n = SỐ ẢNH THỰC
    của organ (đơn vị độc lập thật) -> phản ánh trung thực organ ít mẫu = CI rất rộng."""
    print("\n" + "=" * 96)
    print(f"A3 — PER-ORGAN COVERAGE + Wilson 95% CI (alpha={alpha}, target={1-alpha:.2f}, {scheme}, seeds={seeds})")
    print("=" * 96)
    # gộp coverage per-organ qua tất cả pkl + seeds
    cov_sum = defaultdict(float); cov_cnt = defaultdict(int)
    wid_sum = defaultdict(float); n_img = defaultdict(int)
    for pk in pkls:
        mu, sg, gt, org = load(pk)
        N = len(mu)
        for o in org:
            n_img[o] += 0  # ensure key
        # đếm số ảnh thực mỗi organ (chia đều cho số pkl để không nhân đôi khi avg fold)
        for o in set(org.tolist()):
            n_img[o] = max(n_img[o], int((org == o).sum()))
        for seed in range(seeds):
            rng = np.random.RandomState(1000 + seed)
            idx = rng.permutation(N); n_cal = int(N * cal_ratio)
            cal, tst = idx[:n_cal], idx[n_cal:]
            r_cal = np.abs(gt[cal] - mu[cal]) / sg[cal]
            qg, qbo = assign_q(scheme, r_cal, org[cal], mu[cal], gt[cal], alpha, min_group, n_clusters)
            for i in tst:
                q = qbo.get(org[i], qg)
                lo = max(0.0, mu[i] - q * sg[i]); hi = mu[i] + q * sg[i]
                c = 1.0 if (gt[i] >= lo and gt[i] <= hi) else 0.0
                cov_sum[org[i]] += c; cov_cnt[org[i]] += 1; wid_sum[org[i]] += (hi - lo)
    hdr = f"{'organ':28} | {'n_img':>5} | {'cov':>6} | {'Wilson 95% CI':>16} | {'width':>8} | flag"
    print(hdr); print("-" * len(hdr))
    rows = []
    for o in sorted(cov_cnt, key=lambda o: cov_sum[o] / max(cov_cnt[o], 1)):
        p = cov_sum[o] / cov_cnt[o]; n = n_img[o]
        lo, hi = wilson(round(p * n), n)   # Wilson trên n ẢNH thực
        w = wid_sum[o] / cov_cnt[o]
        # flag: undercover CHẮC CHẮN nếu cả CI trên < target; NHIỄU nếu CI trùm target
        flag = "UNDER (CI<target)" if hi < (1 - alpha) else ("nhiễu (CI trùm target)" if lo < (1 - alpha) else "ok")
        rows.append((o, n, p, lo, hi, w, flag))
    for o, n, p, lo, hi, w, flag in rows:
        print(f"{o:28} | {n:>5} | {p:>6.3f} | [{lo:.3f}, {hi:.3f}] | {w:>8.2f} | {flag}")
    print("-" * len(hdr))
    n_under = sum(1 for r in rows if r[6].startswith("UNDER"))
    n_noise = sum(1 for r in rows if r[6].startswith("nhiễu"))
    print(f"[KẾT LUẬN A3] {n_under} organ UNDER-cover THẬT (CI trên < target) | {n_noise} organ chỉ NHIỄU "
          f"(CI trùm target, thường do ít ảnh). → worst-org point-estimate thấp KHÔNG đồng nghĩa systematic failure.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="1 pkl, hoặc 'a.pkl,b.pkl,c.pkl' (PanNuke 3 fold -> avg)")
    ap.add_argument("--scheme", choices=["cluster", "mondrian", "global"], default="cluster")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cal_ratio", type=float, default=0.5)
    ap.add_argument("--min_group", type=int, default=15)
    ap.add_argument("--n_clusters", type=int, default=5)
    ap.add_argument("--min_organ_imgs", type=int, default=10)
    ap.add_argument("--alpha_a3", type=float, default=0.10)
    args = ap.parse_args()
    pkls = [p.strip() for p in args.preds.split(",") if p.strip()]
    coverage_curve(pkls, args.scheme, args.seeds, args.cal_ratio, args.min_group, args.n_clusters, args.min_organ_imgs)
    per_organ_ci(pkls, args.scheme, args.alpha_a3, args.seeds, args.cal_ratio, args.min_group, args.n_clusters)


if __name__ == "__main__":
    main()

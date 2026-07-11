"""
eval_r2_grouped.py — Đẩy CONDITIONAL coverage của R2 bằng conformal theo NHÓM trên nền σ heteroscedastic.

Câu hỏi: σ per-image của R2 đã kéo worst-org 0.26->0.51. Thêm hiệu chỉnh quantile THEO NHÓM
(Mondrian) trên nền σ đó có đẩy worst-org cao hơn không, mà Winkler không nổi?

3 scheme calibrate (đều split-conformal, đều dùng score r=|gt-mu|/sigma của R2):
  - global   : 1 quantile q chung (đúng eval_r2_conformal — baseline).
  - mondrian : q RIÊNG cho mỗi organ đủ n_cal>=min_group; organ ít mẫu -> fallback q global.
               (Mondrian conformal, Vovk 2005 — bảo đảm coverage TRONG mỗi nhóm đủ mẫu.)
  - cluster  : gom organ thành n_clusters NHÓM ĐỘ KHÓ theo median residual trên CAL (chỉ dùng cal),
               q riêng mỗi nhóm. Nhóm to hơn organ -> quantile ổn định hơn Mondrian-per-organ khi
               ít mẫu/organ. (group-conditional/clustered conformal, Barber et al. 2020.)

Gán nhóm CHỈ TỪ CAL (không rò rỉ test). Test organ chưa thấy trong cal -> fallback global.

Dùng:
  python eval_r2_grouped.py --preds work/student_r2.pkl --seeds 20 --n_clusters 3 --min_group 15
  # so cả KD (global) làm mốc:
  python eval_r2_grouped.py --preds work/student_r2.pkl --kd work/student_kd.pkl --seeds 20
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from typing import Dict, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coverage_transfer import winkler_score, organ_conditional_stats  # noqa: E402
from eval_r2_conformal import load_mu_sigma, conformal_q  # noqa: E402

EPS = 1e-6


def assign_q(scheme, r_cal, organ_cal, mu_cal, gt_cal, alpha, min_group, n_clusters):
    """Trả (q_global, q_by_organ dict). Áp: test image organ o -> q_by_organ.get(o, q_global)."""
    q_global = conformal_q(r_cal, alpha)
    q_by_organ: Dict[str, float] = {}
    if scheme == "global":
        return q_global, q_by_organ
    # gom residual cal theo organ
    r_by_o = defaultdict(list)
    for r, o in zip(r_cal, organ_cal):
        r_by_o[o].append(r)
    if scheme == "mondrian":
        for o, rs in r_by_o.items():
            if len(rs) >= min_group:
                q_by_organ[o] = conformal_q(np.asarray(rs), alpha)
            # else: không set -> fallback q_global
        return q_global, q_by_organ
    if scheme == "cluster":
        # xếp organ theo median residual (độ khó) rồi chia n_clusters nhóm ~đều theo SỐ ORGAN
        organs_sorted = sorted(r_by_o.keys(), key=lambda o: np.median(r_by_o[o]))
        buckets = np.array_split(organs_sorted, n_clusters)
        for bucket in buckets:
            pooled = np.concatenate([np.asarray(r_by_o[o]) for o in bucket]) if len(bucket) else np.array([])
            if len(pooled) == 0:
                continue
            qb = conformal_q(pooled, alpha)
            for o in bucket:
                q_by_organ[o] = qb
        return q_global, q_by_organ
    raise ValueError(scheme)


def eval_scheme(mu, sigma, gt, organs, alpha, seeds, cal_ratio, min_organ_imgs,
                scheme, min_group, n_clusters) -> Dict:
    N = len(mu); target = 1 - alpha
    per_seed, pooled = [], []
    organs = np.asarray(organs, dtype=object)
    for seed in range(seeds):
        rng = np.random.RandomState(1000 + seed)
        idx = rng.permutation(N)
        n_cal = int(N * cal_ratio)
        cal, tst = idx[:n_cal], idx[n_cal:]
        r_cal = np.abs(gt[cal] - mu[cal]) / sigma[cal]
        q_global, q_by_organ = assign_q(scheme, r_cal, organs[cal], mu[cal], gt[cal],
                                        alpha, min_group, n_clusters)
        q_tst = np.array([q_by_organ.get(organs[i], q_global) for i in tst])
        lo = np.maximum(0.0, mu[tst] - q_tst * sigma[tst])
        hi = mu[tst] + q_tst * sigma[tst]
        cov = (gt[tst] >= lo) & (gt[tst] <= hi)
        width = hi - lo
        wink = np.array([winkler_score(lo[j], hi[j], gt[tst][j], alpha) for j in range(len(tst))])
        ae = np.abs(gt[tst] - mu[tst])
        per_seed.append({"coverage": float(cov.mean()), "width": float(width.mean()),
                         "winkler": float(wink.mean()), "mae": float(ae.mean())})
        for j, ti in enumerate(tst):
            pooled.append((organs[ti], int(ti), bool(cov[j]), float(cov[j])))
    def ms(k): v = np.array([d[k] for d in per_seed]); return {"mean": float(v.mean()), "std": float(v.std())}
    cond = organ_conditional_stats(pooled, target, min_organ_imgs)
    return {"scheme": scheme, "coverage": ms("coverage"), "width": ms("width"),
            "winkler": ms("winkler"), "mae": ms("mae"), "conditional": cond}


def run(r2_path, kd_path, alpha, seeds, cal_ratio, min_organ_imgs, min_group, n_clusters):
    mu, sigma, gt, organs = load_mu_sigma(r2_path)
    rows = {}
    for scheme in ("global", "mondrian", "cluster"):
        rows[f"R2-{scheme}"] = eval_scheme(mu, sigma, gt, organs, alpha, seeds, cal_ratio,
                                           min_organ_imgs, scheme, min_group, n_clusters)
    if kd_path:
        kmu, ksig, kgt, korg = load_mu_sigma(kd_path)
        rows["KD-global"] = eval_scheme(kmu, ksig, kgt, korg, alpha, seeds, cal_ratio,
                                        min_organ_imgs, "global", min_group, n_clusters)
    return {"config": {"alpha": alpha, "seeds": seeds, "cal_ratio": cal_ratio,
                       "min_organ_imgs": min_organ_imgs, "min_group": min_group,
                       "n_clusters": n_clusters, "target": 1 - alpha}, "rows": rows}


def pretty(res):
    c = res["config"]
    print("\n" + "=" * 92)
    print(f"R2 GROUPED CONFORMAL | alpha={c['alpha']} target={c['target']:.3f} seeds={c['seeds']} "
          f"| min_group={c['min_group']} n_clusters={c['n_clusters']}")
    print("=" * 92)
    hdr = (f"{'scheme':12} | {'marg.cov':>8} | {'width':>7} | {'Winkler':>13} | {'MAE':>6} | "
           f"{'worst-org':>9} | {'org-gap':>7} | {'#under':>7}")
    print(hdr); print("-" * len(hdr))
    for lab, d in res["rows"].items():
        cd = d["conditional"]
        wo = cd["worst_organ_coverage"]; gap = cd["organ_coverage_gap"]
        wo_s = f"{wo:9.3f}" if wo is not None else f"{'n/a':>9}"
        gap_s = f"{gap:7.3f}" if gap is not None else f"{'n/a':>7}"
        und = f"{cd['n_organs_undercovered']}/{cd['n_organs_eval']}"
        print(f"{lab:12} | {d['coverage']['mean']:8.3f} | {d['width']['mean']:7.2f} | "
              f"{d['winkler']['mean']:6.2f}±{d['winkler']['std']:5.2f} | {d['mae']['mean']:6.2f} | "
              f"{wo_s} | {gap_s} | {und:>7}")
    print("-" * len(hdr))
    # verdict: scheme R2 nào worst-org cao nhất mà Winkler không tệ hơn R2-global có ý nghĩa
    base = res["rows"]["R2-global"]
    bw, bws = base["winkler"]["mean"], base["winkler"]["std"]
    best, best_wo = None, base["conditional"]["worst_organ_coverage"] or -1
    for lab in ("R2-mondrian", "R2-cluster"):
        d = res["rows"][lab]; wo = d["conditional"]["worst_organ_coverage"]
        noninf = d["winkler"]["mean"] <= bw + bws  # trong 1 std của baseline = không tệ hơn có ý nghĩa
        if wo is not None and wo > best_wo and noninf:
            best, best_wo = lab, wo
    print(f"\n[PHÂN TÍCH] R2-global worst-org={base['conditional']['worst_organ_coverage']}")
    if best:
        print(f"  -> '{best}' đẩy worst-org lên {best_wo:.3f} mà Winkler vẫn non-inferior. Có cải thiện.")
    else:
        print("  -> Không scheme nhóm nào vừa tăng worst-org vừa giữ Winkler. σ per-image của R2 đã là "
              "đòn bẩy chính; Mondrian/cluster không thêm được (ít mẫu/organ). Trung thực ghi nhận.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="student_r2.pkl (có mu,sigma)")
    ap.add_argument("--kd", default=None, help="student_kd.pkl để in mốc")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cal_ratio", type=float, default=0.5)
    ap.add_argument("--min_organ_imgs", type=int, default=10)
    ap.add_argument("--min_group", type=int, default=15, help="n_cal tối thiểu để Mondrian dùng q riêng organ")
    ap.add_argument("--n_clusters", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args.preds, args.kd, args.alpha, args.seeds, args.cal_ratio,
              args.min_organ_imgs, args.min_group, args.n_clusters)
    pretty(res)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\nĐã lưu {args.out}")


if __name__ == "__main__":
    main()

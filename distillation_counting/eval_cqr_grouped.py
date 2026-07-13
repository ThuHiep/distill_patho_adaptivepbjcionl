"""
eval_cqr_grouped.py — Chấm baseline QUANTILE (CQR/CHDQR) bằng ĐÚNG protocol conformal + Winkler +
organ_conditional_stats như eval_r2_grouped.py, để SO TRỰC TIẾP với R2 (bảng mục 8).

Khác eval_r2_grouped: R2 conformal trên score chuẩn hoá r=|y−μ|/σ (khoảng ĐỐI XỨNG μ±q·σ).
CQR (Romano 2019) conformal trên score CHÊNH KHOẢNG (khoảng BẤT ĐỐI XỨNG, native của CQR):
    E_i = max(q_lo(x_i) − y_i,  y_i − q_hi(x_i))            (dấu; âm nếu y trong khoảng dự đoán)
    Q   = quantile hữu hạn mẫu mức ⌈(n_cal+1)(1−α)⌉/n_cal của {E_i}
    khoảng test = [q_lo − Q,  q_hi + Q]   (clip lo>=0)
CHDQR: (q_lo,q_hi) đã là cặp highest-density chọn từ lưới quantile (baselines_uq.predict_quantile),
       conformal hoá y hệt CQR trên cặp đó.

Grouping (đăng ký trước, y hệt eval_r2_grouped): global / mondrian (Q riêng mỗi organ đủ mẫu) /
cluster (gom organ theo median |E| độ khó, Q riêng nhóm). Gán nhóm CHỈ TỪ CAL.

MAE dùng μ (= Σdensity, lưu trong pkl) -> cùng thước với R2. Cùng seeds/cal_ratio/min_organ_imgs
-> split trùng khớp R2 (RandomState(1000+seed) trên cùng N & thứ tự ảnh).

Dùng:
  python eval_cqr_grouped.py --preds work/uq_cqr_pannuke_f3.pkl --kd work/student_kd_pannuke_f3_nocolon.pkl \
         --seeds 20 --n_clusters 5 --min_group 15
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
from collections import defaultdict
from typing import Dict, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coverage_transfer import winkler_score, organ_conditional_stats  # noqa: E402
from eval_r2_conformal import load_mu_sigma, conformal_q  # noqa: E402

EPS = 1e-6


def load_quantiles(path):
    """Trả (mu[N], q_lo[N], q_hi[N], gt[N], organs[N]) từ pkl {preds:[{mu,q_lo,q_hi}],gts,organs}."""
    obj = pickle.load(open(path, "rb"))
    preds = obj["preds"]
    gt = np.array([float(np.asarray(g).reshape(-1)[0]) for g in obj["gts"]])
    organs = list(obj.get("organs", ["_all_"] * len(preds)))
    if len(organs) != len(preds):
        organs = ["_all_"] * len(preds)
    mu = np.array([float(p["mu"]) for p in preds])
    q_lo = np.array([float(p["q_lo"]) for p in preds])
    q_hi = np.array([float(p["q_hi"]) for p in preds])
    return mu, q_lo, q_hi, gt, organs


def assign_Q(scheme, E_cal, organ_cal, alpha, min_group, n_clusters):
    """Trả (Q_global, Q_by_organ). Test organ o -> Q_by_organ.get(o, Q_global). E có DẤU (CQR)."""
    Q_global = conformal_q(E_cal, alpha)  # quantile mức ⌈(n+1)(1−α)⌉/n (method='higher')
    Q_by: Dict[str, float] = {}
    if scheme == "global":
        return Q_global, Q_by
    e_by = defaultdict(list)
    for e, o in zip(E_cal, organ_cal):
        e_by[o].append(e)
    if scheme == "mondrian":
        for o, es in e_by.items():
            if len(es) >= min_group:
                Q_by[o] = conformal_q(np.asarray(es), alpha)
        return Q_global, Q_by
    if scheme == "cluster":
        # xếp organ theo median |E| (độ khó khoảng) rồi chia n_clusters nhóm ~đều theo số organ
        organs_sorted = sorted(e_by.keys(), key=lambda o: np.median(np.abs(e_by[o])))
        for bucket in np.array_split(organs_sorted, n_clusters):
            if len(bucket) == 0:
                continue
            pooled = np.concatenate([np.asarray(e_by[o]) for o in bucket])
            Qb = conformal_q(pooled, alpha)
            for o in bucket:
                Q_by[o] = Qb
        return Q_global, Q_by
    raise ValueError(scheme)


def eval_scheme(mu, q_lo, q_hi, gt, organs, alpha, seeds, cal_ratio, min_organ_imgs,
                scheme, min_group, n_clusters):
    N = len(mu); target = 1 - alpha
    organs = np.asarray(organs, dtype=object)
    per_seed, pooled = [], []
    for seed in range(seeds):
        rng = np.random.RandomState(1000 + seed)
        idx = rng.permutation(N); n_cal = int(N * cal_ratio)
        cal, tst = idx[:n_cal], idx[n_cal:]
        E_cal = np.maximum(q_lo[cal] - gt[cal], gt[cal] - q_hi[cal])   # CQR score (có dấu)
        Q_global, Q_by = assign_Q(scheme, E_cal, organs[cal], alpha, min_group, n_clusters)
        Q_tst = np.array([Q_by.get(organs[i], Q_global) for i in tst])
        lo = np.maximum(0.0, q_lo[tst] - Q_tst); hi = q_hi[tst] + Q_tst
        cov = (gt[tst] >= lo) & (gt[tst] <= hi)
        wink = np.array([winkler_score(lo[j], hi[j], gt[tst][j], alpha) for j in range(len(tst))])
        ae = np.abs(gt[tst] - mu[tst])
        per_seed.append({"coverage": float(cov.mean()), "width": float((hi - lo).mean()),
                         "winkler": float(wink.mean()), "mae": float(ae.mean())})
        for j, ti in enumerate(tst):
            pooled.append((organs[ti], int(ti), bool(cov[j]), float(cov[j])))
    def ms(k): v = np.array([d[k] for d in per_seed]); return {"mean": float(v.mean()), "std": float(v.std())}
    cond = organ_conditional_stats(pooled, target, min_organ_imgs)
    return {"scheme": scheme, "coverage": ms("coverage"), "width": ms("width"),
            "winkler": ms("winkler"), "mae": ms("mae"), "conditional": cond,
            "winkler_seeds": [d["winkler"] for d in per_seed]}


def paired_test(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float); diff = a.mean() - b.mean()
    try:
        from scipy.stats import wilcoxon
        if np.allclose(a, b):
            print(f"  {name}: giống hệt (Δ=0)"); return
        _, p = wilcoxon(a, b)
        sig = "CÓ ý nghĩa (p<0.05)" if p < 0.05 else "CHƯA (p>=0.05)"
        print(f"  {name}: Δ(ours−base)={diff:+.2f}  paired-Wilcoxon p={p:.4g} -> {sig}")
    except Exception as e:
        d = a - b; t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        print(f"  {name}: Δ={diff:+.2f} paired-t≈{t:.2f} ({type(e).__name__})")


def run(cqr_path, kd_path, alpha, seeds, cal_ratio, min_organ_imgs, min_group, n_clusters):
    mu, q_lo, q_hi, gt, organs = load_quantiles(cqr_path)
    rows = {}
    for scheme in ("global", "mondrian", "cluster"):
        rows[f"CQR-{scheme}"] = eval_scheme(mu, q_lo, q_hi, gt, organs, alpha, seeds, cal_ratio,
                                            min_organ_imgs, scheme, min_group, n_clusters)
    if kd_path:
        kmu, ksig, kgt, korg = load_mu_sigma(kd_path)
        # KD dùng khoảng đối xứng μ±q·σ (như eval_r2) -> tự dựng qua CQR-view với q_lo=q_hi=μ? Không.
        # Đơn giản: chấm KD bằng conformal r=|y−μ|/σ global (mốc tham chiếu, giống eval_r2_grouped).
        rows["KD-global"] = _eval_musigma(kmu, ksig, kgt, korg, alpha, seeds, cal_ratio,
                                          min_organ_imgs, min_group, n_clusters, "global")
    return {"config": {"alpha": alpha, "seeds": seeds, "cal_ratio": cal_ratio,
                       "min_organ_imgs": min_organ_imgs, "min_group": min_group,
                       "n_clusters": n_clusters, "target": 1 - alpha}, "rows": rows}


def _eval_musigma(mu, sigma, gt, organs, alpha, seeds, cal_ratio, min_organ_imgs,
                  min_group, n_clusters, scheme):
    """Mốc KD: conformal đối xứng r=|y−μ|/σ (y hệt eval_r2_grouped, để mốc trùng)."""
    N = len(mu); target = 1 - alpha; organs = np.asarray(organs, dtype=object)
    per_seed, pooled = [], []
    for seed in range(seeds):
        rng = np.random.RandomState(1000 + seed)
        idx = rng.permutation(N); n_cal = int(N * cal_ratio)
        cal, tst = idx[:n_cal], idx[n_cal:]
        r = np.abs(gt[cal] - mu[cal]) / sigma[cal]; q = conformal_q(r, alpha)
        lo = np.maximum(0.0, mu[tst] - q * sigma[tst]); hi = mu[tst] + q * sigma[tst]
        cov = (gt[tst] >= lo) & (gt[tst] <= hi)
        wink = np.array([winkler_score(lo[j], hi[j], gt[tst][j], alpha) for j in range(len(tst))])
        ae = np.abs(gt[tst] - mu[tst])
        per_seed.append({"coverage": float(cov.mean()), "width": float((hi - lo).mean()),
                         "winkler": float(wink.mean()), "mae": float(ae.mean())})
        for j, ti in enumerate(tst):
            pooled.append((organs[ti], int(ti), bool(cov[j]), float(cov[j])))
    def ms(k): v = np.array([d[k] for d in per_seed]); return {"mean": float(v.mean()), "std": float(v.std())}
    cond = organ_conditional_stats(pooled, target, min_organ_imgs)
    return {"scheme": scheme, "coverage": ms("coverage"), "width": ms("width"),
            "winkler": ms("winkler"), "mae": ms("mae"), "conditional": cond,
            "winkler_seeds": [d["winkler"] for d in per_seed]}


def pretty(res):
    c = res["config"]
    print("\n" + "=" * 92)
    print(f"CQR/CHDQR GROUPED CONFORMAL | alpha={c['alpha']} target={c['target']:.3f} seeds={c['seeds']} "
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
    if "KD-global" in res["rows"]:
        best = min(("CQR-global", "CQR-mondrian", "CQR-cluster"),
                   key=lambda k: res["rows"][k]["winkler"]["mean"])
        print(f"\n[SIGNIFICANCE] paired per-seed (ours = {best} vs KD-global):")
        paired_test(res["rows"][best]["winkler_seeds"], res["rows"]["KD-global"]["winkler_seeds"],
                    "Winkler vs KD-global")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="pkl CQR/CHDQR {mu,q_lo,q_hi}")
    ap.add_argument("--kd", default=None, help="student_kd.pkl mốc")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cal_ratio", type=float, default=0.5)
    ap.add_argument("--min_organ_imgs", type=int, default=10)
    ap.add_argument("--min_group", type=int, default=15)
    ap.add_argument("--n_clusters", type=int, default=5)
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

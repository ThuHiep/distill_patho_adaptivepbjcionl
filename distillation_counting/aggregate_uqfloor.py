#!/usr/bin/env python3
"""Gom UQ-floor NuInsSeg 5-seed -> bảng mean±sd (scheme cluster n=5, khớp R2).

Đọc 20 pkl work/uq_{ensemble,cqr,chdqr,mcdropout}_nuinsseg_s{42..46}.pkl.
- ensemble/mcdropout = {mu,sigma} -> eval_r2_grouped.run(), lấy hàng "R2-cluster".
- cqr/chdqr        = {mu,q_lo,q_hi} -> eval_cqr_grouped.run(), lấy hàng "CQR-cluster".
In marg.cov / Winkler↓ / worst-org↑ (mean±sd 5 seed) để so với R2 (§4.3).
"""
import numpy as np
import eval_r2_grouped as e_r2
import eval_cqr_grouped as e_cqr

REPO, SEEDS, ALPHA, NC = "/workspace/sam3_research", [42, 43, 44, 45, 46], 0.1, 5
METHODS = {                       # method -> (module, nhãn hàng cluster)
    "ensemble":  (e_r2,  "R2-cluster"),
    "mcdropout": (e_r2,  "R2-cluster"),
    "cqr":       (e_cqr, "CQR-cluster"),
    "chdqr":     (e_cqr, "CQR-cluster"),
}


def one(pkl, mod, label):
    r = mod.run(pkl, None, ALPHA, 20, 0.5, 10, 15, NC)["rows"][label]
    return (r["conditional"]["worst_organ_coverage"],
            r["winkler"]["mean"], r["coverage"]["mean"])


def main():
    print(f"{'method':10} | {'marg.cov':>8} | {'Winkler':>13} | {'worst-org':>13}")
    print("-" * 54)
    for m, (mod, label) in METHODS.items():
        wo, wk, cv = [], [], []
        for s in SEEDS:
            try:
                a, b, c = one(f"{REPO}/work/uq_{m}_nuinsseg_s{s}.pkl", mod, label)
            except Exception as ex:
                print(f"  !! {m} s{s}: {ex}"); continue
            if a is not None:
                wo.append(a)
            wk.append(b); cv.append(c)
        wo, wk, cv = np.array(wo), np.array(wk), np.array(cv)
        print(f"{m:10} | {cv.mean():8.3f} | {wk.mean():6.2f}±{wk.std():5.2f} | "
              f"{wo.mean():.3f}±{wo.std():.3f}")


if __name__ == "__main__":
    main()

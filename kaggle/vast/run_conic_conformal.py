"""
Joint multi-class conformal tren CoNIC (K=6) — chung minh PB-JCI TONG QUAT theo K
(khong an may K=5 PanNuke). Within-taxonomy split conformal: cal/test cung bo 6 lop
-> multi-class song (joint coverage 6 lop), khac han cross-taxonomy (sap ve K=1).

Dung conic_predictions.pkl (build tren test split, TypeHead da train tren split khac
-> khong leakage). Voi moi seed: tach cal/test, fit 3 bien the:
  (1) PB-JCI (joint max-statistic)   -> 1 nguong phu dong thoi 6 lop
  (2) Class-stratified Bonferroni    -> alpha/K moi lop (bao thu)
  (3) Marginal per-class (no correction)
So joint_coverage / per-class coverage / macro-width.

Ket qua ky vong: PB-JCI dat joint >=90% voi width HEP HON Bonferroni -> max-stat hieu qua
hon tren K=6 (giong da thay o K=5). Day la bang da-lop thu hai cho paper.

Run:  python run_conic_conformal.py    (CPU, vai giay; chi can conic_predictions.pkl)
"""
from __future__ import annotations
import os, sys, pickle, argparse
import numpy as np

REPO = "/workspace/sam3_research"
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from conformal import (PBAwareJointConformal, ClassStratifiedConformal,        # noqa: E402
                       coverage_per_class, joint_coverage, macro_width,
                       avg_width_per_class, split_calibration_test)
from conic_loader import CONIC_CLASSES, K                                       # noqa: E402

PKL = f"{REPO}/work/conic_predictions.pkl"
ALPHA = 0.1
SEEDS = 5


def stack_intervals(model, test_preds):
    lo, hi = [], []
    for pr in test_preds:
        l, u = model.predict_interval(pr)
        lo.append(l); hi.append(u)
    return np.asarray(lo), np.asarray(hi)


def eval_model(make, preds, gts):
    """make() -> fresh model with .fit(cal_preds,cal_gt). Tra ve dict metric trung binh 5 seed."""
    jc, mw, pcc = [], [], []
    for sd in range(SEEDS):
        cp, cg, tp, tg = split_calibration_test(preds, gts, cal_ratio=0.5, seed=sd)
        model = make().fit(cp, cg)
        lo, hi = stack_intervals(model, tp)
        gt = np.asarray(tg)
        jc.append(joint_coverage(lo, hi, gt) * 100)
        mw.append(macro_width(lo, hi))
        pcc.append(coverage_per_class(lo, hi, gt) * 100)
    return (np.mean(jc), np.std(jc), np.mean(mw), np.mean(pcc, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=PKL)
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    preds = d["preds"]
    gts = [np.asarray(c, dtype=float) for c in d["gt_counts"]]
    classes = d.get("classes", CONIC_CLASSES)
    print(f"CoNIC preds: {len(preds)} imgs | K={K} | classes={classes}")
    print(f"gt mean/class: {np.asarray(d['gt_counts']).mean(0).round(2)}\n")

    methods = [
        ("PB-JCI (joint max-stat)",      lambda: PBAwareJointConformal(alpha=ALPHA)),
        ("Class-strat Bonferroni",       lambda: ClassStratifiedConformal(alpha=ALPHA, bonferroni=True)),
        ("Marginal per-class (no corr.)", lambda: ClassStratifiedConformal(alpha=ALPHA, bonferroni=False)),
    ]

    print("=" * 92)
    print(f"CoNIC K=6 | within-taxonomy split conformal | target joint {int((1-ALPHA)*100)}% | 5 seed")
    print("=" * 92)
    print(f"{'Method':32s} | {'JointCov':>11s} | {'MacroWidth':>10s} | per-class coverage (%)")
    print("-" * 92)
    for name, make in methods:
        jcm, jcs, mw, pcc = eval_model(make, preds, gts)
        pcs = " ".join(f"{v:4.0f}" for v in pcc)
        print(f"{name:32s} | {jcm:5.1f}+/-{jcs:3.1f}% | {mw:10.2f} | {pcs}")
    print("-" * 92)
    print("Lop:", " ".join(f"{c[:4]}" for c in classes))
    print("KY VONG: PB-JCI joint>=90% voi macro-width HEP hon Bonferroni -> max-stat tot hon tren K=6.")
    print("(Within-taxonomy -> multi-class song; tuong phan cross-taxonomy NuInsSeg sap ve K=1.)")


if __name__ == "__main__":
    main()

"""
Joint multi-class conformal tren MoNuSAC (K=4) — dataset da lop SACH cho PathoSAM
(eval-only, da-organ, KHONG trong Lizard). Chung minh PB-JCI tong quat theo K, KHONG an may
K=5 PanNuke, va KHONG leakage (khac CoNIC/CoNSeP deu bi Lizard nuot).

Within-taxonomy split conformal (cal/test cung 4 lop MoNuSAC). So 3 bien the:
  (1) PB-JCI (joint max-stat)  (2) Class-strat Bonferroni  (3) Marginal.
Ky vong: PB-JCI joint>=90% voi macro-width hep hon Bonferroni.

LUU Y trung thuc: Macrophage/Neutrophil hiem -> it diem calib lop hiem -> khoang lop do
rong/bat on; joint max-stat van phu duoc, va day la diem ban (bat dinh lop hiem).

Run:  python run_monusac_conformal.py    (CPU; can monusac_predictions.pkl)
"""
from __future__ import annotations
import sys, pickle, argparse
import numpy as np

REPO = "/workspace/sam3_research"
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from conformal import (PBAwareJointConformal, ClassStratifiedConformal,        # noqa: E402
                       coverage_per_class, joint_coverage, macro_width,
                       split_calibration_test)
from monusac_loader import MONUSAC_CLASSES, K                                   # noqa: E402

PKL = f"{REPO}/work/monusac_predictions.pkl"
ALPHA = 0.1
SEEDS = 5


def eval_model(make, preds, gts):
    jc, mw, pcc = [], [], []
    for sd in range(SEEDS):
        cp, cg, tp, tg = split_calibration_test(preds, gts, cal_ratio=0.5, seed=sd)
        model = make().fit(cp, cg)
        lo, hi = [], []
        for pr in tp:
            l, u = model.predict_interval(pr); lo.append(l); hi.append(u)
        lo, hi, gt = np.asarray(lo), np.asarray(hi), np.asarray(tg)
        jc.append(joint_coverage(lo, hi, gt) * 100); mw.append(macro_width(lo, hi))
        pcc.append(coverage_per_class(lo, hi, gt) * 100)
    return np.mean(jc), np.std(jc), np.mean(mw), np.mean(pcc, axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pkl", default=PKL)
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    preds = d["preds"]; gts = [np.asarray(c, float) for c in d["gt_counts"]]
    classes = d.get("classes", MONUSAC_CLASSES)
    print(f"MoNuSAC preds: {len(preds)} imgs | K={K} | classes={classes}")
    print(f"gt mean/class: {np.asarray(d['gt_counts']).mean(0).round(2)}\n")

    methods = [
        ("PB-JCI (joint max-stat)",       lambda: PBAwareJointConformal(alpha=ALPHA)),
        ("Class-strat Bonferroni",        lambda: ClassStratifiedConformal(alpha=ALPHA, bonferroni=True)),
        ("Marginal per-class (no corr.)", lambda: ClassStratifiedConformal(alpha=ALPHA, bonferroni=False)),
    ]
    print("=" * 88)
    print(f"MoNuSAC K=4 (SACH) | within-taxonomy split conformal | target {int((1-ALPHA)*100)}% | 5 seed")
    print("=" * 88)
    print(f"{'Method':32s} | {'JointCov':>11s} | {'MacroW':>8s} | per-class coverage (%)")
    print("-" * 88)
    for name, make in methods:
        jcm, jcs, mw, pcc = eval_model(make, preds, gts)
        print(f"{name:32s} | {jcm:5.1f}+/-{jcs:3.1f}% | {mw:8.2f} | " +
              " ".join(f"{v:4.0f}" for v in pcc))
    print("-" * 88)
    print("Lop:", " ".join(c[:4] for c in classes))
    print("KY VONG: PB-JCI joint>=90%, macro-width hep hon Bonferroni. MoNuSAC SACH (PathoSAM khong train).")


if __name__ == "__main__":
    main()

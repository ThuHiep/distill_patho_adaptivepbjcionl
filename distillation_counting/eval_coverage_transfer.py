"""
eval_coverage_transfer.py — Instrument ĐO coverage-transfer + CONDITIONAL (organ-wise) coverage.

CÂU HỎI CHẨN ĐOÁN (xem FEASIBILITY_Q1_ASSESSMENT + THIET_KE_THI_NGHIEM):
    Khi ta hiệu chỉnh (calibrate) lớp conformal trên TEACHER (PathoSAM/SAM3) rồi áp
    NGUYÊN quantile đó lên một STUDENT nhẹ (đã distill), coverage 1-alpha có còn giữ
    không, hay bị vỡ (under-coverage)?

Script này KHÔNG train gì. Nhận hai file prediction (teacher, student) cùng schema
`{ "preds":[{scores,probs,K}], "gts":[[...]], "organs":[...] }` (đúng schema repo:
data/pathosam_nuinsseg_preds.pkl) và đo 3 chế độ:

    (T->T)  q hiệu chỉnh trên teacher, áp lên teacher   -> coverage tham chiếu (~1-alpha)
    (T->S)  q hiệu chỉnh trên teacher, áp lên STUDENT   -> CHẨN ĐOÁN: có vỡ không?
    (S->S)  q hiệu chỉnh trên student, áp lên student   -> recalibrate, nên hồi ~1-alpha

⚠️ LƯU Ý LÝ THUYẾT (xem THIET_KE_THI_NGHIEM mục 2): split conformal bảo đảm MARGINAL
coverage theo cấu tạo. Vì vậy MARGINAL coverage của S->S sẽ ~1-alpha cho MỌI student
(kể cả siêu nhẹ) — KHÔNG dùng nó làm bằng chứng "nén phá coverage". Chỗ coverage THỰC SỰ
có thể vỡ dù đã recalibrate là:
   - TRANSFER (T->S, không recalibrate): kịch bản edge không có nhãn tươi.
   - CONDITIONAL coverage (theo organ/subgroup): conformal KHÔNG bảo đảm coverage đều
     theo nhóm. => script này báo cáo coverage THEO ORGAN, worst-organ, và organ-gap.

Cách dùng:
    # sanity check instrument trên dữ liệu teacher thật (student := teacher)
    python eval_coverage_transfer.py --selftest \
        --teacher ../data/pathosam_nuinsseg_preds.pkl

    # chẩn đoán thật (sau khi có student preds từ distill_student_nuinsseg.py trên vast)
    python eval_coverage_transfer.py \
        --teacher ../data/pathosam_nuinsseg_preds.pkl \
        --student work/student_nuinsseg_preds.pkl \
        --alpha 0.1 --seeds 20 --min_organ 8 --out coverage_transfer_results.json
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

# tái dùng đúng module conformal của repo (không viết lại logic score/quantile)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle", "lib"))
from conformal import (  # noqa: E402
    MarginalSplitConformal, pb_count, pb_variance,
)


def load_pred_pkl(path: str) -> Tuple[List[Dict], List[np.ndarray], List[str]]:
    """Nạp file prediction schema repo. Trả (preds, gts, organs). organs=[] nếu không có."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if "preds" in obj and "gts" in obj:
        preds = obj["preds"]
        gts = [np.asarray(g, dtype=float).reshape(-1) for g in obj["gts"]]
    elif "preds" in obj and "gt_counts" in obj:  # schema monusac
        preds = obj["preds"]
        gts = [np.asarray(g, dtype=float).reshape(-1) for g in obj["gt_counts"]]
    else:
        raise ValueError(f"Không nhận ra schema của {path}: keys={list(obj.keys())}")
    organs = list(obj.get("organs", [])) if isinstance(obj.get("organs", []), (list, tuple)) else []
    if len(organs) != len(preds):
        organs = ["_all_"] * len(preds)  # không có organ -> gộp làm một nhóm
    return preds, gts, organs


def winkler_score(lo: float, hi: float, y: float, alpha: float) -> float:
    """Interval score chuẩn (Gneiting & Raftery 2007) — dùng đúng ở paper 1."""
    w = hi - lo
    if y < lo:
        return w + (2.0 / alpha) * (lo - y)
    if y > hi:
        return w + (2.0 / alpha) * (y - hi)
    return w


def eval_regime(q_per_class: np.ndarray, test_preds: List[Dict],
                test_gts: List[np.ndarray], test_organs: List[str],
                test_idx: List[int], alpha: float) -> Dict:
    """Áp quantile q_per_class lên test_preds, đo coverage marginal + trả per-image để gộp organ."""
    K = len(q_per_class)
    cov_flags, joint_flags, widths, wink, abs_err = [], [], [], [], []
    per_image = []  # (organ, img_idx, joint_covered_bool, marginal_covered_frac)
    for pred, gt, organ, gidx in zip(test_preds, test_gts, test_organs, test_idx):
        s = np.asarray(pred["scores"], dtype=float)
        p = np.asarray(pred["probs"], dtype=float)
        if len(s) == 0:
            n_pred = np.zeros(K)
            sigma = np.ones(K)
        else:
            n_pred = pb_count(s, p)
            sigma = np.sqrt(pb_variance(s, p) + 1e-6)
        lower = np.maximum(0.0, n_pred - q_per_class * sigma)
        upper = n_pred + q_per_class * sigma
        covered_k = (gt >= lower) & (gt <= upper)
        cov_flags.append(covered_k)
        joint_flags.append(bool(covered_k.all()))
        widths.append(upper - lower)
        wink.append([winkler_score(lower[k], upper[k], gt[k], alpha) for k in range(K)])
        abs_err.append(np.abs(gt - n_pred))
        per_image.append((organ, int(gidx), bool(covered_k.all()), float(covered_k.mean())))
    cov_flags = np.asarray(cov_flags)          # (N,K)
    widths = np.asarray(widths)
    wink = np.asarray(wink)
    abs_err = np.asarray(abs_err)
    return {
        "marginal_coverage": float(cov_flags.mean()),
        "per_class_coverage": cov_flags.mean(axis=0).tolist(),
        "joint_coverage": float(np.mean(joint_flags)),
        "macro_width": float(widths.mean()),
        "macro_winkler": float(wink.mean()),
        "mae": float(abs_err.mean()),
        "per_image": per_image,          # để gộp conditional coverage qua nhiều seed
    }


def fit_teacher_q(cal_preds: List[Dict], cal_gts: List[np.ndarray],
                  alpha: float) -> np.ndarray:
    conf = MarginalSplitConformal(alpha=alpha).fit(cal_preds, cal_gts)
    return conf.q_per_class


def organ_conditional_stats(per_image_pooled: List[Tuple[str, int, bool, float]],
                            target: float, min_organ_imgs: int) -> Dict:
    """Gộp coverage theo organ. Pool qua seed để GIẢM PHƯƠNG SAI ước lượng, nhưng lọc theo
    số ẢNH DISTINCT (không phải số pooled) để không thổi phồng độ tin cậy cho organ hiếm."""
    by_organ_joint = defaultdict(list)
    by_organ_marg = defaultdict(list)
    by_organ_imgs = defaultdict(set)
    for organ, gidx, joint_cov, marg_cov in per_image_pooled:
        by_organ_joint[organ].append(1.0 if joint_cov else 0.0)
        by_organ_marg[organ].append(marg_cov)
        by_organ_imgs[organ].add(gidx)
    per_organ = {}
    for organ in by_organ_joint:
        n_imgs = len(by_organ_imgs[organ])
        if n_imgs < min_organ_imgs:
            continue  # quá ít ẢNH distinct -> ước lượng coverage không tin cậy, bỏ
        per_organ[organ] = {
            "n_distinct_imgs": n_imgs,
            "n_pooled": len(by_organ_joint[organ]),
            "joint_coverage": float(np.mean(by_organ_joint[organ])),
            "marginal_coverage": float(np.mean(by_organ_marg[organ])),
        }
    if not per_organ:
        return {"per_organ": {}, "worst_organ_coverage": None,
                "best_organ_coverage": None, "organ_coverage_gap": None,
                "n_organs_undercovered": 0, "n_organs_eval": 0}
    covs = {o: per_organ[o]["joint_coverage"] for o in per_organ}
    worst_o = min(covs, key=covs.get)
    best_o = max(covs, key=covs.get)
    under = sum(1 for c in covs.values() if c < target - 0.05)  # <target-5% coi là under
    return {
        "per_organ": per_organ,
        "worst_organ": worst_o,
        "worst_organ_coverage": float(covs[worst_o]),
        "best_organ_coverage": float(covs[best_o]),
        "organ_coverage_gap": float(covs[best_o] - covs[worst_o]),
        "n_organs_undercovered": int(under),
        "n_organs_eval": len(per_organ),
    }


def run(teacher_path: str, student_path: str, alpha: float, seeds: int,
        cal_ratio: float, min_organ_imgs: int) -> Dict:
    preds_T, gts_T, organs_T = load_pred_pkl(teacher_path)
    preds_S, gts_S, _ = load_pred_pkl(student_path)
    assert len(preds_T) == len(preds_S), (
        f"teacher ({len(preds_T)}) và student ({len(preds_S)}) khác số ảnh — "
        "student phải predict trên ĐÚNG tập ảnh (cùng thứ tự) với teacher.")
    gt_mismatch = sum(1 for a, b in zip(gts_T, gts_S) if not np.allclose(a, b))
    if gt_mismatch > 0:
        print(f"[CẢNH BÁO] {gt_mismatch}/{len(gts_T)} ảnh có GT teacher!=student. "
              "Kiểm tra student có chạy đúng tập/thứ tự ảnh không.")
    N = len(preds_T)
    K = len(np.asarray(gts_T[0]).reshape(-1))
    target = 1 - alpha
    regimes = {r: [] for r in ["T->T", "T->S", "S->S"]}
    pooled = {r: [] for r in ["T->T", "T->S", "S->S"]}  # per-image gộp qua seed cho organ-wise
    for seed in range(seeds):
        rng = np.random.RandomState(1000 + seed)
        idx = rng.permutation(N)
        n_cal = int(N * cal_ratio)
        cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

        calT = [preds_T[i] for i in cal_idx]; calTg = [gts_T[i] for i in cal_idx]
        calS = [preds_S[i] for i in cal_idx]; calSg = [gts_S[i] for i in cal_idx]
        tstT = [preds_T[i] for i in test_idx]; tstTg = [gts_T[i] for i in test_idx]
        tstS = [preds_S[i] for i in test_idx]
        tstOrg = [organs_T[i] for i in test_idx]
        tstIdx = [int(i) for i in test_idx]

        q_T = fit_teacher_q(calT, calTg, alpha)
        q_S = fit_teacher_q(calS, calSg, alpha)

        rT = eval_regime(q_T, tstT, tstTg, tstOrg, tstIdx, alpha)
        rTS = eval_regime(q_T, tstS, tstTg, tstOrg, tstIdx, alpha)   # <-- CHẨN ĐOÁN
        rS = eval_regime(q_S, tstS, tstTg, tstOrg, tstIdx, alpha)
        for name, r in (("T->T", rT), ("T->S", rTS), ("S->S", rS)):
            pooled[name].extend(r.pop("per_image"))
            regimes[name].append(r)

    def agg(list_of_d):
        keys_scalar = ["marginal_coverage", "joint_coverage", "macro_width",
                       "macro_winkler", "mae"]
        out = {}
        for k in keys_scalar:
            vals = np.array([d[k] for d in list_of_d])
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        pcc = np.array([d["per_class_coverage"] for d in list_of_d])
        out["per_class_coverage_mean"] = pcc.mean(axis=0).tolist()
        return out

    summary = {}
    for r in regimes:
        summary[r] = agg(regimes[r])
        summary[r]["conditional"] = organ_conditional_stats(pooled[r], target, min_organ_imgs)
    return {
        "config": {"teacher": teacher_path, "student": student_path,
                   "alpha": alpha, "seeds": seeds, "cal_ratio": cal_ratio,
                   "min_organ_imgs": min_organ_imgs, "N_images": N, "K_classes": K,
                   "n_organs": len(set(organs_T)), "target_coverage": target},
        "results": summary,
    }


def pretty(res: Dict):
    a = res["config"]["alpha"]; tgt = 1 - a
    print("\n" + "=" * 82)
    print(f"COVERAGE-TRANSFER | N={res['config']['N_images']} K={res['config']['K_classes']} "
          f"organs={res['config']['n_organs']} alpha={a} target={tgt:.3f} seeds={res['config']['seeds']}")
    print("=" * 82)
    hdr = (f"{'regime':6} | {'marg.cov':>8} | {'joint':>6} | {'width':>7} | {'Winkler':>8} | "
           f"{'MAE':>6} | {'worst-org':>9} | {'org-gap':>7} | {'#under':>6}")
    print(hdr); print("-" * len(hdr))
    for r in ["T->T", "T->S", "S->S"]:
        d = res["results"][r]; c = d["conditional"]
        wo = c["worst_organ_coverage"]; gap = c["organ_coverage_gap"]
        wo_s = f"{wo:9.3f}" if wo is not None else f"{'n/a':>9}"
        gap_s = f"{gap:7.3f}" if gap is not None else f"{'n/a':>7}"
        und = f"{c['n_organs_undercovered']}/{c['n_organs_eval']}"
        print(f"{r:6} | {d['marginal_coverage']['mean']:8.3f} | {d['joint_coverage']['mean']:6.3f} | "
              f"{d['macro_width']['mean']:7.2f} | {d['macro_winkler']['mean']:8.2f} | "
              f"{d['mae']['mean']:6.2f} | {wo_s} | {gap_s} | {und:>6}")
    print("-" * len(hdr))

    R = res["results"]
    tt = R["T->T"]["marginal_coverage"]["mean"]; ts = R["T->S"]["marginal_coverage"]["mean"]
    print(f"\n[MARGINAL]  Δcov (T->T − T->S) = {tt - ts:+.3f}  "
          f"(dương lớn = transfer làm vỡ marginal coverage)")
    # conditional signal (sống sót đòn 'cứ recalibrate')
    for r in ["T->S", "S->S"]:
        c = R[r]["conditional"]
        if c["worst_organ_coverage"] is not None:
            wo = c.get("worst_organ")
            n_wo = c["per_organ"][wo]["n_distinct_imgs"]
            print(f"[CONDITIONAL {r}] worst-organ cov = {c['worst_organ_coverage']:.3f} "
                  f"(organ='{wo}', {n_wo} ảnh distinct), gap = {c['organ_coverage_gap']:.3f}, "
                  f"under-covered organs = {c['n_organs_undercovered']}/{c['n_organs_eval']}")
    print("\nĐọc kết quả (xem THIET_KE_THI_NGHIEM mục 2-4):")
    print("  - MARGINAL S->S sẽ ~target theo cấu tạo — KHÔNG dùng làm bằng chứng 'nén phá coverage'.")
    print("  - Tín hiệu MẠNH & phòng thủ được: (i) T->S marginal tụt (transfer/không nhãn),")
    print("    (ii) worst-organ cov thấp / organ-gap lớn ngay cả ở S->S (conditional — conformal")
    print("    KHÔNG bảo đảm), (iii) width/Winkler phình theo mức nén.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="student := teacher (kiểm tra instrument, cả 3 chế độ phải ~1-alpha)")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cal_ratio", type=float, default=0.5)
    ap.add_argument("--min_organ_imgs", type=int, default=10,
                    help="số ẢNH DISTINCT tối thiểu để 1 organ được tính conditional coverage")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    student = args.teacher if args.selftest else args.student
    if student is None:
        ap.error("cần --student <pkl> hoặc --selftest")
    res = run(args.teacher, student, args.alpha, args.seeds, args.cal_ratio, args.min_organ_imgs)
    pretty(res)
    if args.selftest:
        print("\n[SELFTEST] student==teacher: cả 3 chế độ marginal phải ~bằng nhau và ~1-alpha, "
              "và worst-organ/gap của 3 chế độ cũng phải giống nhau => instrument đúng trên dữ liệu THẬT.")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\nĐã lưu {args.out}")


if __name__ == "__main__":
    main()

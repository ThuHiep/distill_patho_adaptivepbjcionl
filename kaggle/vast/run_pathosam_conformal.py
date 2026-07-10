"""
Run the PB-JCI conformal benchmark on PathoSAM predictions (CPU, minutes).

Loads work/pathosam_predictions.pkl (built by run_pathosam_build_preds.py) and runs the
SAME 6-method x 4-setting benchmark as Phase C, reusing conformal.py unchanged. This is
the PathoSAM version of the main table — proves PB-JCI Online is predictor-agnostic
(works on a strong clean backbone, not just weak SAM3).

Run (CPU fine, no GPU needed):
  micromamba run -p /workspace/penv python run_pathosam_conformal.py
  # or any python with numpy + scipy + the repo's conformal.py on path
"""
from __future__ import annotations
import os, sys, json, pickle
import numpy as np

REPO = "/workspace/sam3_research"
for p in (f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from conformal import (                                    # noqa: E402
    MarginalSplitConformal, AdaptiveConformalInference, ShiftAwareACI,
    PBAwareJointConformal, PBAwareJointConformalOnline, ClassStratifiedConformal,
    RollingShiftDetector, local_coverage_stats,
    coverage_per_class, joint_coverage, avg_width_per_class,
    pb_count, pb_variance,
)

PKL = f"{REPO}/work/pathosam_predictions.pkl"
OUT_JSON = f"{REPO}/work/pathosam_conformal_results.json"

ALPHA, GAMMA_0, LAMBDA, GAMMA_MAX = 0.1, 0.05, 3.0, 0.15
DETECTOR_WINDOW, PBJCI_WINDOW, LOCAL_WINDOW = 100, 300, 100
EVAL_SETTINGS = ["in_dist", "mild_shift", "severe_shift", "temporal_drift"]
METHODS = ["marginal_split", "aci", "sa_aci", "pb_jci", "pb_jci_online", "class_strat"]
METHOD_NAMES = {
    "marginal_split": "Marginal Split", "aci": "ACI (Gibbs-Candes)",
    "sa_aci": "SA-ACI (Ours)", "pb_jci": "PB-Aware JCI (Ours)",
    "pb_jci_online": "PB-JCI Online (Ours)", "class_strat": "Class-Strat Bonf",
}

with open(PKL, "rb") as f:
    D = pickle.load(f)
predictions_by_setting = D["predictions_by_setting"]
gt_counts = np.asarray(D["gt_counts"])
SETTINGS = list(predictions_by_setting.keys())
print(f"Loaded {PKL} | settings={SETTINGS} | N={len(gt_counts)}")


def get_nonconformity_scores(preds, gt_list):
    out = []
    for p, gt in zip(preds, gt_list):
        if len(p["scores"]) == 0:
            out.append(float(abs(gt).max())); continue
        n_p = pb_count(p["scores"], p["probs"])
        sigma = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
        out.append(max(abs(gt[k] - n_p[k]) / sigma[k] for k in range(5)))
    return np.array(out)


def _interval(p, q, K=5):
    if len(p["scores"]) == 0:
        return np.zeros(K), np.zeros(K)
    n_p = pb_count(p["scores"], p["probs"])
    sigma = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
    return np.maximum(0, n_p - q * sigma), n_p + q * sigma


def _score_one(p, gt, K=5):
    if len(p["scores"]) == 0:
        return float(abs(gt).max())
    n_p = pb_count(p["scores"], p["probs"])
    sigma = np.sqrt(pb_variance(p["scores"], p["probs"]) + 1e-6)
    return max(abs(gt[k] - n_p[k]) / sigma[k] for k in range(K))


def _summary(los, his, covered_list, gt_arr, online=False):
    cov_pc = coverage_per_class(los, his, gt_arr)
    width = avg_width_per_class(los, his)
    jc = float(np.mean(covered_list)) if online else joint_coverage(los, his, gt_arr)
    loc = local_coverage_stats(covered_list, window=LOCAL_WINDOW)
    return {"cov_per_class": cov_pc.tolist(), "marginal_coverage": float(cov_pc.mean()),
            "joint_coverage": jc, "width_per_class": width.tolist(),
            "macro_width": float(width.mean()), **loc}


def eval_static_method(method, test_preds, test_gt):
    los, his = [], []
    for p in test_preds:
        p["K"] = 5
        lo, hi = method.predict_interval(p)
        los.append(lo); his.append(hi)
    los, his, gt_arr = np.array(los), np.array(his), np.array(test_gt)
    covered = ((gt_arr >= los) & (gt_arr <= his)).all(axis=1).tolist()
    return _summary(los, his, covered, gt_arr, online=False)


def eval_aci_method(method, test_preds, test_gt, cal_scores, detector=None):
    method.reset(); method.history_scores = list(cal_scores)
    los, his, cov = [], [], []
    for p, gt in zip(test_preds, test_gt):
        q = method.get_quantile()
        lo, hi = _interval(p, q); los.append(lo); his.append(hi)
        c = bool(((gt >= lo) & (gt <= hi)).all()); cov.append(c)
        S = _score_one(p, gt)
        if isinstance(method, ShiftAwareACI):
            d = detector.step(S) if detector is not None else 0.0
            method.update(S, c, delta_t=d)
        else:
            method.update(S, c)
    return _summary(np.array(los), np.array(his), cov, np.array(test_gt), online=True)


def eval_online_window(method, test_preds, test_gt, cal_scores):
    method.warmstart(cal_scores)
    los, his, cov = [], [], []
    for p, gt in zip(test_preds, test_gt):
        q = method.get_quantile()
        lo, hi = _interval(p, q); los.append(lo); his.append(hi)
        cov.append(bool(((gt >= lo) & (gt <= hi)).all()))
        method.update(_score_one(p, gt))
    return _summary(np.array(los), np.array(his), cov, np.array(test_gt), online=True)


def make_split(n, cal_ratio, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n); n_cal = int(n * cal_ratio)
    return idx[:n_cal], idx[n_cal:]


def run_benchmark(cal_seed, verbose=False):
    n = len(gt_counts)
    cal_idx, test_idx = make_split(n, 0.5, cal_seed)
    cal_preds = [predictions_by_setting["in_dist"][i] for i in cal_idx]
    cal_gt = [gt_counts[i] for i in cal_idx]
    test_gt = [gt_counts[i] for i in test_idx]
    test_preds = {s: [predictions_by_setting[s][i] for i in test_idx] for s in SETTINGS}
    cal_scores = get_nonconformity_scores(cal_preds, cal_gt)

    third = len(test_idx) // 3
    drift = (test_preds["in_dist"][:third] + test_preds["mild_shift"][third:2*third]
             + test_preds["severe_shift"][2*third:])
    streams = {"in_dist": (test_preds["in_dist"], test_gt),
               "mild_shift": (test_preds["mild_shift"], test_gt),
               "severe_shift": (test_preds["severe_shift"], test_gt),
               "temporal_drift": (drift, test_gt)}

    msc = MarginalSplitConformal(alpha=ALPHA).fit(cal_preds, cal_gt)
    pb_jci = PBAwareJointConformal(alpha=ALPHA).fit(cal_preds, cal_gt)
    csc = ClassStratifiedConformal(alpha=ALPHA, bonferroni=True).fit(cal_preds, cal_gt)

    res = {s: {} for s in streams}
    for setting, (tp, tg) in streams.items():
        res[setting]["marginal_split"] = eval_static_method(msc, tp, tg)
        res[setting]["pb_jci"] = eval_static_method(pb_jci, tp, tg)
        res[setting]["class_strat"] = eval_static_method(csc, tp, tg)
        res[setting]["aci"] = eval_aci_method(
            AdaptiveConformalInference(alpha_target=ALPHA, gamma=GAMMA_0), tp, tg, cal_scores)
        det = RollingShiftDetector(window=DETECTOR_WINDOW).fit_baseline(cal_scores)
        res[setting]["sa_aci"] = eval_aci_method(
            ShiftAwareACI(alpha_target=ALPHA, gamma_0=GAMMA_0, lambda_=LAMBDA, gamma_max=GAMMA_MAX),
            tp, tg, cal_scores, detector=det)
        res[setting]["pb_jci_online"] = eval_online_window(
            PBAwareJointConformalOnline(alpha=ALPHA, window=PBJCI_WINDOW), tp, tg, cal_scores)
        if verbose:
            print(f"\n=== {setting} ===")
            for m in METHODS:
                r = res[setting][m]
                print(f"  {m:16s}: marg={r['marginal_coverage']:.3f} "
                      f"joint={r['joint_coverage']:.3f} width={r['macro_width']:7.2f} "
                      f"minLocal={r['min_local_cov']:.3f} missRun={r['max_miss_run']}")
    return res, len(cal_idx), len(test_idx)


def ms(vals):
    a = np.asarray(vals, float); return float(a.mean()), float(a.std())


def main():
    results, n_cal, n_test = run_benchmark(42, verbose=True)
    print(f"\nSingle-seed done. Cal={n_cal} Test={n_test}")

    CAL_SEEDS = [42, 100, 200, 300, 400]
    multi = {s: {m: {"marginal_coverage": [], "joint_coverage": [],
                     "macro_width": [], "min_local_cov": []} for m in METHODS}
             for s in EVAL_SETTINGS}
    for sd in CAL_SEEDS:
        r, _, _ = run_benchmark(sd, verbose=False)
        for s in EVAL_SETTINGS:
            for m in METHODS:
                for key in multi[s][m]:
                    multi[s][m][key].append(r[s][m][key])
        print(f"  seed {sd} done")

    print("\n" + "=" * 110)
    print(f"PATHOSAM PB-JCI TABLE (cal seeds={CAL_SEEDS}) | N_test={n_test} | mean+/-std")
    print("=" * 110)
    print(f"\n{'Setting':<15s} | {'Method':<21s} | {'MargCov':>13s} | {'JointCov':>13s} | {'Width':>14s}")
    print("-" * 110)
    agg = {s: {} for s in EVAL_SETTINGS}
    for s in EVAL_SETTINGS:
        for m in METHODS:
            mc = ms(multi[s][m]["marginal_coverage"]); jc = ms(multi[s][m]["joint_coverage"])
            w = ms(multi[s][m]["macro_width"]); ml = ms(multi[s][m]["min_local_cov"])
            agg[s][m] = {"marg": mc, "joint": jc, "width": w, "min_local": ml}
            print(f"{s:<15s} | {METHOD_NAMES[m]:<21s} | {mc[0]*100:>6.1f}+/-{mc[1]*100:>4.1f}% | "
                  f"{jc[0]*100:>6.1f}+/-{jc[1]*100:>4.1f}% | {w[0]:>7.2f}+/-{w[1]:>5.2f}")
        print("-" * 110)

    with open(OUT_JSON, "w") as f:
        json.dump({"config": {"cal_seeds": CAL_SEEDS, "alpha": ALPHA, "n_test": n_test,
                              "pbjci_window": PBJCI_WINDOW},
                   "aggregate": agg, "raw": multi}, f, indent=2)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()

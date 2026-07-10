"""
Cross-dataset conformal for PathoSAM: calibrate PanNuke (clean) -> test NuInsSeg.

Total-count (K=1). Mirrors the SAM3 cross-dataset experiment but with PathoSAM as the
detector — shows PB-JCI Online generalizes across datasets on a 2nd backbone too.

Inputs (CPU, seconds):
  work/pathosam_predictions.pkl       (PanNuke clean-2228, in_dist -> total = Σ s_i)
  work/pathosam_nuinsseg_preds.pkl    (NuInsSeg, built by run_pathosam_nuinsseg.py)

Run:
  micromamba run -p /workspace/penv python run_pathosam_crossdataset.py
  # (or any python with numpy + conformal.py on path; no GPU needed)
"""
from __future__ import annotations
import os, sys, json, pickle
import numpy as np

REPO = "/workspace/sam3_research"
for p in (f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from conformal import (AdaptiveConformalInference, PBAwareJointConformalOnline,  # noqa
                       empirical_quantile, pb_count, pb_variance)

PAN = f"{REPO}/work/pathosam_predictions.pkl"
NU = f"{REPO}/work/pathosam_nuinsseg_preds.pkl"
OUT = f"{REPO}/work/pathosam_crossdataset_results.json"
ALPHA = 0.1

with open(PAN, "rb") as f:
    dpan = pickle.load(f)
with open(NU, "rb") as f:
    dnu = pickle.load(f)

# PanNuke clean in_dist -> total count (K=1): pred total = Σ s_i, GT total = Σ over 5 classes
pan_src = dpan["predictions_by_setting"]["in_dist"]
pan_gtc = np.asarray(dpan["gt_counts"])
pan_preds = [{"scores": np.asarray(p["scores"]),
              "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in pan_src]
pan_gts = [np.array([float(g.sum())]) for g in pan_gtc]

nu_preds = dnu["preds"]
nu_gts = dnu["gts"]
print(f"PanNuke source: {len(pan_preds)} | GT total mean {np.mean([g[0] for g in pan_gts]):.1f}")
print(f"NuInsSeg target: {len(nu_preds)} | GT total mean {np.mean([g[0] for g in nu_gts]):.1f}")


def nonconf(p, gt):
    if len(p["scores"]) == 0:
        return float(abs(gt[0]))
    n = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return abs(gt[0] - n) / sg


def interval(p, q):
    if len(p["scores"]) == 0:
        return 0.0, 0.0
    n = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return max(0.0, n - q * sg), n + q * sg


def cov_width(preds, gts, q):
    los = np.array([interval(p, q)[0] for p in preds])
    his = np.array([interval(p, q)[1] for p in preds])
    g = np.array([gg[0] for gg in gts])
    return float(np.mean((g >= los) & (g <= his))), float(np.mean(his - los))


pan_scores = np.array([nonconf(pan_preds[i], pan_gts[i]) for i in range(len(pan_preds))])
q_cross = empirical_quantile(pan_scores, ALPHA)
print(f"q (cal on PanNuke clean) = {q_cross:.3f}")

split_cov, split_w = cov_width(nu_preds, nu_gts, q_cross)


def stream(kind, nseeds=5):
    covs, ws = [], []
    for sd in range(nseeds):
        order = np.random.RandomState(sd).permutation(len(nu_preds))
        if kind == "aci":
            m = AdaptiveConformalInference(alpha_target=ALPHA, gamma=0.05)
            m.reset(); m.history_scores = list(pan_scores)
        else:
            m = PBAwareJointConformalOnline(alpha=ALPHA, window=300)
            m.warmstart(pan_scores)
        c, w = [], []
        for i in order:
            q = m.get_quantile(); lo, hi = interval(nu_preds[i], q)
            covered = lo <= nu_gts[i][0] <= hi
            c.append(covered); w.append(hi - lo)
            s = nonconf(nu_preds[i], nu_gts[i])
            m.update(s, covered) if kind == "aci" else m.update(s)
        covs.append(np.mean(c)); ws.append(np.mean(w))
    return float(np.mean(covs)), float(np.std(covs)), float(np.mean(ws)), float(np.std(ws))


def indomain(nseeds=5):
    covs, ws = [], []
    for sd in [42, 100, 200, 300, 400][:nseeds]:
        idx = np.random.RandomState(sd).permutation(len(nu_preds))
        ncal = len(idx) // 2
        cal, test = idx[:ncal], idx[ncal:]
        cs = np.array([nonconf(nu_preds[i], nu_gts[i]) for i in cal])
        q = empirical_quantile(cs, ALPHA)
        c, w = cov_width([nu_preds[i] for i in test], [nu_gts[i] for i in test], q)
        covs.append(c); ws.append(w)
    return float(np.mean(covs)), float(np.std(covs)), float(np.mean(ws)), float(np.std(ws))


aci_c, aci_cs, aci_w, aci_ws = stream("aci")
pbo_c, pbo_cs, pbo_w, pbo_ws = stream("pbo")
id_c, id_cs, id_w, id_ws = indomain()

print("\n" + "=" * 78)
print("PathoSAM CROSS-DATASET: PanNuke (cal) -> NuInsSeg (test) | total count, alpha=0.1")
print("=" * 78)
print(f"{'Setting / Method':38s} | {'Coverage':>14s} | {'Width':>10s}")
print("-" * 78)
print(f"{'In-domain split (cal NuInsSeg)':38s} | {id_c*100:>6.1f}+/-{id_cs*100:<4.1f}% | {id_w:>7.2f}")
print(f"{'Cross split (cal PanNuke, no adapt)':38s} | {split_cov*100:>11.1f}% | {split_w:>7.2f}")
print(f"{'Cross ACI (stream feedback)':38s} | {aci_c*100:>6.1f}+/-{aci_cs*100:<4.1f}% | {aci_w:>7.2f}")
print(f"{'Cross PB-JCI Online (stream)':38s} | {pbo_c*100:>6.1f}+/-{pbo_cs*100:<4.1f}% | {pbo_w:>7.2f}")
print("-" * 78)
print(f"\nCoverage drop (in-domain -> cross split): {(id_c-split_cov)*100:+.1f} pp")

with open(OUT, "w") as f:
    json.dump({"in_domain_split": {"coverage": [id_c, id_cs], "width": [id_w, id_ws]},
               "cross_split": {"coverage": split_cov, "width": split_w},
               "cross_aci": {"coverage": [aci_c, aci_cs], "width": [aci_w, aci_ws]},
               "cross_pbjci_online": {"coverage": [pbo_c, pbo_cs], "width": [pbo_w, pbo_ws]},
               "q_cross": float(q_cross), "alpha": ALPHA}, f, indent=2)
print(f"Saved {OUT}")

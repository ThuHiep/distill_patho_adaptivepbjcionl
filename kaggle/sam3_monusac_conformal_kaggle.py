import pickle
from pathlib import Path
import numpy as np

ALPHA = 0.1
SEEDS = 5

def _find(name, roots=("/kaggle/input", "data", "work", ".")):
    for root in roots:
        base = Path(root)
        hits = list(base.rglob(name)) if base.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under {roots}; set the path manually.")

PKL = _find("monusac_predictions.pkl")
print("PKL:", PKL)

def empirical_quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        return float("inf")
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))

def pb_count(scores, probs):
    return (scores[:, None] * probs).sum(axis=0)

def pb_variance(scores, probs):
    w = scores[:, None] * probs
    return (w * (1.0 - w)).sum(axis=0)

class PBAwareJointConformal:

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = 0.0

    def fit(self, cal_preds, cal_gt):
        scores = []
        for pred, gt in zip(cal_preds, cal_gt):
            s, p = pred["scores"], pred["probs"]
            K = len(gt)
            if len(s) == 0:
                S_t = max(abs(gt[k]) / 1.0 for k in range(K))
            else:
                n = pb_count(s, p)
                sg = np.sqrt(pb_variance(s, p) + 1e-6)
                S_t = max(abs(gt[k] - n[k]) / sg[k] for k in range(K))
            scores.append(S_t)
        self.q = empirical_quantile(np.array(scores), self.alpha)
        return self

    def predict_interval(self, pred):
        s, p = pred["scores"], pred["probs"]
        K = pred.get("K", 4)
        if len(s) == 0:
            return np.zeros(K), np.zeros(K)
        n = pb_count(s, p)
        sg = np.sqrt(pb_variance(s, p) + 1e-6)
        return np.maximum(0, n - self.q * sg), n + self.q * sg

class ClassStratifiedConformal:

    def __init__(self, alpha=0.1, bonferroni=True):
        self.alpha = alpha
        self.bonferroni = bonferroni
        self.q_per_class = None

    def fit(self, cal_preds, cal_gt):
        K = cal_gt[0].shape[0]
        alpha_eff = self.alpha / K if self.bonferroni else self.alpha
        spc = [[] for _ in range(K)]
        for pred, gt in zip(cal_preds, cal_gt):
            s, p = pred["scores"], pred["probs"]
            if len(s) == 0:
                continue
            n = pb_count(s, p)
            sg = np.sqrt(pb_variance(s, p) + 1e-6)
            for k in range(K):
                if gt[k] > 0:
                    spc[k].append(abs(gt[k] - n[k]) / sg[k])
        self.q_per_class = np.array([
            empirical_quantile(np.array(spc[k]) if spc[k] else np.array([1.0]), alpha_eff)
            for k in range(K)
        ])
        return self

    def predict_interval(self, pred):
        K = len(self.q_per_class)
        s, p = pred["scores"], pred["probs"]
        if len(s) == 0:
            return np.zeros(K), np.zeros(K)
        n = pb_count(s, p)
        sg = np.sqrt(pb_variance(s, p) + 1e-6)
        return np.maximum(0, n - self.q_per_class * sg), n + self.q_per_class * sg

def joint_coverage(lo, hi, gt):
    return float((((gt >= lo) & (gt <= hi)).all(axis=1)).mean())

def coverage_per_class(lo, hi, gt):
    return ((gt >= lo) & (gt <= hi)).mean(axis=0)

def macro_width(lo, hi):
    return float((hi - lo).mean(axis=0).mean())

def split_calibration_test(preds, gts, cal_ratio=0.5, seed=42):
    n = len(preds)
    idx = np.random.RandomState(seed).permutation(n)
    nc = int(n * cal_ratio)
    ci, ti = idx[:nc], idx[nc:]
    return ([preds[i] for i in ci], [gts[i] for i in ci],
            [preds[i] for i in ti], [gts[i] for i in ti])

with open(PKL, "rb") as f:
    d = pickle.load(f)
PREDS = d["preds"]
GTS = [np.asarray(c, float) for c in d["gt_counts"]]
CLASSES = d.get("classes", ["Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"])
K = len(GTS[0])
print(f"MoNuSAC: {len(PREDS)} imgs | K={K} | classes={CLASSES}")
print(f"gt mean/class: {np.asarray(d['gt_counts']).mean(0).round(2)}\n")

def eval_model(make):
    jc, mw, pcc = [], [], []
    for sd in range(SEEDS):
        cp, cg, tp, tg = split_calibration_test(PREDS, GTS, 0.5, sd)
        m = make().fit(cp, cg)
        lo, hi = zip(*[m.predict_interval(pr) for pr in tp])
        lo, hi, gt = np.asarray(lo), np.asarray(hi), np.asarray(tg)
        jc.append(joint_coverage(lo, hi, gt) * 100)
        mw.append(macro_width(lo, hi))
        pcc.append(coverage_per_class(lo, hi, gt) * 100)
    return np.mean(jc), np.std(jc), np.mean(mw), np.mean(pcc, axis=0)

METHODS = [
    ("Marginal Split",         lambda: ClassStratifiedConformal(ALPHA, bonferroni=False)),
    ("Class-Strat Bonferroni", lambda: ClassStratifiedConformal(ALPHA, bonferroni=True)),
    ("PB-JCI Max-Statistic",   lambda: PBAwareJointConformal(ALPHA)),
]

print("=" * 86)
print(f"MoNuSAC K={K} | within-taxonomy split conformal | target {int((1-ALPHA)*100)}% joint | {SEEDS} seeds")
print("=" * 86)
print(f"{'Method':24s} | {'Joint cov.':>12s} | {'Macro-width':>11s} | per-class cov (%)")
print("-" * 86)
for name, make in METHODS:
    jcm, jcs, mw, pcc = eval_model(make)
    print(f"{name:24s} | {jcm:5.1f}+/-{jcs:3.1f}% | {mw:11.2f} | " +
          " ".join(f"{v:4.0f}" for v in pcc))
print("-" * 86)
print("Classes:", " ".join(c[:4] for c in CLASSES))
print("Joint = all K classes covered simultaneously; macro-width = mean interval width over classes.")

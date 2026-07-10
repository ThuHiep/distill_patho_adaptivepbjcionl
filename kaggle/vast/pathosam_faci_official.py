"""FACI (Fully Adaptive Conformal Inference, Gibbs & Candes 2024, JMLR 25:22-1218)
port VERBATIM tu salesforce/online_conformal/faci.py (Apache-2.0) ve score-space,
cung harness/nonconf/interval/winkler voi pathosam_modern_baselines.py.

FACICore = lop FACI (ban quantile) chep nguyen van logic predict/update tu source:
  gammas=[0.001*2^k]_{k=0..7}, alphas init = 1-cov, log_w=0, I=100, sigma=1/(2I),
  eta = sqrt(3/I)*sqrt((log(I*k)+2)/denom),  denom=((1-a)^2 a^3 + a^2 (1-a)^3)/3.
predict: alpha_t = <softmax(log_w), alphas>; delta = quantile(|resid|, 1-alpha_t).
update(s): beta=mean(resid>=s); pinball loss -> mix weights (logsumexp); alphas adapt.
Warm-start: residuals seed = calibration scores (PanNuke), nhu base.BasePredictor.

So sanh tren CUNG 5 seed, CUNG geometry (n +/- q*sigma) voi PB-JCI/ACI/SAOCP.
CPU, pkl cached.  python pathosam_faci_official.py
"""
from __future__ import annotations
import sys, math, pickle
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kaggle" / "lib"))
sys.path.insert(0, str(REPO / ".saocp_src"))
from conformal import empirical_quantile, pb_count, pb_variance  # noqa
from utils import quantile as oc_quantile, pinball_loss  # noqa  (official online_conformal utils)

ALPHA = 0.1
COVERAGE = 1.0 - ALPHA


def load():
    d = REPO / "data"
    with open(d / "pathosam_predictions.pkl", "rb") as f:
        dpan = pickle.load(f)
    with open(d / "pathosam_nuinsseg_preds.pkl", "rb") as f:
        dnu = pickle.load(f)
    pan_src = dpan["predictions_by_setting"]["in_dist"]
    pan_gtc = np.asarray(dpan["gt_counts"])
    pan_preds = [{"scores": np.asarray(p["scores"]),
                  "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in pan_src]
    pan_gts = [np.array([float(g.sum())]) for g in pan_gtc]
    return pan_preds, pan_gts, dnu["preds"], dnu["gts"]


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


def winkler(lo, hi, y, alpha=ALPHA):
    s = hi - lo
    if y < lo:
        s += (2.0 / alpha) * (lo - y)
    elif y > hi:
        s += (2.0 / alpha) * (y - hi)
    return s


# ===================== port verbatim: faci.py::FACI (quantile version) =====================
class FACICore:
    def __init__(self, calib_resids, coverage=COVERAGE):
        self.coverage = coverage
        self.gammas = np.asarray([0.001 * 2 ** k for k in range(8)])
        self.k = len(self.gammas)
        self.alphas = np.full(self.k, 1 - coverage)
        self.log_w = np.zeros(self.k)
        self.I = 100
        self.sigma = 1.0 / (2 * self.I)
        a = 1 - coverage
        denom = ((1 - a) ** 2 * a ** 3 + a ** 2 * (1 - a) ** 3) / 3
        self.eta = np.sqrt(3 / self.I) * np.sqrt((np.log(self.I * self.k) + 2) / denom)
        self.residuals = [float(r) for r in calib_resids]   # warm-start = calibration

    def predict(self):
        log_w = self.log_w
        alpha = np.dot(np.exp(log_w - logsumexp(log_w)), self.alphas)
        return float(oc_quantile(np.abs(np.asarray(self.residuals)), 1 - alpha))

    def update(self, s):
        res = self.residuals
        if len(res) > math.floor(1 / (1 - self.coverage)):
            beta = float(np.mean(np.asarray(res) >= s))
            losses = pinball_loss(beta, self.alphas, 1 - self.coverage)
            wbar = self.log_w - self.eta * losses
            self.log_w = logsumexp(
                [wbar, np.full(self.k, logsumexp(wbar))],
                b=[[1 - self.sigma], [self.sigma / self.k]], axis=0)
            self.log_w = self.log_w - logsumexp(self.log_w)
            err = self.alphas > beta
            self.alphas = np.clip(self.alphas + self.gammas * ((1 - self.coverage) - err), 0, 1)
        res.append(float(s))


PAN_PREDS, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_SCORES = np.array([nonconf(PAN_PREDS[i], PAN_GTS[i]) for i in range(len(PAN_PREDS))])
T = len(NU_PREDS)


def faci_run(order):
    m = FACICore(PAN_SCORES)
    c, w, wk = [], [], []
    for i in order:
        q = m.predict()
        lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); wk.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, wk


def agg(fn, seeds=5):
    cs, ws, ks = [], [], []
    for sd in range(seeds):
        order = np.random.RandomState(sd).permutation(T)
        c, w, k = fn(order)
        cs.append(np.mean(c) * 100); ws.append(np.mean(w)); ks.append(np.mean(k))
    return np.mean(cs), np.std(cs), np.mean(ws), np.std(ws), np.mean(ks), np.std(ks)


if __name__ == "__main__":
    print(f"PanNuke cal {len(PAN_SCORES)} | NuInsSeg stream T={T} | target {COVERAGE*100:.0f}%")
    print("=" * 72)
    cm, cs, wm, wsd, km, ksd = agg(faci_run)
    print(f"{'FACI (Gibbs-Candes JMLR 2024, official)':40s}")
    print(f"  Coverage = {cm:5.1f} +/- {cs:4.1f} %")
    print(f"  Width    = {wm:6.2f} +/- {wsd:4.2f}")
    print(f"  Winkler  = {km:7.2f} +/- {ksd:4.2f}  (lower=better)")

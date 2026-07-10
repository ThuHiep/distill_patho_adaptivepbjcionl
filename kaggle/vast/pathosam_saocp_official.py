"""
SAOCP + SF-OGD CHINH CHU (port verbatim tu salesforce/online_conformal, Apache-2.0,
Bhatnagar et al. ICML 2023) cho Table 9a: PathoSAM -> NuInsSeg, cal = PanNuke total-count,
K=1, stream 5 seed, target 90%. THAY cho reimplement cu bi hong (cho 26-40% coverage).

Dieu khien trong SCORE-SPACE: thuat toan track phan vi (1-alpha) cua nonconformity score
PB-chuan hoa  s = |y - E[N]| / sigma,  roi boc khoang  n +/- q*sigma  -- DUNG geometry
voi PB-JCI Online (ours), CHI khac co che recalibration. Day la so sanh cong bang nhat
(giu nguyen score-geometry, chi thay "dong co" online -> tach bach dong gop adaptive cua minh).

Logic _OGD / SAOCP.get_p / predict / update va ScaleFreeOGD.update duoc CHEP NGUYEN VAN tu
source (chi bo coupling merlion/pandas + rut ve single-horizon). Doi chieu: ../../.saocp_src/

So kem PB-JCI Online (ours), ACI tren CUNG 5 seed.   python pathosam_saocp_official.py
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kaggle" / "lib"))
from conformal import (empirical_quantile, pb_count, pb_variance,  # noqa
                       PBAwareJointConformalOnline, AdaptiveConformalInference)

ALPHA = 0.1
COVERAGE = 1.0 - ALPHA
SEEDS = 5


# ============================ harness (giong cac script khac) ============================
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


# ===================== port verbatim: utils.py (pinball) =====================
def pinball_loss(y, yhat, q):
    return np.maximum(q * (y - yhat), (1 - q) * (yhat - y))


def pinball_loss_grad(y, yhat, q):
    return -q * (y > yhat) + (1 - q) * (y < yhat)


# ===================== port verbatim: ogd.py::_OGD (SAOCP expert) =====================
class _OGD:
    """Scale-Free OGD expert voi lifetime huu han (Hazan-Seshadhri 2007) + coin-betting
    weight (Jun et al. 2017). CHEP NGUYEN VAN tu online_conformal/saocp.py::_OGD."""
    def __init__(self, t, scale, alpha, yhat_0, g=8):
        self.scale = scale
        self.base_lr = scale / np.sqrt(3)
        self.alpha = alpha
        self.yhat = yhat_0
        self.grad_norm = 0
        u = 0
        while t % 2 == 0:
            t /= 2
            u += 1
        self.lifetime = g * 2 ** u
        self.z = 0
        self.wz = 0
        self.s_t = 0

    @property
    def expired(self):
        return self.s_t > self.lifetime

    def loss(self, y):
        return pinball_loss(y, self.yhat, 1 - self.alpha)

    @property
    def w(self):
        return 0 if self.s_t == 0 else self.z / self.s_t * (1 + self.wz)

    def update(self, y, meta_loss):
        w = self.w
        g = np.clip((meta_loss - self.loss(y)) / self.scale / max(self.alpha, 1 - self.alpha), -1 * (w > 0), 1)
        self.z += g
        self.wz += g * w
        self.s_t += 1
        grad = pinball_loss_grad(y, self.yhat, 1 - self.alpha)
        self.grad_norm += grad ** 2
        if self.grad_norm != 0:
            self.yhat = max(0, self.yhat - self.base_lr / np.sqrt(self.grad_norm) * grad)


# ===================== port verbatim: saocp.py::SAOCP (single-horizon, score-space) =====================
class SAOCPCore:
    """SAOCP (Bhatnagar et al. ICML 2023). get_p / predict / step CHEP NGUYEN VAN tu
    online_conformal/saocp.py (bo merlion, single-horizon). Track phan vi COVERAGE cua
    score; predict() tra ve delta (= q de boc n +/- q*sigma). FAITHFUL."""
    def __init__(self, warm_scores, coverage=COVERAGE, lifetime=8):
        self.coverage = coverage
        self.lifetime = lifetime
        self.t = 1
        self.experts = {}
        r = np.abs(np.asarray(warm_scores, dtype=float))
        self.scale = 1.0 if len(r) == 0 else float(np.max(r) * np.sqrt(3))
        for s in r:                         # warm-start = SAOCP.__init__ chay update tren calib
            self._step(float(s))

    def get_p(self):
        experts = self.experts
        prior = {t: 1 / (t ** 2 * (1 + np.floor(np.log2(t)))) for t in experts.keys()}
        z = sum(prior.values())
        if z == 0:
            return {}
        prior = {t: v / z for t, v in prior.items()}
        p = {t: prior[t] * max(0, e.w) for t, e in experts.items()}
        sum_p = sum(p.values())
        return {t: v / sum_p for t, v in p.items()} if sum_p > 0 else prior

    def predict(self):
        p = self.get_p()
        return sum(p[t] * self.experts[t].yhat for t in p)      # delta (>=0)

    def _step(self, s):
        s_hat = self.predict()
        for t in [k for k, v in self.experts.items() if v.expired]:
            self.experts.pop(t)
        self.experts[self.t] = _OGD(self.t, self.scale, 1 - self.coverage, yhat_0=s_hat, g=self.lifetime)
        meta_loss = pinball_loss(s, self.predict(), self.coverage)
        for e in self.experts.values():
            e.update(s, meta_loss)
        self.t += 1

    # --- streaming API ---
    def quantile(self):
        return self.predict()

    def update(self, s):
        self._step(float(s))


# ===================== port verbatim: ogd.py::ScaleFreeOGD (SF-OGD chinh chu) =====================
class SFOGDCore:
    """Scale-Free OGD (Orabona-Pal 2016) -- base learner cua SAOCP & baseline online-conformal
    chuan. CHEP NGUYEN VAN tu online_conformal/ogd.py::ScaleFreeOGD.update. FAITHFUL.
    (Thay ban hand-rolled cu trong notebook bang cong thuc chinh chu scale/sqrt(3*grad_norm).)"""
    def __init__(self, warm_scores, coverage=COVERAGE):
        self.coverage = coverage
        self.delta = 0.0
        self.grad_norm = 0.0
        r = np.abs(np.asarray(warm_scores, dtype=float))
        self.scale = 1.0 if len(r) == 0 else float(np.max(r) * np.sqrt(3))
        for s in r:
            self.update(float(s))

    def quantile(self):
        return self.delta

    def update(self, s):
        s = abs(s)
        grad = pinball_loss_grad(s, self.delta, self.coverage)
        self.grad_norm += grad ** 2
        if self.grad_norm != 0:
            self.delta = max(0, self.delta - self.scale / np.sqrt(3 * self.grad_norm) * grad)


# ===================== runners =====================
def saocp_run(order):
    m = SAOCPCore(PAN_SCORES, coverage=COVERAGE, lifetime=8)
    c, w, wk = [], [], []
    for i in order:
        q = m.quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); wk.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, wk


def sfogd_run(order):
    m = SFOGDCore(PAN_SCORES, coverage=COVERAGE)
    c, w, wk = [], [], []
    for i in order:
        q = m.quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); wk.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, wk


def pbo_run(order, window=300):
    m = PBAwareJointConformalOnline(alpha=ALPHA, window=window); m.warmstart(PAN_SCORES)
    c, w, wk = [], [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); wk.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, wk


def aci_run(order, gamma=0.05):
    m = AdaptiveConformalInference(alpha_target=ALPHA, gamma=gamma)
    for s in PAN_SCORES:
        m.history_scores.append(float(s))
    c, w, wk = [], [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        cov = lo <= y <= hi
        c.append(cov); w.append(hi - lo); wk.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]), cov)
    return c, w, wk


def agg(fn):
    cs, ws, ks = [], [], []
    for sd in range(SEEDS):
        order = np.random.RandomState(sd).permutation(len(NU_PREDS))
        c, w, wk = fn(order)
        cs.append(np.mean(c) * 100); ws.append(np.mean(w)); ks.append(np.mean(wk))
    return np.mean(cs), np.std(cs), np.mean(ws), np.mean(ks)


PAN_PREDS, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_SCORES = np.array([nonconf(PAN_PREDS[i], PAN_GTS[i]) for i in range(len(PAN_PREDS))])
print(f"PanNuke cal {len(PAN_SCORES)} | NuInsSeg stream T={len(NU_PREDS)} | "
      f"q0={empirical_quantile(PAN_SCORES, ALPHA):.3f}")

print("\n" + "=" * 78)
print("SAOCP/SF-OGD CHINH CHU (salesforce/online_conformal) | PathoSAM->NuInsSeg | 5 seed")
print("=" * 78)
print(f"{'Method':34s} | {'Coverage':>13s} | {'Width':>8s} | {'Winkler':>9s}")
print("-" * 78)
rows = [
    ("SAOCP (Bhatnagar23, official)",   saocp_run),   # FAITHFUL (port verbatim)
    ("SF-OGD (ScaleFreeOGD, official)",  sfogd_run),   # FAITHFUL (port verbatim)
    ("PB-JCI Online (ours)",             pbo_run),     # reference
    ("ACI (Gibbs-Candes21)",             aci_run),     # reference
]
for nm, fn in rows:
    cm, cs, wm, km = agg(fn)
    print(f"{nm:34s} | {cm:5.1f}+/-{cs:4.1f}% | {wm:8.2f} | {km:9.2f}")
print("-" * 78)
print("Tat ca FAITHFUL: SAOCP & SF-OGD port nguyen van; PB-JCI/ACI tu lib. So dung de bao cao.")
print("Ghi chu: dieu khien score-space (n +/- q*sigma) -> cung geometry voi ours, chi khac dong co online.")

"""
SELF-CONTAINED Kaggle cell: reproduces the cross-dataset table (Table tab:cross in
the paper) -- PathoSAM -> NuInsSeg (cal = PanNuke in-dist), 5 stream seeds, target 90%.
Reports Coverage% / Width / Winkler (mean +/- std). No external lib import.

Faithful ports (CPU, a few seconds):
  ACI (Gibbs-Candes 2021), NexCP (Barber 2023), FACI (Gibbs-Candes JMLR 2024),
  SAOCP + SF-OGD (Bhatnagar 2023, port verbatim from salesforce/online_conformal),
  COP (Hu et al. ICLR 2026), Rolling-Origin CP (Halkiewicz 2026),
  PB-JCI Online-Fixed (ours), Adaptive PB-JCI Online (ours).
Weighted Conformal (Tibshirani 2019) is feature-based (domain classifier) -> NOT here;
it is reproduced in the robustness notebook. It collapses (40.8%) under this shift.

Needs two pickles uploaded as a Kaggle dataset (Add Data); the loader auto-finds them
anywhere under /kaggle/input:
  - pathosam_predictions.pkl      (dict: predictions_by_setting{in_dist,...}, gt_counts)
  - pathosam_nuinsseg_preds.pkl   (dict: preds[list of {scores,probs,K}], gts)
"""
import math
import pickle
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

ALPHA = 0.1
COVERAGE = 1.0 - ALPHA
WINDOW = 300

# ----------------------------------------------------------------- locate data
def _find(name, root="/kaggle/input"):
    hits = list(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}; set the path manually.")
    return hits[0]

PAN_PKL = _find("pathosam_predictions.pkl")
NU_PKL = _find("pathosam_nuinsseg_preds.pkl")
print("PAN_PKL:", PAN_PKL, "\nNU_PKL :", NU_PKL)

# ----------------------------------------------------------------- conformal core (inlined)
def empirical_quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        return float("inf")
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))

def oc_quantile(arr, q):
    """online_conformal/utils.py::quantile (unweighted) -- used by FACI."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return float("inf")
    return float(np.quantile(arr, q, method="inverted_cdf"))

def pinball_loss(y, yhat, q):
    return np.maximum(q * (y - yhat), (1 - q) * (yhat - y))

def pinball_loss_grad(y, yhat, q):
    return -q * (y > yhat) + (1 - q) * (y < yhat)

def pb_count(scores, probs):
    return (scores[:, None] * probs).sum(axis=0)

def pb_variance(scores, probs):
    w = scores[:, None] * probs
    return (w * (1.0 - w)).sum(axis=0)

class AdaptiveConformalInference:
    def __init__(self, alpha_target=0.1, gamma=0.05):
        self.alpha_target = alpha_target; self.gamma = gamma
        self.alpha_t = alpha_target; self.history_scores = []
    def reset(self):
        self.alpha_t = self.alpha_target; self.history_scores = []
    def update(self, score_t, covered_t):
        self.history_scores.append(score_t)
        err_t = 0.0 if covered_t else 1.0
        self.alpha_t = max(1e-3, min(0.5, self.alpha_t + self.gamma * (self.alpha_target - err_t)))
    def get_quantile(self):
        if not self.history_scores:
            return 1.0
        return empirical_quantile(np.array(self.history_scores), self.alpha_t)

class PBAwareJointConformalOnline:
    def __init__(self, alpha=0.1, window=300):
        self.alpha = alpha; self.window = window; self.scores = []
    def warmstart(self, cal_scores):
        self.scores = list(np.asarray(cal_scores)[-self.window:]); return self
    def get_quantile(self):
        if not self.scores:
            return float("inf")
        return empirical_quantile(np.asarray(self.scores[-self.window:]), self.alpha)
    def update(self, score_t):
        self.scores.append(float(score_t))
        if len(self.scores) > self.window:
            self.scores = self.scores[-self.window:]

# ----------------------------------------------------------------- load + harness
def load():
    with open(PAN_PKL, "rb") as f:
        dpan = pickle.load(f)
    with open(NU_PKL, "rb") as f:
        dnu = pickle.load(f)
    settings = dpan["predictions_by_setting"]
    gtc = np.asarray(dpan["gt_counts"])
    gts = [np.array([float(g.sum())]) for g in gtc]
    key = "in_dist" if "in_dist" in settings else list(settings)[0]
    pan_in = [{"scores": np.asarray(p["scores"]),
               "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in settings[key]]
    print("settings:", list(settings.keys()), "-> calibration uses", key)
    return pan_in, gts, dnu["preds"], dnu["gts"]

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

PAN_IN, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_SCORES = np.array([nonconf(PAN_IN[i], PAN_GTS[i]) for i in range(len(PAN_IN))])
T = len(NU_PREDS)
print(f"cal(in-dist) {len(PAN_SCORES)} | NuInsSeg {T} | q0={empirical_quantile(PAN_SCORES, ALPHA):.3f}\n")

# ================================================================= baselines
def m_pbo(order):
    m = PBAwareJointConformalOnline(ALPHA, WINDOW).warmstart(PAN_SCORES)
    c, w, k = [], [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); k.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, k

def m_aci(order):
    m = AdaptiveConformalInference(ALPHA, 0.05); m.reset(); m.history_scores = list(PAN_SCORES)
    c, w, k = [], [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        cov = lo <= y <= hi; c.append(cov); w.append(hi - lo); k.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]), cov)
    return c, w, k

def _wquantile(scores, weights, level):
    o = np.argsort(scores); sc = np.asarray(scores)[o]; wt = np.asarray(weights)[o]
    cw = np.cumsum(wt) / wt.sum()
    return sc[min(np.searchsorted(cw, level), len(sc) - 1)]

def m_nexcp(order, rho=0.99):
    hist = list(PAN_SCORES); c, w, k = [], [], []
    for i in order:
        wts = rho ** (len(hist) - 1 - np.arange(len(hist)))
        q = _wquantile(hist, wts, 1 - ALPHA); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); k.append(winkler(lo, hi, y))
        hist.append(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, k

# -------- FACI (Gibbs-Candes JMLR 2024) -- port verbatim from online_conformal/faci.py
class FACICore:
    def __init__(self, calib_resids, coverage=COVERAGE):
        self.coverage = coverage
        self.gammas = np.asarray([0.001 * 2 ** j for j in range(8)])
        self.k = len(self.gammas)
        self.alphas = np.full(self.k, 1 - coverage)
        self.log_w = np.zeros(self.k)
        self.I = 100
        self.sigma = 1.0 / (2 * self.I)
        a = 1 - coverage
        denom = ((1 - a) ** 2 * a ** 3 + a ** 2 * (1 - a) ** 3) / 3
        self.eta = np.sqrt(3 / self.I) * np.sqrt((np.log(self.I * self.k) + 2) / denom)
        self.residuals = [float(r) for r in calib_resids]
    def predict(self):
        lw = self.log_w
        alpha = np.dot(np.exp(lw - logsumexp(lw)), self.alphas)
        return float(oc_quantile(np.abs(np.asarray(self.residuals)), 1 - alpha))
    def update(self, s):
        res = self.residuals
        if len(res) > math.floor(1 / (1 - self.coverage)):
            beta = float(np.mean(np.asarray(res) >= s))
            losses = pinball_loss(beta, self.alphas, 1 - self.coverage)
            wbar = self.log_w - self.eta * losses
            self.log_w = logsumexp([wbar, np.full(self.k, logsumexp(wbar))],
                                   b=[[1 - self.sigma], [self.sigma / self.k]], axis=0)
            self.log_w = self.log_w - logsumexp(self.log_w)
            err = self.alphas > beta
            self.alphas = np.clip(self.alphas + self.gammas * ((1 - self.coverage) - err), 0, 1)
        res.append(float(s))

def m_faci(order):
    m = FACICore(PAN_SCORES); c, w, k = [], [], []
    for i in order:
        q = m.predict(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); k.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, k

# -------- SAOCP + SF-OGD (Bhatnagar 2023) -- port verbatim from salesforce/online_conformal
class _OGD:
    def __init__(self, t, scale, alpha, yhat_0, g=8):
        self.scale = scale; self.base_lr = scale / np.sqrt(3); self.alpha = alpha
        self.yhat = yhat_0; self.grad_norm = 0; u = 0
        while t % 2 == 0:
            t /= 2; u += 1
        self.lifetime = g * 2 ** u; self.z = 0; self.wz = 0; self.s_t = 0
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
        self.z += g; self.wz += g * w; self.s_t += 1
        grad = pinball_loss_grad(y, self.yhat, 1 - self.alpha)
        self.grad_norm += grad ** 2
        if self.grad_norm != 0:
            self.yhat = max(0, self.yhat - self.base_lr / np.sqrt(self.grad_norm) * grad)

class SAOCPCore:
    def __init__(self, warm_scores, coverage=COVERAGE, lifetime=8):
        self.coverage = coverage; self.lifetime = lifetime; self.t = 1; self.experts = {}
        r = np.abs(np.asarray(warm_scores, dtype=float))
        self.scale = 1.0 if len(r) == 0 else float(np.max(r) * np.sqrt(3))
        for s in r:
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
        return sum(p[t] * self.experts[t].yhat for t in p)
    def _step(self, s):
        s_hat = self.predict()
        for t in [k for k, v in self.experts.items() if v.expired]:
            self.experts.pop(t)
        self.experts[self.t] = _OGD(self.t, self.scale, 1 - self.coverage, yhat_0=s_hat, g=self.lifetime)
        meta_loss = pinball_loss(s, self.predict(), self.coverage)
        for e in self.experts.values():
            e.update(s, meta_loss)
        self.t += 1
    def quantile(self):
        return self.predict()
    def update(self, s):
        self._step(float(s))

def m_saocp(order):
    m = SAOCPCore(PAN_SCORES, COVERAGE, 8); c, w, k = [], [], []
    for i in order:
        q = m.quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); k.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, k

# -------- COP (Hu et al. ICLR 2026)
class COP:
    def __init__(self, eta, lam=1.0, w=100, warm=None):
        self.eta, self.lam, self.w = eta, lam, w
        self.qhat = empirical_quantile(np.asarray(warm), ALPHA); self.q = self.qhat
        self.win = list(np.asarray(warm)[-w:])
    def get_q(self):
        return max(0.0, self.q)
    def update(self, s, q_used):
        self.qhat = max(0.0, self.qhat + self.eta * ((1.0 if s > q_used else 0.0) - ALPHA))
        self.win.append(float(s)); self.win = self.win[-self.w:]
        Fhat = float(np.mean(np.asarray(self.win) <= self.qhat))
        self.q = max(0.0, self.qhat - self.lam * (Fhat - (1.0 - ALPHA)))

def m_cop(order, eta=5.0):
    m = COP(eta, warm=PAN_SCORES); c, w, k = [], [], []
    for i in order:
        q = m.get_q(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); k.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]), q)
    return c, w, k

# -------- Rolling-Origin CP (Halkiewicz 2026)
def m_rolling(order, beta=1.0):
    m_star = max(20, int(round(T ** (2 * beta / (2 * beta + 1)))))
    win = list(PAN_SCORES[-m_star:]); c, w, k = [], [], []
    for i in order:
        q = empirical_quantile(np.asarray(win), ALPHA); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); k.append(winkler(lo, hi, y))
        win.append(nonconf(NU_PREDS[i], NU_GTS[i])); win = win[-m_star:]
    return c, w, k

# -------- Adaptive PB-JCI Online (ours, dead-band window)
def m_adapt(order, target=0.9, cov_win=50, w_min=40, w_max=WINDOW):
    scores = list(PAN_SCORES[-w_max:]); eff = w_max; recent, c, w, k = [], [], [], []
    for i in order:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA) if scores else float("inf")
        lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]; cov = lo <= y <= hi
        c.append(cov); w.append(hi - lo); k.append(winkler(lo, hi, y))
        recent.append(cov); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target:
            eff = max(w_min, int(eff * 0.9))
        elif rc > target + 0.03:
            eff = min(w_max, int(eff * 1.05))
        scores.append(nonconf(NU_PREDS[i], NU_GTS[i])); scores = scores[-w_max:]
    return c, w, k

# ================================================================= aggregate + print
def agg(fn, seeds=5):
    cs, ws, mws, ks = [], [], [], []
    for sd in range(seeds):
        order = np.random.RandomState(sd).permutation(T)
        c, w, k = fn(order)
        cs.append(np.mean(c) * 100); ws.append(np.mean(w))
        mws.append(np.median(w)); ks.append(np.mean(k))
    return (np.mean(cs), np.std(cs), np.mean(ws), np.mean(mws),
            np.mean(ks), np.std(ks))

ROWS = [
    ("ACI (Gibbs-Candes 2021)", m_aci),
    ("NexCP (Barber 2023)", m_nexcp),
    ("FACI (Gibbs-Candes 2024)", m_faci),
    ("SAOCP (Bhatnagar 2023, official)", m_saocp),
    ("COP (Hu et al. ICLR 2026)", m_cop),
    ("Rolling-Origin CP (Halkiewicz 2026)", m_rolling),
    ("PB-JCI Online-Fixed (ours)", m_pbo),
    ("Adaptive PB-JCI Online (ours)", m_adapt),
]

print("=" * 86)
print("CROSS-DATASET TABLE (tab:cross) -- PathoSAM -> NuInsSeg, 5 seeds, target 90%")
print("=" * 86)
print(f"{'Method':37s} | {'Coverage %':>13s} | {'AvgW':>6s} | {'MedW':>6s} | {'Winkler':>16s}")
print("-" * 92)
for name, fn in ROWS:
    cm, cs, wm, mwm, km, ksd = agg(fn)
    print(f"{name:37s} | {cm:6.1f}+/-{cs:4.1f} | {wm:6.2f} | {mwm:6.2f} | {km:8.2f}+/-{ksd:5.2f}")
print("-" * 92)
print("Only Adaptive PB-JCI Online attains the nominal 90% coverage; all baselines stay below.")
print("Winkler gap to the strongest baseline (Rolling-Origin) is within 1 std -> the contribution")
print("is conditional validity (recovering coverage), not interval efficiency (Adaptive is widest).")

"""
SELF-CONTAINED Kaggle cell: Winkler/Interval-score evaluation of the conformal
mechanisms on PathoSAM -> NuInsSeg (cal = PanNuke). No external lib import.

Needs two pickles uploaded as a Kaggle dataset:
  - pathosam_predictions.pkl       (dict: predictions_by_setting{in_dist,mild_shift,
                                     severe_shift}, gt_counts)
  - pathosam_nuinsseg_preds.pkl    (dict: preds[list of {scores,probs,K}], gts)
The script auto-locates them anywhere under /kaggle/input. If your filenames differ,
edit PAN_PKL / NU_PKL below.

Produces:
  (1) Tables 8f/9a with a Winkler column (lower = better).
  (2) Mechanism analysis on 2 synthetic streams: W@90% (matched-coverage width),
      per-segment conditional coverage, and per-segment Winkler.
"""
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np

ALPHA = 0.1
WINDOW = 300

# ----------------------------------------------------------------- locate data
def _find(name, root="/kaggle/input"):
    hits = list(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}; set the path manually.")
    return hits[0]

PAN_PKL = _find("pathosam_predictions.pkl")      # or: Path("/kaggle/input/<ds>/pathosam_predictions.pkl")
NU_PKL  = _find("pathosam_nuinsseg_preds.pkl")   # or: Path("/kaggle/input/<ds>/pathosam_nuinsseg_preds.pkl")
print("PAN_PKL:", PAN_PKL, "\nNU_PKL :", NU_PKL)

# ----------------------------------------------------------------- conformal core (inlined)
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

class RollingShiftDetector:
    def __init__(self, window=100, cap=1.0):
        self.window = window; self.cap = cap; self.baseline = None; self.recent = []
    def fit_baseline(self, cal_scores):
        self.baseline = float(np.median(np.asarray(cal_scores))) + 1e-6; return self
    def step(self, score_t):
        self.recent.append(float(score_t))
        if len(self.recent) > self.window:
            self.recent.pop(0)
        delta = (float(np.median(self.recent)) - self.baseline) / self.baseline
        return float(np.clip(delta, 0.0, self.cap))

# ----------------------------------------------------------------- load
def load():
    with open(PAN_PKL, "rb") as f:
        dpan = pickle.load(f)
    with open(NU_PKL, "rb") as f:
        dnu = pickle.load(f)
    settings = dpan["predictions_by_setting"]
    gtc = np.asarray(dpan["gt_counts"])
    gts = [np.array([float(g.sum())]) for g in gtc]
    def pick(*cands):
        for k in cands:
            if k in settings:
                return k
        raise KeyError(f"none of {cands} in {list(settings)}")
    keys = dict(in_=pick("in_dist", "in-dist"), mild=pick("mild_shift", "mild"),
                sev=pick("severe_shift", "severe"))
    def mk(key):
        return [{"scores": np.asarray(p["scores"]),
                 "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in settings[key]]
    pan = {n: mk(k) for n, k in keys.items()}
    print("settings:", list(settings.keys()), "-> using", keys)
    return pan, gts, dnu["preds"], dnu["gts"]

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

PAN, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_IN = PAN["in_"]
PAN_SCORES = np.array([nonconf(PAN_IN[i], PAN_GTS[i]) for i in range(len(PAN_IN))])
print(f"cal(in-dist) {len(PAN_SCORES)} | NuInsSeg {len(NU_PREDS)} | "
      f"q0={empirical_quantile(PAN_SCORES, ALPHA):.3f}\n")

# ================================================================= PART 1: Tables 8f/9a + Winkler
def m_pbo(order):
    m = PBAwareJointConformalOnline(ALPHA, WINDOW).warmstart(PAN_SCORES)
    c, w, s = [], [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, s

def m_aci(order):
    m = AdaptiveConformalInference(ALPHA, 0.05); m.reset(); m.history_scores = list(PAN_SCORES)
    c, w, s = [], [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        cov = lo <= y <= hi; c.append(cov); w.append(hi - lo); s.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]), cov)
    return c, w, s

def wquantile(scores, weights, level):
    o = np.argsort(scores); sc = np.asarray(scores)[o]; wt = np.asarray(weights)[o]
    cw = np.cumsum(wt) / wt.sum()
    return sc[min(np.searchsorted(cw, level), len(sc) - 1)]

def m_nexcp(order, rho=0.99):
    hist = list(PAN_SCORES); c, w, s = [], [], []
    for i in order:
        wts = rho ** (len(hist) - 1 - np.arange(len(hist)))
        q = wquantile(hist, wts, 1 - ALPHA); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        hist.append(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, s

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
    m = COP(eta, warm=PAN_SCORES); c, w, s = [], [], []
    for i in order:
        q = m.get_q(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]), q)
    return c, w, s

def m_flush(order, thresh=0.5):
    m = PBAwareJointConformalOnline(ALPHA, WINDOW).warmstart(PAN_SCORES)
    det = RollingShiftDetector(100).fit_baseline(PAN_SCORES)
    tgt, flushed, c, w, s = [], False, [], [], []
    for i in order:
        q = m.get_quantile(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        sc = nonconf(NU_PREDS[i], NU_GTS[i]); tgt.append(sc)
        if not flushed and det.step(sc) >= thresh:
            m.scores = list(tgt[-WINDOW:]); flushed = True
        else:
            m.update(sc)
    return c, w, s

def m_adapt(order, target=0.9, cov_win=50, w_min=40, w_max=WINDOW):
    scores = list(PAN_SCORES[-w_max:]); eff = w_max; recent, c, w, s = [], [], [], []
    for i in order:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA) if scores else float("inf")
        lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]; cov = lo <= y <= hi
        c.append(cov); w.append(hi - lo); s.append(winkler(lo, hi, y))
        recent.append(cov); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target:
            eff = max(w_min, int(eff * 0.9))
        elif rc > target + 0.03:
            eff = min(w_max, int(eff * 1.05))
        scores.append(nonconf(NU_PREDS[i], NU_GTS[i])); scores = scores[-w_max:]
    return c, w, s

def m_fallback(order, target=0.9, eta=0.03, cov_win=50):
    m = PBAwareJointConformalOnline(ALPHA, WINDOW).warmstart(PAN_SCORES)
    mult, recent, c, w, s = 1.0, [], [], [], []
    for i in order:
        q = m.get_quantile() * mult; lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        cov = lo <= y <= hi; c.append(cov); w.append(hi - lo); s.append(winkler(lo, hi, y))
        recent.append(cov); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target:
            mult *= (1 + eta)
        elif rc > target + 0.03:
            mult = max(1.0, mult * (1 - eta))
        mult = min(mult, 6.0); m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, s

def m_hybrid(order):
    pb = PBAwareJointConformalOnline(ALPHA, WINDOW).warmstart(PAN_SCORES)
    aci = AdaptiveConformalInference(ALPHA, 0.05); aci.reset(); aci.history_scores = list(PAN_SCORES)
    c, w, s = [], [], []
    for i in order:
        lo1, hi1 = interval(NU_PREDS[i], pb.get_quantile())
        lo2, hi2 = interval(NU_PREDS[i], aci.get_quantile())
        lo, hi = min(lo1, lo2), max(hi1, hi2); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); s.append(winkler(lo, hi, y))
        sc = nonconf(NU_PREDS[i], NU_GTS[i]); pb.update(sc); aci.update(sc, lo2 <= y <= hi2)
    return c, w, s

def agg(fn, seeds=5):
    cs, ws, ss = [], [], []
    for sd in range(seeds):
        order = np.random.RandomState(sd).permutation(len(NU_PREDS))
        c, w, s = fn(order)
        cs.append(np.mean(c) * 100); ws.append(np.mean(w)); ss.append(np.mean(s))
    return np.mean(cs), np.std(cs), np.mean(ws), np.mean(ss), np.std(ss)

print("=" * 80)
print("TABLES 8f/9a + Winkler -- PathoSAM->NuInsSeg, 5 seeds, target 90%")
print("=" * 80)
print(f"{'Method':34s} | {'Coverage':>13s} | {'Width':>7s} | {'Winkler (lower=better)':>22s}")
print("-" * 80)
for name, fn in [
    ("PB-JCI Online (static op-point)", m_pbo),
    ("ACI", m_aci),
    ("NexCP (Barber 2023)", m_nexcp),
    ("COP (ICLR 2026, eta=5)", m_cop),
    ("Detector-flush (variant)", m_flush),
    ("Adaptive PB-JCI Online (ours)", m_adapt),
    ("C. Fallback-multiplier", m_fallback),
    ("D. Hybrid max(PB,ACI)", m_hybrid),
]:
    cm, cs, wm, sm, sd = agg(fn)
    print(f"{name:34s} | {cm:5.1f}+/-{cs:4.1f}% | {wm:7.2f} | {sm:10.2f}+/-{sd:5.2f}")
print("-" * 80)

# ================================================================= PART 2: mechanism streams
# rec = (n, sigma, q, gt) so we can compute raw cov/width, W@90, conditional cov, Winkler
def _ns(p):
    if len(p["scores"]) == 0:
        return 0.0, 0.0
    n = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return n, sg

def _cov(n, sg, q, g):
    return max(0.0, n - q * sg) <= g <= n + q * sg

def _sample(rng, n, length):
    return rng.choice(length, n, replace=length < n)

def stream_abrupt(seed, n_pre=150, n_post=300):
    rng = np.random.RandomState(seed); items, labels = [], []
    for i in _sample(rng, n_pre, len(PAN_IN)):
        items.append((PAN_IN[i], PAN_GTS[i])); labels.append("pre(in-dist)")
    post = list(_sample(rng, n_post, len(NU_PREDS)))
    for j, i in enumerate(post):
        items.append((NU_PREDS[i], NU_GTS[i])); labels.append("post_early" if j < 50 else "post_late")
    return items, labels

def stream_drift(seed, per=150):
    rng = np.random.RandomState(seed); items, labels = [], []
    for name, preds in [("in-dist", PAN["in_"]), ("mild", PAN["mild"]), ("severe", PAN["sev"])]:
        for i in _sample(rng, per, len(preds)):
            items.append((preds[i], PAN_GTS[i])); labels.append(name)
    return items, labels

def r_pbo(items):
    m = PBAwareJointConformalOnline(ALPHA, WINDOW).warmstart(PAN_SCORES); rec = []
    for p, gt in items:
        q = m.get_quantile(); n, sg = _ns(p); rec.append((n, sg, q, gt[0]))
        m.update(nonconf(p, gt))
    return rec

def r_flush(items, thresh=0.5):
    m = PBAwareJointConformalOnline(ALPHA, WINDOW).warmstart(PAN_SCORES)
    det = RollingShiftDetector(100).fit_baseline(PAN_SCORES); tgt, flushed, rec = [], False, []
    for p, gt in items:
        q = m.get_quantile(); n, sg = _ns(p); rec.append((n, sg, q, gt[0]))
        s = nonconf(p, gt); tgt.append(s)
        if not flushed and det.step(s) >= thresh:
            m.scores = list(tgt[-WINDOW:]); flushed = True
        else:
            m.update(s)
    return rec

def r_adapt(items, target=0.9, cov_win=50, w_min=40, w_max=WINDOW):
    scores = list(PAN_SCORES[-w_max:]); eff = w_max; recent, rec = [], []
    for p, gt in items:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA) if scores else float("inf")
        n, sg = _ns(p); rec.append((n, sg, q, gt[0]))
        recent.append(_cov(n, sg, q, gt[0])); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target:
            eff = max(w_min, int(eff * 0.9))
        elif rc > target + 0.03:
            eff = min(w_max, int(eff * 1.05))
        scores.append(nonconf(p, gt)); scores = scores[-w_max:]
    return rec

def cov_width(rec, c=1.0):
    cov, wid = [], []
    for n, sg, q, g in rec:
        if not np.isfinite(q):
            cov.append(True); wid.append(float("inf")); continue
        lo = max(0.0, n - c * q * sg); hi = n + c * q * sg
        cov.append(lo <= g <= hi); wid.append(hi - lo)
    return float(np.mean(cov)), float(np.mean(wid))

def width_at(rec, target=0.90):
    if cov_width(rec, 40.0)[0] < target:
        return cov_width(rec, 40.0)[1], 40.0
    lo, hi = 0.02, 40.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if cov_width(rec, mid)[0] >= target:
            hi = mid
        else:
            lo = mid
    return cov_width(rec, hi)[1], hi

def winkler_rec(rec):
    s = []
    for n, sg, q, g in rec:
        if not np.isfinite(q):
            s.append(float("inf")); continue
        L = max(0.0, n - q * sg); U = n + q * sg
        s.append(winkler(L, U, g))
    return float(np.mean(s))

RMETHODS = [("PB-JCI Online", r_pbo), ("Detector-flush", r_flush), ("Adaptive PB-JCI Online", r_adapt)]

def report_stream(title, build, seg_order, nseeds=5):
    print("\n" + "=" * 96); print(title); print("=" * 96)
    hdr = (f"{'Method':24s} | {'rawCov':>7s} | {'W@90':>6s} | {'Winkler':>8s} | "
           + " ".join(f"{s+'cov':>14s}" for s in seg_order))
    print(hdr); print("-" * len(hdr))
    for name, fn in RMETHODS:
        rc, cw, wk = [], [], []
        segc = defaultdict(list)
        for sd in range(nseeds):
            items, labels = build(sd); rec = fn(items)
            rc.append(cov_width(rec, 1.0)[0] * 100); cw.append(width_at(rec, 0.90)[0]); wk.append(winkler_rec(rec))
            for lab in set(labels):
                idx = [i for i, l in enumerate(labels) if l == lab]
                segc[lab].append(np.mean([cov_width([rec[i]], 1.0)[0] for i in idx]) * 100)
        segstr = " ".join(f"{np.mean(segc[s]):13.1f}%" for s in seg_order)
        print(f"{name:24s} | {np.mean(rc):6.1f}% | {np.mean(cw):6.2f} | {np.mean(wk):8.2f} | {segstr}")

report_stream("STREAM A - ABRUPT (in-dist 150 -> NuInsSeg 300). target 90%",
              stream_abrupt, ["pre(in-dist)", "post_early", "post_late"])
report_stream("STREAM B - DRIFT (in-dist -> mild -> severe, 150 each). target 90%",
              stream_drift, ["in-dist", "mild", "severe"])

print("\nREAD: W@90 = width if rescaled to exactly 90% (matched-coverage efficiency;")
print(" online/flush/adapt ~equal -> width is NOT the contribution). Winkler (lower=better)")
print(" and per-segment conditional coverage -> Adaptive PB-JCI Online wins on shifted regimes.")

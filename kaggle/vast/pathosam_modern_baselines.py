"""
Baseline HIEN DAI 2023-2026 cho bang 9a (PathoSAM -> NuInsSeg, cal = PanNuke total-count,
K=1, stream 5 seed). Cung harness/nonconf/interval voi pathosam_cop_baseline.py.
Xuat Coverage / Width / Winkler de ghep thang vao Table 9a.

Ba baseline:
  (1) Rolling-Origin CP (Halkiewicz 2026)  -- FAITHFUL: cua so truot voi kich thuoc
      ly thuyet m* = round(T^(2/3)) (Lipschitz drift, beta=1). Chinh la online-window
      voi window dat theo ly thuyet thay vi W=300.
  (2) SAOCP (Bhatnagar et al. ICML 2023)   -- REIMPLEMENT: aggregation cua cac expert
      SF-OGD voi geometric lifetimes (strongly-adaptive). *** Kiem chung lai voi repo
      goc github tac gia truoc khi nop ***.
  (3) AdaptNC (arXiv 2602.01629, 2026)      -- APPROXIMATE PLACEHOLDER: paper chua co
      trong tay; day chi la xap xi y tuong "dong thoi adapt scale score + nguong".
      *** KHONG bao cao nhu AdaptNC chinh thuc; thay bang code tac gia khi co ***.

So kem PB-JCI Online (ours), ACI, NexCP tren CUNG 5 seed de doi chieu.
CPU, pkl cached.  python pathosam_modern_baselines.py
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "kaggle" / "lib"))
from conformal import empirical_quantile, pb_count, pb_variance, PBAwareJointConformalOnline, AdaptiveConformalInference  # noqa

ALPHA = 0.1
SEEDS = 5


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


PAN_PREDS, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_SCORES = np.array([nonconf(PAN_PREDS[i], PAN_GTS[i]) for i in range(len(PAN_PREDS))])
T = len(NU_PREDS)
print(f"PanNuke cal {len(PAN_SCORES)} | NuInsSeg stream T={T} | q0={empirical_quantile(PAN_SCORES, ALPHA):.3f}")


# ===================== (1) Rolling-Origin CP (Halkiewicz 2026) =====================
def rolling_origin_run(order, beta=1.0):
    m_star = max(20, int(round(T ** (2 * beta / (2 * beta + 1)))))   # Lipschitz: T^(2/3)
    win = list(PAN_SCORES[-m_star:])
    c, w, wk = [], [], []
    for i in order:
        q = empirical_quantile(np.asarray(win), ALPHA)
        lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); wk.append(winkler(lo, hi, y))
        win.append(nonconf(NU_PREDS[i], NU_GTS[i])); win = win[-m_star:]
    return c, w, wk, m_star


# ===================== (2) SAOCP (Bhatnagar et al. 2023) =====================
class SFOGD:
    """Mot expert: scale-free OGD tren pinball loss cho phan vi (1-alpha)."""
    def __init__(self, q0, alpha=ALPHA):
        self.q = float(q0); self.alpha = alpha; self.G2 = 1e-8
    def predict(self):
        return max(0.0, self.q)
    def update(self, s):
        tau = 1.0 - self.alpha
        g = (1.0 if s < self.q else 0.0) - tau         # subgradient quantile loss wrt q
        self.G2 += g * g
        self.q = self.q - g / np.sqrt(self.G2)


def sfogd_run(order):
    """SF-OGD online conformal (Scale-Free OGD tren pinball) -- la BASE LEARNER cua SAOCP
    va ban than la mot baseline online-conformal chuan. FAITHFUL.
    Luu y: SAOCP DAY DU (strongly-adaptive aggregation cua nhieu SF-OGD) manh hon cai nay;
    reimplement aggregation khong on dinh -> dung CODE GOC (github Bhatnagar) cho so SAOCP that."""
    m = SFOGD(empirical_quantile(PAN_SCORES, ALPHA))
    c, w, wk = [], [], []
    for i in order:
        q = m.predict(); lo, hi = interval(NU_PREDS[i], q); y = NU_GTS[i][0]
        c.append(lo <= y <= hi); w.append(hi - lo); wk.append(winkler(lo, hi, y))
        m.update(nonconf(NU_PREDS[i], NU_GTS[i]))
    return c, w, wk


# ===================== (3) AdaptNC (xap xi -- PLACEHOLDER) =====================
def adaptnc_run(order, eta_q=0.3, eta_a=0.05, w=100):
    """XAP XI y tuong: dong thoi (a) adapt nguong qua OGD pinball, (b) adapt mot he so
    scale 'a' nhan vao sigma de score-geometry co gian theo shift. KHONG phai AdaptNC
    chinh thuc -- thay bang code tac gia khi co. Verify truoc khi bao cao."""
    q = empirical_quantile(PAN_SCORES, ALPHA)
    a = 1.0                                   # he so scale geometry
    win = list(PAN_SCORES[-w:]); c, wd, wk = [], [], []
    for i in order:
        p = NU_PREDS[i]
        if len(p["scores"]) == 0:
            lo, hi = 0.0, 0.0
        else:
            n = pb_count(p["scores"], p["probs"])[0]
            sg = a * np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
            lo, hi = max(0.0, n - q * sg), n + q * sg
        y = NU_GTS[i][0]
        c.append(lo <= y <= hi); wd.append(hi - lo); wk.append(winkler(lo, hi, y))
        s = nonconf(NU_PREDS[i], NU_GTS[i])
        err = 1.0 if not (lo <= y <= hi) else 0.0
        q = max(0.0, q + eta_q * (err - ALPHA))                  # threshold adapt
        win.append(s); win = win[-w:]
        cv = np.std(win) / (np.mean(win) + 1e-6)
        a = float(np.clip(a * (1.0 + eta_a * (cv - 0.5)), 0.3, 5.0))      # scale adapt (capped)
    return c, wd, wk


# ===================== reference (cung seed) =====================
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
    extra = None
    for sd in range(SEEDS):
        order = np.random.RandomState(sd).permutation(T)
        out = fn(order)
        c, w, wk = out[0], out[1], out[2]
        if len(out) > 3:
            extra = out[3]
        cs.append(np.mean(c) * 100); ws.append(np.mean(w)); ks.append(np.mean(wk))
    return np.mean(cs), np.std(cs), np.mean(ws), np.mean(ks), extra


print("\n" + "=" * 74)
print("Baseline hien dai | PathoSAM->NuInsSeg | target 90% | 5 seed | Winkler thap=tot")
print("=" * 74)
print(f"{'Method':30s} | {'Coverage':>13s} | {'Width':>8s} | {'Winkler':>9s}")
print("-" * 74)
rows = [
    ("Rolling-Origin (Halkiewicz26)", rolling_origin_run),  # FAITHFUL
    ("SF-OGD (SAOCP base, Bhatn23)",  sfogd_run),            # FAITHFUL base learner
    ("AdaptNC (2026) *APPROX",        adaptnc_run),          # placeholder, khong bao cao
    ("PB-JCI Online (ours)",          pbo_run),
    ("ACI (Gibbs-Candes21)",          aci_run),
]
for nm, fn in rows:
    cm, cs, wm, km, extra = agg(fn)
    tag = f"  [m*={extra}]" if extra is not None else ""
    print(f"{nm:30s} | {cm:5.1f}+/-{cs:4.1f}% | {wm:8.2f} | {km:9.2f}{tag}")
print("-" * 74)
print("DUNG DUOC (faithful): Rolling-Origin, SF-OGD, PB-JCI Online, ACI.")
print("KHONG bao cao: AdaptNC (*APPROX, can code goc 2602.01629);")
print("  SAOCP day du (strongly-adaptive aggregation) -> chay CODE GOC (github Bhatnagar23).")

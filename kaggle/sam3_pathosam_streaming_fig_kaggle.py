"""
SELF-CONTAINED Kaggle cell: two-panel streaming figure (RetroAdj/Barber-style).
Left  (a): rolling LOCAL COVERAGE over an abrupt-shift stream vs the 90% target.
Right (b): rolling LOCAL WIDTH over the same stream.
Backbone PathoSAM; stream = PanNuke in-dist -> NuInsSeg at a change-point.
No external lib import. CPU, a few seconds.

Needs two pickles uploaded as a Kaggle dataset (Add Data); the loader auto-finds them
anywhere under /kaggle/input:
  - pathosam_predictions.pkl      (dict: predictions_by_setting{in_dist,...}, gt_counts)
  - pathosam_nuinsseg_preds.pkl   (dict: preds[list of {scores,probs,K}], gts)

Saves figures/F2_streaming_coverage.png to /kaggle/working and shows it inline.
Story: static conformal keeps narrow intervals (panel b, red ~18) and therefore misses
on the harder domain (panel a, red ~25%); Adaptive widens just enough (panel b, green)
to hold coverage (panel a, green ~90%). Width is the price of coverage, not the goal.
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.1
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

def pb_count(scores, probs):
    return (scores[:, None] * probs).sum(axis=0)

def pb_variance(scores, probs):
    w = scores[:, None] * probs
    return (w * (1.0 - w)).sum(axis=0)

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
    key = "in_dist" if "in_dist" in settings else list(settings)[0]
    gtc = np.asarray(dpan["gt_counts"])
    pan = [{"scores": np.asarray(p["scores"]),
            "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in settings[key]]
    pgt = [np.array([float(g.sum())]) for g in gtc]
    print("settings:", list(settings.keys()), "-> calibration uses", key)
    return pan, pgt, dnu["preds"], dnu["gts"]


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


PAN_IN, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_SCORES = np.array([nonconf(PAN_IN[i], PAN_GTS[i]) for i in range(len(PAN_IN))])

# ----------------------------------------------------------------- stream + methods
def stream_abrupt(seed=0, n_pre=150, n_post=300):
    rng = np.random.RandomState(seed); items = []
    for i in rng.choice(len(PAN_IN), n_pre):
        items.append((PAN_IN[i], PAN_GTS[i]))
    for i in rng.choice(len(NU_PREDS), n_post):
        items.append((NU_PREDS[i], NU_GTS[i]))
    return items, n_pre


def run_static(items):
    q = empirical_quantile(PAN_SCORES, ALPHA)
    c, w = [], []
    for p, gt in items:
        lo, hi = interval(p, q); c.append(int(lo <= gt[0] <= hi)); w.append(hi - lo)
    return c, w


def run_online(items):
    m = PBAwareJointConformalOnline(ALPHA, WINDOW).warmstart(PAN_SCORES); c, w = [], []
    for p, gt in items:
        lo, hi = interval(p, m.get_quantile()); c.append(int(lo <= gt[0] <= hi)); w.append(hi - lo)
        m.update(nonconf(p, gt))
    return c, w


def run_adapt(items, target=0.9, cov_win=50, w_min=40):
    scores = list(PAN_SCORES[-WINDOW:]); eff = WINDOW; recent, c, w = [], [], []
    for p, gt in items:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA)
        lo, hi = interval(p, q); cov = int(lo <= gt[0] <= hi); c.append(cov); w.append(hi - lo)
        recent.append(cov); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target:
            eff = max(w_min, int(eff * 0.9))
        elif rc > target + 0.03:
            eff = min(WINDOW, int(eff * 1.05))
        scores.append(nonconf(p, gt)); scores = scores[-WINDOW:]
    return c, w


def rolling(x, win=50, scale=1.0):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - win + 1):i + 1].mean() for i in range(len(x))]) * scale


METHODS = [
    ("Static conformal (fixed)", run_static, "#d62728", 2.0),
    ("PB-JCI Online-Fixed", run_online, "#1f77b4", 2.0),
    ("Adaptive PB-JCI Online (ours)", run_adapt, "#2ca02c", 2.4),
]

# ----------------------------------------------------------------- plot
items, cp = stream_abrupt(seed=0)
res = {name: fn(items) for name, fn, _, _ in METHODS}
fig, (axc, axw) = plt.subplots(1, 2, figsize=(11.0, 4.3))

for name, _, col, lw in METHODS:
    c, w = res[name]
    axc.plot(rolling(c, scale=100.0), label=name, c=col, lw=lw)
    axw.plot(rolling(w, scale=1.0), label=name, c=col, lw=lw)

axc.axvline(cp, ls="--", c="black", lw=1.1)
axc.text(cp - 6, 33, "switch to NuInsSeg\n(change-point)", fontsize=8, ha="right")
axc.axhline(90, ls=":", c="gray")
axc.text(200, 96, "target 90%", fontsize=8, color="gray")
axc.set_xlabel("Step in stream"); axc.set_ylabel("Local coverage (%) [window 50]")
axc.set_title("(a) Local coverage over time")
axc.set_ylim(20, 100)

axw.axvline(cp, ls="--", c="black", lw=1.1)
axw.text(cp - 6, 72, "switch to NuInsSeg\n(change-point)", fontsize=8, ha="right")
axw.set_xlabel("Step in stream"); axw.set_ylabel("Local width [window 50]")
axw.set_title("(b) Local interval width over time")

# single shared legend below both panels (outside the axes, no overlap)
handles, labels = axc.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.08, 1, 1])
out = Path("/kaggle/working/F2_streaming_coverage.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print("wrote", out)
plt.show()

"""
SELF-CONTAINED Kaggle cell: figures for the report (PathoSAM -> NuInsSeg).
No external lib import. Needs the two pickles uploaded as a Kaggle Dataset
(auto-located under /kaggle/input):
  - pathosam_predictions.pkl
  - pathosam_nuinsseg_preds.pkl
Produces 3 figures (display inline + save PNG to working dir):
  F1 coverage-Winkler trade-off | F2 streaming coverage | F3 conditional coverage
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ALPHA, WINDOW = 0.1, 300

# ----------------------------------------------------------------- locate data
def _find(name, root="/kaggle/input"):
    hits = list(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}; set the path manually.")
    return hits[0]

PAN_PKL = _find("pathosam_predictions.pkl")
NU_PKL  = _find("pathosam_nuinsseg_preds.pkl")

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

class PBOnline:
    def __init__(self, window=WINDOW):
        self.window = window; self.scores = []
    def warmstart(self, cal):
        self.scores = list(np.asarray(cal)[-self.window:]); return self
    def get_q(self):
        return empirical_quantile(np.asarray(self.scores[-self.window:]), ALPHA) if self.scores else float("inf")
    def update(self, s):
        self.scores.append(float(s)); self.scores = self.scores[-self.window:]

# ----------------------------------------------------------------- load + helpers
def load():
    with open(PAN_PKL, "rb") as f:
        dpan = pickle.load(f)
    with open(NU_PKL, "rb") as f:
        dnu = pickle.load(f)
    s = dpan["predictions_by_setting"]; gtc = np.asarray(dpan["gt_counts"])
    gts = [np.array([float(g.sum())]) for g in gtc]
    mk = lambda k: [{"scores": np.asarray(p["scores"]),
                     "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in s[k]]
    pan = {"in": mk("in_dist"), "mild": mk("mild_shift"), "sev": mk("severe_shift")}
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

PAN, PAN_GTS, NU_PREDS, NU_GTS = load()
PAN_IN = PAN["in"]
PAN_SCORES = np.array([nonconf(PAN_IN[i], PAN_GTS[i]) for i in range(len(PAN_IN))])
print(f"loaded: PanNuke cal {len(PAN_SCORES)} | NuInsSeg {len(NU_PREDS)}")

def run_static(items):
    q = empirical_quantile(PAN_SCORES, ALPHA)
    out = []
    for p, gt in items:
        lo, hi = interval(p, q); out.append(int(lo <= gt[0] <= hi))
    return out

def run_online(items):
    m = PBOnline().warmstart(PAN_SCORES); c = []
    for p, gt in items:
        lo, hi = interval(p, m.get_q()); c.append(int(lo <= gt[0] <= hi)); m.update(nonconf(p, gt))
    return c

def run_adapt(items, target=0.9, cov_win=50, w_min=40):
    scores = list(PAN_SCORES[-WINDOW:]); eff = WINDOW; recent, c = [], []
    for p, gt in items:
        q = empirical_quantile(np.asarray(scores[-eff:]), ALPHA)
        lo, hi = interval(p, q); cov = int(lo <= gt[0] <= hi); c.append(cov)
        recent.append(cov); recent = recent[-cov_win:]; rc = np.mean(recent)
        if rc < target: eff = max(w_min, int(eff * 0.9))
        elif rc > target + 0.03: eff = min(WINDOW, int(eff * 1.05))
        scores.append(nonconf(p, gt)); scores = scores[-WINDOW:]
    return c

def rolling(c, w=50):
    c = np.asarray(c, float)
    return np.array([c[max(0, i - w + 1):i + 1].mean() for i in range(len(c))]) * 100

# === FIG 1 ===
# coverage vs Winkler (verified numbers, 5 seeds, from the Winkler table notebook)
pts = [
    ("PB-JCI Online (static op-pt)", 81.8, 125.96, "#888888", "o", 90),
    ("ACI", 84.0, 129.55, "#d62728", "o", 90),
    ("NexCP (2023)", 84.7, 119.56, "#9467bd", "o", 90),
    ("COP (ICLR 2026)", 87.9, 113.13, "#ff7f0e", "o", 90),
    ("Detector-flush (variant)", 88.7, 110.07, "#1f77b4", "o", 90),
    ("Adaptive PB-JCI Online (ours)", 90.0, 108.67, "#2ca02c", "*", 320),
]
fig, ax = plt.subplots(figsize=(8.2, 5.2))
for name, cov, wk, c, mk, sz in pts:
    ax.scatter(cov, wk, s=sz, c=c, marker=mk, edgecolors="black",
               linewidths=1.5 if mk == "*" else 0.6, zorder=3, label=name)
ax.axvline(90, ls="--", c="gray", lw=1)
ax.text(90.1, ax.get_ylim()[1] * 0.995, "target 90%", va="top", fontsize=8, color="gray")
ax.annotate("tốt hơn", xy=(89.5, 110), xytext=(86.5, 122),
            arrowprops=dict(arrowstyle="->", color="#2ca02c"), color="#2ca02c", fontsize=10)
ax.set_xlabel("Coverage (%)  — gần 90% là tốt")
ax.set_ylabel("Winkler / Interval score  — THẤP là tốt")
ax.set_title("Trade-off coverage–efficiency dưới extreme shift (PathoSAM → NuInsSeg)")
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
fig.tight_layout(); fig.savefig("F1_coverage_winkler.png", dpi=150); plt.show()

# === FIG 2 ===
rng = np.random.RandomState(0)
items = [(PAN_IN[i], PAN_GTS[i]) for i in rng.choice(len(PAN_IN), 150)] + \
        [(NU_PREDS[i], NU_GTS[i]) for i in rng.choice(len(NU_PREDS), 300)]
cp = 150
fig, ax = plt.subplots(figsize=(8.4, 4.8))
ax.plot(rolling(run_static(items)), label="Conformal tĩnh (cố định)", c="#d62728", lw=2)
ax.plot(rolling(run_online(items)), label="PB-JCI Online (cửa sổ cố định)", c="#1f77b4", lw=2)
ax.plot(rolling(run_adapt(items)), label="Adaptive PB-JCI Online (ours)", c="#2ca02c", lw=2.6)
ax.axvline(cp, ls="--", c="black", lw=1.2)
ax.text(cp + 3, 99.5, "← đổi sang NuInsSeg (change-point)", fontsize=9, va="top")
ax.axhline(90, ls=":", c="gray"); ax.text(448, 92.5, "target 90%", fontsize=8, color="gray", ha="right")
ax.set_xlabel("Bước trong stream"); ax.set_ylabel("Rolling coverage (%) [cửa sổ 50]")
ax.set_title("Coverage theo thời gian dưới shift đột ngột\n(tĩnh sụp tại change-point; adaptive kéo về 90%)")
ax.set_ylim(20, 103); ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
fig.tight_layout(); fig.savefig("F2_streaming_coverage.png", dpi=150); plt.show()

# === FIG 3 ===
rng = np.random.RandomState(0); per = 150; items, labels = [], []
for nm, k in [("in-dist", "in"), ("mild", "mild"), ("severe", "sev")]:
    for i in rng.choice(len(PAN[k]), per):
        items.append((PAN[k][i], PAN_GTS[i])); labels.append(nm)
segs = ["in-dist", "mild", "severe"]
def seg_cov(c):
    c = np.asarray(c, float)
    return [c[[i for i, l in enumerate(labels) if l == s]].mean() * 100 for s in segs]
cs, ca = seg_cov(run_static(items)), seg_cov(run_adapt(items))
x = np.arange(len(segs)); w = 0.36
fig, ax = plt.subplots(figsize=(7.4, 4.8))
ax.bar(x - w / 2, cs, w, label="Conformal tĩnh", color="#d62728")
ax.bar(x + w / 2, ca, w, label="Adaptive PB-JCI Online", color="#2ca02c")
ax.axhline(90, ls="--", c="gray"); ax.text(-0.45, 91.5, "target 90%", fontsize=8, color="gray")
for i, v in enumerate(cs): ax.text(i - w / 2, v + 1, f"{v:.0f}", ha="center", fontsize=9)
for i, v in enumerate(ca): ax.text(i + w / 2, v + 1, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(segs); ax.set_ylim(0, 108)
ax.set_xlabel("Regime (shift tăng dần →)"); ax.set_ylabel("Coverage theo regime (%)")
ax.set_title("Conditional coverage: tĩnh under-cover khi shift mạnh,\nadaptive giữ ~90% ở mọi regime")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout(); fig.savefig("F3_conditional_coverage.png", dpi=150); plt.show()

# === FIG 4 ===
# So baseline hiện đại trực diện (Bảng 9a). Panel trái: coverage (gồm 2 baseline sụp);
# panel phải: Winkler (chỉ method không sụp; thấp = tốt).
cov_rows = [
    ("Naive PB\n(không conformal)", 10.4, "#777777"),
    ("Weighted Conformal\n(Tibshirani'19)", 40.8, "#777777"),
    ("PB-JCI Online (static)", 81.8, "#bbbbbb"),
    ("ACI", 84.0, "#bbbbbb"),
    ("NexCP (2023)", 84.7, "#bbbbbb"),
    ("COP (ICLR 2026)", 87.9, "#bbbbbb"),
    ("Adaptive PB-JCI Online\n(ours)", 90.0, "#2ca02c"),
]
wk_rows = [
    ("ACI", 129.55, "#bbbbbb"),
    ("PB-JCI Online (static)", 125.96, "#bbbbbb"),
    ("NexCP (2023)", 119.56, "#bbbbbb"),
    ("COP (ICLR 2026)", 113.13, "#ff7f0e"),
    ("Detector-flush (variant)", 110.07, "#1f77b4"),
    ("Adaptive PB-JCI Online (ours)", 108.67, "#2ca02c"),
]
fig, (axc, axw) = plt.subplots(1, 2, figsize=(13.5, 5.2))
# panel A: coverage
names = [r[0] for r in cov_rows]; vals = [r[1] for r in cov_rows]; cols = [r[2] for r in cov_rows]
y = np.arange(len(names))[::-1]
axc.barh(y, vals, color=cols, edgecolor="black", linewidth=0.5)
axc.axvline(90, ls="--", c="red", lw=1.2); axc.text(90.5, y.max() + 0.3, "target 90%", color="red", fontsize=8)
for yi, v in zip(y, vals): axc.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=9)
axc.set_yticks(y); axc.set_yticklabels(names, fontsize=9); axc.set_xlim(0, 105)
axc.set_xlabel("Coverage (%)"); axc.set_title("(a) Coverage — naive/weighted SỤP,\nchỉ conformal online giữ được")
# panel B: Winkler
names2 = [r[0] for r in wk_rows]; vals2 = [r[1] for r in wk_rows]; cols2 = [r[2] for r in wk_rows]
y2 = np.arange(len(names2))[::-1]
axw.barh(y2, vals2, color=cols2, edgecolor="black", linewidth=0.5)
for yi, v in zip(y2, vals2): axw.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=9)
axw.set_yticks(y2); axw.set_yticklabels(names2, fontsize=9); axw.set_xlim(105, 133)
axw.set_xlabel("Winkler / Interval score  (THẤP = tốt)")
axw.set_title("(b) Winkler — Adaptive PB-JCI Online\nthấp nhất (vượt cả COP'26)")
fig.suptitle("So với baseline uncertainty/conformal hiện đại (PathoSAM → NuInsSeg)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("F4_vs_baselines.png", dpi=150); plt.show()
print("done: F1/F2/F3/F4 saved to working dir")

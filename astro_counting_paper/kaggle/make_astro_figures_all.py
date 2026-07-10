# =====================================================================================
# make_astro_figures_all.py — sinh NHIEU STYLE hinh recovery TU JSON da luu (khong chay SDSS).
# Muc dich: xem canh nhau roi CHON 1 style. Tat ca doc part2.curves (rolling coverage/method).
#   fig_A_focus.png : focus+context — baseline CLOUD (dai xam min-max) + Adaptive dam (khuyen nghi)
#   fig_B_dev.png   : deviation-from-target (cov - 90%), can giua 0 -> bien do dao dong doc truc tiep
#   fig_C_box.png   : boxplot phan bo rolling-coverage/method (spread = dao dong), Adaptive noi bat
# BO Static split-CP khoi ca 3 (chi so cac baseline HIEN DAI voi Adaptive).
# =====================================================================================
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANDIDATES = [
    "/kaggle/input/datasets/hipinhththu/astro-results/astro_diagnostic_results.json",
    "/kaggle/input/astro-results/astro_diagnostic_results.json",
    "results/astro_diagnostic_results.json",
    "astro_diagnostic_results.json",
]
JSON_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if JSON_PATH is None:
    raise FileNotFoundError("Khong thay astro_diagnostic_results.json — sua CANDIDATES.")
print("[load]", JSON_PATH)
D = json.load(open(JSON_PATH))
COVERAGE = 1 - 0.1
cp = D["part2"]["cp"]
curves = {k: np.asarray(v, float) for k, v in D["part2"]["curves"].items()}
T = max(len(v) for v in curves.values())
OUT_DIR = "figures" if os.path.isdir("figures") else ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")

ADAPT = "Adaptive PB-JCI Online (ours)"
STATIC = "Static split-CP (no update)"
# baseline hien dai = tat ca tru Adaptive va Static, giu thu tu on dinh
BASE_ORDER = ["ACI (Gibbs-Candes 2021)", "NexCP (Barber 2023)", "FACI (Gibbs-Candes 2024)",
              "SAOCP (Bhatnagar 2023)", "COP (Hu 2026)", "Rolling-Origin CP (2026)",
              "PB-JCI Online-Fixed (ours)"]
BASES = [n for n in BASE_ORDER if n in curves]
short = lambda n: n.split(" (")[0]

def save(fig, stem):
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{stem}.{ext}"); fig.savefig(p, dpi=200, bbox_inches="tight")
    print("WROTE", os.path.join(OUT_DIR, stem + ".png")); plt.close(fig)

# ---------------------------------------------------------------- A. FOCUS + CONTEXT (cloud)
# Baseline = dai xam [min,max] qua cac baseline tai moi t (context "dam may dao dong")
# + duong xam manh mo tung baseline; Adaptive = duong cam dam noi bat.
M = np.vstack([curves[n][:T] for n in BASES])
bmin, bmax, bmean = M.min(0), M.max(0), M.mean(0)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.axvspan(cp, T, color="#D55E00", alpha=0.05)
ax.fill_between(np.arange(T), bmin, bmax, color="#9AA0A6", alpha=0.30, lw=0,
                label="baseline cloud (min-max)", zorder=1)
for n in BASES:                                          # tung baseline mo, cung mau xam
    ax.plot(curves[n][:T], color="#9AA0A6", lw=0.7, alpha=0.55, zorder=2)
ax.plot(bmean, color="#5F6368", lw=1.0, ls="-", alpha=0.9, label="baseline mean", zorder=3)
ax.plot(curves[ADAPT][:T], color="#D55E00", lw=2.6, label=short(ADAPT), zorder=6)
ax.axhline(COVERAGE, color="0.2", ls="--", lw=1, label=f"target {COVERAGE:.0%}")
ax.axvline(cp, color="0.5", ls=":", lw=0.9)
ax.set_xlabel("stream step (khoi cam = shallow survey)"); ax.set_ylabel("rolling coverage (w=50)")
ax.set_ylim(0.4, 1.04); ax.legend(ncol=2, fontsize=8, loc="lower left")
ax.set_title("A. Focus+context: Adaptive vs baseline cloud (deep->shallow)", fontsize=9.5)
fig.tight_layout(); save(fig, "fig_A_focus")

# ---------------------------------------------------------------- B. DEVIATION FROM TARGET
# Ve (rolling cov - target): bien do dao dong doc truc tiep quanh 0. Baseline xam, Adaptive dam.
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.axvspan(cp, T, color="#D55E00", alpha=0.05)
ax.axhline(0, color="0.2", ls="--", lw=1, label="target (0 = dung 90%)")
for n in BASES:
    ax.plot(curves[n][:T] - COVERAGE, color="#9AA0A6", lw=0.9, alpha=0.6, zorder=2)
ax.plot([], [], color="#9AA0A6", lw=1.2, alpha=0.7, label="baselines hien dai")   # proxy legend
ax.plot(curves[ADAPT][:T] - COVERAGE, color="#D55E00", lw=2.4, label=short(ADAPT), zorder=6)
ax.axvline(cp, color="0.5", ls=":", lw=0.9)
ax.set_xlabel("stream step"); ax.set_ylabel("rolling coverage - target")
ax.set_ylim(-0.5, 0.15); ax.legend(fontsize=8, loc="lower left")
ax.set_title("B. Deviation-from-target: bien do dao dong (0 = dat 90%)", fontsize=9.5)
fig.tight_layout(); save(fig, "fig_B_dev")

# ---------------------------------------------------------------- C. BOXPLOT (spread = dao dong)
# Moi method 1 box cua phan bo rolling-coverage qua t (chi lay post-shift de doc recovery).
# Box hep + can 90% = it dao dong / on dinh. Adaptive to mau noi bat.
seg = slice(cp, T)                                       # chi vung post-shift (shallow)
order = BASES + [ADAPT]
data = [curves[n][seg] for n in order]
meds = [np.median(d) for d in data]
srt = np.argsort(meds)                                   # sap theo median cho de doc
order = [order[i] for i in srt]; data = [data[i] for i in srt]
fig, ax = plt.subplots(figsize=(7, 3.8))
bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6, showfliers=False)
for i, n in enumerate(order):
    is_ad = (n == ADAPT); face = "#D55E00" if is_ad else "#C8CBD0"
    bp["boxes"][i].set(facecolor=face, alpha=0.95 if is_ad else 0.8, edgecolor="0.3")
    bp["medians"][i].set(color="white" if is_ad else "0.3", lw=2 if is_ad else 1)
ax.axhline(COVERAGE, color="#0072B2", ls="--", lw=1.2, label=f"target {COVERAGE:.0%}")
ax.set_xticklabels([short(n) for n in order], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("rolling coverage (w=50), post-shift"); ax.set_ylim(0.3, 1.04)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("C. Boxplot: spread rolling-coverage/method (hep+giua 90% = it dao dong)", fontsize=9.5)
fig.tight_layout(); save(fig, "fig_C_box")

print("\nDONE — xem fig_A_focus / fig_B_dev / fig_C_box (.png) roi chon 1 style.")
print("Ghi chu: two-panel (coverage + width theo thoi gian) va band mean+-std qua seeds")
print("         CAN re-run pipeline de luu width-curves + per-seed curves (JSON hien khong co).")

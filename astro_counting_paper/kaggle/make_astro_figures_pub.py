# =====================================================================================
# make_astro_figures_pub.py — ban PUBLICATION-QUALITY (thay make_astro_figures_all.py).
# Sua loi tham my: despine, luoi ngang mo, palette Paul Tol, baseline gom 1 mau xam,
# SAOCP tach duong rieng (over-cover), 1 accent cho Adaptive, nhan tieng Anh.
# Doc part2.curves tu JSON (khong chay lai SDSS). Xuat 2 hinh:
#   fig_pub_recovery.png : focus+context (hinh chinh)
#   fig_pub_box.png      : boxplot spread post-shift (hinh phu)
# =====================================================================================
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- style tap chi (ap dung toan cuc) ----
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a", "text.color": "#1a1a1a",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white",
})
# Paul Tol palette (colorblind-safe, chuan hoc thuat)
TOL = dict(blue="#4477AA", cyan="#66CCEE", green="#228833", yellow="#CCBB44",
           red="#EE6677", purple="#AA3377", grey="#BBBBBB", wine="#882255", teal="#44AA99")
OURS   = "#CC3311"     # accent Adaptive: vermilion tram (Tol "vibrant red"), khong choi
BASELN = "#B0B4BA"     # baseline gom 1 mau xam mo
SAOCPC = "#AA3377"     # SAOCP: tim tram, duong rieng (over-cover)
SHIFTBG = "#F4F4F2"    # nen vung post-shift (rat nhat)

CANDIDATES = [
    "/kaggle/input/datasets/hipinhththu/astro-results/astro_diagnostic_results.json",
    "/kaggle/input/astro-results/astro_diagnostic_results.json",
    "results/astro_diagnostic_results.json", "astro_diagnostic_results.json",
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
SAOCP = "SAOCP (Bhatnagar 2023)"
LABEL = {"ACI (Gibbs-Candes 2021)": "ACI", "NexCP (Barber 2023)": "NexCP",
         "FACI (Gibbs-Candes 2024)": "FACI", SAOCP: "SAOCP",
         "COP (Hu 2026)": "COP", "Rolling-Origin CP (2026)": "Rolling-Origin",
         "PB-JCI Online-Fixed (ours)": "PB-JCI Fixed", ADAPT: "Adaptive PB-JCI (ours)"}
# baseline "thuong" = tat ca tru Adaptive, SAOCP, Static
NORMAL = [n for n in LABEL if n not in (ADAPT, SAOCP) and n in curves]

def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"))
    print("WROTE", os.path.join(OUT_DIR, stem + ".png")); plt.close(fig)

# =============================================================== HINH CHINH: focus+context
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.xaxis.grid(False)                                          # chi luoi ngang
ax.axvspan(cp, T, color=SHIFTBG, zorder=0)                   # vung shallow (nen rat nhat)
# baseline thuong: gom 1 mau xam, mot proxy legend duy nhat
for n in NORMAL:
    ax.plot(curves[n][:T], color=BASELN, lw=1.1, alpha=0.9, zorder=2, solid_capstyle="round")
ax.plot([], [], color=BASELN, lw=1.6, label="Modern baselines")   # proxy
# SAOCP: duong rieng (minh hoa over-cover)
if SAOCP in curves:
    ax.plot(curves[SAOCP][:T], color=SAOCPC, lw=1.4, ls=(0, (5, 2)), alpha=0.9,
            zorder=3, label="SAOCP (over-covers)")
# Adaptive: accent, day, tren cung
ax.plot(curves[ADAPT][:T], color=OURS, lw=2.6, zorder=6,
        solid_capstyle="round", label="Adaptive PB-JCI (ours)")
ax.axhline(COVERAGE, color="#333333", ls=(0, (4, 3)), lw=1.1, zorder=4, label="Target 90%")
ax.axvline(cp, color="#666666", ls=":", lw=1.0, zorder=1)
ax.annotate("shallow survey\n(post-shift)", xy=(cp + (T - cp) * 0.5, 0.45),
            ha="center", va="center", fontsize=8.5, color="#8a8a8a")
ax.set_xlabel("Stream step"); ax.set_ylabel("Rolling coverage (window = 50)")
ax.set_xlim(0, T); ax.set_ylim(0.4, 1.03)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.legend(loc="lower left", ncol=2, handlelength=1.8, columnspacing=1.4, borderaxespad=0.3)
ax.set_title("Coverage recovery after a deep→shallow survey shift", pad=8)
fig.tight_layout(); save(fig, "fig_pub_recovery")

# =============================================================== HINH PHU: boxplot spread
seg = slice(cp, T)
order = [n for n in LABEL if n not in ("Static split-CP (no update)",) and n in curves]
data = [curves[n][seg] for n in order]
srt = np.argsort([np.median(d) for d in data])
order = [order[i] for i in srt]; data = [data[i] for i in srt]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.xaxis.grid(False)
bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.62, showfliers=False,
                medianprops=dict(lw=1.4), whiskerprops=dict(color="#666666", lw=1.0),
                capprops=dict(color="#666666", lw=1.0), boxprops=dict(lw=0.8))
for i, n in enumerate(order):
    is_ad = (n == ADAPT); is_sa = (n == SAOCP)
    face = OURS if is_ad else (SAOCPC if is_sa else BASELN)
    bp["boxes"][i].set(facecolor=face, alpha=0.9 if (is_ad or is_sa) else 0.75, edgecolor="#555555")
    bp["medians"][i].set(color="white" if (is_ad or is_sa) else "#444444")
ax.axhline(COVERAGE, color=TOL["blue"], ls=(0, (4, 3)), lw=1.2, label="Target 90%")
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels([LABEL[n] for n in order], rotation=28, ha="right")
ax.set_ylabel("Rolling coverage (post-shift)"); ax.set_ylim(0.3, 1.03)
ax.legend(loc="lower left")
ax.set_title("Coverage stability by method (tight box near 90% = stable)", pad=8)
fig.tight_layout(); save(fig, "fig_pub_box")

print("\nDONE — fig_pub_recovery (chinh) + fig_pub_box (phu).")

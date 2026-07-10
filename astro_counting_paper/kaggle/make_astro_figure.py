# =====================================================================================
# make_astro_figure.py — VE LAI hinh recovery TU JSON da luu (khong chay lai SDSS).
# Dung khi da co astro_diagnostic_results.json (vd Kaggle dataset input).
# Bo Static split-CP -> so cac baseline HIEN DAI voi Adaptive PB-JCI Online.
# =====================================================================================
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- tim JSON: uu tien Kaggle dataset input, roi thu muc local ---
CANDIDATES = [
    "/kaggle/input/datasets/hipinhththu/astro-results/astro_diagnostic_results.json",
    "/kaggle/input/astro-results/astro_diagnostic_results.json",
    "results/astro_diagnostic_results.json",
    "astro_diagnostic_results.json",
]
JSON_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if JSON_PATH is None:
    raise FileNotFoundError("Khong thay astro_diagnostic_results.json — sua CANDIDATES cho dung path.")
print("[load]", JSON_PATH)
D = json.load(open(JSON_PATH))

COVERAGE = 1 - 0.1                                   # target 90% (nhu ALPHA=0.1 goc)
cp   = D["part2"]["cp"]
curves = D["part2"]["curves"]                        # {ten method: [rolling coverage ...]}
T = max(len(v) for v in curves.values())

# --- output: uu tien thu muc figures/, roi /kaggle/working ---
OUT_DIR = "figures" if os.path.isdir("figures") else ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")

# --- mau: baseline hien dai mo, Adaptive dam noi bat (BO Static) ---
col = {"ACI (Gibbs-Candes 2021)": "#0072B2", "NexCP (Barber 2023)": "#56B4E9",
       "FACI (Gibbs-Candes 2024)": "#009E73", "SAOCP (Bhatnagar 2023)": "#E69F00",
       "COP (Hu 2026)": "#CC79A7", "Rolling-Origin CP (2026)": "#8C564B",
       "PB-JCI Online-Fixed (ours)": "#999999", "Adaptive PB-JCI Online (ours)": "#D55E00"}

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.axvspan(cp, T, color="#D55E00", alpha=0.05)          # vung shallow (post-shift)
for name in col:
    if name not in curves:                              # phong khi JSON thieu 1 method
        print("  (bo qua, khong co trong JSON):", name); continue
    is_ad = "Adaptive" in name
    ax.plot(curves[name], color=col[name], lw=2.4 if is_ad else 1.0,
            label=name.split(" (")[0], zorder=6 if is_ad else 3, alpha=1.0 if is_ad else 0.7)
ax.axhline(COVERAGE, color="0.2", ls="--", lw=1, label=f"target {COVERAGE:.0%}")
ax.axvline(cp, color="0.5", ls=":", lw=0.9)
ax.set_xlabel("stream step (khoi cam = shallow survey)")
ax.set_ylabel("rolling coverage (w=50)")
ax.set_ylim(0.4, 1.04); ax.legend(ncol=3, fontsize=7, loc="lower left")
ax.set_title("Astro: baseline hien dai vs Adaptive PB-JCI Online (deep->shallow shift)", fontsize=9.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(OUT_DIR, f"astro_recovery.{ext}")
    fig.savefig(p, dpi=200, bbox_inches="tight"); print("WROTE", p)

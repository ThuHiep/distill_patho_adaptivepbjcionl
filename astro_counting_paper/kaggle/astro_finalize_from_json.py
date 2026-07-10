# =====================================================================================
# astro_finalize_from_json.py — TU JSON da luu, xuat 2 thu cho paper (KHONG chay lai SDSS):
#   (1) Bang mean +- std (coverage, Winkler) moi method  -> dien vao Bang 1 cua paper.
#   (2) Hinh Coverage-vs-DEPTH (fig_pub_depth.png)        -> hinh curve reviewer thich.
# JSON da luu san 'crosstable' (co cov_std/wink_std) va 'depth_sweep'.
# =====================================================================================
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "axes.edgecolor": "#222222",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#ECECEC", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white",
    "lines.solid_capstyle": "round",
})
# Okabe-Ito (colorblind-safe): xanh=de xuat, cam=SAOCP(over-cover), xam dam=Static, xam nhat=baseline nen
OURS, STATICC, SAOCPC, BASELN = "#0072B2", "#4D4D4D", "#E69F00", "#BFBFBF"

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
OUT_DIR = "figures" if os.path.isdir("figures") else ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
ADAPT = "Adaptive PB-JCI Online (ours)"; SAOCP = "SAOCP (Bhatnagar 2023)"

# ---------------------------------------------------------- (1) mean +- std
tab = D["crosstable"]
ORDER = ["Static split-CP (no update)", "ACI (Gibbs-Candes 2021)", "NexCP (Barber 2023)",
         "FACI (Gibbs-Candes 2024)", "SAOCP (Bhatnagar 2023)", "COP (Hu 2026)",
         "Rolling-Origin CP (2026)", "PB-JCI Online-Fixed (ours)", ADAPT]
print("\n=== Coverage / Winkler (mean +- std qua 5 seed) — dien vao Bang 1 ===")
print(f"{'Method':30s} | {'Coverage %':>14s} | {'Winkler':>14s}")
print("-" * 64)
for n in ORDER:
    if n not in tab: continue
    d = tab[n]
    cs = d.get("cov_std", 0.0); ws = d.get("wink_std", 0.0)
    print(f"{n.split(' (')[0]:30s} | {d['cov']:6.1f} +- {cs:4.1f}   | {d['wink']:7.2f} +- {ws:5.2f}")
print("-" * 64)
print("LaTeX goi y (Adaptive): $%.1f \\pm %.1f$ coverage, Winkler $%.2f \\pm %.2f$"
      % (tab[ADAPT]['cov'], tab[ADAPT].get('cov_std', 0), tab[ADAPT]['wink'], tab[ADAPT].get('wink_std', 0)))

# ---------------------------------------------------------- (2) Coverage vs DEPTH
ds = D["depth_sweep"]                                   # {depth_str: {static:..., covs:{name:cov}}}
depths = sorted(ds.keys(), key=float)
xs = [float(x) for x in depths]
methods = list(ds[depths[0]]["covs"].keys())
fig, ax = plt.subplots(figsize=(5.4, 3.5))
ax.axhline(90, color="#9A9A9A", ls=(0, (3, 3)), lw=0.9, zorder=1, label="Target (90%)")
for m in methods:                                                      # xanh=Adaptive, cam=SAOCP, xam nhat=con lai
    is_ad = "Adaptive" in m; is_sa = "SAOCP" in m
    ys = [ds[d]["covs"][m] for d in depths]
    ax.plot(xs, ys,
            color=OURS if is_ad else (SAOCPC if is_sa else BASELN),
            lw=1.9 if is_ad else (1.3 if is_sa else 0.9),
            marker=("o" if is_ad else ("^" if is_sa else None)), ms=4 if is_ad else 3.5,
            alpha=1.0 if (is_ad or is_sa) else 0.6,
            zorder=6 if is_ad else (4 if is_sa else 2),
            label=("Adaptive PB-JCI" if is_ad else ("SAOCP" if is_sa else None)))
ax.plot(xs, [ds[d]["static"] for d in depths],
        color=STATICC, lw=1.3, ls=(0, (5, 2)), marker="s", ms=3.5, zorder=5, label="Static split-CP")
ax.plot([], [], color=BASELN, lw=1.0, label="Other baselines")
ax.set_xlabel("Shift strength (DEPTH)"); ax.set_ylabel("Joint coverage (%)")
ax.set_xticks(xs); ax.set_ylim(0, 100); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.margins(x=0.04)
ax.legend(loc="lower left", handlelength=1.9, borderaxespad=0.4, labelspacing=0.35)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT_DIR, f"fig_pub_depth.{ext}"))
print("\nWROTE", os.path.join(OUT_DIR, "fig_pub_depth.png")); plt.close(fig)

# ---------------------------------------------------------- (3) fig_pub_recovery (baseline vs Adaptive)
cp = D["part2"]["cp"]; curves = {k: np.asarray(v, float) for k, v in D["part2"]["curves"].items()}
T = max(len(v) for v in curves.values())
STATIC = "Static split-CP (no update)"
GREY = ["ACI (Gibbs-Candes 2021)", "NexCP (Barber 2023)", "FACI (Gibbs-Candes 2024)",
        "COP (Hu 2026)", "Rolling-Origin CP (2026)", "PB-JCI Online-Fixed (ours)"]
fig, ax = plt.subplots(figsize=(6.2, 3.3))
ax.axvspan(cp, T, color="#F7F7F7", zorder=0)                                # nen rat nhat cho giai doan shallow
ax.axhline(0.9, color="#9A9A9A", ls=(0, (3, 3)), lw=0.9, zorder=1, label="Target (90%)")
for n in GREY:                                                              # baseline con lai: xam nhat lam nen
    if n in curves: ax.plot(curves[n][:T], color=BASELN, lw=0.9, alpha=0.55, zorder=2)
ax.plot([], [], color=BASELN, lw=1.0, label="Other baselines")
if STATIC in curves:                                                        # Static: xam dam, net dut
    ax.plot(curves[STATIC][:T], color=STATICC, lw=1.3, ls=(0, (5, 2)), zorder=4, label="Static split-CP")
if SAOCP in curves:                                                         # SAOCP: cam (over-cover)
    ax.plot(curves[SAOCP][:T], color=SAOCPC, lw=1.3, zorder=4, label="SAOCP")
ax.plot(curves[ADAPT][:T], color=OURS, lw=1.9, zorder=6, label="Adaptive PB-JCI")
ax.axvline(cp, color="#777777", ls="-", lw=0.8, zorder=3)                   # thoi diem shift
ax.text(cp + 5, 0.985, "shift", fontsize=7.5, color="#777777", va="top", ha="left")
ax.set_xlabel("Stream step"); ax.set_ylabel("Rolling coverage")
ax.set_xlim(0, T - 1); ax.set_ylim(0.45, 1.0); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.legend(loc="lower left", ncol=2, handlelength=1.9, borderaxespad=0.4, labelspacing=0.35)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT_DIR, f"fig_pub_recovery.{ext}"))
print("WROTE", os.path.join(OUT_DIR, "fig_pub_recovery.png"))

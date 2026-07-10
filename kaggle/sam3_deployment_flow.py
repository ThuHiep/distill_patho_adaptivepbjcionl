"""
Deployment-lifecycle diagram for PB-JCI (no data needed).
With labels  -> Adaptive PB-JCI Online self-calibrates back to ~90%.
Without labels -> feature shift detector raises an alert -> expert audit ->
offline recalibration. Both paths yield a valid 90% interval and loop back.
  python kaggle/sam3_deployment_flow.py  ->  figures/F5_deployment_flow.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(exist_ok=True)

BLUE, GREEN, ORANGE, RED, GRAY = "#cfe2f3", "#d9ead3", "#fce5cd", "#f4cccc", "#e8e8e8"


def box(ax, x, y, text, w=3.0, h=0.8, fc=BLUE, fs=8.0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.04", fc=fc, ec="black", lw=1.1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3)


def diamond(ax, x, y, text, w=2.6, h=1.5, fc="#fff2cc", fs=8.0):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                 closed=True, fc=fc, ec="black", lw=1.1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, p1, p2, label="", lc="black", off=(0.0, 0.0)):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=lc, lw=1.5,
                                mutation_scale=14, shrinkA=1, shrinkB=1), zorder=1)
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1]
        ax.text(mx, my, label, fontsize=8.0, fontweight="bold", color=lc,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"), zorder=4)


fig, ax = plt.subplots(figsize=(8.4, 7.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

LX, RX = 2.7, 7.1

# main spine
box(ax, LX, 9.30, "New histopathology image", w=3.0, h=0.66)
box(ax, LX, 8.15, "Predict count +\n90% prediction interval", w=3.2, h=0.92)
diamond(ax, LX, 6.70, "Label feedback\navailable?")

# no-label (offline) path
box(ax, LX, 5.15, "Feature shift detector\n(MMD$^2$ / Wasserstein / Energy)\nno labels needed", w=3.5, h=1.06, fc=ORANGE)
box(ax, LX, 3.85, "Alert: domain shift detected", w=3.2, h=0.66, fc=RED)
box(ax, LX, 2.70, "Expert audit +\ncollect new-domain labels", w=3.2, h=0.92, fc=GRAY)
box(ax, LX, 1.55, "Offline recalibration\nand redeploy", w=3.0, h=0.86, fc=GRAY)

# with-label (online) path
box(ax, RX, 6.70, "Adaptive PB-JCI Online\nif recent coverage $<90\\%$:\nshrink window $\\to$ adapt $\\to\\ {\\sim}90\\%$",
    w=3.5, h=1.5, fc=GREEN)

# convergence
box(ax, 5.0, 0.50, "Valid 90% prediction interval   $\\circlearrowleft$   return to monitoring",
    w=6.8, h=0.74, fc=BLUE)

# arrows
arrow(ax, (LX, 8.97), (LX, 8.61))
arrow(ax, (LX, 7.69), (LX, 7.45))
arrow(ax, (4.00, 6.70), (5.35, 6.70), label="yes", lc="#38761d", off=(0, 0.28))
arrow(ax, (LX, 5.95), (LX, 5.68), label="no", lc="#990000", off=(0.45, 0.02))
arrow(ax, (LX, 4.62), (LX, 4.18))
arrow(ax, (LX, 3.52), (LX, 3.16))
arrow(ax, (LX, 2.24), (LX, 1.98))
arrow(ax, (LX, 1.12), (LX, 0.87))
arrow(ax, (RX, 5.95), (RX, 0.87))   # online path -> valid interval (long right spine)

fig.tight_layout()
fig.savefig(FIGDIR / "F5_deployment_flow.png", dpi=200, bbox_inches="tight")
print(f"wrote {FIGDIR / 'F5_deployment_flow.png'}")

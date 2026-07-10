# =====================================================================================
# make_astro_dataset_viz.py — VISUALIZE DATASET + BACKBONE (SDSS DR17 catalog).
# SDSS query = CATALOG (khong co anh) -> visualize cac dai luong THAT method dung, + (tuy chon)
# keo anh cutout that tu SkyServer de "thay mat" du lieu.
# Chay Kaggle (Internet On) voi USE_SDSS=True. Offline: sim fallback de test bo cuc.
#   fig_pub_dataset.png : 4-panel (sky map | SNR dist | completeness+shift | probPSF split)
#   fig_pub_cutouts.png : dai anh SDSS that (chi khi FETCH_CUTOUTS=True + internet)
# =====================================================================================
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

USE_SDSS = True           # Kaggle: True = query SDSS that. Doi False de test bo cuc offline.
FETCH_CUTOUTS = True      # keo anh cutout THAT tu SkyServer (can internet + requests + PIL).
# LUU Y: FETCH_CUTOUTS chi co nghia khi USE_SDSS=True (ra/dec that). Neu de USE_SDSS=False thi
#        cutout se lay theo toa do SIM ngau nhien + nhan sao/thien ha GIA -> KHONG dung. Tren
#        Kaggle giu ca hai = True. Test offline: dat CA HAI = False.
N_CUTOUTS = 4
SDSS_REGION = dict(ra0=150.0, ra1=152.0, dec0=0.0, dec1=2.0)
SNR0, SNRW, DEPTH_FACTOR = 5.0, 1.5, 3.0

# ---- style tap chi (dong bo make_astro_figures_pub.py) ----
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "axes.grid": True, "grid.color": "#ECECEC", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white",
})
C_STAR, C_GAL, C_ACC, C_SHIFT = "#4477AA", "#CC7722", "#CC3311", "#AA3377"
OUT_DIR = "figures" if os.path.isdir("figures") else ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")

def _ensure(mod, pip_name=None):
    """Tu cai package neu Kaggle chua co (khoi phai them cell !pip rieng)."""
    import importlib, subprocess, sys
    try:
        importlib.import_module(mod)
    except ImportError:
        print(f"[setup] cai {pip_name or mod} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or mod], check=True)

def load_catalog():
    """Tra ve dict flat: ra, dec, is_star(bool), probPSF, snr, mag."""
    if USE_SDSS:
        _ensure("astroquery")
        from astroquery.sdss import SDSS
        r = SDSS_REGION
        sql = (f"SELECT p.ra,p.dec,p.type,p.probPSF_r,p.psfFlux_r,p.psfFluxIvar_r FROM PhotoObjAll p "
               f"WHERE p.ra BETWEEN {r['ra0']} AND {r['ra1']} AND p.dec BETWEEN {r['dec0']} AND {r['dec1']} "
               f"AND p.mode=1 AND p.clean=1 AND p.type IN (3,6) AND p.psfFluxIvar_r>0 AND p.psfFlux_r>0")
        t = SDSS.query_sql(sql, data_release=17)
        flux = np.asarray(t["psfFlux_r"], float); ivar = np.asarray(t["psfFluxIvar_r"], float)
        return dict(ra=np.asarray(t["ra"], float), dec=np.asarray(t["dec"], float),
                    is_star=np.asarray(t["type"]) == 6,
                    probPSF=np.clip(np.asarray(t["probPSF_r"], float), 0, 1),
                    snr=flux * np.sqrt(ivar), mag=22.5 - 2.5 * np.log10(np.clip(flux, 1e-3, None)))
    # ---- sim fallback (co ra/dec de test bo cuc) ----
    rng = np.random.default_rng(0); N = 12000; r = SDSS_REGION
    ra = rng.uniform(r["ra0"], r["ra1"], N); dec = rng.uniform(r["dec0"], r["dec1"], N)
    is_star = rng.random(N) < 0.42; mag = 18.5 + rng.exponential(1.5, N)
    snr = 10 ** (0.4 * (23.5 - mag))
    sep = 1 / (1 + np.exp(-(snr - 3.0)))
    probPSF = np.clip(np.where(is_star, 0.5 + 0.5 * sep, 0.5 - 0.5 * sep) + rng.normal(0, 0.05, N), 0, 1)
    return dict(ra=ra, dec=dec, is_star=is_star, probPSF=probPSF, snr=snr, mag=mag)

cat = load_catalog()
ra, dec, star = cat["ra"], cat["dec"], cat["is_star"]
snr, probPSF, mag = cat["snr"], cat["probPSF"], cat["mag"]
print(f"[cat] {len(ra)} nguon | sao={int(star.sum())} thien ha={int((~star).sum())} | median SNR={np.median(snr):.1f}")

# ================================================================ 4-PANEL DATASET OVERVIEW
fig = plt.figure(figsize=(9.6, 6.4))
gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24)

# (a) sky map RA/Dec
ax = fig.add_subplot(gs[0, 0]); ax.xaxis.grid(False); ax.yaxis.grid(False)
ax.scatter(ra[~star], dec[~star], s=2, c=C_GAL, alpha=0.35, lw=0, label="Galaxy")
ax.scatter(ra[star], dec[star], s=2, c=C_STAR, alpha=0.45, lw=0, label="Star")
ax.set_xlabel("RA (deg)"); ax.set_ylabel("Dec (deg)"); ax.set_title("(a) Sky map of catalog sources")
lg = ax.legend(markerscale=4, loc="upper right");
ax.set_aspect("equal", adjustable="box")

# (b) SNR distribution (log) + nguong 5-sigma + vi tri sau shift
ax = fig.add_subplot(gs[0, 1]); ax.xaxis.grid(False)
bins = np.logspace(np.log10(max(snr.min(), 1)), np.log10(np.percentile(snr, 99.5)), 45)
ax.hist(snr[star], bins=bins, color=C_STAR, alpha=0.6, label="Star")
ax.hist(snr[~star], bins=bins, color=C_GAL, alpha=0.55, label="Galaxy")
ax.axvline(SNR0, color=C_ACC, ls="--", lw=1.3, label="5σ detection")
ax.axvline(SNR0 * DEPTH_FACTOR, color=C_SHIFT, ls=":", lw=1.3,
           label=f"eff. limit after shift (×{DEPTH_FACTOR:g})")
ax.set_xscale("log"); ax.set_xlabel("SNR (psfFlux·√ivar)"); ax.set_ylabel("count")
ax.set_title("(b) Real SNR distribution"); ax.legend(loc="upper right")

# (c) completeness model + cu shift (giai thich co che)
ax = fig.add_subplot(gs[1, 0])
xs = np.logspace(np.log10(1), np.log10(300), 300)
pdeep = 1 / (1 + np.exp(-(xs - SNR0) / SNRW))
pshal = 1 / (1 + np.exp(-(xs / DEPTH_FACTOR - SNR0) / SNRW))
ax.plot(xs, pdeep, color=C_STAR, lw=2.2, label="deep survey (CAL)")
ax.plot(xs, pshal, color=C_SHIFT, lw=2.2, ls="--", label="shallow survey (TEST)")
ax.fill_between(xs, pshal, pdeep, color=C_SHIFT, alpha=0.10)
ax.axvline(SNR0, color="#888", ls=":", lw=1)
ax.set_xscale("log"); ax.set_xlabel("SNR"); ax.set_ylabel("p(detect)")
ax.set_ylim(-0.03, 1.03); ax.set_title("(c) Completeness model = induced depth shift")
ax.legend(loc="lower right")

# (d) probPSF split (backbone classification confidence)
ax = fig.add_subplot(gs[1, 1]); ax.xaxis.grid(False)
b2 = np.linspace(0, 1, 41)
ax.hist(probPSF[star], bins=b2, color=C_STAR, alpha=0.6, label="true Star (type=6)")
ax.hist(probPSF[~star], bins=b2, color=C_GAL, alpha=0.55, label="true Galaxy (type=3)")
ax.set_xlabel("probPSF_r  (backbone P[star])"); ax.set_ylabel("count")
ax.set_title("(d) Backbone star/galaxy confidence"); ax.legend(loc="upper center")

fig.suptitle("SDSS DR17 PhotoObj — dataset & backbone overview", fontsize=11.5, y=0.995)
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT_DIR, f"fig_pub_dataset.{ext}"))
print("WROTE", os.path.join(OUT_DIR, "fig_pub_dataset.png")); plt.close(fig)

# ================================================================ (tuy chon) ANH CUTOUT THAT
if FETCH_CUTOUTS:
    try:
        _ensure("requests"); _ensure("PIL", "pillow")
        import requests, io
        from PIL import Image
        # chon vai nguon sang nhat: 2 sao + 2 thien ha
        idx_s = np.where(star)[0][np.argsort(snr[star])[::-1][:N_CUTOUTS // 2]]
        idx_g = np.where(~star)[0][np.argsort(snr[~star])[::-1][:N_CUTOUTS - N_CUTOUTS // 2]]
        picks = list(idx_s) + list(idx_g)
        fig, axes = plt.subplots(1, len(picks), figsize=(2.4 * len(picks), 2.7))
        if len(picks) == 1: axes = [axes]
        for ax, i in zip(axes, picks):
            url = ("https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?"
                   f"ra={ra[i]:.5f}&dec={dec[i]:.5f}&scale=0.2&width=128&height=128")
            im = Image.open(io.BytesIO(requests.get(url, timeout=30).content))
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{'Star' if star[i] else 'Galaxy'}\nSNR={snr[i]:.0f}", fontsize=8.5)
        fig.suptitle("Real SDSS image cutouts (illustration only — not used by method)", fontsize=9.5)
        fig.tight_layout()
        for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT_DIR, f"fig_pub_cutouts.{ext}"))
        print("WROTE", os.path.join(OUT_DIR, "fig_pub_cutouts.png")); plt.close(fig)
    except Exception as e:
        print("(bo qua cutouts:", e, ")")

print("\nDONE — Kaggle: dat USE_SDSS=True (+ FETCH_CUTOUTS=True neu muon anh that).")

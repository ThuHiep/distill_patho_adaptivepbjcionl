"""
Builder -> sam3_pathosam_trial.ipynb

DISCOVERY notebook for PathoSAM (computational-cell-analytics/patho-sam, built on
micro-sam). Goal = answer 3 questions before committing to a full pipeline:

  (1) Does the GENERALIST give nucleus TYPE (5-class) or instance-only?
  (2) How good is its counting on PanNuke Fold 3 (MAE vs GT) — i.e. is it the
      "strong predictor" we want vs SAM3 (~weak)?
  (3) Tissue distribution of Fold 3 -> how many COLON images (the only possible
      leakage via Lizard) to exclude.

Leakage note: PathoSAM generalist is trained on CoNSeP/CPM17/Lizard/MoNuSeg/
MoNuSAC/TNBC — NOT PanNuke (documented) -> fold-clean by construction. Only
residual = Lizard contains PanNuke-colon -> handle by excluding colon tissue.

Internet ON (downloads weights). GPU T4. Attach: hipinhththu/pannuke.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "sam3_pathosam_trial.ipynb"
LIB_DIR = Path(__file__).parent / "lib"
PANNUKE_LOADER = "%%writefile pannuke_loader.py\n" + (LIB_DIR / "pannuke_loader.py").read_text(encoding="utf-8")

def md(*lines):
    src = [ln if ln.endswith("\n") else ln + "\n" for ln in lines]
    if src: src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(body):
    lines = body.lstrip("\n").splitlines(keepends=True)
    if lines: lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

cells = []
cells.append(md(
    "# PathoSAM trial — strong CLEAN backbone for PB-JCI (discovery)",
    "",
    "Answer 3 questions: (1) does the generalist give 5-class TYPE or instance-only?",
    "(2) counting MAE on PanNuke Fold 3 vs SAM3? (3) how many colon images to exclude",
    "(Lizard overlap)?",
    "",
    "PathoSAM generalist trained on 6 datasets **excluding PanNuke** (documented) → fold-clean.",
    "Only residual = Lizard contains PanNuke-colon → exclude colon tissue.",
    "",
    "**Internet ON** (downloads weights). GPU T4. Attach `hipinhththu/pannuke`.",
))

cells.append(md(
    "## 00 — Install patho-sam / micro-sam FROM SOURCE (not on PyPI) + clone loader",
    "",
    "micro-sam is a conda-ecosystem package (not on PyPI). We install it + deps from",
    "GitHub source via pip. `nifty`/`elf` are conda-only; if a later import fails on them,",
    "we fall back (AIS inference often doesn't need nifty). Errors are surfaced, not hidden.",
))
cells.append(code('''
import subprocess, sys, os
def pipi(*pkgs):
    r = subprocess.run([sys.executable,"-m","pip","install","-q",*pkgs],
                       capture_output=True, text=True)
    tag = "OK " if r.returncode == 0 else "FAIL"
    print(f"[{tag}] pip install {' '.join(pkgs)[:70]}")
    if r.returncode != 0:
        print("   ", (r.stderr or r.stdout).strip().splitlines()[-1][:200])
    return r.returncode == 0

WORK = "/kaggle/working"; REPO = f"{WORK}/sam3_research"
if not os.path.exists(REPO):
    subprocess.run(["git","clone","https://github.com/duonguwu/sam3_research.git",REPO], check=True)

# deps from source (order matters): segment-anything -> elf -> torch-em -> micro-sam -> patho-sam
pipi("segment-anything")
pipi("git+https://github.com/constantinpape/elf.git")
pipi("git+https://github.com/constantinpape/torch-em.git")
pipi("git+https://github.com/computational-cell-analytics/micro-sam.git")
pipi("git+https://github.com/computational-cell-analytics/patho-sam.git")
pipi("-U","numpy>=2")   # guard numpy ABI

import torch, numpy as np
print("\\ntorch", torch.__version__, "| numpy", np.__version__, "| CUDA", torch.cuda.is_available())

# probe what actually imports
for mod in ["segment_anything","elf","torch_em","micro_sam","patho_sam"]:
    try:
        __import__(mod); print(f"  import {mod:18s} OK")
    except Exception as e:
        print(f"  import {mod:18s} FAIL -> {repr(e)[:120]}")
'''))

cells.append(code(PANNUKE_LOADER))
cells.append(code('''
import sys
if "." not in sys.path: sys.path.insert(0, ".")
from pannuke_loader import PanNukeFold, DEFAULT_ROOT, CELL_TYPES
fold3 = PanNukeFold(DEFAULT_ROOT, 3)
print(f"Fold 3: {len(fold3)} images")
'''))

cells.append(md(
    "## 01 — Discover available histopathology models (confirm the model name)",
    "",
    "micro-sam registers patho-sam models. Print the registry so we use the exact name",
    "(expected something like `vit_l_histopathology`).",
))
cells.append(code('''
try:
    from micro_sam.util import models
    reg = models()
    names = [n for n in reg.urls.keys()] if hasattr(reg, "urls") else list(reg)
    histo = [n for n in names if "histo" in n.lower() or "patho" in n.lower()]
    print("histopathology-related model names:")
    for n in histo: print("  ", n)
    if not histo:
        print("(none matched 'histo/patho'; full list sample:)", names[:20])
except Exception as e:
    print("could not introspect registry:", repr(e)[:200])
    print("-> we'll just try 'vit_l_histopathology' below.")
'''))

cells.append(md(
    "## 02 — Load generalist predictor + automatic instance segmenter (AIS)",
    "",
    "Generalists were trained with the instance-segmentation decoder → use `amg=False`",
    "(AIS). Downloads weights on first call.",
))
cells.append(code('''
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "vit_l_histopathology"   # change if cell 01 shows a different name
predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_NAME, device=device, amg=False)
print("loaded:", MODEL_NAME)

def pathosam_count(img_rgb):
    """img_rgb: HxWx3 uint8 -> instance label map + count."""
    inst = automatic_instance_segmentation(
        predictor=predictor, segmenter=segmenter, input_path=img_rgb, ndim=2,
    )
    inst = np.asarray(inst)
    n = int(len(np.unique(inst)) - (1 if (inst == 0).any() else 0))
    return inst, n
print("inference fn ready.")
'''))

cells.append(md("## 03 — Smoke: 5 Fold-3 images → PathoSAM count vs GT (+ tissue)"))
cells.append(code('''
import time
t0 = time.time()
for i in range(5):
    s = fold3[i]
    inst, n = pathosam_count(s["image"])
    gt = int(s["counts"].sum())
    print(f"  img {i} | tissue={s['tissue']:14s} | GT={gt:3d} | PathoSAM={n:3d} | inst_map {inst.shape}")
print(f"\\n5 images in {time.time()-t0:.1f}s. If counts track GT (tens) -> strong predictor works.")
'''))

cells.append(md(
    "## 04 — Does it output TYPE (5-class)? (instance vs semantic)",
    "",
    "Check whether the result carries per-nucleus type, or only instance labels. If",
    "instance-only → we add OUR TypeHead (trained Fold 1+2) for per-class; else use directly.",
))
cells.append(code('''
s = fold3[0]
inst, n = pathosam_count(s["image"])
print("instance map: unique labels =", len(np.unique(inst)), "-> instance segmentation present.")
print("dtype:", inst.dtype, "| values are nucleus IDs, not class labels.")
# Probe for any semantic/type API in patho_sam
try:
    import patho_sam, pkgutil
    subs = [m.name for m in pkgutil.iter_modules(patho_sam.__path__)]
    print("patho_sam submodules:", subs)
    print("-> look for 'semantic'/'classification' to see if a 5-class model exists (likely PanNuke-trained = LEAKY).")
except Exception as e:
    print("patho_sam introspization failed:", repr(e)[:150])
print("\\nCONCLUSION: generalist gives INSTANCE only (no 5-class) is the expected case ->")
print("for per-class we attach our own TypeHead trained on Fold 1+2 (clean).")
'''))

cells.append(md("## 05 — Fold-3 tissue distribution (colon = the only leakage to exclude)"))
cells.append(code('''
from collections import Counter
tissues = Counter(fold3[i]["tissue"] for i in range(len(fold3)))
print("Fold 3 tissue distribution:")
for t, c in tissues.most_common():
    flag = "  <-- COLON: exclude for PathoSAM (Lizard overlap)" if "colon" in t.lower() else ""
    print(f"  {t:18s}: {c:4d}{flag}")
colon_n = sum(c for t, c in tissues.items() if "colon" in t.lower())
print(f"\\nColon images: {colon_n}/{len(fold3)} ({100*colon_n/len(fold3):.1f}%) -> exclude these for")
print("a provably-clean PathoSAM evaluation; the rest never seen by PathoSAM.")
'''))

cells.append(md(
    "## Next (after this trial)",
    "",
    "- If counting MAE << SAM3 → PathoSAM is the strong predictor we wanted.",
    "- Full run → extract per-instance (s_i from mask score, p_ik from OUR TypeHead F1+2)",
    "  → `pathosam_preds.pkl` (same schema) → re-run PB-JCI.",
    "- Cross-dataset for PathoSAM: **PanNuke(no-colon) → NuInsSeg only** (CoNSeP is in PathoSAM's training → leaky).",
))
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info": {"name":"python","version":"3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT.name}: {len(cells)} cells")

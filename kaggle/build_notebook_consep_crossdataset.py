"""
Builder -> sam3_consep_crossdataset.ipynb

SECOND cross-dataset target (after NuInsSeg): CoNSeP (Graham 2019, HoVer-Net).
Total-count conformal (K=1). CALIBRATE on PanNuke (source) -> TEST on CoNSeP (target).

Why CoNSeP strengthens the paper: turns Table 6b from a single cross-dataset point
(NuInsSeg) into TWO independent real-shift targets -> the 90%->60% collapse +
adaptive recovery is shown to GENERALIZE, not a NuInsSeg artifact.

Pipeline mirrors Phase E:
  - unzip consep.zip -> Train/Test, images 1000x1000 + Labels .mat (inst_map, inst_centroid)
  - TILE each 1000x1000 into 4x4 = 16 patches of 250x250 (~656 patches total, ~ NuInsSeg 665)
    so per-patch count scale MATCHES PanNuke (256x256) -> shift is a DOMAIN shift, not a
    count-scale artifact.
  - GT count per tile = #instances whose centroid falls in that tile (conserves total).
  - SAM3 + A2 LoRA seed42 total-count inference -> consep_preds.pkl
  - In-domain CoNSeP conformal (5 cal seeds) + CROSS-DATASET (cal PanNuke -> test CoNSeP).

Attach datasets:
  - the dataset holding consep.zip (e.g. upload OpenDataLab CoNSeP)
  - hipinhththu/sam3-native-pt, hipinhththu/sam3-q1-multiseed-ckpts (LoRA seed42 + PanNuke pkl)

GPU T4 for inference (656 patches, fast); conformal is CPU seconds.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "sam3_consep_crossdataset.ipynb"
LIB_DIR = Path(__file__).parent / "lib"

def _read(name: str) -> str:
    return (LIB_DIR / name).read_text(encoding="utf-8")

METRICS    = "%%writefile metrics.py\n"    + _read("metrics.py")
LORA_SAM3  = "%%writefile lora_sam3.py\n"  + _read("lora_sam3.py")
SAM3_TRAIN = "%%writefile sam3_train.py\n" + _read("sam3_train.py")
CONFORMAL  = "%%writefile conformal.py\n"  + _read("conformal.py")

def md(*lines: str) -> dict:
    src = [ln if ln.endswith("\n") else ln + "\n" for ln in lines]
    if src:
        src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(body: str) -> dict:
    lines = body.lstrip("\n").splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}

cells: list[dict] = []

cells.append(md(
    "# Cross-dataset #2 — calibrate PanNuke, test CoNSeP (total count)",
    "",
    "**Second real-shift target** (after NuInsSeg). CoNSeP = 41 H&E tiles 1000x1000",
    "(Graham 2019), tiled into 4x4 = 16 patches of 250x250 (~656 patches) so per-patch",
    "count scale matches PanNuke -> a genuine **domain shift**, not a scale artifact.",
    "",
    "Same story as Table 6b: split conformal calibrated on PanNuke under-covers on CoNSeP;",
    "ACI / PB-JCI Online (streaming feedback) recover. Confirms the effect generalizes.",
    "",
    "**Attach:** consep.zip dataset, `hipinhththu/sam3-native-pt`,",
    "`hipinhththu/sam3-q1-multiseed-ckpts` (LoRA seed42 + `phase_C_preds_seed42.pkl`).",
))

cells.append(md("## 00 — Setup"))
cells.append(code('''
import subprocess, sys, os, glob, time, json, zipfile
import numpy as np
import torch
from PIL import Image
import scipy.io as sio
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())

WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/sam3_research"
SAM3_DIR = f"{REPO_DIR}/sam3"
CHECKPOINT_PATH = "/kaggle/input/datasets/hipinhththu/sam3-native-pt/sam3.pt"

LORA_CANDIDATES = [
    "/kaggle/input/datasets/hipinhththu/sam3-q1-multiseed-ckpts/sam3_lora_seed42_final.pt",
    "/kaggle/input/sam3-q1-multiseed-ckpts/sam3_lora_seed42_final.pt",
    "/kaggle/input/datasets/hipinhththu/phase-a2-lora-weights/sam3_lora_rank16_final.pt",
]
LORA_PATH = next((p for p in LORA_CANDIDATES if os.path.exists(p)), None)
if LORA_PATH is None:
    hits = glob.glob("/kaggle/input/**/sam3_lora_seed42_final.pt", recursive=True) \\
        or glob.glob("/kaggle/input/**/sam3_lora_rank16_final.pt", recursive=True)
    LORA_PATH = hits[0] if hits else None

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone",
                    "https://github.com/duonguwu/sam3_research.git", REPO_DIR], check=True)
assert os.path.exists(CHECKPOINT_PATH), "Attach hipinhththu/sam3-native-pt"
assert LORA_PATH, "Attach sam3-q1-multiseed-ckpts (sam3_lora_seed42_final.pt)"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", SAM3_DIR], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scikit-image", "scipy", "opencv-python", "einops", "tqdm"], check=True)
print("OK setup | LoRA:", LORA_PATH)
'''))

cells.append(md("## Helper modules"))
cells.append(code(METRICS))
cells.append(code(LORA_SAM3))
cells.append(code(SAM3_TRAIN))
cells.append(code(CONFORMAL))
cells.append(code('''
import sys
for p in [".", WORK, SAM3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
from lora_sam3 import inject_lora, load_lora_state, DEFAULT_LORA_TARGETS
from sam3_train import make_transform, encode_image_frozen, encode_text, forward_decoder_with_grad
from conformal import (AdaptiveConformalInference, PBAwareJointConformalOnline,
                       empirical_quantile, pb_count, pb_variance)
print("Helpers loaded.")
'''))

cells.append(md(
    "## 01 — Locate CoNSeP (handles BOTH zipped and Kaggle-auto-extracted)",
    "",
    "Kaggle usually auto-extracts an uploaded `.zip` -> the dataset is already a folder",
    "tree. We first look for an already-extracted `Train/` dir under /kaggle/input; if",
    "none, we find `consep.zip` and extract it to working. Either way we end with `BASE`",
    "= the dir holding `{Train,Test}/{Images,Labels}`.",
))
cells.append(code('''
def find_base_extracted():
    # any dir under /kaggle/input that has a Train subdir with Images/ in it
    for tr in glob.glob("/kaggle/input/**/Train", recursive=True):
        if os.path.isdir(os.path.join(tr, "Images")):
            return os.path.dirname(tr)
    return None

def find_zip():
    for pat in ["/kaggle/input/**/consep.zip", "/kaggle/input/**/CoNSeP*.zip",
                "/kaggle/input/**/*onsep*.zip"]:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None

BASE = find_base_extracted()
if BASE is None:                          # not auto-extracted -> find + unzip the .zip
    zpath = find_zip()
    assert zpath, "Neither an extracted Train/ dir nor consep.zip found - attach CoNSeP"
    print("zip:", zpath)
    EX = f"{WORK}/consep_data"
    if not os.path.isdir(EX):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(EX)
        print("extracted ->", EX)
    trs = [os.path.dirname(t) for t in glob.glob(f"{EX}/**/Train", recursive=True)
           if os.path.isdir(os.path.join(t, "Images"))]
    assert trs, "No Train/Images found after unzip"
    BASE = trs[0]
print("CoNSeP base:", BASE, "| has", os.listdir(BASE))
'''))

cells.append(md(
    "## 02 — Index + tile (1000x1000 -> 16x 250x250), GT count per tile",
    "",
    "GT count per tile = #instances whose centroid lands in the tile (each instance",
    "assigned to exactly one tile -> tile counts sum to the whole-image count).",
))
cells.append(code('''
TILE = 250            # 1000 / 250 = 4 -> 4x4 = 16 tiles per image
GRID = 1000 // TILE   # = 4

def list_consep(base):
    items = []
    for split in ["Train", "Test"]:
        idir, ldir = os.path.join(base, split, "Images"), os.path.join(base, split, "Labels")
        if not os.path.isdir(idir):
            continue
        for f in sorted(os.listdir(idir)):
            if f.lower().endswith((".png", ".tif", ".tiff")):
                stem = os.path.splitext(f)[0]
                mat = os.path.join(ldir, stem + ".mat")
                if os.path.exists(mat):
                    items.append({"image": os.path.join(idir, f), "mat": mat,
                                  "split": split, "stem": stem})
    return items

items = list_consep(BASE)
print(f"Indexed {len(items)} CoNSeP tiles (expect 41: 27 Train + 14 Test)")

def tile_counts(mat_path):
    m = sio.loadmat(mat_path)
    inst = m["inst_map"]
    n_total = int(len(np.unique(inst)) - 1)        # exclude background 0
    cent = np.asarray(m["inst_centroid"], dtype=float)  # (N,2) = (x=col, y=row)
    if cent.size == 0:
        return np.zeros(GRID * GRID, dtype=int), n_total
    x, y = cent[:, 0], cent[:, 1]
    col = np.clip((x // TILE).astype(int), 0, GRID - 1)
    row = np.clip((y // TILE).astype(int), 0, GRID - 1)
    tid = row * GRID + col
    counts = np.bincount(tid, minlength=GRID * GRID)
    return counts, n_total

# sanity: tile counts must conserve the whole-image instance count
c0, n0 = tile_counts(items[0]["mat"])
print(f"sample {items[0]['stem']}: whole={n0}  sum(tiles)={c0.sum()}  conserved={c0.sum()==n0}")
assert c0.sum() == n0, "centroid orientation wrong - tiles do not conserve total count"
print("tile counts (4x4):\\n", c0.reshape(GRID, GRID))
'''))

cells.append(md("## 03 — Build SAM3 + A2 LoRA; total-count inference fn"))
cells.append(code('''
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam3_image_model(device=device, eval_mode=True,
                               checkpoint_path=CHECKPOINT_PATH, load_from_HF=False)
model.eval()
inject_lora(model, target_module_names=DEFAULT_LORA_TARGETS, r=16, alpha=32, dropout=0.0)
load_lora_state(model, LORA_PATH)
for p in model.parameters():
    p.requires_grad = False

transform = make_transform(resolution=1008)
find_stage = FindStage(
    img_ids=torch.tensor([0], device=device, dtype=torch.long),
    text_ids=torch.tensor([0], device=device, dtype=torch.long),
    input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
    input_points=None, input_points_mask=None)
INFER_PROMPT = "cell"
SCORE_THRESH = 0.3
print("SAM3 + A2 LoRA ready.")

@torch.no_grad()
def predict_total(pil_img):
    bb = encode_image_frozen(model, transform, pil_img, device=device)
    bb.update(encode_text(model, INFER_PROMPT, device=device))
    outputs = forward_decoder_with_grad(model, bb, find_stage, model._get_dummy_prompt())
    cls_prob = outputs["pred_logits"].float().sigmoid()
    pres = outputs["presence_logit_dec"].float().sigmoid().unsqueeze(1)
    prob = (cls_prob * pres).squeeze(-1).squeeze(0)
    keep = prob > SCORE_THRESH
    return prob[keep].cpu().numpy() if keep.sum() > 0 else np.zeros(0)
'''))

cells.append(md("## 04 — Smoke: first image, 16 tiles"))
cells.append(code('''
it0 = items[0]
img0 = np.asarray(Image.open(it0["image"]).convert("RGB"))
cnts0, _ = tile_counts(it0["mat"])
for r in range(GRID):
    for c in range(GRID):
        crop = img0[r*TILE:(r+1)*TILE, c*TILE:(c+1)*TILE]
        sc = predict_total(Image.fromarray(crop))
        print(f"  tile({r},{c}) GT={cnts0[r*GRID+c]:3d} | n_det={len(sc):3d} | pred={sc.sum():6.1f}")
print("\\nIf pred is in a sane ballpark vs GT -> run full.")
'''))

cells.append(md("## 05 — Full inference over all tiles (cached) -> consep_preds.pkl"))
cells.append(code('''
import pickle
from tqdm import tqdm
CACHE = f"{WORK}/consep_preds.pkl"

if os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        data = pickle.load(f)
    preds, gts, src = data["preds"], data["gts"], data["src"]
    print(f"Loaded cache: {len(preds)} tiles")
else:
    preds, gts, src = [], [], []
    t0 = time.time()
    for it in tqdm(items, desc="CoNSeP infer"):
        img = np.asarray(Image.open(it["image"]).convert("RGB"))
        cnts, _ = tile_counts(it["mat"])
        for r in range(GRID):
            for c in range(GRID):
                crop = img[r*TILE:(r+1)*TILE, c*TILE:(c+1)*TILE]
                sc = predict_total(Image.fromarray(crop))
                preds.append({"scores": sc, "probs": np.ones((len(sc), 1)), "K": 1})
                gts.append(np.array([float(cnts[r*GRID+c])]))
                src.append(f"{it['stem']}_r{r}c{c}")
    with open(CACHE, "wb") as f:
        pickle.dump({"preds": preds, "gts": gts, "src": src}, f)
    print(f"Done {len(preds)} tiles in {(time.time()-t0)/60:.1f} min -> {CACHE}")

gts_arr = np.array([g[0] for g in gts])
pred_cnt = np.array([p["scores"].sum() for p in preds])
mae = np.abs(gts_arr - pred_cnt).mean()
print(f"GT/tile mean={gts_arr.mean():.1f} (min {gts_arr.min():.0f}, max {gts_arr.max():.0f})")
print(f"Pred/tile mean={pred_cnt.mean():.1f} | Total-count MAE = {mae:.2f}")
'''))

cells.append(md("## 06 — Total-count nonconformity / interval (K=1)"))
cells.append(code('''
ALPHA = 0.1

def nonconf(p, gt):
    if len(p["scores"]) == 0: return float(abs(gt[0]))
    n  = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return abs(gt[0] - n) / sg

def interval(p, q):
    if len(p["scores"]) == 0: return 0.0, 0.0
    n  = pb_count(p["scores"], p["probs"])[0]
    sg = np.sqrt(pb_variance(p["scores"], p["probs"])[0] + 1e-6)
    return max(0.0, n - q * sg), n + q * sg

def cov_width(P, G, q):
    los = np.array([interval(p, q)[0] for p in P])
    his = np.array([interval(p, q)[1] for p in P])
    g   = np.array([gg[0] for gg in G])
    return float(np.mean((g >= los) & (g <= his))), float(np.mean(his - los))
'''))

cells.append(md("## 07 — Load PanNuke (source) total-count from phase_C_preds_seed42.pkl"))
cells.append(code('''
def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None

pan_path = find("phase_C_preds_seed42.pkl")
assert pan_path, "phase_C_preds_seed42.pkl not found - attach sam3-q1-multiseed-ckpts"
with open(pan_path, "rb") as f:
    dpan = pickle.load(f)
pan_src = dpan["predictions_by_setting"]["in_dist"]
pan_gtc = np.asarray(dpan["gt_counts"])
pan_preds = [{"scores": np.asarray(p["scores"]),
              "probs": np.ones((len(p["scores"]), 1)), "K": 1} for p in pan_src]
pan_gts = [np.array([float(g.sum())]) for g in pan_gtc]
print(f"PanNuke source: {len(pan_preds)} patches | GT total mean={np.mean([g[0] for g in pan_gts]):.1f}")
print(f"CoNSeP target : {len(preds)} tiles | GT total mean={gts_arr.mean():.1f}")
'''))

cells.append(md(
    "## 08 — CROSS-DATASET: calibrate PanNuke -> test CoNSeP",
    "",
    "Split (no adapt) = honest coverage drop. ACI / PB-JCI Online (window=300, same as",
    "all phases) warm-start on PanNuke then stream CoNSeP with feedback (5 stream seeds).",
))
cells.append(code('''
pan_scores = np.array([nonconf(pan_preds[i], pan_gts[i]) for i in range(len(pan_preds))])
q_cross = empirical_quantile(pan_scores, ALPHA)
print(f"q (calibrated on PanNuke) = {q_cross:.3f}")

split_cov, split_w = cov_width(preds, gts, q_cross)

def stream(kind, nseeds=5):
    covs, ws = [], []
    for sd in range(nseeds):
        order = np.random.RandomState(sd).permutation(len(preds))
        if kind == "aci":
            m = AdaptiveConformalInference(alpha_target=ALPHA, gamma=0.05)
            m.reset(); m.history_scores = list(pan_scores)
        else:
            m = PBAwareJointConformalOnline(alpha=ALPHA, window=300)
            m.warmstart(pan_scores)
        c, w = [], []
        for i in order:
            q = m.get_quantile(); lo, hi = interval(preds[i], q)
            covered = lo <= gts[i][0] <= hi
            c.append(covered); w.append(hi - lo)
            s = nonconf(preds[i], gts[i])
            m.update(s, covered) if kind == "aci" else m.update(s)
        covs.append(np.mean(c)); ws.append(np.mean(w))
    return float(np.mean(covs)), float(np.std(covs)), float(np.mean(ws)), float(np.std(ws))

aci_c, aci_cs, aci_w, aci_ws = stream("aci")
pbo_c, pbo_cs, pbo_w, pbo_ws = stream("pbo")
print("\\nCROSS-DATASET (cal PanNuke -> test CoNSeP):")
print(f"  Split (no adapt) : cov {split_cov*100:.1f}% | width {split_w:.2f}")
print(f"  ACI (stream)     : cov {aci_c*100:.1f}+/-{aci_cs*100:.1f}% | width {aci_w:.2f}+/-{aci_ws:.2f}")
print(f"  PB-JCI Online    : cov {pbo_c*100:.1f}+/-{pbo_cs*100:.1f}% | width {pbo_w:.2f}+/-{pbo_ws:.2f}")
'''))

cells.append(md("## 09 — In-domain CoNSeP reference (calibrate CoNSeP, 5 seeds)"))
cells.append(code('''
def indomain(nseeds=5):
    covs, ws = [], []
    for sd in [42, 100, 200, 300, 400][:nseeds]:
        idx = np.random.RandomState(sd).permutation(len(preds))
        ncal = len(idx) // 2
        cal, test = idx[:ncal], idx[ncal:]
        cs = np.array([nonconf(preds[i], gts[i]) for i in cal])
        q = empirical_quantile(cs, ALPHA)
        c, w = cov_width([preds[i] for i in test], [gts[i] for i in test], q)
        covs.append(c); ws.append(w)
    return float(np.mean(covs)), float(np.std(covs)), float(np.mean(ws)), float(np.std(ws))

id_c, id_cs, id_w, id_ws = indomain()
print(f"In-domain split (cal CoNSeP): cov {id_c*100:.1f}+/-{id_cs*100:.1f}% | width {id_w:.2f}")
'''))

cells.append(md("## 10 — Summary table + save"))
cells.append(code('''
print("=" * 78)
print("CROSS-DATASET #2: PanNuke (cal) -> CoNSeP (test) | total count, alpha=0.1")
print("=" * 78)
print(f"{'Setting / Method':38s} | {'Coverage':>14s} | {'Width':>10s}")
print("-" * 78)
print(f"{'In-domain split (cal CoNSeP)':38s} | {id_c*100:>6.1f}+/-{id_cs*100:<4.1f}% | {id_w:>7.2f}")
print(f"{'Cross split (cal PanNuke, no adapt)':38s} | {split_cov*100:>11.1f}% | {split_w:>7.2f}")
print(f"{'Cross ACI (stream feedback)':38s} | {aci_c*100:>6.1f}+/-{aci_cs*100:<4.1f}% | {aci_w:>7.2f}")
print(f"{'Cross PB-JCI Online (stream)':38s} | {pbo_c*100:>6.1f}+/-{pbo_cs*100:<4.1f}% | {pbo_w:>7.2f}")
print("-" * 78)
drop = (id_c - split_cov) * 100
print(f"\\nCoverage DROP (in-domain -> cross split): {drop:+.1f} pp")

out = {
    "dataset": "CoNSeP", "n_tiles": len(preds), "tile_px": TILE, "total_count_MAE": float(mae),
    "in_domain_split":   {"coverage": [id_c, id_cs], "width": [id_w, id_ws]},
    "cross_split":       {"coverage": split_cov, "width": split_w},
    "cross_aci":         {"coverage": [aci_c, aci_cs], "width": [aci_w, aci_ws]},
    "cross_pbjci_online":{"coverage": [pbo_c, pbo_cs], "width": [pbo_w, pbo_ws]},
    "q_cross": float(q_cross), "alpha": ALPHA, "pbjci_window": 300,
}
with open(f"{WORK}/consep_crossdataset_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\\nSaved: {WORK}/consep_crossdataset_results.json  (+ consep_preds.pkl)")
'''))

cells.append(md(
    "## Notes",
    "",
    "- **CoNSeP = 2nd independent cross-dataset target** -> Table 6b becomes PanNuke->{NuInsSeg, CoNSeP}.",
    "- Total-count (K=1): no cell-type taxonomy mapping needed (CoNSeP types differ from PanNuke).",
    "- Same window=300, same streaming-feedback assumption as NuInsSeg -> directly comparable.",
    "- Send `consep_crossdataset_results.json` to add the CoNSeP rows to PAPER_TABLES Table 6b.",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT.name}: {len(cells)} cells")

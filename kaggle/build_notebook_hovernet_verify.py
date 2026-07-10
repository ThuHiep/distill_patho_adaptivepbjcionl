"""
Builder -> sam3_pannuke_hovernet_verify.ipynb

GOAL: determine which PanNuke fold the released HoVer-Net checkpoint
(hovernet_fast_pannuke_type_tf2pytorch) was HELD OUT from — because the
checkpoint carries NO training metadata (only the 'desc' weight dict), so the
fold is undocumented. If we don't know the held-out fold we risk LEAKAGE
(testing conformal on a fold the backbone trained on).

METHOD = memorization-gap test: run the checkpoint on PanNuke Fold 1/2/3,
measure per-fold counting MAE + foreground Dice. A model performs noticeably
BETTER on folds it trained on. The fold with the clearly WORSE metric is the
held-out (safe) one.
  - If Fold 3 is clearly worst  -> Fold 3 held out -> SAFE to use as our conformal cal/test set.
  - If all folds ~equal/good    -> trained on all  -> LEAKY -> must retrain on Fold 1+2.

Uses the vqdang/hover_net repo (native format of this checkpoint).
GPU T4. Attach: hipinhththu/pannuke + the dataset holding the checkpoint file.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "sam3_pannuke_hovernet_verify.ipynb"
LIB_DIR = Path(__file__).parent / "lib"

def _read(name: str) -> str:
    return (LIB_DIR / name).read_text(encoding="utf-8")

PANNUKE_LOADER = "%%writefile pannuke_loader.py\n" + _read("pannuke_loader.py")
METRICS        = "%%writefile metrics.py\n"        + _read("metrics.py")

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
    "# HoVer-Net checkpoint — which PanNuke fold is held out? (memorization-gap test)",
    "",
    "The released `hovernet_fast_pannuke_type` checkpoint has **no training metadata**",
    "(only a weight dict), so we don't know its train/test fold split. Using it on a fold",
    "it trained on = **leakage**. Here we find the held-out fold empirically: run on",
    "Fold 1/2/3 and compare per-fold **counting MAE + foreground Dice**. The fold the model",
    "is clearly **worst** on is the held-out (safe) one.",
    "",
    "**Decision:** if **Fold 3** is clearly worst → safe to use (it's our conformal set).",
    "If all folds look equally good → trained on all → leaky → must retrain on Fold 1+2.",
    "",
    "**Attach:** `hipinhththu/pannuke` + the dataset holding the checkpoint file",
    "(`hovernet_fast_pannuke_type_tf2pytorch`). GPU T4.",
))

cells.append(md("## 00 — Setup: clone vqdang/hover_net, deps, find checkpoint"))
cells.append(code('''
import subprocess, sys, os, glob, time, json
import numpy as np
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())

WORK = "/kaggle/working"
HOVER_DIR = f"{WORK}/hover_net"
if not os.path.isdir(HOVER_DIR):
    subprocess.run(["git", "clone", "https://github.com/vqdang/hover_net.git", HOVER_DIR], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scipy", "scikit-image", "opencv-python-headless"], check=True)

# locate the checkpoint file (no extension, ~150MB) anywhere under /kaggle/input
def find_ckpt():
    pats = ["/kaggle/input/**/hovernet_fast_pannuke_type*",
            "/kaggle/input/**/*hovernet*pannuke*"]
    for p in pats:
        for h in glob.glob(p, recursive=True):
            if os.path.isfile(h) and os.path.getsize(h) > 50_000_000:
                return h
    return None
CKPT = find_ckpt()
assert CKPT, "HoVer-Net checkpoint not found - attach the dataset holding it"
print("Checkpoint:", CKPT, f"({os.path.getsize(CKPT)/1e6:.0f} MB)")

if HOVER_DIR not in sys.path:
    sys.path.insert(0, HOVER_DIR)
'''))

cells.append(md("## Helper modules (PanNuke loader + metrics)"))
cells.append(code(PANNUKE_LOADER))
cells.append(code(METRICS))
cells.append(code('''
import sys
if "." not in sys.path: sys.path.insert(0, ".")
from pannuke_loader import PanNukeFold, DEFAULT_ROOT
from metrics import binary_dice
print("helpers loaded.")
'''))

cells.append(md(
    "## 01 — Build HoVer-Net (fast, nr_types=6) + load checkpoint",
    "",
    "PanNuke = 5 cell types + background = `nr_types=6`. Weights come from the `'desc'`",
    "key of the checkpoint (vqdang native format).",
))
cells.append(code('''
from models.hovernet.net_desc import HoVerNet
from models.hovernet.post_proc import process

device = "cuda" if torch.cuda.is_available() else "cpu"
net = HoVerNet(input_ch=3, nr_types=6, mode="fast")
sd = torch.load(CKPT, map_location="cpu")
sd = sd["desc"] if isinstance(sd, dict) and "desc" in sd else sd
missing, unexpected = net.load_state_dict(sd, strict=False)
print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected (want ~0/0)")
if missing[:3]:    print("  e.g. missing:", missing[:3])
if unexpected[:3]: print("  e.g. unexpected:", unexpected[:3])
net = net.to(device).eval()
print("HoVer-Net ready.")
'''))

cells.append(md(
    "## 02 — Inference fn (reflect-pad 256->384, center-crop output to 256)",
    "",
    "Fast mode uses valid convolutions (output < input). We reflect-pad to 384 so the",
    "valid output safely covers the original 256x256, then center-crop the output back to",
    "256x256 (center-aligned → no need to know the exact border size). `NORM` is picked by",
    "the smoke test below.",
))
cells.append(code('''
NORM = "raw"  # HoVerNet.forward divides by 255 INTERNALLY -> feed raw 0-255 (smoke confirms)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

def _prep(img256, norm):
    x = np.pad(img256, ((64, 64), (64, 64), (0, 0)), mode="reflect").astype(np.float32)  # 384
    if norm == "div255":
        x = x / 255.0
    elif norm == "imagenet":
        x = (x / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    # raw: leave as 0..255
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    return t

def _cc(a, size=256):
    h, w = a.shape[:2]
    s0, s1 = (h - size) // 2, (w - size) // 2
    return a[s0:s0 + size, s1:s1 + size]

@torch.no_grad()
def hover_predict(img256, norm=None, want_tp=False):
    norm = norm or NORM
    out = net(_prep(img256, norm))
    out = {k: v for k, v in out.items()}  # OrderedDict -> dict
    tp = torch.softmax(out["tp"], dim=1)[0].permute(1, 2, 0).cpu().numpy()   # H,W,6
    npr = torch.softmax(out["np"], dim=1)[0, 1].cpu().numpy()                # H,W  fg prob
    hv = out["hv"][0].permute(1, 2, 0).cpu().numpy()                         # H,W,2
    tp, npr, hv = _cc(tp), _cc(npr), _cc(hv)
    type_arg = tp.argmax(-1)[..., None].astype(np.float32)
    pred_map = np.concatenate([type_arg, npr[..., None], hv], axis=-1)        # H,W,4
    inst_map, inst_info = process(pred_map, nr_types=6)
    count = len(inst_info)
    fg = (npr > 0.5).astype(np.uint8)
    return (count, fg, tp) if want_tp else (count, fg)
print("inference fn ready.")
'''))

cells.append(md(
    "## 03 — Smoke: pick NORM (counts must be in a sane ballpark vs GT)",
    "",
    "Try the 3 normalizations on a few Fold-1 images; the right one yields nucleus counts",
    "in the same ballpark as GT (tens per patch), not ~0 or absurd.",
))
cells.append(code('''
f1 = PanNukeFold(DEFAULT_ROOT, 1)
print(f"Fold 1: {len(f1)} images")
idxs = [0, 1, 2, 3, 4]
for norm in ["div255", "imagenet", "raw"]:
    rows = []
    for i in idxs:
        s = f1[i]
        c, _ = hover_predict(s["image"], norm=norm)
        rows.append((c, int(s["counts"].sum())))
    preds = [r[0] for r in rows]; gts = [r[1] for r in rows]
    mae = np.mean([abs(p - g) for p, g in zip(preds, gts)])
    print(f"  norm={norm:9s} preds={preds} gts={gts}  MAE={mae:.1f}")
print("\\n-> set NORM below to the mode whose preds best track GT, then re-run cell 02.")
'''))
cells.append(code('''
# HoVerNet normalizes internally (imgs/255 inside forward) -> "raw" is correct.
# div255/imagenet will give ~0 nuclei (double-normalized black input).
NORM = "raw"
print("NORM =", NORM)
'''))

cells.append(md(
    "## 04 — Per-fold metrics (counting MAE + foreground Dice)",
    "",
    "Subsample `PER_FOLD` images/fold for speed (the gap signal is clear well before all",
    "images). The held-out fold shows **higher MAE / lower Dice**.",
))
cells.append(code('''
from tqdm import tqdm
PER_FOLD = None   # None = ALL images/fold (tighter, ~40 min); or an int to subsample

def eval_fold(fold_id):
    fold = PanNukeFold(DEFAULT_ROOT, fold_id)
    if PER_FOLD is None:
        sel = np.arange(len(fold))
    else:
        sel = np.random.RandomState(0).permutation(len(fold))[:min(PER_FOLD, len(fold))]
    n = len(sel)
    abs_err, dices, gtc, prc = [], [], [], []
    for i in tqdm(sel, desc=f"Fold {fold_id}", leave=False):
        s = fold[int(i)]
        c, fg = hover_predict(s["image"])
        gt = int(s["counts"].sum())
        abs_err.append(abs(c - gt)); gtc.append(gt); prc.append(c)
        gt_fg = (s["masks"] > 0).any(0).astype(np.uint8)
        dices.append(binary_dice(fg, gt_fg))
    return {"n": n, "MAE": float(np.mean(abs_err)),
            "Dice": float(np.mean(dices)),
            "gt_mean": float(np.mean(gtc)), "pred_mean": float(np.mean(prc))}

results = {}
t0 = time.time()
for fid in [1, 2, 3]:
    results[fid] = eval_fold(fid)
    r = results[fid]
    print(f"Fold {fid}: MAE={r['MAE']:.2f} | Dice={r['Dice']:.3f} | "
          f"GT/pred mean={r['gt_mean']:.1f}/{r['pred_mean']:.1f} | n={r['n']}")
print(f"Done in {(time.time()-t0)/60:.1f} min")
'''))

cells.append(md("## 05 — Verdict: which fold is held out?"))
cells.append(code('''
maes  = {f: results[f]["MAE"]  for f in results}
dices = {f: results[f]["Dice"] for f in results}
worst_mae  = max(maes,  key=maes.get)     # highest MAE  = most likely held-out
worst_dice = min(dices, key=dices.get)    # lowest Dice  = most likely held-out

print("=" * 60)
print("MEMORIZATION-GAP VERDICT")
print("=" * 60)
for f in [1, 2, 3]:
    print(f"  Fold {f}: MAE={maes[f]:.2f}  Dice={dices[f]:.3f}")
print("-" * 60)
mae_spread = (max(maes.values()) - min(maes.values()))
print(f"Worst (=> held-out candidate) by MAE : Fold {worst_mae}")
print(f"Worst (=> held-out candidate) by Dice: Fold {worst_dice}")
print(f"MAE spread across folds: {mae_spread:.2f}")
print()
if worst_mae == 3 and worst_dice == 3:
    print(">>> Fold 3 is clearly worst on BOTH metrics -> Fold 3 was HELD OUT -> SAFE to "
          "use this checkpoint for our Fold-3 conformal experiments.")
elif worst_mae == worst_dice:
    print(f">>> Fold {worst_mae} is the held-out one (both metrics agree), NOT Fold 3. "
          "Options: (a) switch our conformal set to this fold, or (b) retrain on the "
          "other two folds with Fold 3 held out.")
else:
    print(">>> Metrics DISAGREE or spread is small -> inconclusive / possibly trained on "
          "all folds -> treat as LEAKY -> retrain HoVer-Net on Fold 1+2 (Fold 3 held out).")

with open(f"{WORK}/hovernet_fold_verify.json", "w") as fjs:
    json.dump({"per_fold": results, "worst_by_mae": worst_mae,
               "worst_by_dice": worst_dice, "mae_spread": mae_spread, "norm": NORM}, fjs, indent=2)
print("\\nSaved: hovernet_fold_verify.json")
'''))

cells.append(md(
    "## Notes",
    "",
    "- A *clear* gap (Fold 3 worst by both MAE and Dice) is the green light. A small/ambiguous",
    "  spread should be treated conservatively as leaky → retrain.",
    "- If Fold 3 is confirmed held out, the next notebook reuses `hover_predict(..., want_tp=True)`",
    "  to extract per-nucleus (s_i from np-branch over the instance, p_ik from the tp softmax)",
    "  → build `hovernet_preds.pkl` in the same schema as `phase_C_preds_seed*.pkl` → re-run conformal.",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT.name}: {len(cells)} cells")

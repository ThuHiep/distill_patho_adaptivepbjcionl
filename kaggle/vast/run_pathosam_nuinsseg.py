"""
Build PathoSAM total-count predictions on NuInsSeg (cross-dataset target).

PathoSAM trained on CoNSeP/CPM17/Lizard/MoNuSeg/MoNuSAC/TNBC — NuInsSeg is NOT among
them → clean cross-dataset target. (CoNSeP IS in PathoSAM's training → would be leaky,
so PathoSAM cross-dataset = NuInsSeg only.) Total-count (K=1): NuInsSeg has instance
masks but no cell type.

For each NuInsSeg image: PathoSAM AIS -> per-instance s_i (foreground prob) -> save
scores; GT total = #unique nonzero in mask. Output schema matches phase_E pkl so
run_pathosam_crossdataset.py can consume it:
  work/pathosam_nuinsseg_preds.pkl = {"preds":[{scores,probs(1),K:1}], "gts":[[total]], "organs":[...]}

Run (needs ipateam/nuinsseg downloaded to data/nuinsseg):
  kaggle datasets download -d ipateam/nuinsseg --unzip -p data/nuinsseg/
  micromamba run -p /workspace/penv python run_pathosam_nuinsseg.py
"""
from __future__ import annotations
import os, sys, glob, time, pickle
import numpy as np
from PIL import Image
import torch

REPO = "/workspace/sam3_research"
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from pathosam_lib import load_pathosam, pathosam_instances  # noqa: E402

OUT = f"{REPO}/work/pathosam_nuinsseg_preds.pkl"
os.makedirs(f"{REPO}/work", exist_ok=True)
IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")

NUINSSEG_CANDS = [
    f"{REPO}/data/nuinsseg",
    f"{REPO}/data/nuinsseg/NuInsSeg",
    "/kaggle/input/datasets/ipateam/nuinsseg",
]


def _find_mask_dir(organ_dir):
    for name in os.listdir(organ_dir):
        full = os.path.join(organ_dir, name); low = name.lower()
        if os.path.isdir(full) and "label" in low and "mask" in low and "modif" not in low:
            return full
    for name in os.listdir(organ_dir):
        full = os.path.join(organ_dir, name)
        if os.path.isdir(full) and "label" in name.lower():
            return full
    return None


def _load_mask(path):
    try:
        import tifffile
        if path.lower().endswith((".tif", ".tiff")):
            return np.asarray(tifffile.imread(path))
    except Exception:
        pass
    return np.asarray(Image.open(path))


def build_index(root):
    tissue_dirs = glob.glob(os.path.join(root, "**", "tissue images"), recursive=True)
    samples = []
    for tdir in tissue_dirs:
        organ_dir = os.path.dirname(tdir)
        organ = os.path.basename(organ_dir)
        mdir = _find_mask_dir(organ_dir)
        if mdir is None:
            continue
        masks = {os.path.splitext(f)[0]: os.path.join(mdir, f) for f in os.listdir(mdir)}
        for f in sorted(os.listdir(tdir)):
            if not f.lower().endswith(IMG_EXT):
                continue
            stem = os.path.splitext(f)[0]
            if stem in masks:
                samples.append({"organ": organ, "image": os.path.join(tdir, f),
                                "mask": masks[stem]})
    return samples


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    root = next((c for c in NUINSSEG_CANDS if os.path.isdir(c)), None)
    if root is None:
        td = glob.glob(f"{REPO}/data/**/tissue images", recursive=True)
        root = os.path.dirname(os.path.dirname(td[0])) if td else None
    assert root, ("NuInsSeg not found. Download: kaggle datasets download -d "
                  "ipateam/nuinsseg --unzip -p data/nuinsseg/")
    print(f"NuInsSeg root: {root}")

    samples = build_index(root)
    assert samples, "No (image,mask) pairs found under NuInsSeg root."
    print(f"Indexed {len(samples)} pairs across {len(set(s['organ'] for s in samples))} organs")

    predictor, segmenter = load_pathosam(device)
    preds, gts, organs = [], [], []
    t0 = time.time()
    for k, s in enumerate(samples):
        img = np.asarray(Image.open(s["image"]).convert("RGB"))
        masks, scores, feat = pathosam_instances(img, predictor, segmenter)
        m = _load_mask(s["mask"])
        gt = int(len(np.unique(m)) - (1 if (m == 0).any() else 0))
        preds.append({"scores": scores.astype(np.float32),
                      "probs": np.ones((len(scores), 1), np.float32), "K": 1})
        gts.append([float(gt)])
        organs.append(s["organ"])
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(samples)} | {(time.time()-t0)/(k+1):.2f}s/img")

    with open(OUT, "wb") as f:
        pickle.dump({"preds": preds, "gts": gts, "organs": organs}, f)
    # sanity total-count MAE
    est = np.array([p["scores"].sum() for p in preds])
    gtv = np.array([g[0] for g in gts])
    print(f"\nSaved {OUT} | {len(preds)} patches")
    print(f"NuInsSeg total-count MAE (PathoSAM, Σs_i) = {np.abs(est-gtv).mean():.2f} "
          f"(GT mean {gtv.mean():.1f})")


if __name__ == "__main__":
    main()

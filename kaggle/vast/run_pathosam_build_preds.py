"""
Build PathoSAM predictions on CLEAN Fold 3 (exclude colon) for PB-JCI conformal.

For each image x {in_dist, mild HSV shift, severe blur shift}:
  PathoSAM AIS -> instances -> s_i (foreground prob) + pooled ViT feature
  -> TypeHead(PathoSAM) -> p_ik  => per-image {"scores":(N,), "probs":(N,5), "K":5}

Output (same schema as Phase C phase_C_predictions.pkl, so run_pathosam_conformal.py
reuses conformal.py unchanged):
  /workspace/sam3_research/work/pathosam_predictions.pkl
    { "predictions_by_setting": {setting: [per-image dict,...]},
      "gt_counts": (M,5), "indices": (M,), "settings": {...} }

Leakage: PathoSAM never trained on PanNuke; colon excluded (Lizard overlap) -> clean.

Run (after TypeHead trained):
  micromamba run -p /workspace/penv python run_pathosam_build_preds.py
"""
from __future__ import annotations
import os, sys, time, pickle
import numpy as np
import torch

REPO = "/workspace/sam3_research"
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from pannuke_loader import PanNukeFold, CELL_TYPES                 # noqa: E402
from type_head import TypeHead                                    # noqa: E402
from shift_detector import apply_hsv_jitter, apply_blur_shift     # noqa: E402
from pathosam_lib import load_pathosam, pathosam_instances, pool_features  # noqa: E402

DATA_ROOT = f"{REPO}/data/pannuke"
TH_PATH = f"{REPO}/checkpoints/type_head_pathosam.pt"
OUT = f"{REPO}/work/pathosam_predictions.pkl"
os.makedirs(f"{REPO}/work", exist_ok=True)

SETTINGS = {
    "in_dist":      {"augment": None},
    "mild_shift":   {"augment": ("hsv", "moderate")},
    "severe_shift": {"augment": ("blur", "severe")},
}


def aug(img_np, setting):
    a = SETTINGS[setting]["augment"]
    if a is None:
        return img_np
    kind, sev = a
    return apply_hsv_jitter(img_np, sev) if kind == "hsv" else apply_blur_shift(img_np, sev)


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    fold3 = PanNukeFold(DATA_ROOT, 3)
    # clean subset: drop colon (only possible Lizard leakage for PathoSAM)
    clean = [i for i in range(len(fold3)) if "colon" not in str(fold3[i]["tissue"]).lower()]
    rng = np.random.RandomState(42)
    indices = np.array(clean)[rng.permutation(len(clean))]
    print(f"Fold 3: {len(fold3)} total | clean (no colon) = {len(indices)}")

    predictor, segmenter = load_pathosam(device)
    head = TypeHead(in_dim=256, hidden_dim=128, num_classes=5).to(device)
    head.load_state_dict(torch.load(TH_PATH, map_location=device))
    head.eval()
    print(f"TypeHead loaded: {TH_PATH}")

    gt_counts = np.array([fold3[int(i)]["counts"] for i in indices], dtype=np.float32)
    print(f"gt_counts {gt_counts.shape} | mean/class {gt_counts.mean(0).round(2)}")

    preds_by_setting = {s: [] for s in SETTINGS}
    for setting in SETTINGS:
        t0 = time.time()
        for k, i in enumerate(indices):
            img = aug(fold3[int(i)]["image"], setting)
            masks, scores, feat = pathosam_instances(img, predictor, segmenter)
            if len(masks) == 0:
                preds_by_setting[setting].append(
                    {"scores": np.zeros(0, np.float32), "probs": np.zeros((0, 5), np.float32), "K": 5})
                continue
            pooled = pool_features(feat, masks, device)
            probs = head(pooled).softmax(-1).cpu().numpy().astype(np.float32)
            preds_by_setting[setting].append(
                {"scores": scores.astype(np.float32), "probs": probs, "K": 5})
            if (k + 1) % 200 == 0:
                print(f"  {setting} {k+1}/{len(indices)} | {(time.time()-t0)/(k+1):.2f}s/img")
        print(f"[{setting}] done {len(indices)} imgs in {(time.time()-t0)/60:.1f} min")

    with open(OUT, "wb") as f:
        pickle.dump({"predictions_by_setting": preds_by_setting,
                     "gt_counts": gt_counts, "indices": indices,
                     "settings": SETTINGS}, f)
    print(f"\nSaved {OUT}")
    # quick sanity: PathoSAM count-MAE (total) on in_dist
    pc = preds_by_setting["in_dist"]
    est = np.array([(p["scores"][:, None] * p["probs"]).sum() if len(p["scores"]) else 0.0
                    for p in pc])
    mae = np.abs(est - gt_counts.sum(1)).mean()
    print(f"PathoSAM total-count MAE (in_dist, with TypeHead) = {mae:.2f}")


if __name__ == "__main__":
    main()

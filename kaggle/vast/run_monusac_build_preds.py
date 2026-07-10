"""
Build PathoSAM predictions tren MoNuSAC TEST split (K=4) cho joint conformal da lop SACH.
Mirror run_conic_build_preds.py. Dung split da luu boi run_monusac_typehead_train.py
(tach theo patient) -> TypeHead khong nhin test.

Output: /workspace/sam3_research/work/monusac_predictions.pkl
  {"preds":[{"scores":(N,),"probs":(N,4),"K":4}], "gt_counts":(M,4), "indices":(M,),
   "classes": MONUSAC_CLASSES}
Run (sau khi train TypeHead):  python run_monusac_build_preds.py --seed 0
"""
from __future__ import annotations
import os, sys, time, pickle, argparse
import numpy as np
import torch

REPO = "/workspace/sam3_research"
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from monusac_loader import MonusacSet, monusac_split, MONUSAC_CLASSES, K   # noqa: E402
from type_head import TypeHead                                            # noqa: E402
from pathosam_lib import load_pathosam, pathosam_instances, pool_features  # noqa: E402

PKL = f"{REPO}/data/monusac_converted.pkl"
TH_PATH = f"{REPO}/checkpoints/type_head_monusac.pt"
OUT = f"{REPO}/work/monusac_predictions.pkl"
os.makedirs(f"{REPO}/work", exist_ok=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--frac-cal", type=float, default=0.5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"device={device}")

    split_f = f"{REPO}/work/monusac_split_seed{args.seed}.npy"
    if os.path.exists(split_f):
        test_idx = np.asarray(np.load(split_f, allow_pickle=True).item()["test"])
        print(f"loaded test split {split_f}: {len(test_idx)} imgs")
    else:
        _, test_idx = monusac_split(PKL, frac_cal=args.frac_cal, seed=args.seed)
        print("WARN: split file missing -> re-derived (phai cung seed voi luc train)")

    ds = MonusacSet(PKL, test_idx)
    predictor, segmenter = load_pathosam(device)
    head = TypeHead(in_dim=256, hidden_dim=128, num_classes=K).to(device)
    head.load_state_dict(torch.load(TH_PATH, map_location=device)); head.eval()
    print(f"TypeHead(K={K}) loaded: {TH_PATH}")

    gt_counts = np.array([ds[i]["counts"] for i in range(len(ds))], dtype=np.float32)
    print(f"gt_counts {gt_counts.shape} | mean/class {gt_counts.mean(0).round(2)}")

    preds = []; t0 = time.time()
    for i in range(len(ds)):
        masks, scores, feat = pathosam_instances(ds[i]["image"], predictor, segmenter)
        if len(masks) == 0:
            preds.append({"scores": np.zeros(0, np.float32),
                          "probs": np.zeros((0, K), np.float32), "K": K}); continue
        pooled = pool_features(feat, masks, device)
        probs = head(pooled).softmax(-1).cpu().numpy().astype(np.float32)
        preds.append({"scores": scores.astype(np.float32), "probs": probs, "K": K})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ds)} | {(time.time()-t0)/(i+1):.2f}s/img")
    print(f"done {len(ds)} imgs in {(time.time()-t0)/60:.1f} min")

    with open(OUT, "wb") as f:
        pickle.dump({"preds": preds, "gt_counts": gt_counts,
                     "indices": test_idx, "classes": MONUSAC_CLASSES}, f)
    print(f"\nSaved {OUT}")
    est = np.array([(p["scores"][:, None] * p["probs"]).sum(0) if len(p["scores"])
                    else np.zeros(K) for p in preds])
    print("per-class count MAE:", dict(zip(MONUSAC_CLASSES,
          np.abs(est - gt_counts).mean(0).round(2).tolist())))
    print(f"total-count MAE = {np.abs(est.sum(1) - gt_counts.sum(1)).mean():.2f}")


if __name__ == "__main__":
    main()

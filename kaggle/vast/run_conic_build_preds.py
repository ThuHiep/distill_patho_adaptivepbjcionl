"""
Build PathoSAM predictions tren CoNIC TEST split (K=6) cho joint conformal da lop.
Mirror run_pathosam_build_preds.py. Dung split da luu boi run_conic_typehead_train.py
(cal/test tach theo source) -> TypeHead KHONG nhin test -> khong leakage.

Moi anh test:
  PathoSAM AIS -> instances -> s_i (foreground prob) + pooled ViT feat
  -> TypeHead(conic, K=6) -> p_ik  => {"scores":(N,), "probs":(N,6), "K":6}

Output: /workspace/sam3_research/work/conic_predictions.pkl
  { "preds": [per-image dict,...], "gt_counts": (M,6), "indices": (M,),
    "classes": CONIC_CLASSES }
Schema tuong thich conformal.py (PBAwareJointConformal) — chay run_conic_conformal.py.

Run (sau khi train TypeHead):
  micromamba run -p /workspace/penv python run_conic_build_preds.py
"""
from __future__ import annotations
import os, sys, time, pickle
import numpy as np
import torch

REPO = "/workspace/sam3_research"
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from conic_loader import ConicSet, conic_split, CONIC_CLASSES, K   # noqa: E402
from type_head import TypeHead                                     # noqa: E402
from pathosam_lib import load_pathosam, pathosam_instances, pool_features  # noqa: E402

CONIC_ROOT = f"{REPO}/data/conic"
TH_PATH = f"{REPO}/checkpoints/type_head_conic.pt"
OUT = f"{REPO}/work/conic_predictions.pkl"
os.makedirs(f"{REPO}/work", exist_ok=True)


@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac-cal", type=float, default=0.5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    split_f = f"{REPO}/work/conic_split_seed{args.seed}.npy"
    if os.path.exists(split_f):
        sp = np.load(split_f, allow_pickle=True).item()
        test_idx = np.asarray(sp["test"])
        print(f"loaded test split {split_f}: {len(test_idx)} imgs")
    else:
        _, test_idx = conic_split(CONIC_ROOT, frac_cal=args.frac_cal, seed=args.seed)
        print("WARN: split file missing -> re-derived (must match seed used in training)")

    ds = ConicSet(CONIC_ROOT, test_idx)
    predictor, segmenter = load_pathosam(device)
    head = TypeHead(in_dim=256, hidden_dim=128, num_classes=K).to(device)
    head.load_state_dict(torch.load(TH_PATH, map_location=device))
    head.eval()
    print(f"TypeHead(K={K}) loaded: {TH_PATH}")

    gt_counts = np.array([ds[i]["counts"] for i in range(len(ds))], dtype=np.float32)
    print(f"gt_counts {gt_counts.shape} | mean/class {gt_counts.mean(0).round(2)}")

    preds = []
    t0 = time.time()
    for i in range(len(ds)):
        img = ds[i]["image"]
        masks, scores, feat = pathosam_instances(img, predictor, segmenter)
        if len(masks) == 0:
            preds.append({"scores": np.zeros(0, np.float32),
                          "probs": np.zeros((0, K), np.float32), "K": K})
            continue
        pooled = pool_features(feat, masks, device)
        probs = head(pooled).softmax(-1).cpu().numpy().astype(np.float32)
        preds.append({"scores": scores.astype(np.float32), "probs": probs, "K": K})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ds)} | {(time.time()-t0)/(i+1):.2f}s/img")
    print(f"done {len(ds)} imgs in {(time.time()-t0)/60:.1f} min")

    with open(OUT, "wb") as f:
        pickle.dump({"preds": preds, "gt_counts": gt_counts,
                     "indices": test_idx, "classes": CONIC_CLASSES}, f)
    print(f"\nSaved {OUT}")
    # sanity: per-class count-MAE (E[N_k] = sum_i s_i * p_ik)
    est = np.array([(p["scores"][:, None] * p["probs"]).sum(0) if len(p["scores"])
                    else np.zeros(K) for p in preds])
    mae = np.abs(est - gt_counts).mean(0)
    print("per-class count MAE:", dict(zip(CONIC_CLASSES, mae.round(2).tolist())))
    print(f"total-count MAE = {np.abs(est.sum(1) - gt_counts.sum(1)).mean():.2f}")


if __name__ == "__main__":
    main()

"""
Train a TypeHead for PathoSAM detections — CLEAN (Fold 1+2 only).

PathoSAM (generalist) gives instance masks but no nucleus TYPE. We attach a small
TypeHead(256,128,5) on the mask-pooled ViT embedding. Type supervision = majority vote
of the PanNuke GT type-map over each predicted instance's pixels (fast + robust; a pred
instance covering mostly background is skipped as a false positive).

Two phases (both cached to disk so retraining the head is free):
  1. EXTRACT: run PathoSAM AIS on Fold 1+2 -> (pooled_feature 256-d, type_label) pairs.
  2. TRAIN:   fit TypeHead with cross-entropy.

Output: /workspace/sam3_research/checkpoints/type_head_pathosam.pt

Run:
  micromamba run -p /workspace/penv python run_pathosam_typehead_train.py
Needs Fold 1 + Fold 2 present under data/pannuke (re-download if you deleted them).
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn

REPO = "/workspace/sam3_research"
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from pannuke_loader import PanNukeFold, CELL_TYPES          # noqa: E402
from type_head import TypeHead                              # noqa: E402
from pathosam_lib import load_pathosam, pathosam_instances, pool_features  # noqa: E402

DATA_ROOT = f"{REPO}/data/pannuke"
CKPT_OUT = f"{REPO}/checkpoints/type_head_pathosam.pt"
CACHE = f"{REPO}/work/pathosam_typehead_cache.npz"
os.makedirs(f"{REPO}/checkpoints", exist_ok=True)
os.makedirs(f"{REPO}/work", exist_ok=True)

MIN_FG_FRAC = 0.30   # a pred instance must overlap GT foreground by >=30% to get a type


def gt_type_map(sample) -> np.ndarray:
    """(H,W) int8: 0..4 = the 5 PanNuke types, -1 = background."""
    mpt = sample["masks"]                      # (5,H,W) instance ids
    H, W = mpt.shape[1:]
    tmap = np.full((H, W), -1, dtype=np.int8)
    for t in range(5):
        tmap[mpt[t] > 0] = t
    return tmap


def extract(device, n_cap=0):
    feats, labels = [], []
    predictor, segmenter = load_pathosam(device)
    for fold in (1, 2):
        ds = PanNukeFold(DATA_ROOT, fold)
        n = len(ds) if n_cap in (0, None) else min(n_cap, len(ds))
        t0 = time.time()
        for i in range(n):
            s = ds[i]
            masks, scores, feat = pathosam_instances(s["image"], predictor, segmenter)
            if len(masks) == 0:
                continue
            tmap = gt_type_map(s)
            pooled = pool_features(feat, masks, device).cpu().numpy()
            for j, m in enumerate(masks):
                vals = tmap[m]
                fg = vals[vals >= 0]
                if len(fg) < MIN_FG_FRAC * m.sum():
                    continue                    # mostly background -> false positive
                lab = int(np.bincount(fg, minlength=5).argmax())
                feats.append(pooled[j]); labels.append(lab)
            if (i + 1) % 100 == 0:
                dt = time.time() - t0
                print(f"  fold{fold} {i+1}/{n} | pairs={len(labels)} | "
                      f"{dt/(i+1):.2f}s/img")
        print(f"fold{fold} done: {len(labels)} labelled instances so far")
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    np.savez(CACHE, X=X, y=y)
    print(f"cached {X.shape} features -> {CACHE}")
    return X, y


def train_head(X, y, device, epochs=60, lr=1e-3, bs=512):
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    cls_count = np.bincount(y, minlength=5)
    print("class distribution:", dict(zip(CELL_TYPES, cls_count.tolist())))
    w = torch.tensor((cls_count.sum() / (5 * np.maximum(cls_count, 1))),
                     dtype=torch.float32, device=device)

    head = TypeHead(in_dim=X.shape[1], hidden_dim=128, num_classes=5).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    n = len(yt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        head.train()
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            opt.zero_grad()
            loss = lossf(head(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 10 == 0 or ep == 0:
            head.eval()
            with torch.no_grad():
                acc = (head(Xt).argmax(1) == yt).float().mean().item()
            print(f"  epoch {ep+1:3d} | loss {tot/n:.4f} | train acc {acc:.3f}")
    torch.save(head.state_dict(), CKPT_OUT)
    print(f"saved TypeHead -> {CKPT_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="cap imgs/fold (0=all)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--reuse-cache", action="store_true",
                    help="skip extraction, train from cached features")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    if args.reuse_cache and os.path.exists(CACHE):
        d = np.load(CACHE)
        X, y = d["X"], d["y"]
        print(f"loaded cache {X.shape}")
    else:
        X, y = extract(device, n_cap=args.n)
    train_head(X, y, device, epochs=args.epochs)


if __name__ == "__main__":
    main()

"""
Train TypeHead(256,128,4) cho PathoSAM tren MoNuSAC — dataset da lop SACH (K=4, eval-only).
Mirror run_conic_typehead_train.py: type supervision = majority vote class-map MoNuSAC tren
moi instance PathoSAM; train tren CAL split (tach theo patient, monusac_split).

Output: /workspace/sam3_research/checkpoints/type_head_monusac.pt
Run:
  python run_monusac_typehead_train.py            (sau khi co data/monusac_converted.pkl)
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

from monusac_loader import MonusacSet, monusac_split, MONUSAC_CLASSES, K   # noqa: E402
from type_head import TypeHead                                            # noqa: E402
from pathosam_lib import load_pathosam, pathosam_instances, pool_features  # noqa: E402

PKL = f"{REPO}/data/monusac_converted.pkl"
CKPT_OUT = f"{REPO}/checkpoints/type_head_monusac.pt"
CACHE = f"{REPO}/work/monusac_typehead_cache.npz"
os.makedirs(f"{REPO}/checkpoints", exist_ok=True)
os.makedirs(f"{REPO}/work", exist_ok=True)
MIN_FG_FRAC = 0.30


def extract(device, cal_idx, n_cap=0):
    feats, labels = [], []
    predictor, segmenter = load_pathosam(device)
    ds = MonusacSet(PKL, cal_idx)
    n = len(ds) if n_cap in (0, None) else min(n_cap, len(ds))
    t0 = time.time()
    for i in range(n):
        s = ds[i]
        masks, scores, feat = pathosam_instances(s["image"], predictor, segmenter)
        if len(masks) == 0:
            continue
        tmap = s["type_map"]
        pooled = pool_features(feat, masks, device).cpu().numpy()
        for j, m in enumerate(masks):
            vals = tmap[m]; fg = vals[vals >= 0]
            if len(fg) < MIN_FG_FRAC * m.sum():
                continue
            labels.append(int(np.bincount(fg, minlength=K).argmax())); feats.append(pooled[j])
        if (i + 1) % 100 == 0:
            print(f"  cal {i+1}/{n} | pairs={len(labels)} | {(time.time()-t0)/(i+1):.2f}s/img")
    X = np.asarray(feats, np.float32); y = np.asarray(labels, np.int64)
    np.savez(CACHE, X=X, y=y); print(f"cached {X.shape} -> {CACHE}")
    return X, y


def train_head(X, y, device, epochs=60, lr=1e-3, bs=512):
    Xt = torch.from_numpy(X).to(device); yt = torch.from_numpy(y).to(device)
    cc = np.bincount(y, minlength=K)
    print("class distribution:", dict(zip(MONUSAC_CLASSES, cc.tolist())))
    w = torch.tensor(cc.sum() / (K * np.maximum(cc, 1)), dtype=torch.float32, device=device)
    head = TypeHead(in_dim=X.shape[1], hidden_dim=128, num_classes=K).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w); n = len(yt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device); tot = 0.0; head.train()
        for k in range(0, n, bs):
            idx = perm[k:k + bs]; opt.zero_grad()
            loss = lossf(head(Xt[idx]), yt[idx]); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 10 == 0 or ep == 0:
            head.eval()
            with torch.no_grad():
                acc = (head(Xt).argmax(1) == yt).float().mean().item()
            print(f"  epoch {ep+1:3d} | loss {tot/n:.4f} | acc {acc:.3f}")
    torch.save(head.state_dict(), CKPT_OUT); print(f"saved TypeHead(K={K}) -> {CKPT_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--frac-cal", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"device={device}")
    cal_idx, test_idx = monusac_split(PKL, frac_cal=args.frac_cal, seed=args.seed)
    np.save(f"{REPO}/work/monusac_split_seed{args.seed}.npy",
            {"cal": cal_idx, "test": test_idx}, allow_pickle=True)
    print(f"split saved (cal={len(cal_idx)}, test={len(test_idx)})")
    if args.reuse_cache and os.path.exists(CACHE):
        d = np.load(CACHE); X, y = d["X"], d["y"]; print(f"loaded cache {X.shape}")
    else:
        X, y = extract(device, cal_idx, n_cap=args.n)
    train_head(X, y, device, epochs=args.epochs)


if __name__ == "__main__":
    main()

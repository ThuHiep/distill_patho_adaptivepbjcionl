#!/usr/bin/env python3
"""transfer_feature_distill.py — student TÍ HON hấp thụ feature Phikon + CỔNG LỌC (phễu).

Gate §0.6: thông tin TỒN TẠI trên Phikon (probe +0.15 low-label). Naive distill vào efflite0 bắt
được ~nửa (+0.07) — capacity-gap nuốt nửa vì student phí capacity khớp CẢ nền lẫn nhân.

Ý: PHỄU lọc feature teacher trước khi vào student, 2 cổng (chỉ dùng nhãn count, KHÔNG point/mask):
  - density-gate (spatial): trọng số distill theo density student (detach) -> dồn capacity vào vùng NHÂN.
  - reliability-gate (per-image): ảnh mà Phikon-probe-count lệch GT -> teacher-feature kém tin -> hạ distill.
So: count-only / naive-distill / gated(both) ở 25% nhãn + shift (leave-organ-out), MULTI-SEED.

Chạy Kaggle (Internet ON, GPU; ĐỪNG pip install):
  !python transfer_feature_distill.py --root /kaggle/input/datasets/ipateam/nuinsseg
"""
import argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from distill_student_nuinsseg import build_index, find_root, _load_mask
from distill_student_r2 import DensitySigmaUNet

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def gt_from_mask(p):
    m = _load_mask(p)
    return int(len(np.unique(m)) - (1 if (m == 0).any() else 0))


@torch.no_grad()
def cache_phikon(imgs224, dev, bs=32):
    """(N,3,224,224)[0,1] -> feature dày Phikon (N,768,14,14) f16 + pooled (N,768) cho probe."""
    from transformers import AutoModel
    ph = AutoModel.from_pretrained("owkin/phikon").to(dev).eval()
    dense, pooled = [], []
    for i in range(0, len(imgs224), bs):
        x = ((imgs224[i:i + bs].to(dev)) - MEAN.to(dev)) / STD.to(dev)
        h = ph(pixel_values=x).last_hidden_state[:, 1:, :]        # (B,196,768)
        B = x.shape[0]
        dense.append(h.transpose(1, 2).reshape(B, 768, 14, 14).half().cpu())
        pooled.append(h.mean(1).float().cpu())
    del ph
    if dev == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(dense), torch.cat(pooled).numpy()


def phikon_reliability(pooled, gts, tr_idx, dev, seed=0):
    """Per-image reliability = teacher-feature dự đoán count đúng tới đâu (2-fold cross-fit, leak-free).
    Trả mảng rel trên toàn bộ ảnh (chỉ dùng ở train_idx); rel in [0,1], cao=đáng tin."""
    tr = np.array(tr_idx)
    rel_full = np.ones(len(gts), np.float32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr); half = len(perm) // 2
    folds = [(perm[:half], perm[half:]), (perm[half:], perm[:half])]
    for fit, pred in folds:
        Xf = torch.tensor(pooled[fit], device=dev); yf = torch.tensor(gts[fit], device=dev).float()
        net = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        for _ in range(200):
            opt.zero_grad(); ((net(Xf).squeeze(1) - yf) ** 2).mean().backward(); opt.step()
        with torch.no_grad():
            pr = net(torch.tensor(pooled[pred], device=dev)).squeeze(1).cpu().numpy()
        err = np.abs(pr - gts[pred])
        s = np.median(err) + 1e-6
        rel_full[pred] = np.exp(-err / s).astype(np.float32)          # đáng tin -> ~1
    return rel_full


def dens_and_feat(model, x):
    y, fs = model._features(x)
    density = F.relu(model.dens(y))
    mu = density.sum(dim=(1, 2, 3))
    return mu, density, fs


def train_student(imgs, gts, tr_idx, phikon, dev, epochs, w_feat, mode,
                  rel=None, warmup=10, bs=16, seed=0):
    """mode: none(count-only) | naive | gated(density+reliability)."""
    torch.manual_seed(seed)
    use_distill = mode != "none"
    model = DensitySigmaUNet(32, backbone="efficientnet_lite0").to(dev)
    adapter = None
    if use_distill:
        adapter = nn.LazyConv2d(768, 1).to(dev)
        with torch.no_grad():
            _, _, fs0 = dens_and_feat(model, imgs[tr_idx[:2]].to(dev))
            adapter(fs0)
    params = list(model.parameters()) + (list(adapter.parameters()) if use_distill else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    tr = np.array(tr_idx)
    for ep in range(epochs):
        np.random.shuffle(tr)
        model.train()
        gate_on = use_distill and (mode == "naive" or ep >= warmup)   # gated: warm-up naive trước
        for i in range(0, len(tr), bs):
            idx = tr[i:i + bs]
            x = imgs[idx].to(dev)
            gt = torch.tensor(gts[idx], device=dev, dtype=torch.float32)
            mu, density, fs = dens_and_feat(model, x)
            loss = (mu - gt).abs().mean()
            if use_distill:
                fp = adapter(fs)
                pf = phikon[idx].float().to(dev)                       # (B,768,14,14)
                fp = F.interpolate(fp, size=pf.shape[-2:], mode="bilinear", align_corners=False)
                fp = F.normalize(fp, dim=1); pf = F.normalize(pf, dim=1)
                cos = 1 - (fp * pf).sum(1)                             # (B,14,14) khoảng cách
                if mode == "gated" and gate_on:
                    # spatial density-gate (detach) -> dồn vào vùng nhân
                    w = F.interpolate(density.detach(), size=pf.shape[-2:], mode="bilinear",
                                      align_corners=False)[:, 0]       # (B,14,14)
                    w = w / (w.flatten(1).sum(1).view(-1, 1, 1) + 1e-6)
                    fl = (w * cos).flatten(1).sum(1)                   # (B,) spatial-weighted
                    if rel is not None:                               # per-image reliability-gate
                        fl = fl * torch.tensor(rel[idx], device=dev)
                    feat_loss = fl.mean()
                else:
                    feat_loss = cos.mean()
                loss = loss + w_feat * feat_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    return model


@torch.no_grad()
def eval_r2(model, imgs, gts, idx, dev):
    model.eval()
    mus = [float(dens_and_feat(model, imgs[i:i + 1].to(dev))[0][0]) for i in idx]
    pr = np.array(mus); yt = gts[idx]
    ss_res = ((yt - pr) ** 2).sum(); ss_tot = ((yt - yt.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot if ss_tot > 0 else float("nan")), np.abs(pr - yt).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--w_feat", type=float, default=30.0)
    ap.add_argument("--frac", type=float, default=0.25, help="ngân sách nhãn (regime gap lớn nhất)")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")

    samples = build_index(args.root or find_root())
    im256, im224, gts, organs = [], [], [], []
    for s in samples:
        raw = Image.open(s["image"]).convert("RGB")
        im256.append(np.asarray(raw.resize((256, 256), Image.BILINEAR), np.float32) / 255.)
        im224.append(np.asarray(raw.resize((224, 224), Image.BILINEAR), np.float32) / 255.)
        gts.append(float(gt_from_mask(s["mask"]))); organs.append(s["organ"])
    im256 = torch.from_numpy(np.stack(im256)).permute(0, 3, 1, 2)
    im224 = torch.from_numpy(np.stack(im224)).permute(0, 3, 1, 2)
    gts = np.array(gts); organs = np.array(organs)
    print(f"[data] {len(gts)} ảnh | {len(set(organs))} mô")
    print("[phikon] cache feature ..."); phikon, pooled = cache_phikon(im224, dev)
    print(f"[phikon] dense {tuple(phikon.shape)}")

    seeds = [int(s) for s in args.seeds.split(",")]
    modes = [("count-only", "none"), ("naive-distill", "naive"), ("gated-phễu", "gated")]
    results = {m: [] for m, _ in modes}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        orgs = sorted(set(organs)); rng.shuffle(orgs)
        te_org = set(orgs[:max(1, len(orgs) // 5)])
        te = np.where([o in te_org for o in organs])[0]
        tr_all = np.where([o not in te_org for o in organs])[0]
        k = max(30, int(len(tr_all) * args.frac))
        tr = rng.choice(tr_all, size=min(k, len(tr_all)), replace=False)
        rel = phikon_reliability(pooled, gts, tr, dev, seed=seed)
        for name, mode in modes:
            m = train_student(im256, gts, tr, phikon, dev, args.epochs, args.w_feat,
                              mode, rel=rel, seed=seed)
            r2, mae = eval_r2(m, im256, gts, te, dev)
            results[name].append(r2)
            print(f"  seed {seed} | {name:14s} R²={r2:+.3f} MAE={mae:.2f}")

    print(f"\n=== {int(args.frac*100)}% nhãn, mô-chưa-thấy — R² mean±sd qua {len(seeds)} seed ===")
    base = np.mean(results["count-only"])
    for name, _ in modes:
        a = np.array(results[name])
        print(f"  {name:14s} {a.mean():+.3f} ± {a.std():.3f}   Δ vs count-only {a.mean()-base:+.3f}")
    print("\nĐỌC: gated-phễu > naive-distill > count-only -> CỔNG có giá trị (method thật). "
          "gated ≈ naive -> cổng không giúp, ghi honest.")


if __name__ == "__main__":
    main()

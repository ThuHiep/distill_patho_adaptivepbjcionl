#!/usr/bin/env python3
"""transfer_feature_distill.py — BƯỚC TRANSFERABILITY: student TÍ HON có hấp thụ được thặng dư
feature Phikon không? (gate §0.6 mới chứng minh thông tin TỒN TẠI trên Phikon 86M đông lạnh).

So trên efflite0 (DensitySigmaUNet, count-only density) ở CÙNG split leave-organ-out:
  (A) count-only               : chỉ |Σdensity - GT|
  (B) count + feature-distill  : + cosine-distill feature sâu student -> feature dày Phikon (14x14x768)
Eval R² trên MÔ-CHƯA-THẤY, full-label + 25% (nơi gate cho gap lớn nhất).

Thắng (B>A ở shift/low-label) -> nhồi được vào tí hon -> xây phễu (error-gating). Không -> capacity-gap
nuốt hết -> thặng dư không chuyển được vào 3.6M.

Chạy Kaggle (Internet ON, GPU; ĐỪNG pip install kẻo hỏng torch-CUDA):
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
    """(N,3,224,224)[0,1] -> feature dày Phikon (N,768,14,14) float16."""
    from transformers import AutoModel
    ph = AutoModel.from_pretrained("owkin/phikon").to(dev).eval()
    out = []
    for i in range(0, len(imgs224), bs):
        x = imgs224[i:i + bs].to(dev)
        x = (x - MEAN.to(dev)) / STD.to(dev)
        h = ph(pixel_values=x).last_hidden_state[:, 1:, :]     # (B,196,768) bỏ CLS
        B = x.shape[0]
        fm = h.transpose(1, 2).reshape(B, 768, 14, 14)         # patch -> spatial
        out.append(fm.half().cpu())
    del ph; torch.cuda.empty_cache() if dev == "cuda" else None
    return torch.cat(out)


def dens_and_feat(model, x):
    """Trả (mu, density, fs) — replicate forward nhưng lấy cả feature sâu fs để distill."""
    y, fs = model._features(x)
    density = F.relu(model.dens(y))
    mu = density.sum(dim=(1, 2, 3))
    return mu, fs


def train_student(imgs, gts, tr_idx, phikon, dev, epochs, w_feat, use_distill, bs=16, seed=0):
    torch.manual_seed(seed)
    model = DensitySigmaUNet(32, backbone="efficientnet_lite0").to(dev)
    adapter = None
    if use_distill:
        adapter = nn.LazyConv2d(768, 1).to(dev)
        with torch.no_grad():                       # materialize LazyConv2d -> có params trước optimizer
            _, fs0 = dens_and_feat(model, imgs[tr_idx[:2]].to(dev))
            adapter(fs0)
    params = list(model.parameters()) + (list(adapter.parameters()) if use_distill else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    tr = np.array(tr_idx)
    for ep in range(epochs):
        np.random.shuffle(tr)
        model.train()
        for i in range(0, len(tr), bs):
            idx = tr[i:i + bs]
            x = imgs[idx].to(dev)
            gt = torch.tensor(gts[idx], device=dev, dtype=torch.float32)
            mu, fs = dens_and_feat(model, x)
            loss = (mu - gt).abs().mean()
            if use_distill:
                fp = adapter(fs)
                pf = phikon[idx].float().to(dev)
                fp = F.interpolate(fp, size=pf.shape[-2:], mode="bilinear", align_corners=False)
                fp = F.normalize(fp, dim=1); pf = F.normalize(pf, dim=1)
                loss = loss + w_feat * (1 - (fp * pf).sum(1)).mean()   # cosine distill
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    return model


@torch.no_grad()
def eval_r2(model, imgs, gts, idx, dev):
    model.eval()
    mus = []
    for i in idx:
        mu, _ = dens_and_feat(model, imgs[i:i + 1].to(dev))
        mus.append(float(mu[0]))
    pr = np.array(mus); yt = gts[idx]
    ss_res = ((yt - pr) ** 2).sum(); ss_tot = ((yt - yt.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, np.abs(pr - yt).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--w_feat", type=float, default=30.0, help="trọng số cosine feature-distill (tunable)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")

    # data: student 256 + phikon 224 + gt + organ
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

    print("[phikon] cache feature dày ...")
    phikon = cache_phikon(im224, dev)
    print(f"[phikon] {tuple(phikon.shape)}")

    # leave-organ-out (giống gate)
    rng = np.random.default_rng(args.seed)
    orgs = sorted(set(organs)); rng.shuffle(orgs)
    te_org = set(orgs[:max(1, len(orgs) // 5)])
    te = np.where([o in te_org for o in organs])[0]
    tr_all = np.where([o not in te_org for o in organs])[0]
    print(f"[split] test mô: {sorted(te_org)} | train {len(tr_all)} / test {len(te)}")

    for frac in (1.0, 0.25):
        k = max(30, int(len(tr_all) * frac))
        tr = rng.choice(tr_all, size=min(k, len(tr_all)), replace=False)
        print(f"\n=== {int(frac*100)}% nhãn ({len(tr)} ảnh) — R² trên mô-chưa-thấy ===")
        for tag, use in [("(A) count-only     ", False), ("(B) +feature-distill", True)]:
            m = train_student(im256, gts, tr, phikon, dev, args.epochs, args.w_feat, use, seed=args.seed)
            r2, mae = eval_r2(m, im256, gts, te, dev)
            print(f"  {tag}: R²={r2:+.3f}  MAE={mae:.2f}")
    print("\nĐỌC: (B)>(A) ở shift/low-label -> student tí hon HẤP THỤ được thặng dư -> xây phễu. "
          "Không hơn -> capacity-gap nuốt hết.")


if __name__ == "__main__":
    main()

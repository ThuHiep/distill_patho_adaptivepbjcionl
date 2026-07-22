#!/usr/bin/env python3
"""gate_feature_surplus.py — GATE: feature pathology-FM có THẶNG DƯ so với ImageNet không?

Quyết định cả hướng feature-distill/phễu (kỷ luật chống-PACT). Nếu feature ĐÔNG LẠNH của
pathology-FM (Phikon) KHÔNG dự đoán count tốt hơn ImageNet ở MÔ-CHƯA-THẤY (shift) -> distill
(nhồi feature đó vào student) chắc chắn vô ích -> DỪNG. Nếu thắng -> feature có thặng dư -> xây phễu.

Điều kiện cần (upper-bound giá trị distill): linear/MLP-probe trên feature đông lạnh.
So SẠCH: Phikon (ViT-B/16, pathology SSL) vs ImageNet ViT-B/16 — CÙNG kiến trúc, khác pretrain.
Shift = leave-organ-out trong NuInsSeg (train tập mô, test mô chưa thấy). + đường cong low-label.

Chạy Kaggle (Internet ON): !python gate_feature_surplus.py --root /kaggle/input/datasets/ipateam/nuinsseg
"""
import argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from distill_student_nuinsseg import build_index, find_root, _load_mask

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def gt_from_mask(path):
    m = _load_mask(path)
    return int(len(np.unique(m)) - (1 if (m == 0).any() else 0))


@torch.no_grad()
def extract(model_kind, model, imgs224, dev, bs=32):
    """imgs224: (N,3,224,224) float[0,1] CHƯA normalize. Trả mean-patch-token feature (N,D).
    Cả 2 model đều ViT-B/16 -> last_hidden_state (B,197,D); bỏ CLS[0], mean patch [1:]."""
    feats = []
    for i in range(0, len(imgs224), bs):
        x = imgs224[i:i + bs].to(dev)
        x = (x - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)   # cả Phikon lẫn timm ViT dùng ImageNet stats
        if model_kind == "phikon":
            out = model(pixel_values=x).last_hidden_state          # (B,197,768)
            f = out[:, 1:, :].mean(1)                              # mean patch tokens
        else:  # timm vit
            tok = model.forward_features(x)                        # (B,197,768) cho vit_base_patch16_224
            f = tok[:, 1:, :].mean(1) if tok.dim() == 3 else tok
        feats.append(f.float().cpu())
    return torch.cat(feats).numpy()


def probe_r2(Ftr, ytr, Fte, yte, epochs=300, dev="cuda", seed=0):
    """MLP nhỏ 768->256->1 count-only; trả R² trên test."""
    torch.manual_seed(seed)
    D = Ftr.shape[1]
    net = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.tensor(Ftr, device=dev); Ytr = torch.tensor(ytr, device=dev).float()
    Xte = torch.tensor(Fte, device=dev)
    for _ in range(epochs):
        opt.zero_grad()
        p = net(Xtr).squeeze(1)
        loss = ((p - Ytr) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pr = net(Xte).squeeze(1).cpu().numpy()
    yte = np.asarray(yte, float)
    ss_res = ((yte - pr) ** 2).sum(); ss_tot = ((yte - yte.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.abs(pr - yte).mean()
    return r2, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = "cuda"

    # 1) data: raw img (resize 224) + gt count + organ
    samples = build_index(args.root or find_root())
    imgs, gts, organs = [], [], []
    for s in samples:
        im = Image.open(s["image"]).convert("RGB").resize((224, 224), Image.BILINEAR)
        imgs.append(np.asarray(im, np.float32) / 255.)
        gts.append(float(gt_from_mask(s["mask"]))); organs.append(s["organ"])
    imgs = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)   # (N,3,224,224)
    gts = np.array(gts); organs = np.array(organs)
    print(f"[data] {len(gts)} ảnh | {len(set(organs))} mô | GT mean {gts.mean():.1f}")

    # 2) load 2 feature extractor ĐÔNG LẠNH, cùng ViT-B/16
    import timm
    from transformers import AutoModel
    phikon = AutoModel.from_pretrained("owkin/phikon").to(dev).eval()
    imnet = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k",
                              pretrained=True, num_classes=0).to(dev).eval()
    for p in list(phikon.parameters()) + list(imnet.parameters()):
        p.requires_grad_(False)

    # 3) trích + cache feature 1 lần
    print("[feat] trích Phikon ..."); Fp = extract("phikon", phikon, imgs, dev)
    print("[feat] trích ImageNet-ViT ..."); Fi = extract("timm", imnet, imgs, dev)
    print(f"[feat] Phikon {Fp.shape} | ImageNet {Fi.shape}")

    # 4) LEAVE-ORGAN-OUT (shift): 80% mô train, 20% mô test (mô chưa thấy)
    rng = np.random.default_rng(args.seed)
    orgs = sorted(set(organs)); rng.shuffle(orgs)
    n_te_org = max(1, len(orgs) // 5)
    te_org = set(orgs[:n_te_org])
    te = np.array([o in te_org for o in organs]); tr = ~te
    print(f"\n[split] test mô ({n_te_org}): {sorted(te_org)} | train {tr.sum()} / test {te.sum()} ảnh")

    print("\n=== GATE: R² trên MÔ-CHƯA-THẤY (shift) — full label ===")
    for name, F in [("ImageNet-ViT", Fi), ("Phikon (pathology)", Fp)]:
        r2, mae = probe_r2(F[tr], gts[tr], F[te], gts[te], dev=dev)
        print(f"  {name:22s} R²={r2:+.3f}  MAE={mae:.2f}")

    # 5) LOW-LABEL: giảm dần ảnh train, eval cùng test-mô
    print("\n=== LOW-LABEL: R² trên mô-chưa-thấy theo % nhãn train ===")
    tr_idx = np.where(tr)[0]
    for frac in (0.1, 0.25, 0.5, 1.0):
        k = max(20, int(len(tr_idx) * frac))
        sub = rng.choice(tr_idx, size=min(k, len(tr_idx)), replace=False)
        row = []
        for name, F in [("ImageNet", Fi), ("Phikon", Fp)]:
            r2, _ = probe_r2(F[sub], gts[sub], F[te], gts[te], dev=dev)
            row.append(f"{name} {r2:+.3f}")
        print(f"  {int(frac*100):3d}% ({len(sub):3d} ảnh): " + " | ".join(row))

    print("\nĐỌC: nếu Phikon > ImageNet ở mô-chưa-thấy (nhất là low-label) -> feature pathology "
          "CÓ thặng dư -> xây phễu feature-distill. Nếu KHÔNG -> distill chết, dừng.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""method_composite_linearity.py — BƯỚC 1: test LOSS composite-linearity (ngôi sao) tối giản.

Premise P1 đã PASS (premise_saturation_test): density head count-only SATURATE dưới overlap
(Δsat 0→26.5→39.9→50.9% theo k). Loss này ép head hồi ĐÚNG số nhân distinct từ ảnh chồng-lấp:

  L_comp = | count( overlay_min(k ảnh) ) − Σ GT(k ảnh) |     (k∈{2,3,4}, count-only, ΣGT đã biết)

Overlay = pixelwise-min (nhân tối mọi lớp hiện, mật độ tăng) → tạo crowding có ΣGT biết, KHÔNG mask.
Đây là đòn count-only đánh điểm yếu Triple-U-net/CP-Net (mask+watershed under-split ở mô dày).

★ CỔNG SỐNG-CHẾT: anti-saturation học trên overlay TỔNG HỢP có chuyển sang mô DÀY THẬT không?
Script đo cả 3: (a) Δsat overlay giảm? (b) MAE mô-dày-thật + worst-organ R² [TRANSFER]; (c) R² tổng.
So baseline count-only vs +composite, CÙNG split, multi-seed, paired.

Chạy Kaggle (GPU; ĐỪNG pip install):  !python method_composite_linearity.py --root /kaggle/input/datasets/ipateam/nuinsseg
"""
import argparse
import numpy as np
from PIL import Image
import torch
from distill_student_nuinsseg import build_index, find_root, _load_mask, IMG_SIZE
from distill_student_r2 import DensitySigmaUNet
from r2_losses import count_from_density


def gt_from_mask(path):
    m = _load_mask(path)
    return int(len(np.unique(m)) - (1 if (m == 0).any() else 0))


def overlay_min(batch_imgs):
    """batch_imgs: (k,3,H,W) tensor[0,1] -> (3,H,W) min theo lớp (chồng-lấp, giữ nhân tối)."""
    return batch_imgs.min(dim=0)[0]


def dens_count(model, x):
    return count_from_density(model(x)[0])


def train_model(imgs, gts, tr_idx, dev, epochs, use_comp, w_comp, bs=16, seed=0, kmax=4, n_comp=4):
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DensitySigmaUNet(32, backbone="efficientnet_lite0").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr = np.array(tr_idx)
    for ep in range(epochs):
        order = rng.permutation(tr)
        model.train()
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            x = imgs[b].to(dev)
            gt = torch.tensor(gts[b], device=dev, dtype=torch.float32)
            loss = (dens_count(model, x) - gt).abs().mean()
            if use_comp:                                     # composite-linearity (overlay ΣGT biết)
                comp_x, comp_t = [], []
                for _ in range(n_comp):
                    k = rng.integers(2, kmax + 1)
                    pick = rng.choice(tr, size=int(k), replace=False)
                    comp_x.append(overlay_min(imgs[pick].to(dev)))
                    comp_t.append(float(gts[pick].sum()))
                cx = torch.stack(comp_x)
                ct = torch.tensor(comp_t, device=dev, dtype=torch.float32)
                loss = loss + w_comp * (dens_count(model, cx) - ct).abs().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


@torch.no_grad()
def preds(model, imgs, idx, dev):
    model.eval()
    return np.array([float(dens_count(model, imgs[i:i + 1].to(dev))[0]) for i in idx])


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum(); ss_tot = ((true - true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def worst_organ_r2(pred, true, organs_te, min_n=8):
    vals = []
    for o in set(organs_te):
        m = organs_te == o
        if m.sum() >= min_n:
            vals.append(r2(pred[m], true[m]))
    return min(vals) if vals else float("nan")


@torch.no_grad()
def sat_at_k(model, imgs, gts, te, dev, k=4, ntup=200, seed=0):
    """Δsat% overlay ở mức k (đo lại saturation sau train)."""
    model.eval(); rng = np.random.default_rng(seed); sp, ov = [], []
    for _ in range(ntup):
        pick = rng.choice(te, size=k, replace=False)
        sp.append(sum(float(dens_count(model, imgs[i:i + 1].to(dev))[0]) for i in pick))
        ov.append(float(dens_count(model, overlay_min(imgs[pick].to(dev))[None])[0]))
    sp, ov = np.array(sp), np.array(ov)
    return 100 * (sp - ov).mean() / max(sp.mean(), 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--w_comp", type=float, default=1.0)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")

    samples = build_index(args.root or find_root())
    ims, gts, organs = [], [], []
    for s in samples:
        im = np.asarray(Image.open(s["image"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                        np.float32) / 255.
        ims.append(im); gts.append(float(gt_from_mask(s["mask"]))); organs.append(s["organ"])
    ims = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2)
    gts = np.array(gts); organs = np.array(organs)
    # mô dày thật = organ có GT trung bình cao nhất
    org_mean = {o: gts[organs == o].mean() for o in set(organs)}
    dense_orgs = sorted(org_mean, key=org_mean.get, reverse=True)[:4]
    print(f"[data] {len(gts)} ảnh | mô-dày (GT cao nhất): {[(o, round(org_mean[o])) for o in dense_orgs]}")

    seeds = [int(s) for s in args.seeds.split(",")]
    R = {m: {"r2": [], "worst": [], "densemae": [], "sat": []} for m in ["baseline", "+composite"]}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(gts)); n_te = len(gts) // 5
        te, tr = idx[:n_te], idx[n_te:]
        dense_mask = np.array([organs[i] in dense_orgs for i in te])
        for name, uc in [("baseline", False), ("+composite", True)]:
            m = train_model(ims, gts, tr, dev, args.epochs, uc, args.w_comp, seed=seed)
            pr = preds(m, ims, te, dev); yt = gts[te]
            R[name]["r2"].append(r2(pr, yt))
            R[name]["worst"].append(worst_organ_r2(pr, yt, organs[te]))
            R[name]["densemae"].append(np.abs(pr[dense_mask] - yt[dense_mask]).mean() if dense_mask.any() else np.nan)
            R[name]["sat"].append(sat_at_k(m, ims, gts, te, dev, seed=seed))
        print(f"  seed {seed} done")

    print(f"\n=== BƯỚC 1: composite-linearity vs baseline count-only ({len(seeds)} seed, paired) ===")
    print(f"  {'metric':22s} {'baseline':>16s} {'+composite':>16s} {'Δ':>8s}")
    def row(lab, key, better_up=True, pct=False):
        b, c = np.array(R["baseline"][key]), np.array(R["+composite"][key])
        d = (c - b).mean()
        good = (d > 0) if better_up else (d < 0)
        star = "  ✅" if good else "  ✗"
        u = "%" if pct else ""
        print(f"  {lab:22s} {b.mean():>7.3f}±{b.std():.3f}{u:1s} {c.mean():>7.3f}±{c.std():.3f}{u:1s} {d:>+8.3f}{star}")
    row("R² tổng ↑", "r2", True)
    row("worst-organ R² ↑ [TRANSFER]", "worst", True)
    row("MAE mô-dày ↓ [TRANSFER]", "densemae", False)
    row("Δsat% overlay k=4 ↓", "sat", False, pct=True)
    print("\nĐỌC — CỔNG SỐNG-CHẾT = 2 dòng [TRANSFER]:")
    print("  worst-organ R² TĂNG & MAE mô-dày GIẢM  -> anti-saturation CHUYỂN sang mô thật -> method THẬT -> thêm module.")
    print("  Δsat giảm nhưng [TRANSFER] KHÔNG cải thiện -> chỉ vá ảnh-ghép, mô thật vô cảm -> method RỖNG, dừng.")


if __name__ == "__main__":
    main()

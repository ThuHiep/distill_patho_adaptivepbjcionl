#!/usr/bin/env python3
"""premise_saturation_test.py — PREMISE P1 (sống-chết của method mới chống crowding).

Bối cảnh (đọc CODE đối thủ, không chỉ paper): họ hematoxylin-aware (Triple U-net, CP-Net) đếm bằng
chuỗi stain->mask->watershed; code lộ stain SATURATE ở vùng OD cao (percentile+exp nén dải) và
watershed UNDER-SPLIT nhân dính -> undercount ở mô dày = LỖI CẤU TRÚC. Method của ta: học đếm trực
tiếp, count-only, + hiệu chỉnh khối->số theo crowding, dạy bằng composite-linearity (không mask).

P1 QUYẾT ĐỊNH: tỉ lệ khối->số có THẬT SỰ bẻ cong dưới overlap không — tức density head hiện tại có
undercount TĂNG DẦN khi ta chồng nhiều ảnh (mật độ nhân/overlap tăng, SỐ nhân distinct = Σ vẫn biết)?
  - CÓ (gap tăng theo k) -> saturation có thật -> module hiệu chỉnh + composite-linearity CÓ tín hiệu
    để học -> đòn đánh CP-Net/Triple-U-net đứng vững -> XÂY.
  - KHÔNG (gap phẳng) -> không có saturation để khai thác -> DỪNG, khỏi phí (kỷ luật chống-PACT/tiling).

Cách dựng chồng-lấp count-only, giữ SỐ distinct = Σ GT:
  - OVERLAY (min-blend): composite = pixelwise-min của k ảnh -> nhân tối của MỌI lớp đều hiện (chồng lấp),
    nền sáng chỉ ở nơi mọi lớp đều sáng. k nhân distinct = Σ GT (nhân vẫn là các thực thể riêng dù ảnh
    gộp thành mảng) -> counter đúng phải hồi được Σ; head saturate thì hồi THIẾU.
  - TILE (grid downscale) = đối chứng: nhân nhỏ đi nhưng KHÔNG chồng -> nếu gap-tile << gap-overlay thì
    overlap (không phải resolution) là thủ phạm.

KHÁC ý tiling đã GIẾT: kia cắt-ảnh-có-sẵn rồi upscale (Σ-các-ô của 1 forward vốn cộng-tính = rỗng);
đây GHÉP ảnh RỜI thành ảnh dày MỚI rồi forward LẠI -> nếu saturate thì count(ghép) < Σ = tín hiệu THẬT.

Chạy Kaggle (GPU; ĐỪNG pip install):  !python premise_saturation_test.py --root /kaggle/input/datasets/ipateam/nuinsseg
"""
import argparse
import numpy as np
from PIL import Image
import torch
from distill_student_nuinsseg import build_index, find_root, _load_mask, IMG_SIZE
from distill_student_r2 import train
from r2_losses import count_from_density


def gt_from_mask(path):
    m = _load_mask(path)
    return int(len(np.unique(m)) - (1 if (m == 0).any() else 0))


@torch.no_grad()
def count(model, img01, dev):
    """img01: (H,W,3) float[0,1] -> mu (float)."""
    t = torch.from_numpy(img01.transpose(2, 0, 1).astype(np.float32))[None].to(dev)
    return float(count_from_density(model(t)[0])[0])


def overlay_min(imgs):
    """chồng-lấp = pixelwise-min -> nhân (tối) của mọi lớp hiện; mật độ tăng theo số lớp."""
    return np.minimum.reduce(imgs)


def tile_grid(imgs):
    """đối chứng: xếp k ảnh downscale vào 1 canvas 256 (nhân nhỏ đi, KHÔNG chồng)."""
    k = len(imgs); g = int(np.ceil(np.sqrt(k))); cell = IMG_SIZE // g
    canvas = np.ones((IMG_SIZE, IMG_SIZE, 3), np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, g)
        small = np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((cell, cell), Image.BILINEAR),
                           np.float32) / 255.
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = small
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ks", default="1,2,3,4")
    ap.add_argument("--ntup", type=int, default=250, help="số bộ ngẫu nhiên mỗi k")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")

    samples = build_index(args.root or find_root())
    imgs, gts, organs = [], [], []
    for s in samples:
        im = np.asarray(Image.open(s["image"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                        np.float32) / 255.
        imgs.append(im); gts.append(float(gt_from_mask(s["mask"]))); organs.append(s["organ"])
    imgs = np.stack(imgs); gts = np.array(gts)
    print(f"[data] {len(gts)} ảnh | GT mean {gts.mean():.1f}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(imgs)); n_te = len(imgs) // 5
    te, tr = idx[:n_te], idx[n_te:]

    # train count-only efflite0 (w_density=0, KHÔNG mask/teacher) — head density-sum thuần
    data = [{"img": (imgs[i] * 255).astype(np.uint8),
             "density": np.zeros((IMG_SIZE, IMG_SIZE), np.float32),
             "gt": gts[i], "organ": organs[i]} for i in range(len(imgs))]
    model = train(data, dev, args.epochs, 32, 1e-3, list(tr),
                  0.0, 1.0, 0.01, 0.5, 16, True, "poisson", "efficientnet_lite0")
    model.eval()

    ks = [int(x) for x in args.ks.split(",")]
    print(f"\n=== P1: saturation khối->số theo mức chồng k (test {len(te)} ảnh, {args.ntup} bộ/k) ===")
    print(f"{'k':>2s} | {'Σpred_parts':>11s} {'overlay':>8s} {'Δsat%':>6s} | {'tile':>8s} {'Δtile%':>6s} | {'ΣGT':>6s}")
    for k in ks:
        sp, ov, tl, sg = [], [], [], []
        for _ in range(args.ntup):
            pick = rng.choice(te, size=k, replace=False)
            ims = [imgs[i] for i in pick]
            sp.append(sum(count(model, im, dev) for im in ims))
            ov.append(count(model, overlay_min(ims), dev))
            tl.append(count(model, tile_grid(ims), dev))
            sg.append(float(gts[pick].sum()))
        sp, ov, tl, sg = map(np.array, (sp, ov, tl, sg))
        dsat = 100 * (sp - ov).mean() / max(sp.mean(), 1e-6)          # saturation thuần (control per-ảnh)
        dtile = 100 * (sp - tl).mean() / max(sp.mean(), 1e-6)
        print(f"{k:>2d} | {sp.mean():>11.1f} {ov.mean():>8.1f} {dsat:>6.1f} | {tl.mean():>8.1f} {dtile:>6.1f} | {sg.mean():>6.1f}")

    print("\nĐỌC: Δsat% (Σpred_parts − overlay) TĂNG DẦN theo k -> head SATURATE dưới overlap -> P1 ĐÚNG,"
          " module hiệu chỉnh + composite-linearity có tín hiệu -> XÂY.")
    print("     Δsat% ~phẳng (≈0) -> KHÔNG saturation -> DỪNG hướng này.")
    print("     Nếu Δtile% << Δsat% -> overlap (không phải resolution) là thủ phạm (củng cố cơ chế).")


if __name__ == "__main__":
    main()

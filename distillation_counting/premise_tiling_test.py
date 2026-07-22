#!/usr/bin/env python3
"""premise_tiling_test.py — KIỂM TIỀN ĐỀ additivity/tiling TRƯỚC KHI xây STC² (kỷ luật chống-PACT).

Câu hỏi QUYẾT ĐỊNH (sống-chết của ý tưởng loss mới):
  Với ảnh DÀY và raw có ĐỘ PHÂN GIẢI GỐC > 256, đếm(Σ ô cắt ở native-res) có GẦN GT hơn
  đếm(ảnh downscale-256) không?
  - CÓ  -> resolution/tiling mang tín hiệu thật ở crowding -> loss tiling (STC²) đáng xây.
  - KHÔNG -> tiling/upscale không thêm gì -> BỎ (khỏi tốn công vô ích như PACT).

Lập luận: cắt 256²->upscale = nội suy, KHÔNG thêm info. Chỉ ô cắt ở native-res (raw>256) mới
cho model (nhìn 256²) nhiều pixel/nhân hơn -> tách nhân dày tốt hơn. Nên test PHẢI dùng raw native.

Setup: train efflite0 count-only (w_density=0, KHÔNG teacher/mask-target) trên 80% NuInsSeg raw;
test 20%. Mỗi ảnh test:  whole = model(resize(raw,256)).sum();
  tiles = Σ_ô model(resize(ô_native,256)).sum()  (ô cắt ở raw native-res).
Báo theo bin density + đếm ảnh raw>256 (chỗ tín hiệu resolution mới có nghĩa).
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
def _count(model, arr_rgb, dev):
    im = np.asarray(Image.fromarray(arr_rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    t = torch.from_numpy(im.astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
    return float(count_from_density(model(t)[0])[0])


@torch.no_grad()
def tiled_count(model, raw, grid, dev):
    H, W = raw.shape[:2]
    th, tw = H // grid, W // grid
    tot = 0.0
    for i in range(grid):
        for j in range(grid):
            tot += _count(model, raw[i * th:(i + 1) * th, j * tw:(j + 1) * tw], dev)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="thư mục NuInsSeg raw (vd /kaggle/input/datasets/ipateam/nuinsseg)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = "cuda"

    samples = build_index(args.root or find_root())
    recs = []
    for s in samples:
        raw = np.asarray(Image.open(s["image"]).convert("RGB"))
        recs.append({"raw": raw, "gt": float(gt_from_mask(s["mask"])),
                     "organ": s["organ"], "hw": int(max(raw.shape[:2]))})
    hw = np.array([r["hw"] for r in recs])
    print(f"[data] {len(recs)} ảnh | raw max-cạnh: min {hw.min()} med {int(np.median(hw))} "
          f"max {hw.max()} | số ảnh raw>256: {int((hw > 256).sum())}/{len(recs)}")
    if (hw > 256).sum() == 0:
        print("⚠️ KHÔNG ảnh raw nào >256 -> tiling KHÔNG có resolution thật -> tiền đề coi như BÁC ngay.")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(recs)); n_te = len(recs) // 5
    te, tr = idx[:n_te], idx[n_te:]

    def mk(i):
        r = recs[i]
        img = np.asarray(Image.fromarray(r["raw"]).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        return {"img": img.astype(np.uint8),
                "density": np.zeros((IMG_SIZE, IMG_SIZE), np.float32),  # count-only: KHÔNG dùng
                "gt": r["gt"], "organ": r["organ"]}
    data = [mk(i) for i in range(len(recs))]
    model = train(data, dev, args.epochs, 32, 1e-3, list(tr),
                  0.0, 1.0, 0.01, 0.5, 16, True, "poisson", "efficientnet_lite0")
    model.eval()

    rows = []
    for i in te:
        r = recs[i]
        rows.append((r["gt"], _count(model, r["raw"], dev), tiled_count(model, r["raw"], args.grid, dev), r["hw"]))
    a = np.array(rows); gt, cw, ct, rhw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    print(f"\n=== TIỀN ĐỀ: whole vs tiles (native-res, grid {args.grid}x{args.grid}) theo density ===")
    print(f"{'bin':10s} {'n':>3s} {'GT̄':>6s} {'r>256':>6s} {'whole(err)':>14s} {'tiles(err)':>14s} {'Δ(t-w)':>7s}")
    for lab, m in [("Thấp<=20", gt <= 20), ("TB 21-50", (gt > 20) & (gt <= 50)), ("Cao>50", gt > 50)]:
        if m.sum() == 0:
            continue
        print(f"{lab:10s} {int(m.sum()):>3d} {gt[m].mean():>6.1f} {int((rhw[m]>256).sum()):>6d} "
              f"{cw[m].mean():>7.1f}({np.abs(cw[m]-gt[m]).mean():>4.1f}) "
              f"{ct[m].mean():>7.1f}({np.abs(ct[m]-gt[m]).mean():>4.1f}) "
              f"{np.mean(ct[m]-cw[m]):>+7.1f}")
    print(f"\nTỔNG whole MAE={np.abs(cw-gt).mean():.2f} | tiles MAE={np.abs(ct-gt).mean():.2f}")
    print("ĐỌC: nếu bin CAO có tiles-err < whole-err (Δ>0, GẦN GT hơn) -> TIỀN ĐỀ ĐÚNG, STC² đáng xây.")
    print("     nếu tiles KHÔNG tốt hơn (hoặc tệ hơn) ở bin cao -> BÁC, bỏ hướng tiling.")


if __name__ == "__main__":
    main()

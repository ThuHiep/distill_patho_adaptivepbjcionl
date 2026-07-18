#!/usr/bin/env python3
"""CryoNuSeg (ipateam) -> images/*.png + gt_counts.csv (image,gt_count).

Native: 'tissue images/*.tif' (512x512 H&E) + 'Annotator 1 (biologist)/label masks modify/*.tif'
(instance-labeled, cùng basename). GT count/ảnh = số instance ID != 0 trong label mask.
CryoNuSeg = clean OOD (KHÔNG trong PathoSAM training / Lizard). Dataset thứ 3 cho transfer (N5).
Thay MoNuSAC (native 1024 -> resize 256 co nhân 4x, μ sập); CryoNuSeg native 512 -> squish 2x nhẹ hơn.

Chạy: python prep_cryonuseg_counts.py --root ../data/cryonuseg --out ../work/cryonuseg_png
-> images/*.png + gt_counts.csv. Rồi:
  eval_cross_dataset.py --train_dataset nuinsseg --detach_mu \
    --test_images_dir <out>/images --test_gt_csv <out>/gt_counts.csv
"""
import os, csv, glob, argparse
import numpy as np
from PIL import Image


def read_tif(path):
    try:
        import tifffile
        return tifffile.imread(path)
    except Exception:
        return np.asarray(Image.open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/cryonuseg")
    ap.add_argument("--annotator", default="Annotator 1 (biologist)/label masks modify",
                    help="thư mục instance-labeled mask (tương đối trong root)")
    ap.add_argument("--out", default="../work/cryonuseg_png")
    args = ap.parse_args()

    img_dir_in = os.path.join(args.root, "tissue images")
    mask_dir = os.path.join(args.root, args.annotator)
    out_img = os.path.join(args.out, "images")
    os.makedirs(out_img, exist_ok=True)

    rows, skipped = [], 0
    for ip in sorted(glob.glob(os.path.join(img_dir_in, "*.tif"))):
        name = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(mask_dir, name + ".tif")
        if not os.path.exists(mp):
            print(f"[skip] không có mask: {name}"); skipped += 1; continue
        arr = read_tif(ip)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        arr = np.ascontiguousarray(arr[..., :3]).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(out_img, name + ".png"))
        m = read_tif(mp)
        cnt = int(len(np.unique(m[m != 0])))          # số instance ID != 0
        rows.append((name, cnt))

    with open(os.path.join(args.out, "gt_counts.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "gt_count"])
        for name, cnt in rows:
            w.writerow([name, cnt])
    counts = np.array([c for _, c in rows], dtype=float)
    print(f"XONG -> {out_img} ({len(rows)} ảnh, skip {skipped}) + gt_counts.csv")
    print(f"count/ảnh: mean={counts.mean():.1f} min={counts.min():.0f} "
          f"max={counts.max():.0f} tổng={counts.sum():.0f}")


if __name__ == "__main__":
    main()

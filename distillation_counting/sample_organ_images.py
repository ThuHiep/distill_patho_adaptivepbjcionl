#!/usr/bin/env python3
"""sample_organ_images.py — lưu lưới ảnh RAW của các MÔ chỉ định để NHÌN TẬN MẮT đặc điểm tế bào.

Không cần dataset raw — đọc thẳng density-cache (đã chứa img 256² + organ + gt).
Dùng để kiểm: mô lympho/tạo máu (spleen/thymus/kidney) có phải "nhân nhỏ, dày, chồng lấp" không.

VD:
  python sample_organ_images.py --cache /kaggle/input/**/teacher_density_nuinsseg.pkl \
      --organs "spleen,thymus,kidney,tonsile" --per 4 --out /kaggle/working/organ_samples.png
"""
import argparse, glob, pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="glob tới density cache (img+organ+gt)")
    ap.add_argument("--organs", default="spleen,thymus,kidney,tonsile",
                    help="chuỗi con tên mô, phân tách phẩy")
    ap.add_argument("--per", type=int, default=4, help="số ảnh mỗi mô")
    ap.add_argument("--out", default="/kaggle/working/organ_samples.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = glob.glob(args.cache, recursive=True)
    assert files, f"không thấy cache: {args.cache}"
    data = pickle.load(open(files[0], "rb"))
    print(f"[cache] {files[0]} | N={len(data)}")

    wants = [o.strip().lower() for o in args.organs.split(",") if o.strip()]
    rows = []
    for key in wants:
        hits = [d for d in data if key in str(d["organ"]).lower()]
        hits = sorted(hits, key=lambda d: -float(np.ravel(d["gt"])[0]))[:args.per]  # đông nhất trước
        if not hits:
            print(f"  ⚠️ không có mô khớp '{key}'"); continue
        rows.append((key, hits))

    ncol = args.per
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3.2))
    if nrow == 1:
        axes = axes[None, :]
    for r, (key, hits) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r][c]; ax.axis("off")
            if c < len(hits):
                d = hits[c]
                ax.imshow(d["img"])
                ax.set_title(f"{d['organ']}\nGT={float(np.ravel(d['gt'])[0]):.0f} nhân", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[saved] {args.out} — tải về nhìn: mô lympho có 'nhân nhỏ dày chồng lấp' không?")


if __name__ == "__main__":
    main()

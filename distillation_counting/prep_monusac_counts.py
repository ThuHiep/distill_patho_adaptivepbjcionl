"""
prep_monusac_counts.py — MoNuSAC (data/monusac_converted.pkl) -> folder ảnh PNG + gt_counts.csv.
Dùng làm dataset OOD THỨ 3 (all-OOD fair table): mọi model train PanNuke -> test zero-shot MoNuSAC
(TCGA, KHÁC site/organ PanNuke+NuInsSeg -> OOD thật cho TẤT CẢ, kể cả student).

item schema (đã verify): {'image': PNG bytes 1024x1024, 'inst': PNG, 'type_map': PNG,
                          'counts': (4,) float32 per-class, 'source': TCGA id, 'encoded': True}
GT count/ảnh = counts.sum() (tổng nhân 4 lớp) — cùng định nghĩa "tổng nhân" như student/heavy harness.

Chạy: python prep_monusac_counts.py --pkl ../data/monusac_converted.pkl --out ../work/monusac_png
-> images/*.png + gt_counts.csv (cột image,gt_count). Rồi:
  - heavy/lightweight net: dump_cellvit_counts.py / dump NuLite --images_dir <out>/images
  - student: predict PanNuke-trained student trên <out>/images (OOD)
  - chấm: eval_heavy_count.py --gt <out>/gt_counts.csv --preds <net>.csv
"""
from __future__ import annotations
import argparse, csv, io, os, pickle
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="../data/monusac_converted.pkl")
    ap.add_argument("--out", default="../work/monusac_png")
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, "rb"))
    items = d["items"]
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    counts_all = []
    seen = {}
    for i, it in enumerate(items):
        # decode ảnh PNG bytes -> RGB
        img = Image.open(io.BytesIO(it["image"])).convert("RGB")
        # tên duy nhất: source + index (source TCGA có thể lặp nhiều patch)
        src = str(it.get("source", "img")).replace("/", "_")
        name = f"{src}_{i:04d}"
        if name in seen:
            name = f"{name}_{seen[name]}"
        seen[name] = seen.get(name, 0) + 1
        img.save(os.path.join(img_dir, name + ".png"))
        gt = float(np.asarray(it["counts"]).sum())   # tổng nhân 4 lớp
        rows.append([name, gt])
        counts_all.append(gt)

    with open(os.path.join(args.out, "gt_counts.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "gt_count"]); w.writerows(rows)

    c = np.asarray(counts_all)
    print(f"XONG -> {img_dir} ({len(rows)} ảnh) + gt_counts.csv")
    print(f"GT count/ảnh: min={c.min():.0f} max={c.max():.0f} mean={c.mean():.1f} median={np.median(c):.0f} "
          f"(N={len(c)}, tổng nhân={c.sum():.0f})")
    print(f"ảnh đầu: {rows[0][0]}.png -> {rows[0][1]:.0f} nhân | size 1 ảnh:", Image.open(
        os.path.join(img_dir, rows[0][0] + ".png")).size)


if __name__ == "__main__":
    main()

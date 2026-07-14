"""
prep_nuinsseg_as_pannuke.py — Chuẩn bị NuInsSeg thành folder ảnh + GT count cho Phần B (count-MAE
heavy net leak-free). NuInsSeg là OOD với mọi checkpoint PanNuke (CellViT/LKCell/NuLite) → KHÔNG leak.

LÀM GÌ:
  - Duyệt NuInsSeg (build_index đã có), mỗi ảnh: lưu PNG + đọc GT count (số nhân thật).
  - GT count = len(unique(instance_mask)) − background — Y HỆT student (distill_student_nuinsseg.py:174)
    → so sánh count-MAE CÔNG BẰNG với student & teacher.
  - Xuất: <out>/images/*.png, <out>/gt_counts.csv (image,gt,organ), <out>/types.csv (image,tissue).
    (KHÔNG ghi labels/.npy kiểu PanNuke: count chỉ cần instance DỰ ĐOÁN của model; GT của họ chỉ để PQ,
     mà format npy của CellViT khác — tránh viết mù. PQ không cần cho trục count.)

HAI MODE (rủi ro duy nhất — TEST vài ảnh trên vast rồi chốt, xem md Bước 2):
  --mode resize : resize cả ảnh về 256×256 (đơn giản; MẤT resolution → có thể gộp nhân sát nhau → undercount).
  --mode tile   : cắt lưới 256 không chồng (pad tới bội 256); count ảnh = TỔNG count các tile
                  (lỗi biên: nhân nằm mép tile bị cắt/đếm đôi — heavy net PanNuke path KHÔNG overlap-merge).
  → Chạy CẢ HAI trên ~5 ảnh, mắt thường đối chiếu detect vs GT, chọn mode bám GT nhất, GHI RÕ trong paper.

Chạy:
  python prep_nuinsseg_as_pannuke.py --out ../work/nuinsseg_png --mode resize
  python prep_nuinsseg_as_pannuke.py --out ../work/nuinsseg_png_tile --mode tile
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_student_nuinsseg import build_index, find_root, _load_mask  # noqa: E402


def gt_count_from_mask(m: np.ndarray) -> int:
    """Y HỆT student: số instance = #nhãn unique trừ nền 0."""
    return int(len(np.unique(m)) - (1 if (m == 0).any() else 0))


def save_resize(img: np.ndarray, size: int):
    return [(Image.fromarray(img).resize((size, size), Image.BILINEAR), "")]  # 1 ảnh, hậu tố rỗng


def save_tiles(img: np.ndarray, size: int):
    """Cắt lưới không chồng, pad phản chiếu tới bội `size`. Trả list (PIL, suffix _rIcJ)."""
    H, W = img.shape[:2]
    ph, pw = (-H) % size, (-W) % size
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    Hh, Ww = img.shape[:2]
    out = []
    for i in range(0, Hh, size):
        for j in range(0, Ww, size):
            out.append((Image.fromarray(img[i:i+size, j:j+size]), f"_r{i//size}c{j//size}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="NuInsSeg root (mặc định find_root())")
    ap.add_argument("--out", required=True, help="thư mục xuất")
    ap.add_argument("--mode", choices=["resize", "tile"], default="resize")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    root = args.root or find_root()
    samples = build_index(root)
    print(f"NuInsSeg root={root}  #images={len(samples)}  mode={args.mode} size={args.size}")

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    gt_rows, type_rows, tile_rows = [], [], []
    n_nuc = 0
    for k, s in enumerate(samples):
        organ = s["organ"]
        img = np.asarray(Image.open(s["image"]).convert("RGB"))
        m = _load_mask(s["mask"])
        gt = gt_count_from_mask(m)
        n_nuc += gt
        # tên duy nhất: organ + stem (stem có thể trùng giữa các organ)
        stem = f"{organ}__{os.path.splitext(os.path.basename(s['image']))[0]}".replace(" ", "_")
        pieces = save_resize(img, args.size) if args.mode == "resize" else save_tiles(img, args.size)
        for pil, suf in pieces:
            name = f"{stem}{suf}"
            pil.save(os.path.join(img_dir, name + ".png"))
            type_rows.append([name, organ])
            if args.mode == "tile":
                tile_rows.append([name, stem])   # tile -> ảnh gốc (để cộng count)
        gt_rows.append([stem, gt, organ])         # GT count theo ẢNH GỐC (full-res)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(samples)} imgs, {n_nuc} nuclei")

    with open(os.path.join(args.out, "gt_counts.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["image", "gt", "organ"], *gt_rows])
    with open(os.path.join(args.out, "types.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["image", "tissue"], *type_rows])
    if args.mode == "tile":
        with open(os.path.join(args.out, "tiles_map.csv"), "w", newline="") as f:
            csv.writer(f).writerows([["tile", "image"], *tile_rows])
    print(f"XONG -> {args.out}  (gt_counts.csv {len(gt_rows)} ảnh, {n_nuc} nhân, images/ {len(type_rows)} file)")
    print("Bước tiếp: chạy CellViT/LKCell inference trên images/, dump image->len(instance_types) ra preds.csv,")
    print("rồi: python eval_heavy_count.py --gt <out>/gt_counts.csv --preds preds.csv [--tiles_map <out>/tiles_map.csv]")


if __name__ == "__main__":
    main()

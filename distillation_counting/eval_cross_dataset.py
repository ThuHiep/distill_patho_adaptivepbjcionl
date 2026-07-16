"""
eval_cross_dataset.py — CROSS-DATASET TRANSFER (train trên dataset A -> predict dataset B).

Câu hỏi Q1 (generalization của distillation): (μ, σ) mà student HỌC được từ distillation trên
dataset A có CHUYỂN sang một dataset B khác (khác phòng lab, khác nhuộm, khác dải count) không —
hay chỉ khớp riêng A? Đây là bằng chứng transfer mạnh cho một paper distillation.

Thiết kế (leak-free BẨM SINH: A và B là 2 dataset khác nhau -> không thể rò ảnh):
  1. Train MỘT DensitySigmaUNet trên TOÀN BỘ dataset A (density teacher của A + count GT của A).
  2. Predict trên TOÀN BỘ dataset B -> pkl {preds:[{mu,sigma}], gts, organs} (schema chuẩn repo).
  3. Chấm bằng eval_r2_grouped.py trên pkl đó: split conformal HIỆU CHỈNH TRÊN CAL CỦA B
     (chuẩn domain-transfer: recalibrate ở target). -> đo MAE + Winkler + worst-org TRÊN B.

Cùng backbone / loss / sigma_mode / detach_mu như bảng chính -> so trực tiếp với in-domain.
KHÔNG dùng nhãn B lúc train (chỉ dùng ở bước conformal-cal, đúng split-conformal).

⚠️ Kích thước ảnh: cả 2 cache đều lưu ảnh 256×256 (build_*_density resize về IMG_SIZE=256) ->
   input đồng nhất, model train-A áp thẳng lên B được.
⚠️ colon: nếu A HOẶC B là PanNuke, dùng --exclude_tissue colon y hệt bảng chính (leak teacher).

Chạy trên vast (2 chiều):
  # PanNuke -> NuInsSeg
  python eval_cross_dataset.py --train_dataset pannuke --test_dataset nuinsseg \
      --exclude_tissue colon --detach_mu --out work/xfer_pannuke2nuinsseg.pkl
  python eval_r2_grouped.py --preds work/xfer_pannuke2nuinsseg.pkl --seeds 20 --n_clusters 5

  # NuInsSeg -> PanNuke (chiều ngược lại)
  python eval_cross_dataset.py --train_dataset nuinsseg --test_dataset pannuke \
      --exclude_tissue colon --detach_mu --out work/xfer_nuinsseg2pannuke.pkl
  python eval_r2_grouped.py --preds work/xfer_nuinsseg2pannuke.pkl --seeds 20 --n_clusters 3
"""
from __future__ import annotations
import argparse, os, pickle, sys
import numpy as np
import torch
from PIL import Image

REPO = os.environ.get("REPO", "/workspace/sam3_research")
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from distill_student_nuinsseg import build_index, find_root  # noqa: E402
from distill_student_r2 import (  # noqa: E402
    train, predict_r2, build_pannuke_density, build_teacher_density,
)


def _load_dataset(name, args, device):
    """Trả list mẫu {img, density, gt, organ, [fold]} cho dataset `name` (dùng ĐÚNG cache như bảng chính)."""
    tag = "gt" if args.use_gt_density else "teacher"
    if name == "pannuke":
        folds = [int(x) for x in args.pannuke_folds.split(",")]
        fstr = "".join(str(x) for x in sorted(folds))
        cache = f"{REPO}/work/{tag}_density_pannuke_f{fstr}.pkl"
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        data = build_pannuke_density(args.pannuke_root, folds, device, cache, use_gt=args.use_gt_density)
    else:
        cache = f"{REPO}/work/{tag}_density_nuinsseg.pkl"
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        if os.path.exists(cache):
            samples = None
            print(f"[A/{name}] cache có sẵn -> bỏ qua build_index")
        else:
            samples = build_index(find_root())
            print(f"[A/{name}] indexed {len(samples)} pairs")
        data = build_teacher_density(samples, device, cache, use_gt=args.use_gt_density)
    # loại tissue (colon) nếu dataset này là PanNuke — y hệt bảng chính
    if args.exclude_tissue and name == "pannuke":
        ex = [t.strip().lower() for t in args.exclude_tissue.split(",") if t.strip()]
        before = len(data)
        data = [d for d in data if not any(e in str(d["organ"]).lower() for e in ex)]
        print(f"[EXCLUDE/{name}] bỏ tissue chứa {ex}: {before} -> {len(data)} ảnh")
    return data


def _load_test_folder(images_dir, gt_csv):
    """Test set generic từ folder ảnh + gt_counts.csv (image,gt_count) — cho MoNuSAC (prep_monusac_counts.py)
    hoặc bất kỳ dataset OOD nào. Resize 256 (input student). organ='_all_' (chỉ cần count MAE)."""
    import csv, glob
    gt = {}
    with open(gt_csv) as f:
        r = csv.reader(f); next(r)
        for row in r:
            gt[row[0]] = float(row[1])
    data = []
    for p in sorted(glob.glob(os.path.join(images_dir, "*.png"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name not in gt:
            continue
        img = np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)).astype(np.uint8)
        data.append({"img": img, "gt": gt[name], "organ": "_all_"})
    print(f"[TEST-FOLDER] {images_dir}: {len(data)} ảnh (resize 256, OOD zero-shot)")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dataset", choices=["nuinsseg", "pannuke"], required=True)
    ap.add_argument("--test_dataset", choices=["nuinsseg", "pannuke"], default=None,
                    help="dataset test built-in; HOẶC dùng --test_images_dir cho folder ngoài (MoNuSAC).")
    ap.add_argument("--test_images_dir", default=None, help="folder ảnh test ngoài (MoNuSAC) — cần --test_gt_csv")
    ap.add_argument("--test_gt_csv", default=None, help="gt_counts.csv (image,gt_count) cho --test_images_dir")
    ap.add_argument("--pannuke_root", default=f"{REPO}/data/pannuke")
    ap.add_argument("--pannuke_folds", default="1,2,3")
    ap.add_argument("--exclude_tissue", default=None,
                    help="loại tissue (áp CHO PHÍA PanNuke), vd 'colon' — y hệt bảng chính (leak teacher).")
    ap.add_argument("--use_gt_density", action="store_true",
                    help="baseline SUPERVISED: density target từ GT thay vì teacher PathoSAM.")
    # train (mirror distill_student_r2 -> cùng công thức bảng chính)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--student_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--w_density", type=float, default=1.0)
    ap.add_argument("--w_count", type=float, default=0.01)
    ap.add_argument("--w_nll", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sigma_mode", choices=["poisson", "raw", "nb"], default="poisson")
    ap.add_argument("--detach_mu", action="store_true")
    ap.add_argument("--dump_feat", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=f"{REPO}/work/xfer_preds.pkl")
    args = ap.parse_args()

    use_folder = bool(args.test_images_dir)
    assert use_folder or args.test_dataset, "cần --test_dataset HOẶC --test_images_dir + --test_gt_csv"
    assert use_folder or args.train_dataset != args.test_dataset, \
        "cross-dataset: train và test phải KHÁC dataset (in-domain đã có ở bảng chính)."
    if use_folder:
        assert args.test_gt_csv, "--test_images_dir cần kèm --test_gt_csv"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    tgt = args.test_images_dir if use_folder else args.test_dataset
    print(f"device={device} | TRANSFER {args.train_dataset} -> {tgt}")

    # 1) train trên TOÀN BỘ A (test là dataset khác -> không leak ảnh)
    train_data = _load_dataset(args.train_dataset, args, device)
    print(f"[TRAIN] {args.train_dataset}: {len(train_data)} ảnh (train toàn bộ)")
    model = train(train_data, device, args.epochs, args.student_ch, args.lr,
                  list(range(len(train_data))), args.w_density, args.w_count, args.w_nll,
                  args.beta, args.bs, args.detach_mu, args.sigma_mode)

    # 2) predict trên TOÀN BỘ B (built-in dataset hoặc folder ngoài như MoNuSAC)
    if use_folder:
        test_data = _load_test_folder(args.test_images_dir, args.test_gt_csv)
    else:
        test_data = _load_dataset(args.test_dataset, args, device)
    print(f"[TEST] {tgt}: {len(test_data)} ảnh (predict toàn bộ)")
    out = predict_r2(model, test_data, device, dump_feat=args.dump_feat)
    pickle.dump(out, open(args.out, "wb"))

    mu = np.array([p["mu"] for p in out["preds"]])
    sg = np.array([p["sigma"] for p in out["preds"]])
    gt = np.array([g[0] for g in out["gts"]])
    print(f"[OUT] saved {args.out} (N={len(mu)}) | transfer MAE={np.abs(mu-gt).mean():.2f} "
          f"| sigma mean={sg.mean():.2f} std={sg.std():.2f}")
    print(f"  -> chấm conformal (cal trên chính B): "
          f"python eval_r2_grouped.py --preds {args.out} --seeds 20")


if __name__ == "__main__":
    main()

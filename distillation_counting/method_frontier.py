#!/usr/bin/env python3
"""method_frontier.py — ĐƯỜNG CONG params–accuracy (count-only, in-domain NuInsSeg).

Mục tiêu (yêu cầu của cô): bảng Bước-2 accuracy CÔNG BẰNG + hiệu suất-tham số. Câu chuyện:
"1.9M cách xa, nhưng lớn thêm CHÚT (3.6–~12M) đã bám sát SOTA, với 10–200× ít tham số".
Đây là frontier NỘI BỘ: cùng khung DensitySigmaUNet count-only, chỉ đổi BACKBONE -> đo R²/MAE theo params.
(External NuLite-T/StarDist ghép sau làm mốc fair.)

Chú ý: distill KHÔNG dùng ở đây — đã chứng minh distill HẠI accuracy in-domain (efflite0 0.925->0.512).
Accuracy = count-only + chọn/scale backbone (thay đổi CẤU TRÚC). Distill chỉ còn ở nhánh σ/UQ.

Chạy Kaggle (GPU, Internet ON cho pretrained; ĐỪNG pip install):
  !python method_frontier.py --root /kaggle/input/datasets/ipateam/nuinsseg
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from distill_student_nuinsseg import build_index, find_root, _load_mask, IMG_SIZE
from distill_student_r2 import DensitySigmaUNet, train
from r2_losses import count_from_density


def gt_from_mask(path):
    m = _load_mask(path)
    return int(len(np.unique(m)) - (1 if (m == 0).any() else 0))


def _fold_base(root, fold):
    f = f"fold{fold}"
    for c in (root / f / f"Fold {fold}", root / f"Fold {fold}"):
        if (c / "images" / f / "images.npy").exists():
            return c
    raise FileNotFoundError(f"Không thấy Fold {fold} dưới {root}")


def _pannuke_counts(base, fold):
    d = base / "images" / f"fold{fold}"
    if (d / "counts.npy").exists():
        return np.load(d / "counts.npy")
    cache = Path.cwd() / f"pannuke_counts_fold{fold}.npy"
    if cache.exists():
        return np.load(cache)
    mpath = base / "masks" / f"fold{fold}" / "masks.npy"
    print(f"[pannuke] tính counts từ {mpath} (một lần) ...")
    masks = np.load(mpath, mmap_mode="r"); n = masks.shape[0]
    counts = np.zeros((n, 5), np.int32)
    for i in range(n):
        m = np.asarray(masks[i, :, :, :5], np.int32)
        for k in range(5):
            counts[i, k] = int(np.unique(m[:, :, k]).size - 1)
    try:
        np.save(cache, counts)
    except Exception:
        pass
    return counts


def load_pannuke_data(root, folds, exclude, max_imgs, seed=0):
    """Trả list dict {img uint8 256, density zeros, gt, organ} — đúng format train()/eval."""
    root = Path(root); out = []
    for fold in folds:
        base = _fold_base(root, fold)
        d = base / "images" / f"fold{fold}"
        imgs = np.load(d / "images.npy", mmap_mode="r")
        tp = d / "types.npy"
        types = np.load(tp, allow_pickle=True) if tp.exists() else np.array(["na"] * len(imgs))
        gt = _pannuke_counts(base, fold).sum(1).astype(np.float32)
        for i in range(len(imgs)):
            t = str(types[i])
            if exclude and exclude.lower() in t.lower():
                continue
            im = np.asarray(imgs[i])
            if im.max() <= 1.5:
                im = (im * 255)
            out.append({"img": im.astype(np.uint8),
                        "density": np.zeros((IMG_SIZE, IMG_SIZE), np.float32),
                        "gt": float(gt[i]), "organ": t})
    if max_imgs and len(out) > max_imgs:
        r = np.random.default_rng(seed).choice(len(out), max_imgs, replace=False)
        out = [out[i] for i in r]
    return out


def n_params(backbone):
    m = DensitySigmaUNet(32, backbone=backbone)
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def eval_r2_mae(model, data, idx, dev):
    model.eval(); pr, yt = [], []
    for i in idx:
        x = torch.from_numpy(data[i]["img"].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
        pr.append(float(count_from_density(model(x)[0])[0])); yt.append(data[i]["gt"])
    pr, yt = np.array(pr), np.array(yt)
    ss_res = ((yt - pr) ** 2).sum(); ss_tot = ((yt - yt.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, np.abs(pr - yt).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="NuInsSeg root")
    ap.add_argument("--dataset", default="pannuke", choices=["pannuke", "nuinsseg"])
    ap.add_argument("--pannuke_root", default=None)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--exclude_tissue", default="colon")
    ap.add_argument("--max_imgs", type=int, default=0, help="cap ảnh TRAIN (0=hết) để giảm runtime")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--backbones",
                    default="tinyunet,mobilenetv3_small_100,efficientnet_lite0,efficientnet_b1,resnet18,resnet34")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev} | dataset {args.dataset}")

    if args.dataset == "pannuke":
        tr_folds = [int(x) for x in args.train_folds.split(",")]
        data = load_pannuke_data(args.pannuke_root, tr_folds, args.exclude_tissue, args.max_imgs)
        n_tr = len(data)
        data += load_pannuke_data(args.pannuke_root, [int(args.test_fold)], args.exclude_tissue, 0)
        tr_fixed = list(range(n_tr)); te_fixed = list(range(n_tr, len(data)))
        print(f"[data] PanNuke train fold{tr_folds}={n_tr} / test fold{args.test_fold}={len(te_fixed)} (no-{args.exclude_tissue})")
    else:
        samples = build_index(args.root or find_root())
        data = []
        for s in samples:
            im = np.asarray(Image.open(s["image"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
            data.append({"img": im.astype(np.uint8),
                         "density": np.zeros((IMG_SIZE, IMG_SIZE), np.float32),
                         "gt": float(gt_from_mask(s["mask"])), "organ": s["organ"]})
        tr_fixed = te_fixed = None
        print(f"[data] NuInsSeg {len(data)} ảnh (random split)")
    n = len(data)

    seeds = [int(s) for s in args.seeds.split(",")]
    backbones = args.backbones.split(",")
    rows = []
    for bb in backbones:
        try:
            npar = n_params(bb)
        except Exception as e:
            print(f"[skip] {bb}: build lỗi ({e})"); continue
        r2s, maes = [], []
        for seed in seeds:
            if tr_fixed is not None:
                tr, te = tr_fixed, te_fixed                       # PanNuke: fold split cố định
            else:
                rng = np.random.default_rng(seed); idx = rng.permutation(n); n_te = n // 5
                te, tr = idx[:n_te], idx[n_te:]
            try:
                model = train(data, dev, args.epochs, 32, 1e-3, list(tr),
                              0.0, 1.0, 0.01, 0.5, 16, True, "poisson", bb)   # count-only (w_density=0)
                r2, mae = eval_r2_mae(model, data, te, dev)
            except Exception as e:
                print(f"[skip] {bb} seed{seed}: train lỗi ({e})"); r2, mae = np.nan, np.nan
            r2s.append(r2); maes.append(mae)
        rows.append((bb, npar, np.nanmean(r2s), np.nanstd(r2s), np.nanmean(maes), np.nanstd(maes)))
        print(f"  {bb:26s} {npar/1e6:5.1f}M  R² {np.nanmean(r2s):+.3f}±{np.nanstd(r2s):.3f}  "
              f"MAE {np.nanmean(maes):.2f}")

    rows.sort(key=lambda r: r[1])
    print(f"\n=== FRONTIER count-only in-domain ({len(seeds)} seed) ===")
    print(f"  {'backbone':26s} {'params':>8s} {'R² ↑':>16s} {'MAE ↓':>12s}")
    for bb, npar, r2m, r2s, maem, maes in rows:
        print(f"  {bb:26s} {npar/1e6:7.2f}M {r2m:+.3f}±{r2s:.3f}   {maem:6.2f}±{maes:.2f}")
    print("\nĐỌC: R² tăng theo params tới đâu thì bão hoà? -> chọn điểm 'bám SOTA, params nhỏ nhất' làm PACT.")
    print("     Ghép NuLite-T(12M)/StarDist in-domain vào frontier này -> bảng công bằng cuối.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""premise_spatial_grounding.py — PREMISE TẦNG 1 cho Magnitude-Decoupled Spatial Distillation.

Ý tưởng mới (grounded vào vì sao PACT chết): PACT distill teacher-density copy CẢ magnitude ->
nuốt count-sai-OOD teacher -> 0.512. Fix = tách MAGNITUDE (đếm, từ GT) khỏi SHAPE (không gian, từ FM,
chuẩn hoá bỏ magnitude). Nhưng TRƯỚC khi đụng FM, phải test tiền đề: SPATIAL GROUNDING có đáng không?

Cảnh báo từ data cũ: count-only (0.925) >= gt-density-supervised (0.881) trên NuInsSeg -> spatial grounding
CHƯA chắc giúp. Test SẠCH bằng ORACLE (GT-shape, upper bound) trên PanNuke (mask sẵn có):

  baseline : L = |ΣD − GT|                                   (count-only)
  +spatial : L = |ΣD − GT| + w · CE( D/ΣD , GT_shape )       (magnitude từ GT + SHAPE từ GT, chuẩn hoá)

GT_shape = density GT chuẩn hoá sum=1 (mỗi nhân góp 1/area). CE đẩy mass của D về ĐÚNG chỗ nhân,
KHÔNG ép magnitude (magnitude vẫn do count-loss lo). Đây là oracle của "spatial shape từ teacher".

ĐỌC: +spatial > baseline (R² tăng, p<0.05) -> spatial grounding CÓ GIÁ TRỊ -> sang tầng 2 (teacher-shape).
      +spatial <= baseline -> grounding vô ích kể cả oracle -> DỪNG cả ý tưởng, khỏi phí FM.

Chạy Kaggle (GPU; ĐỪNG pip install):
  !python premise_spatial_grounding.py --pannuke_root /kaggle/input/datasets/hipinhththu/pannuke
"""
import argparse
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from distill_student_r2 import DensitySigmaUNet
from r2_losses import count_from_density

IMG = 256


def _fold_base(root, fold):
    f = f"fold{fold}"
    for c in (root / f / f"Fold {fold}", root / f"Fold {fold}"):
        if (c / "images" / f / "images.npy").exists():
            return c
    raise FileNotFoundError(f"Không thấy Fold {fold} dưới {root}")


def _gt_density(mask5):
    """mask5: (256,256,5) instance-labeled -> density (256,256), mỗi nhân góp 1/area."""
    d = np.zeros((IMG, IMG), np.float32)
    for k in range(5):
        lab = mask5[:, :, k]
        for iid in np.unique(lab):
            if iid == 0:
                continue
            m = lab == iid
            a = int(m.sum())
            if a > 0:
                d[m] += 1.0 / a
    return d


def _fast_count(mask5):
    return int(sum(np.unique(mask5[:, :, k]).size - 1 for k in range(5)))


def load_fold(root, fold, exclude, want_shape, max_imgs=0, seed=0):
    """Trả (imgs uint8 (M,256,256,3), gts (M,), shapes (M,256,256) f16 sum=1 hoặc None)."""
    base = _fold_base(root, fold)
    d = base / "images" / f"fold{fold}"
    imgs = np.load(d / "images.npy", mmap_mode="r")
    tp = d / "types.npy"
    types = np.load(tp, allow_pickle=True) if tp.exists() else np.array(["na"] * len(imgs))
    cp = d / "counts.npy"
    counts = np.load(cp) if cp.exists() else None
    need_mask = want_shape or counts is None
    masks = np.load(base / "masks" / f"fold{fold}" / "masks.npy", mmap_mode="r") if need_mask else None
    keep = [i for i in range(len(imgs)) if not (exclude and exclude.lower() in str(types[i]).lower())]
    if max_imgs and len(keep) > max_imgs:
        keep = list(np.random.default_rng(seed).choice(keep, max_imgs, replace=False))
    oi, og, osh = [], [], []
    for j, i in enumerate(keep):
        im = np.asarray(imgs[i])
        if im.max() <= 1.5:
            im = im * 255
        oi.append(im.astype(np.uint8))
        if want_shape:
            dens = _gt_density(np.asarray(masks[i, :, :, :5], np.int32))
            og.append(float(dens.sum())); s = dens.sum()
            osh.append((dens / s if s > 0 else dens).astype(np.float16))
        elif counts is not None:
            og.append(float(counts[i].sum()))
        else:
            og.append(float(_fast_count(np.asarray(masks[i, :, :, :5], np.int32))))
        if (j + 1) % 500 == 0:
            print(f"  [fold{fold}] {j+1}/{len(keep)}")
    imgs_a = np.stack(oi)
    shapes = np.stack(osh) if want_shape else None
    return imgs_a, np.array(og, np.float32), shapes


def density_map(model, x):
    return model(x)[0]                        # (B,1,H,W)


def train_arm(imgs, gts, shapes, tr, dev, epochs, mode, w_sp, bs=16, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DensitySigmaUNet(32, backbone="efficientnet_lite0").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr = np.array(tr)
    imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.
    for ep in range(epochs):
        order = rng.permutation(tr); model.train()
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            x = imgs_t[b].to(dev)
            gt = torch.tensor(gts[b], device=dev, dtype=torch.float32)
            D = density_map(model, x)                          # (B,1,H,W)
            loss = (D.sum(dim=(1, 2, 3)) - gt).abs().mean()
            if mode == "spatial":
                dn = D / (D.sum(dim=(2, 3), keepdim=True) + 1e-6)   # chuẩn hoá bỏ magnitude
                gs = torch.tensor(shapes[b], device=dev, dtype=torch.float32).unsqueeze(1)
                ce = -(gs * torch.log(dn + 1e-8)).sum(dim=(1, 2, 3)).mean()   # đẩy mass về chỗ nhân
                loss = loss + w_sp * ce
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model, imgs_t


@torch.no_grad()
def eval_r2(model, imgs_t, gts, dev):
    model.eval()
    pr = np.array([float(count_from_density(density_map(model, imgs_t[i:i + 1].to(dev)))[0])
                   for i in range(len(gts))])
    ss_res = ((gts - pr) ** 2).sum(); ss_tot = ((gts - gts.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot if ss_tot > 0 else float("nan")), np.abs(pr - gts).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--exclude_tissue", default="colon")
    ap.add_argument("--max_imgs", type=int, default=1500, help="cap ảnh train (build shape tốn)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--w_sp", type=float, default=1.0)
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.pannuke_root)
    print(f"[device] {dev} | w_sp {args.w_sp}")

    tr_imgs, tr_gts, tr_sh = [], [], []
    for f in [int(x) for x in args.train_folds.split(",")]:
        print(f"[load] train fold{f} (+shape từ mask) ...")
        ia, ga, sa = load_fold(root, f, args.exclude_tissue, True, args.max_imgs, seed=42)
        tr_imgs.append(ia); tr_gts.append(ga); tr_sh.append(sa)
    tr_imgs = np.concatenate(tr_imgs); tr_gts = np.concatenate(tr_gts); tr_sh = np.concatenate(tr_sh)
    print(f"[load] test fold{args.test_fold} ...")
    te_imgs, te_gts, _ = load_fold(root, int(args.test_fold), args.exclude_tissue, False, 0)
    print(f"[data] train {len(tr_gts)} / test {len(te_gts)} | GT count mean {tr_gts.mean():.1f}")

    seeds = [int(s) for s in args.seeds.split(",")]
    R = {"baseline": [], "spatial": []}
    for seed in seeds:
        for mode in ["baseline", "spatial"]:
            model, tr_t = train_arm(tr_imgs, tr_gts, tr_sh, range(len(tr_gts)), dev,
                                    args.epochs, mode, args.w_sp, seed=seed)
            te_t = torch.from_numpy(te_imgs).permute(0, 3, 1, 2).float() / 255.
            r2, mae = eval_r2(model, te_t, te_gts, dev)
            R[mode].append((r2, mae))
        print(f"  seed {seed}: baseline R² {R['baseline'][-1][0]:+.3f} | spatial R² {R['spatial'][-1][0]:+.3f}")

    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    b = np.array([x[0] for x in R["baseline"]]); s = np.array([x[0] for x in R["spatial"]])
    bm = np.array([x[1] for x in R["baseline"]]); sm = np.array([x[1] for x in R["spatial"]])
    p = float("nan")
    if wilcoxon is not None and len(b) >= 5 and np.any(s != b):
        try:
            p = wilcoxon(s, b).pvalue
        except Exception:
            pass
    print(f"\n=== PREMISE tầng 1: spatial grounding (oracle GT-shape) — {len(seeds)} seed ===")
    print(f"  baseline (count-only)   R² {b.mean():+.3f}±{b.std():.3f}  MAE {bm.mean():.2f}")
    print(f"  +spatial  (count+shape) R² {s.mean():+.3f}±{s.std():.3f}  MAE {sm.mean():.2f}")
    print(f"  Δ R² {(s-b).mean():+.3f}  (#thắng {int((s>b).sum())}/{len(b)}, p={p:.3g}) | Δ MAE {(sm-bm).mean():+.2f}")
    print("\nĐỌC: Δ R²>0 (p<0.05) -> spatial grounding CÓ GIÁ TRỊ -> sang tầng 2 (teacher-shape thay oracle).")
    print("     Δ R²<=0 -> grounding vô ích kể cả oracle GT -> DỪNG ý tưởng, khỏi phí FM.")


if __name__ == "__main__":
    main()

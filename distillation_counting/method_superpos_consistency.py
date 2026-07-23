#!/usr/bin/env python3
"""method_superpos_consistency.py — CONTRIBUTION: Local Superposition-Consistency (tự-giám-sát, một-phía).

Nền (đã verify): density-counting + nhãn-mức-ảnh under-constrain không gian -> saturate ở vùng dày
(undercount tăng đơn điệu theo overlap: P1 0→26→40→51%). Bước 1 (ràng buộc count TOÀN CỤC) sửa được
saturation NHƯNG bơm bias lên toàn cục -> R²↓ (giúp dày, hại thưa). Gốc lỗi = ràng buộc KHÔNG cục bộ.

Nguyên lý (của mình): bộ đếm density phải CỘNG-TÍNH dưới chồng ảnh, CỤC BỘ theo vùng:
    D(overlay(A,B)) ≥ D(A)+D(B)  theo từng ô     (hiện cộng THIẾU = saturate)
Loss một-phía theo ô (target = density phần, detach, self-sup):
    L = mean_cells relu( pool_P(D_A)+pool_P(D_B) − pool_P(D_overlay) )
  - CỤC BỘ (P=4×4) -> KHÔNG bias toàn cục (ô thưa vốn thoả).   - MỘT-PHÍA -> không ép over-count (occlusion).
  - TỰ-GIÁM-SÁT -> dùng ảnh KHÔNG nhãn (label-efficiency); property tổng quát cross-dataset.

Ablation cô lập biến "locality": baseline / global-selfsup (P=1) / local-selfsup (P=4). Cả 2 self-sup
một-phía, chỉ khác độ mịn -> chứng minh locality có gỡ bias không MÀ vẫn giảm undercount mô dày.

Gate NGHIÊM (pre-register): local vs baseline -> R²-tổng KHÔNG tụt (|Δ|<0.02) VÀ MAE-mô-dày giảm
  -> nguyên lý target được -> METHOD THẬT. R² vẫn tụt -> DỪNG honest.

Chạy Kaggle (GPU; ĐỪNG pip install):  !python method_superpos_consistency.py --root /kaggle/input/datasets/ipateam/nuinsseg
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from distill_student_nuinsseg import build_index, find_root, _load_mask, IMG_SIZE
from distill_student_r2 import DensitySigmaUNet
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


def load_pannuke(root, folds, exclude, max_imgs, seed=0):
    """Load PanNuke trực tiếp từ .npy (img + counts.sum + tissue), KHÔNG cần lib/teacher/mask.
    counts.npy (N,5) do precompute_pannuke_counts tạo. Loại tissue `exclude` (vd colon: leak teacher)."""
    root = Path(root); ims, gts, tis = [], [], []
    for fold in folds:
        d = _fold_base(root, fold) / "images" / f"fold{fold}"
        imgs = np.load(d / "images.npy")
        types = np.load(d / "types.npy", allow_pickle=True)
        counts = np.load(d / "counts.npy")               # (N,5)
        gt = counts.sum(1).astype(np.float32)
        for i in range(len(imgs)):
            t = str(types[i])
            if exclude and exclude.lower() in t.lower():
                continue
            ims.append(imgs[i]); gts.append(float(gt[i])); tis.append(t)
    ims = np.stack(ims).astype(np.float32)
    if ims.max() > 1.5:
        ims /= 255.
    gts, tis = np.array(gts), np.array(tis)
    if max_imgs and len(ims) > max_imgs:
        r = np.random.default_rng(seed).choice(len(ims), max_imgs, replace=False)
        ims, gts, tis = ims[r], gts[r], tis[r]
    return torch.from_numpy(ims).permute(0, 3, 1, 2), gts, tis


def density(model, x):
    return model(x)[0]                       # (B,1,H,W) >=0, có grad


def poolP(D, P):
    return F.adaptive_avg_pool2d(D, P)       # (B,1,P,P) trung bình mỗi ô (tỉ lệ với sum ô)


def train_arm(imgs, gts, tr, dev, epochs, P, w_sup, bs=16, seed=0):
    """P=0: baseline (không consistency). P>=1: superposition-consistency một-phía độ mịn PxP."""
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DensitySigmaUNet(32, backbone="efficientnet_lite0").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr = np.array(tr)
    for ep in range(epochs):
        order = rng.permutation(tr); model.train()
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            x = imgs[b].to(dev)
            gt = torch.tensor(gts[b], device=dev, dtype=torch.float32)
            D = density(model, x)                                  # grad (cho count loss)
            loss = (count_from_density(D) - gt).abs().mean()
            if P >= 1 and x.shape[0] >= 2:                         # superposition-consistency
                perm = torch.randperm(x.shape[0], device=dev)
                O = torch.minimum(x, x[perm])                      # overlay (min-blend) = chồng-lấp
                D_O = density(model, O)                            # grad
                Ddet = D.detach()
                tgt = poolP(Ddet, P) + poolP(Ddet[perm], P)        # tổng phần (self-sup, detach)
                pred = poolP(D_O, P)
                loss = loss + w_sup * F.relu(tgt - pred).mean()    # MỘT-PHÍA: chỉ kéo undercount lên
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


@torch.no_grad()
def preds(model, imgs, idx, dev):
    model.eval()
    return np.array([float(count_from_density(density(model, imgs[i:i + 1].to(dev)))[0]) for i in idx])


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum(); ss_tot = ((true - true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def worst_organ_r2(pred, true, org_te, min_n=8):
    vals = [r2(pred[org_te == o], true[org_te == o]) for o in set(org_te) if (org_te == o).sum() >= min_n]
    return min(vals) if vals else float("nan")


@torch.no_grad()
def sat_k4(model, imgs, te, dev, ntup=200, seed=0):
    model.eval(); rng = np.random.default_rng(seed); sp, ov = [], []
    for _ in range(ntup):
        pick = rng.choice(te, size=4, replace=False)
        sp.append(sum(float(count_from_density(density(model, imgs[i:i + 1].to(dev)))[0]) for i in pick))
        O = imgs[pick].to(dev).min(dim=0)[0][None]
        ov.append(float(count_from_density(density(model, O))[0]))
    sp, ov = np.array(sp), np.array(ov)
    return 100 * (sp - ov).mean() / max(sp.mean(), 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="NuInsSeg root (dataset=nuinsseg)")
    ap.add_argument("--dataset", default="nuinsseg", choices=["nuinsseg", "pannuke"])
    ap.add_argument("--pannuke_root", default=None)
    ap.add_argument("--folds", default="3", help="PanNuke folds, vd '3' (clean) hoặc '1,2,3'")
    ap.add_argument("--exclude_tissue", default="colon", help="loại tissue PanNuke (leak teacher)")
    ap.add_argument("--max_imgs", type=int, default=0, help="cap số ảnh (0=hết) để giảm runtime")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--w_sup", type=float, default=1.0)
    ap.add_argument("--min_n", type=int, default=8, help="min ảnh/organ cho worst-organ (PanNuke để 20)")
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev} | dataset {args.dataset} | w_sup {args.w_sup}")

    if args.dataset == "pannuke":
        folds = [int(x) for x in args.folds.split(",")]
        ims, gts, organs = load_pannuke(args.pannuke_root, folds, args.exclude_tissue, args.max_imgs)
        print(f"[data] PanNuke fold{folds} no-{args.exclude_tissue}: {len(gts)} ảnh")
    else:
        samples = build_index(args.root or find_root())
        ims_l, gts, organs = [], [], []
        for s in samples:
            im = np.asarray(Image.open(s["image"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                            np.float32) / 255.
            ims_l.append(im); gts.append(float(gt_from_mask(s["mask"]))); organs.append(s["organ"])
        ims = torch.from_numpy(np.stack(ims_l)).permute(0, 3, 1, 2)
        gts, organs = np.array(gts), np.array(organs)
    org_mean = {o: gts[organs == o].mean() for o in set(organs)}
    dense_orgs = sorted(org_mean, key=org_mean.get, reverse=True)[:4]
    print(f"[data] {len(gts)} ảnh | {len(set(organs))} mô | mô-dày: {[(o, round(org_mean[o])) for o in dense_orgs]}")

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [("baseline", 0), ("global-selfsup", 1), ("local-selfsup", 4)]
    R = {a: {"r2": [], "worst": [], "densemae": [], "sat": []} for a, _ in arms}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(gts)); n_te = len(gts) // 5
        te, tr = idx[:n_te], idx[n_te:]
        dmask = np.array([organs[i] in dense_orgs for i in te])
        for name, P in arms:
            m = train_arm(ims, gts, tr, dev, args.epochs, P, args.w_sup, seed=seed)
            pr = preds(m, ims, te, dev); yt = gts[te]
            R[name]["r2"].append(r2(pr, yt))
            R[name]["worst"].append(worst_organ_r2(pr, yt, organs[te], args.min_n))
            R[name]["densemae"].append(np.abs(pr[dmask] - yt[dmask]).mean() if dmask.any() else np.nan)
            R[name]["sat"].append(sat_k4(m, ims, te, dev, seed=seed))
        print(f"  seed {seed} done")

    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    def pval(a, b):
        d = np.array(a) - np.array(b)
        if wilcoxon is None or len(d) < 5 or not np.any(d != 0):
            return float("nan")
        try:
            return wilcoxon(a, b).pvalue
        except Exception:
            return float("nan")

    print(f"\n=== Local Superposition-Consistency ({len(seeds)} seed, paired) ===")
    print(f"  {'metric':22s} {'baseline':>16s} {'global(P1)':>16s} {'local(P4)':>16s}")
    for lab, key in [("R² tổng ↑", "r2"), ("worst-organ R² ↑", "worst"),
                     ("MAE mô-dày ↓ [KEY]", "densemae"), ("Δsat% overlay ↓", "sat")]:
        b, g, l = (np.array(R[a][key]) for a, _ in arms)
        print(f"  {lab:22s} {b.mean():>8.3f}±{b.std():.3f} {g.mean():>8.3f}±{g.std():.3f} {l.mean():>8.3f}±{l.std():.3f}")

    base = R["baseline"]
    print("\n  === significance vs baseline (paired Wilcoxon) ===")
    for name in ["global-selfsup", "local-selfsup"]:
        dr2 = np.array(R[name]["r2"]) - np.array(base["r2"])
        dma = np.array(R[name]["densemae"]) - np.array(base["densemae"])
        pr2, pma = pval(R[name]["r2"], base["r2"]), pval(R[name]["densemae"], base["densemae"])
        print(f"  {name:16s} ΔR² {dr2.mean():+.3f} (p={pr2:.3g}, #win {int((dr2>0).sum())}/{len(dr2)}) | "
              f"ΔMAE-mô-dày {dma.mean():+.2f} (p={pma:.3g}, #win {int((dma<0).sum())}/{len(dma)})")
    print("\nĐỌC: local — ΔR²>0 (p<0.05) VÀ ΔMAE-mô-dày<0 (p<0.05) = METHOD THẬT, significant.")
    print("     baseline<global<local đơn điệu + local > global significant = LOCALITY là mấu chốt (giả thuyết đúng).")
    print("     Nếu p>0.05 (chưa đủ power) → tăng seed / kiểm PanNuke trước khi kết luận.")


if __name__ == "__main__":
    main()

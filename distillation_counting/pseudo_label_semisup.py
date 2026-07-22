#!/usr/bin/env python3
"""pseudo_label_semisup.py — PIVOT sau khi feature-distill funnel CHẾT (8-seed p=0.74, w_feat sweep đổi dấu).

Chẩn đoán vì sao funnel chết (§0.7):
  - Kênh feature-mimicry ĐẤU capacity: ép student 3.6M khớp feature 768-chiều Phikon = bất khả.
  - reliability-gate áp NHẦM: hạ trọng số NHÃN THẬT trong regime khan nhãn = tự bắn chân.

GIỮ cái tốt / THAY cái xấu:
  - GIỮ: surplus FM có thật (probe Phikon @25% OOD 0.838 > student 0.688) + tín hiệu confidence + count-only.
  - THAY kênh: probe làm TEACHER OUTPUT (teacher THẬT mạnh hơn student — điều kiện distill hoạt động),
    KHÔNG mimic feature -> student học count như nó vốn giỏi, không đấu capacity.
  - THAY việc của cổng: confidence -> CHỌN pseudo-label đáng tin trên ảnh KHÔNG nhãn (thêm data),
    thay vì hạ nhãn thật (vứt data).

Thiết kế (semi-supervised, count-only, đúng hook label-efficiency):
  1. leave-organ-out; 25% ảnh train-organ = LABELED, phần còn lại = UNLABELED pool.
  2. Teacher = ENSEMBLE K probe (MLP trên Phikon pooled đông lạnh, bootstrap) trên LABELED.
  3. pseudo-count(unlabeled) = mean ensemble; confidence = 1/std ensemble (bất định deep-ensemble, KHÔNG cần GT).
  4. confident-gate: giữ pseudo có std < median (nửa tin nhất).
  5. student efflite0 count-only train trên: labeled-only | +pseudo-all | +pseudo-confident. Eval test-organ.

Self-validating (in để verify tiền đề NGAY):
  - teacher(probe) R² trên test-organ: teacher CÓ mạnh hơn student baseline không?
  - pseudo R² trên unlabeled pool (all vs confident): cổng CÓ chọn pseudo tốt hơn không? (unlabeled có GT thật để chấm).

Chạy Kaggle (Internet ON, GPU; ĐỪNG pip install):
  !python pseudo_label_semisup.py --root /kaggle/input/datasets/ipateam/nuinsseg
"""
import argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from distill_student_nuinsseg import build_index, find_root, _load_mask
from distill_student_r2 import DensitySigmaUNet
from r2_losses import count_from_density

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def gt_from_mask(p):
    m = _load_mask(p)
    return int(len(np.unique(m)) - (1 if (m == 0).any() else 0))


@torch.no_grad()
def cache_phikon_pooled(imgs224, dev, bs=32):
    """(N,3,224,224)[0,1] -> pooled Phikon (N,768) (mean patch tokens) — nhẹ, chỉ cho probe."""
    from transformers import AutoModel
    ph = AutoModel.from_pretrained("owkin/phikon").to(dev).eval()
    pooled = []
    for i in range(0, len(imgs224), bs):
        x = ((imgs224[i:i + bs].to(dev)) - MEAN.to(dev)) / STD.to(dev)
        h = ph(pixel_values=x).last_hidden_state[:, 1:, :]        # (B,196,768)
        pooled.append(h.mean(1).float().cpu())
    del ph
    if dev == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(pooled).numpy()


def probe_ensemble(pooled, gts, lab_idx, pred_idx, dev, k=5, epochs=200, seed=0):
    """ENSEMBLE k probe (MLP 768->128->1) bootstrap trên LABELED. Trả (mean, std) count trên pred_idx.
    std = bất định ensemble (KHÔNG cần GT) -> tín hiệu confidence pseudo-label."""
    lab = np.array(lab_idx)
    Xall = torch.tensor(pooled, device=dev)
    yl = gts[lab]
    preds = []
    for j in range(k):
        rng = np.random.default_rng(seed * 100 + j)
        bs_idx = rng.choice(lab, size=len(lab), replace=True)             # bootstrap
        Xf = torch.tensor(pooled[bs_idx], device=dev)
        yf = torch.tensor(gts[bs_idx], device=dev).float()
        torch.manual_seed(seed * 100 + j)
        net = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            opt.zero_grad(); ((net(Xf).squeeze(1) - yf) ** 2).mean().backward(); opt.step()
        with torch.no_grad():
            preds.append(net(Xall[pred_idx]).squeeze(1).cpu().numpy())
    P = np.stack(preds)                                                   # (k, n_pred)
    return P.mean(0), P.std(0)


def r2_mae(pred, true):
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    ss_res = ((true - pred) ** 2).sum(); ss_tot = ((true - true.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot if ss_tot > 0 else float("nan")), np.abs(pred - true).mean()


def train_student(imgs, targets, idx, weights, dev, epochs, warmup_idx, warmup=15, bs=16, seed=0):
    """count-only |mu-target|*weight. warmup_idx = tập train trong warm-up (chỉ labeled) rồi mới thêm pseudo."""
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DensitySigmaUNet(32, backbone="efficientnet_lite0").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    idx = np.array(idx); warmup_idx = np.array(warmup_idx)
    for ep in range(epochs):
        cur = warmup_idx if ep < warmup else idx
        order = rng.permutation(cur)
        model.train()
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            x = imgs[b].to(dev)
            tgt = torch.tensor(targets[b], device=dev, dtype=torch.float32)
            w = torch.tensor(weights[b], device=dev, dtype=torch.float32)
            mu = count_from_density(model(x)[0])
            loss = ((mu - tgt).abs() * w).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


@torch.no_grad()
def eval_r2(model, imgs, gts, idx, dev):
    model.eval()
    pr = np.array([float(count_from_density(model(imgs[i:i + 1].to(dev))[0])[0]) for i in idx])
    return r2_mae(pr, gts[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--frac", type=float, default=0.25, help="tỉ lệ ảnh train-organ CÓ nhãn")
    ap.add_argument("--w_pseudo", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=5, help="số probe trong ensemble teacher")
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")

    samples = build_index(args.root or find_root())
    im256, im224, gts, organs = [], [], [], []
    for s in samples:
        raw = Image.open(s["image"]).convert("RGB")
        im256.append(np.asarray(raw.resize((256, 256), Image.BILINEAR), np.float32) / 255.)
        im224.append(np.asarray(raw.resize((224, 224), Image.BILINEAR), np.float32) / 255.)
        gts.append(float(gt_from_mask(s["mask"]))); organs.append(s["organ"])
    im256 = torch.from_numpy(np.stack(im256)).permute(0, 3, 1, 2)
    im224 = torch.from_numpy(np.stack(im224)).permute(0, 3, 1, 2)
    gts = np.array(gts); organs = np.array(organs)
    print(f"[data] {len(gts)} ảnh | {len(set(organs))} mô")
    print("[phikon] cache pooled ..."); pooled = cache_phikon_pooled(im224, dev)

    seeds = [int(s) for s in args.seeds.split(",")]
    modes = ["labeled-only", "pseudo-all", "pseudo-confident"]
    res = {m: [] for m in modes}
    teach_r2, pq_all, pq_conf = [], [], []          # sanity: teacher & pseudo quality
    for seed in seeds:
        rng = np.random.default_rng(seed)
        orgs = sorted(set(organs)); rng.shuffle(orgs)
        te_org = set(orgs[:max(1, len(orgs) // 5)])
        te = np.where([o in te_org for o in organs])[0]
        tr_all = np.where([o not in te_org for o in organs])[0]
        k = max(30, int(len(tr_all) * args.frac))
        lab = rng.choice(tr_all, size=min(k, len(tr_all)), replace=False)
        unlab = np.array([i for i in tr_all if i not in set(lab)])

        # teacher ensemble: chấm test-organ (sanity mạnh hơn student?) + pseudo trên unlabeled
        t_te_mean, _ = probe_ensemble(pooled, gts, lab, te, dev, k=args.k, seed=seed)
        r2t, _ = r2_mae(t_te_mean, gts[te]); teach_r2.append(r2t)
        p_mean, p_std = probe_ensemble(pooled, gts, lab, unlab, dev, k=args.k, seed=seed)
        conf = p_std < np.median(p_std)                                   # nửa tin nhất
        r2pa, _ = r2_mae(p_mean, gts[unlab]); pq_all.append(r2pa)
        r2pc, _ = r2_mae(p_mean[conf], gts[unlab[conf]]); pq_conf.append(r2pc)

        # target/weight arrays cho toàn bộ ảnh (student chỉ đọc ở index được cấp)
        targets = gts.astype(np.float32).copy()
        targets[unlab] = p_mean.astype(np.float32)                        # pseudo cho unlabeled
        weights = np.ones(len(gts), np.float32)

        line = [f"seed {seed}"]
        for mode in modes:
            if mode == "labeled-only":
                idx = lab
            elif mode == "pseudo-all":
                idx = np.concatenate([lab, unlab])
            else:  # pseudo-confident
                idx = np.concatenate([lab, unlab[conf]])
            w = weights.copy()
            w[unlab] = args.w_pseudo                                      # pseudo nhẹ hơn nếu muốn
            m = train_student(im256, targets, idx, w, dev, args.epochs, warmup_idx=lab, seed=seed)
            r2, _ = eval_r2(m, im256, gts, te, dev)
            res[mode].append(r2); line.append(f"{mode.split('-')[-1]} {r2:+.3f}")
        print("  " + " | ".join(line) + f"  [teacher {r2t:+.3f}]")

    print(f"\n=== SANITY tiền đề ({len(seeds)} seed) ===")
    print(f"  teacher(probe) R² test-organ : {np.mean(teach_r2):+.3f}±{np.std(teach_r2):.3f}"
          f"  (student labeled-only {np.mean(res['labeled-only']):+.3f}) "
          f"-> teacher {'MẠNH HƠN' if np.mean(teach_r2) > np.mean(res['labeled-only']) else 'KHÔNG hơn'}")
    print(f"  pseudo R² unlabeled  all     : {np.mean(pq_all):+.3f}±{np.std(pq_all):.3f}")
    print(f"  pseudo R² unlabeled  confident: {np.mean(pq_conf):+.3f}±{np.std(pq_conf):.3f}"
          f" -> cổng {'CHỌN TỐT HƠN' if np.mean(pq_conf) > np.mean(pq_all) else 'KHÔNG lọc tốt'}")

    print(f"\n=== STUDENT R² test-organ, {int(args.frac*100)}% nhãn — {len(seeds)} seed ===")
    base = np.array(res["labeled-only"])
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    print(f"  {'mode':16s} {'R² mean±sd':>14s} {'Δ vs lab':>9s} {'#thắng':>7s} {'p(Wilcoxon)':>12s}")
    for mode in modes:
        a = np.array(res[mode]); d = a - base
        if mode == "labeled-only":
            print(f"  {mode:16s} {a.mean():+.3f}±{a.std():.3f}")
            continue
        wins = int((d > 0).sum())
        p = float("nan")
        if wilcoxon is not None and len(d) >= 5 and np.any(d != 0):
            try:
                p = wilcoxon(a, base).pvalue
            except Exception:
                pass
        print(f"  {mode:16s} {a.mean():+.3f}±{a.std():.3f} {d.mean():+9.3f} {wins:>4d}/{len(d)} {p:>12.4g}")
    print("\nĐỌC: pseudo-confident > labeled-only (paired, p<0.05) = FM-pseudo-label CÓ giá trị label-efficiency. "
          "pseudo-all vs confident = cổng confidence có lọc được lỗi teacher không. "
          "Nếu teacher KHÔNG mạnh hơn student ở SANITY -> pseudo vô nghĩa, dừng.")


if __name__ == "__main__":
    main()

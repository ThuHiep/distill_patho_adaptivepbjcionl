#!/usr/bin/env python3
"""pseudo_label_semisup.py — TEST CÔNG BẰNG cuối cho FM-pseudo-label (v2, transductive + teacher mạnh).

3 vòng trước (funnel / w_feat / pseudo-v1) đều +0.05 KHÔNG significant vì 2 lỗi thiết kế:
  (1) teacher bị làm YẾU (ensemble bootstrap, hidden 128) -> 0.719 thay vì 0.838 (§0.6 full-data).
  (2) pseudo pool KHÔNG phủ vùng shift: unlabeled toàn mô-đã-thấy, test là mô-CHƯA-thấy
      -> nhồi data mô-cũ không dạy student gì về mô mới = bỏ đói pseudo-label đúng chỗ nó có ích.

v2 sửa cả hai (giữ tốt / thay xấu):
  - TEACHER MẠNH: 1 probe full-data (768->256->1, 300ep) như §0.6 (0.838).
  - TRANSDUCTIVE (SSL công bằng): ảnh test-organ CHIA ĐÔI -> nửa vào pool UNLABELED (student thấy ẢNH,
    KHÔNG thấy nhãn, nhận pseudo), nửa để EVAL (không train cả pseudo). Pseudo giờ PHỦ vùng shift.
  - BỎ confidence-gate (v1 đã chứng minh vô dụng: pseudo-all == pseudo-confident).

Câu hỏi đúng: FM pseudo-label trên ảnh KHÔNG-NHÃN của mô-mới có giúp student thích nghi shift không?
Self-validating: teacher R² trên eval (mạnh hơn student?), pseudo R² trên te_pool (chất lượng pseudo VÙNG SHIFT).

Stop-rule pre-register: p<0.05 -> method label-efficiency thật (đi tiếp đường cong 10/25/50%).
                        không -> pseudo = mục phụ (ổn định), DỪNG.

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
    """(N,3,224,224)[0,1] -> pooled Phikon (N,768) (mean patch tokens) cho probe."""
    from transformers import AutoModel
    ph = AutoModel.from_pretrained("owkin/phikon").to(dev).eval()
    pooled = []
    for i in range(0, len(imgs224), bs):
        x = ((imgs224[i:i + bs].to(dev)) - MEAN.to(dev)) / STD.to(dev)
        h = ph(pixel_values=x).last_hidden_state[:, 1:, :]
        pooled.append(h.mean(1).float().cpu())
    del ph
    if dev == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(pooled).numpy()


def strong_probe(pooled, gts, lab_idx, pred_idx, dev, epochs=300, seed=0):
    """TEACHER MẠNH: 1 probe MLP 768->256->1 trên TOÀN BỘ labeled (như §0.6, đạt ~0.838).
    Trả pseudo-count trên pred_idx (numpy)."""
    lab = np.array(lab_idx)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    Xf = torch.tensor(pooled[lab], device=dev); yf = torch.tensor(gts[lab], device=dev).float()
    for _ in range(epochs):
        opt.zero_grad(); ((net(Xf).squeeze(1) - yf) ** 2).mean().backward(); opt.step()
    with torch.no_grad():
        return net(torch.tensor(pooled[pred_idx], device=dev)).squeeze(1).cpu().numpy()


def r2_mae(pred, true):
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    ss_res = ((true - pred) ** 2).sum(); ss_tot = ((true - true.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot if ss_tot > 0 else float("nan")), np.abs(pred - true).mean()


def train_student(imgs, targets, idx, dev, epochs, warmup_idx, warmup=15, bs=16, seed=0):
    """count-only |mu-target|. warm-up chỉ labeled rồi mới thêm pseudo (tránh student non học pseudo sớm)."""
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
            mu = count_from_density(model(x)[0])
            loss = (mu - tgt).abs().mean()
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
    modes = ["labeled-only", "pseudo-transductive"]
    res = {m: [] for m in modes}
    teach_r2, pq_shift, pq_indom = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        orgs = sorted(set(organs)); rng.shuffle(orgs)
        te_org = set(orgs[:max(1, len(orgs) // 5)])
        te_all = np.where([o in te_org for o in organs])[0]
        rng.shuffle(te_all)
        half = len(te_all) // 2
        te_pool, te_eval = te_all[:half], te_all[half:]        # pool=unlabeled(pseudo), eval=held-out
        tr_all = np.where([o not in te_org for o in organs])[0]
        k = max(30, int(len(tr_all) * args.frac))
        lab = rng.choice(tr_all, size=min(k, len(tr_all)), replace=False)
        tr_unlab = np.array([i for i in tr_all if i not in set(lab)])
        pool = np.concatenate([tr_unlab, te_pool])             # pseudo pool PHỦ cả in-domain lẫn shift

        # teacher mạnh: pseudo cho pool + chấm eval (sanity teacher>student?)
        t_eval = strong_probe(pooled, gts, lab, te_eval, dev, seed=seed)
        r2t, _ = r2_mae(t_eval, gts[te_eval]); teach_r2.append(r2t)
        p_pool = strong_probe(pooled, gts, lab, pool, dev, seed=seed)
        r2ps, _ = r2_mae(p_pool[len(tr_unlab):], gts[te_pool]); pq_shift.append(r2ps)   # pseudo VÙNG SHIFT
        r2pi, _ = r2_mae(p_pool[:len(tr_unlab)], gts[tr_unlab]); pq_indom.append(r2pi)

        targets = gts.astype(np.float32).copy()
        targets[pool] = p_pool.astype(np.float32)

        line = [f"seed {seed}"]
        for mode in modes:
            idx = lab if mode == "labeled-only" else np.concatenate([lab, pool])
            m = train_student(im256, targets, idx, dev, args.epochs, warmup_idx=lab, seed=seed)
            r2, _ = eval_r2(m, im256, gts, te_eval, dev)
            res[mode].append(r2); line.append(f"{mode.split('-')[0]} {r2:+.3f}")
        print("  " + " | ".join(line) + f"  [teacher {r2t:+.3f}]")

    print(f"\n=== SANITY tiền đề ({len(seeds)} seed) ===")
    print(f"  teacher(probe) R² eval : {np.mean(teach_r2):+.3f}±{np.std(teach_r2):.3f}"
          f"  (student labeled-only {np.mean(res['labeled-only']):+.3f}) "
          f"-> teacher {'MẠNH HƠN' if np.mean(teach_r2) > np.mean(res['labeled-only']) else 'KHÔNG hơn'}")
    print(f"  pseudo R² VÙNG SHIFT (te_pool) : {np.mean(pq_shift):+.3f}±{np.std(pq_shift):.3f}  "
          f"(in-domain {np.mean(pq_indom):+.3f}) -> pseudo trên mô-mới {'DÙNG ĐƯỢC' if np.mean(pq_shift) > 0.3 else 'YẾU'}")

    print(f"\n=== STUDENT R² eval (mô-mới held-out), {int(args.frac*100)}% nhãn — {len(seeds)} seed ===")
    base = np.array(res["labeled-only"])
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    print(f"  {'mode':20s} {'R² mean±sd':>14s} {'Δ vs lab':>9s} {'#thắng':>7s} {'p(Wilcoxon)':>12s}")
    for mode in modes:
        a = np.array(res[mode]); d = a - base
        if mode == "labeled-only":
            print(f"  {mode:20s} {a.mean():+.3f}±{a.std():.3f}")
            continue
        wins = int((d > 0).sum()); p = float("nan")
        if wilcoxon is not None and len(d) >= 5 and np.any(d != 0):
            try:
                p = wilcoxon(a, base).pvalue
            except Exception:
                pass
        print(f"  {mode:20s} {a.mean():+.3f}±{a.std():.3f} {d.mean():+9.3f} {wins:>4d}/{len(d)} {p:>12.4g}")
    print("\nĐỌC: pseudo-transductive > labeled-only (paired, p<0.05) = FM-pseudo-label CHỮA shift = method thật. "
          "Nếu teacher không mạnh hơn HOẶC pseudo vùng-shift yếu -> pseudo hết cửa, DỪNG.")


if __name__ == "__main__":
    main()

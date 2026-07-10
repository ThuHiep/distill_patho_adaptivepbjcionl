"""
distill_student_pbud.py — Trainer Paper 2: distill PathoSAM -> student nhẹ với loss PBUD/CCAD.

CHẠY: vast.ai (GPU + PathoSAM + NuInsSeg). Loss (pbud_losses.py) đã test local 16/16.
Điểm chạm PathoSAM/NuInsSeg copy từ distill_student_nuinsseg.py (đã có, bạn đã chạy KD baseline).

Khác distill_student_nuinsseg.py (KD foreground thuần):
  - Cache thêm TEACHER LABEL IMAGE (instance) + teacher per-instance score s_T.
  - Lúc train: student existence per-instance s_i^S = ROI-pool prob của student trên TỪNG teacher
    instance (khả vi) -> tính PB moment (mu,var) -> loss PBUD (distill mean+VARIANCE) / CCAD (cân
    bằng conditional coverage theo organ).
  - Lúc infer: student dùng CONNECTED COMPONENTS của chính nó (self-contained) -> preds schema.

⚠️ GHI RÕ (không giấu): v1 này TRAIN theo teacher proposals nhưng INFER theo proposals của student.
Vì cả hai cùng suy từ 1 foreground map của student, hiệu chỉnh per-instance existence kỳ vọng chuyển
được sang inference. Đây là lựa chọn v1 để loss KHẢ VI; ablation/khắc phục để mở.

--loss: kd | pbud | ccad | pbud_ccad
Cách chạy:
  python distill_student_pbud.py --loss pbud --student_ch 32 --epochs 60 --out work/student_pbud.pkl
Rồi đo: python eval_coverage_transfer.py --teacher ../data/pathosam_nuinsseg_preds.pkl \
             --student work/student_pbud.pkl --seeds 20
"""
from __future__ import annotations
import argparse, os, pickle, sys, time
import numpy as np
from PIL import Image
import torch

REPO = os.environ.get("REPO", "/workspace/sam3_research")
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

# tái dùng phần đã có
from distill_student_nuinsseg import (  # noqa: E402
    build_index, find_root, TinyUNet, dice_bce_loss, student_predict, IMG_SIZE, _load_mask,
)
from pbud_losses import pb_moments, pbud_loss, ccad_loss  # noqa: E402


# ---------- Phase A: teacher targets (fg map + instance label + s_T) ----------
@torch.no_grad()
def build_teacher_targets_pbud(samples, device, cache):
    from pathosam_lib import load_pathosam, pathosam_instances  # verified
    import torch.nn.functional as F
    if os.path.exists(cache):
        print(f"[A] load cache {cache}")
        return pickle.load(open(cache, "rb"))
    predictor, segmenter = load_pathosam(device)
    data = []
    t0 = time.time()
    for k, s in enumerate(samples):
        img = np.asarray(Image.open(s["image"]).convert("RGB"))
        H, W = img.shape[:2]
        masks, scores, _ = pathosam_instances(img, predictor, segmenter)
        fg = np.asarray(segmenter._foreground, dtype=np.float32)
        # label image (instance) ở IMG_SIZE
        label = np.zeros((IMG_SIZE, IMG_SIZE), np.int16)
        for i, m in enumerate(masks):
            mr = np.asarray(Image.fromarray(m.astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), Image.NEAREST)).astype(bool)
            label[mr] = i + 1
        img_r = np.asarray(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        fg_t = torch.from_numpy(fg)[None, None]
        fg_r = F.interpolate(fg_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                             align_corners=False)[0, 0].numpy()
        fg_r = np.clip(fg_r, 0.0, 1.0)
        m = _load_mask(s["mask"])
        gt = int(len(np.unique(m)) - (1 if (m == 0).any() else 0))
        data.append({"img": img_r.astype(np.uint8), "fg": fg_r.astype(np.float32),
                     "label": label, "s_T": np.asarray(scores, np.float32),
                     "gt": float(gt), "organ": s["organ"]})
        if (k + 1) % 100 == 0:
            print(f"[A] {k+1}/{len(samples)} {(time.time()-t0)/(k+1):.2f}s/img")
    pickle.dump(data, open(cache, "wb"))
    print(f"[A] saved {cache}")
    return data


# ---------- student per-instance existence (khả vi) từ teacher label ----------
def student_instance_scores(prob_map: torch.Tensor, label: np.ndarray):
    """prob_map:(H,W) khả vi. label:(H,W) int (teacher instances). Trả s_S:(n,) khả vi, khớp thứ tự
    id 1..n. Instance nào không còn pixel (do resize) -> bỏ để khớp s_T tương ứng (trả cả mask id)."""
    ids = [i for i in np.unique(label) if i != 0]
    lab_t = torch.from_numpy(label.astype(np.int64)).to(prob_map.device)
    s_list, keep = [], []
    for i in ids:
        m = (lab_t == i)
        area = m.sum()
        if area < 1:
            continue
        s_list.append((prob_map * m).sum() / area)
        keep.append(i - 1)  # index vào s_T
    if not s_list:
        return prob_map.new_zeros(0), np.zeros(0, int)
    return torch.stack(s_list), np.asarray(keep, int)


def train(data, device, epochs, ch, loss_kind, lr, train_idx, alpha, beta, gamma, bs=6):
    model = TinyUNet(ch).to(device)
    print(f"[B] student ch={ch} params={sum(p.numel() for p in model.parameters())/1e6:.3f}M "
          f"loss={loss_kind}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        np.random.shuffle(train_idx)
        logs = []
        for i in range(0, len(train_idx), bs):
            idxs = train_idx[i:i + bs]
            imgs = torch.from_numpy(
                np.stack([data[j]["img"] for j in idxs]).astype(np.float32) / 255.0
            ).permute(0, 3, 1, 2).to(device)
            logits = model(imgs)                 # (B,1,H,W)
            prob = torch.sigmoid(logits)
            total = imgs.new_zeros(())
            mu_b, var_b, gt_b, grp_b = [], [], [], []
            for bi, j in enumerate(idxs):
                d = data[j]
                if loss_kind == "kd":
                    fg = torch.from_numpy(d["fg"])[None, None].to(device)
                    total = total + dice_bce_loss(logits[bi:bi+1], fg)
                    continue
                s_S, keep = student_instance_scores(prob[bi, 0], d["label"])
                p_S = torch.ones(len(s_S), 1, device=device)
                s_T = torch.from_numpy(d["s_T"][keep]).to(device) if len(keep) else \
                    torch.zeros(0, device=device)
                p_T = torch.ones(len(s_T), 1, device=device)
                gt = torch.tensor([d["gt"]], device=device)
                out = pbud_loss(s_S, p_S, s_T, p_T, gt, alpha, beta, gamma)
                total = total + out["loss"]
                mu, var = pb_moments(s_S, p_S)
                mu_b.append(mu); var_b.append(var); gt_b.append(gt); grp_b.append(d["organ"])
            if loss_kind in ("ccad", "pbud_ccad") and len(gt_b) >= 2:
                total = total + ccad_loss(mu_b, var_b, gt_b, grp_b)["loss"]
            total = total / len(idxs)
            opt.zero_grad(); total.backward(); opt.step()
            logs.append(float(total.detach()))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[B] ep {ep+1}/{epochs} loss={np.mean(logs):.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["kd", "pbud", "ccad", "pbud_ccad"], default="pbud")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--student_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--min_area", type=int, default=5)
    ap.add_argument("--cache", default=f"{REPO}/work/teacher_targets_pbud_nuinsseg.pkl")
    ap.add_argument("--out", default=f"{REPO}/work/student_pbud_nuinsseg_preds.pkl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    samples = build_index(find_root())
    print(f"indexed {len(samples)} pairs")
    data = build_teacher_targets_pbud(samples, device, args.cache)
    model = train(data, device, args.epochs, args.student_ch, args.loss, args.lr,
                  list(range(len(data))), args.alpha, args.beta, args.gamma)
    out = student_predict(model, data, device, args.thresh, args.min_area)
    pickle.dump(out, open(args.out, "wb"))
    est = np.array([p["scores"].sum() for p in out["preds"]])
    gtv = np.array([g[0] for g in out["gts"]])
    print(f"[C] saved {args.out} | MAE={np.abs(est-gtv).mean():.2f}")


if __name__ == "__main__":
    main()

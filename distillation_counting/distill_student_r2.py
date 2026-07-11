"""
distill_student_r2.py — Trainer Paper 2 / R2: Distributional Count Distillation.

Student nhẹ xuất TRỰC TIẾP (density_map, log_sigma). count mu = sum(density); sigma = exp(log_sigma).
Train/eval CÙNG đo (mu, sigma) -> aligned (sửa misalignment của PBUD/CCAD, xem
PHAN_TICH_LOI_va_THIET_KE_LAI.md mục 1 & 6). Loss = r2_losses.r2_loss (đã test local 17/17).

Teacher targets (Phase A):
  - density_T(x) = sum_i [x in mask_i] / area_i   (density-map counting: sum = số instance).
    Không cần dot-annotation; dùng FULL mask teacher -> bền hơn centroid Gaussian.
  - gt = số nhân thật từ mask NuInsSeg.

Student (Phase B): DensitySigmaUNet = TinyUNet backbone + density head (softplus, >=0)
  + log_sigma head (global pool bottleneck -> scalar/ảnh).

Inference (Phase C): mu = sum(density_S), sigma = exp(log_sigma). Output schema:
  {preds:[{mu,sigma}], gts:[[gt]], organs:[...]}  -> đo bằng eval_r2_conformal.py.

Baseline so sánh (CỔNG): student_kd.pkl (distill_student_nuinsseg.py) — eval_r2_conformal.py
tự suy mu=sum(scores), sigma=sqrt(PB var) từ scores, nên KD và R2 so ĐƯỢC trên cùng conformal.

Chạy trên vast:
  python distill_student_r2.py --epochs 80 --student_ch 32 --out work/student_r2.pkl
  python eval_r2_conformal.py --preds work/student_r2.pkl --seeds 20
"""
from __future__ import annotations
import argparse, os, pickle, sys, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.environ.get("REPO", "/workspace/sam3_research")
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

from distill_student_nuinsseg import (  # noqa: E402
    build_index, find_root, DoubleConv, IMG_SIZE, _load_mask,
)
from r2_losses import r2_loss, count_from_density  # noqa: E402


# ===================== Student: density + log_sigma =====================
class DensitySigmaUNet(nn.Module):
    """TinyUNet backbone -> (density_map>=0, log_sigma scalar/ảnh).
    ch=32 => ~1.9M params (như TinyUNet) + head log_sigma nhỏ."""
    def __init__(self, ch=32):
        super().__init__()
        self.d1 = DoubleConv(3, ch);          self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(ch, ch * 2);     self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(ch * 2, ch * 4); self.p3 = nn.MaxPool2d(2)
        self.bott = DoubleConv(ch * 4, ch * 8)
        self.u3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2); self.c3 = DoubleConv(ch * 8, ch * 4)
        self.u2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2); self.c2 = DoubleConv(ch * 4, ch * 2)
        self.u1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2);     self.c1 = DoubleConv(ch * 2, ch)
        self.dens = nn.Conv2d(ch, 1, 1)            # -> softplus = density >=0
        # log_sigma head: pool bottleneck (ch*8) -> MLP -> scalar
        self.sig = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch * 8, ch), nn.ReLU(inplace=True), nn.Linear(ch, 1))
        # init log_sigma bias ~ log(15): count NuInsSeg ~ chục -> sigma khởi động hợp lý
        nn.init.constant_(self.sig[-1].bias, 2.7)

    def forward(self, x):
        x1 = self.d1(x); x2 = self.d2(self.p1(x1)); x3 = self.d3(self.p2(x2))
        xb = self.bott(self.p3(x3))
        y = self.c3(torch.cat([self.u3(xb), x3], 1))
        y = self.c2(torch.cat([self.u2(y), x2], 1))
        y = self.c1(torch.cat([self.u1(y), x1], 1))
        density = F.relu(self.dens(y))             # (B,1,H,W) >=0 (nền=0 chính xác, như CSRNet)
        log_sigma = self.sig(xb).squeeze(1)        # (B,)
        return density, log_sigma


# ===================== Phase A: teacher density targets =====================
@torch.no_grad()
def build_teacher_density(samples, device, cache, use_gt=False):
    """use_gt=False: density target = instance PathoSAM (distill từ foundation model).
    use_gt=True : density target = instance GT NuInsSeg (baseline SUPERVISED, không dùng teacher)
                  -> tách được GIÁ TRỊ của foundation-model teacher."""
    if os.path.exists(cache):
        print(f"[A] load cache {cache}")
        return pickle.load(open(cache, "rb"))
    if not use_gt:
        from pathosam_lib import load_pathosam, pathosam_instances  # verified
        predictor, segmenter = load_pathosam(device)
    data = []
    t0 = time.time()
    for k, s in enumerate(samples):
        img = np.asarray(Image.open(s["image"]).convert("RGB"))
        m = _load_mask(s["mask"])
        gt = int(len(np.unique(m)) - (1 if (m == 0).any() else 0))
        dens = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
        if use_gt:
            # instance từ label GT: mỗi id != 0 là 1 nhân
            for iid in np.unique(m):
                if iid == 0:
                    continue
                mr = np.asarray(Image.fromarray((m == iid).astype(np.uint8)).resize(
                    (IMG_SIZE, IMG_SIZE), Image.NEAREST)).astype(bool)
                a = int(mr.sum())
                if a > 0:
                    dens[mr] += 1.0 / a
        else:
            masks, scores, _ = pathosam_instances(img, predictor, segmenter)
            for mask in masks:
                mr = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
                    (IMG_SIZE, IMG_SIZE), Image.NEAREST)).astype(bool)
                a = int(mr.sum())
                if a > 0:
                    dens[mr] += 1.0 / a
        img_r = np.asarray(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        data.append({"img": img_r.astype(np.uint8), "density": dens,
                     "gt": float(gt), "organ": s["organ"]})
        if (k + 1) % 100 == 0:
            print(f"[A] {k+1}/{len(samples)} {(time.time()-t0)/(k+1):.2f}s/img")
    pickle.dump(data, open(cache, "wb"))
    print(f"[A] saved {cache}")
    return data


# ===================== Phase B: train =====================
def train(data, device, epochs, ch, lr, train_idx, w_density, w_count, w_nll, beta, bs,
          detach_mu=False):
    model = DensitySigmaUNet(ch).to(device)
    print(f"[B] DensitySigmaUNet ch={ch} params={sum(p.numel() for p in model.parameters())/1e6:.3f}M "
          f"w=(dens {w_density}, count {w_count}, nll {w_nll}) beta={beta} detach_mu={detach_mu}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        np.random.shuffle(train_idx)
        logs = {"loss": [], "L_density": [], "L_count": [], "L_nll": [], "sigma": []}
        for i in range(0, len(train_idx), bs):
            idxs = train_idx[i:i + bs]
            imgs = torch.from_numpy(
                np.stack([data[j]["img"] for j in idxs]).astype(np.float32) / 255.0
            ).permute(0, 3, 1, 2).to(device)
            dT = torch.from_numpy(
                np.stack([data[j]["density"] for j in idxs]))[:, None].to(device)
            gt = torch.tensor([data[j]["gt"] for j in idxs], device=device, dtype=torch.float32)
            dens_S, log_sigma = model(imgs)
            out = r2_loss(dens_S, dT, log_sigma, gt, w_density, w_count, w_nll, beta, detach_mu)
            opt.zero_grad(); out["loss"].backward(); opt.step()
            for key in ("loss", "L_density", "L_count", "L_nll"):
                logs[key].append(float(out[key]))
            logs["sigma"].append(float(out["sigma"].mean()))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[B] ep {ep+1}/{epochs} loss={np.mean(logs['loss']):.4f} "
                  f"Ldens={np.mean(logs['L_density']):.4f} Lcount={np.mean(logs['L_count']):.3f} "
                  f"Lnll={np.mean(logs['L_nll']):.3f} sigma~{np.mean(logs['sigma']):.2f}")
    return model


# ===================== Phase C: inference -> (mu, sigma) =====================
@torch.no_grad()
def predict_r2(model, data, device):
    model.eval()
    preds, gts, organs = [], [], []
    for d in data:
        img = torch.from_numpy(d["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        dens_S, log_sigma = model(img)
        mu = float(count_from_density(dens_S)[0])
        sigma = float(torch.exp(log_sigma)[0])
        preds.append({"mu": mu, "sigma": sigma})
        gts.append([d["gt"]]); organs.append(d["organ"])
    return {"preds": preds, "gts": gts, "organs": organs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--student_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_density", type=float, default=1.0)
    ap.add_argument("--w_count", type=float, default=0.01, help="L_count là |mu-GT| (thang chục) -> trọng số nhỏ")
    ap.add_argument("--w_nll", type=float, default=0.01, help="L_nll thang lớn -> trọng số nhỏ để cân với density MSE")
    ap.add_argument("--beta", type=float, default=0.5, help="beta-NLL (Seitzer 2022)")
    ap.add_argument("--detach_mu", action="store_true",
                    help="tách mu khỏi NLL (NLL chỉ dạy sigma) — sửa NLL-coupling làm hỏng MAE")
    ap.add_argument("--use_gt_density", action="store_true",
                    help="baseline SUPERVISED: density target từ GT NuInsSeg thay vì teacher PathoSAM")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--cache", default=f"{REPO}/work/teacher_density_nuinsseg.pkl")
    ap.add_argument("--out", default=f"{REPO}/work/student_r2_nuinsseg_preds.pkl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    samples = build_index(find_root())
    print(f"indexed {len(samples)} pairs")
    cache = args.cache
    if args.use_gt_density and cache == f"{REPO}/work/teacher_density_nuinsseg.pkl":
        cache = f"{REPO}/work/gt_density_nuinsseg.pkl"  # cache RIÊNG để không đè teacher density
    data = build_teacher_density(samples, device, cache, use_gt=args.use_gt_density)
    model = train(data, device, args.epochs, args.student_ch, args.lr, list(range(len(data))),
                  args.w_density, args.w_count, args.w_nll, args.beta, args.bs, args.detach_mu)
    out = predict_r2(model, data, device)
    pickle.dump(out, open(args.out, "wb"))
    mu = np.array([p["mu"] for p in out["preds"]])
    sg = np.array([p["sigma"] for p in out["preds"]])
    gtv = np.array([g[0] for g in out["gts"]])
    print(f"[C] saved {args.out} | MAE={np.abs(mu-gtv).mean():.2f} "
          f"| sigma mean={sg.mean():.2f} std={sg.std():.2f} (std>0 => heteroscedastic)")


if __name__ == "__main__":
    main()

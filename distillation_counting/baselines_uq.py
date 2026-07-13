"""
baselines_uq.py — Baseline UQ HIỆN ĐẠI trên CÙNG student nhẹ (~1.9M) để so trục reliability
với R2 (Distributional Count Distillation). Fair compute: mọi baseline dùng đúng backbone
DoubleConv/U-Net + đúng protocol leak-free (PanNuke test_fold, NuInsSeg cross-fit) + đúng teacher
density cache. Khác nhau CHỈ ở cách sinh bất định per-ảnh (μ,σ) hoặc khoảng (q_lo,q_hi):

  method=mcdropout : MC-Dropout (Gal & Ghahramani 2016). Thêm Dropout2d vào decoder; T forward
                     ngẫu nhiên lúc test (dropout BẬT, BN giữ running-stats) -> μ=mean count,
                     σ=std count. Bất định EPISTEMIC. -> pkl {mu,sigma} -> eval_r2_grouped.py.
  method=ensemble  : Deep Ensembles (Lakshminarayanan et al. 2017). M student R2 khác seed;
                     mixture: μ*=mean μ_m, σ*²=mean(σ_m²)+mean(μ_m²)−μ*² (aleatoric+epistemic).
                     -> pkl {mu,sigma} -> eval_r2_grouped.py.
  method=cqr       : Conformalized Quantile Regression (Romano et al. 2019). 2 quantile head
                     (τ=α/2, 1−α/2) pinball loss; conformal hoá E=max(q_lo−y, y−q_hi) ở eval.
                     -> pkl {mu,q_lo,q_hi} -> eval_cqr_grouped.py.
  method=chdqr     : Conformalized High-Density QR (arXiv 2411.01266, 2024). Học lưới quantile;
                     lúc test chọn CẶP (τ_lo,τ_hi) khối lượng ≥1−α cho khoảng NGẮN NHẤT (highest-
                     density cho phân phối đơn đỉnh) rồi conformal hoá. -> pkl {mu,q_lo,q_hi}.

Ablation σ (không cần file này): raw-σ = distill_student_r2.py --sigma_mode raw (đã hỗ trợ).

CHẤM: (μ,σ)-type -> eval_r2_grouped.py (y hệt bảng mục 8). quantile-type -> eval_cqr_grouped.py
(cùng seeds/cal_ratio/grouping/Winkler/organ_conditional_stats -> so TRỰC TIẾP). MAE mọi baseline
dùng μ = Σdensity (cùng backbone đếm) -> công bằng.

Chạy trên vast (ví dụ PanNuke leak-free no-colon fold3):
  python baselines_uq.py --method mcdropout --dataset pannuke --test_fold 3 --exclude_tissue colon \
      --out work/uq_mcdropout_pannuke_f3.pkl
  python baselines_uq.py --method ensemble  --dataset pannuke --test_fold 3 --exclude_tissue colon \
      --M 5 --out work/uq_ensemble_pannuke_f3.pkl
  python baselines_uq.py --method cqr       --dataset pannuke --test_fold 3 --exclude_tissue colon \
      --out work/uq_cqr_pannuke_f3.pkl
  python baselines_uq.py --method chdqr     --dataset pannuke --test_fold 3 --exclude_tissue colon \
      --out work/uq_chdqr_pannuke_f3.pkl
"""
from __future__ import annotations
import argparse, os, pickle, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.environ.get("REPO", "/workspace/sam3_research")
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from distill_student_nuinsseg import (  # noqa: E402
    DoubleConv, IMG_SIZE, build_index, find_root, assign_kfold,
)
from distill_student_r2 import (  # noqa: E402
    DensitySigmaUNet, train as train_r2, predict_r2,
    build_pannuke_density, build_teacher_density,
)
from r2_losses import count_from_density  # noqa: E402

EPS = 1e-6


# ============================ Kiến trúc ============================
class MCDropoutUNet(nn.Module):
    """U-Net density (như TinyUNet/DensitySigmaUNet backbone) + Dropout2d trong decoder cho MC-Dropout.
    Chỉ xuất density (>=0); bất định lấy từ T forward ngẫu nhiên (dropout BẬT lúc test)."""
    def __init__(self, ch=32, p_drop=0.2):
        super().__init__()
        self.d1 = DoubleConv(3, ch);          self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(ch, ch * 2);     self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(ch * 2, ch * 4); self.p3 = nn.MaxPool2d(2)
        self.bott = DoubleConv(ch * 4, ch * 8); self.drop_b = nn.Dropout2d(p_drop)
        self.u3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2); self.c3 = DoubleConv(ch * 8, ch * 4)
        self.u2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2); self.c2 = DoubleConv(ch * 4, ch * 2)
        self.u1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2);     self.c1 = DoubleConv(ch * 2, ch)
        self.drop3 = nn.Dropout2d(p_drop); self.drop2 = nn.Dropout2d(p_drop); self.drop1 = nn.Dropout2d(p_drop)
        self.dens = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        x1 = self.d1(x); x2 = self.d2(self.p1(x1)); x3 = self.d3(self.p2(x2))
        xb = self.drop_b(self.bott(self.p3(x3)))
        y = self.drop3(self.c3(torch.cat([self.u3(xb), x3], 1)))
        y = self.drop2(self.c2(torch.cat([self.u2(y), x2], 1)))
        y = self.drop1(self.c1(torch.cat([self.u1(y), x1], 1)))
        return F.relu(self.dens(y))  # (B,1,H,W) >=0


class QuantileUNet(nn.Module):
    """U-Net density + head quantile (cho CQR/CHDQR). density -> μ=Σ (đếm, giữ backbone như R2).
    Quantile head: pool bottleneck -> MLP -> T offset ĐƠN ĐIỆU quanh μ (không cross):
      q_k = μ.detach() + off_k, off tăng dần (base + cumsum(softplus(gaps))).
    μ detach: quantile học ĐỘ RỘNG quanh count, không kéo lệch μ (nhất quán detach_mu của R2)."""
    def __init__(self, ch=32, n_taus=2):
        super().__init__()
        self.n_taus = n_taus
        self.d1 = DoubleConv(3, ch);          self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(ch, ch * 2);     self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(ch * 2, ch * 4); self.p3 = nn.MaxPool2d(2)
        self.bott = DoubleConv(ch * 4, ch * 8)
        self.u3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2); self.c3 = DoubleConv(ch * 8, ch * 4)
        self.u2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2); self.c2 = DoubleConv(ch * 4, ch * 2)
        self.u1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2);     self.c1 = DoubleConv(ch * 2, ch)
        self.dens = nn.Conv2d(ch, 1, 1)
        self.qh = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch * 8, ch), nn.ReLU(inplace=True), nn.Linear(ch, n_taus))

    def forward(self, x):
        x1 = self.d1(x); x2 = self.d2(self.p1(x1)); x3 = self.d3(self.p2(x2))
        xb = self.bott(self.p3(x3))
        y = self.c3(torch.cat([self.u3(xb), x3], 1))
        y = self.c2(torch.cat([self.u2(y), x2], 1))
        y = self.c1(torch.cat([self.u1(y), x1], 1))
        density = F.relu(self.dens(y))
        mu = density.sum(dim=(1, 2, 3))                      # (B,)
        raw = self.qh(xb)                                    # (B,T)
        base = raw[:, :1]                                    # (B,1)
        if self.n_taus > 1:
            gaps = F.softplus(raw[:, 1:])                    # (B,T-1) >0 -> đơn điệu tăng
            off = torch.cat([base, base + torch.cumsum(gaps, dim=1)], dim=1)
        else:
            off = base
        q = mu.detach().unsqueeze(1) + off                  # (B,T) quanh count
        return density, q


# ============================ Loss/train ============================
def _batch(data, idxs, device):
    imgs = torch.from_numpy(
        np.stack([data[j]["img"] for j in idxs]).astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2).to(device)
    dT = torch.from_numpy(np.stack([data[j]["density"] for j in idxs]))[:, None].to(device)
    gt = torch.tensor([data[j]["gt"] for j in idxs], device=device, dtype=torch.float32)
    return imgs, dT, gt


def train_pointwise(data, device, epochs, ch, lr, train_idx, w_density, w_count, bs, p_drop):
    """Train MCDropoutUNet: density MSE (KD teacher) + w_count*|Σdensity−GT|. Dropout BẬT khi train."""
    model = MCDropoutUNet(ch, p_drop=p_drop).to(device)
    print(f"[mcdropout] params={sum(p.numel() for p in model.parameters())/1e6:.3f}M p_drop={p_drop}")
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train()
    idx = list(train_idx)
    for ep in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            imgs, dT, gt = _batch(data, idx[i:i + bs], device)
            dens = model(imgs); mu = count_from_density(dens)
            loss = w_density * F.mse_loss(dens, dT) + w_count * (mu - gt).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def pinball(q, y, tau):
    """Pinball/quantile loss: mean(max(tau*(y−q), (tau−1)*(y−q))). q,y:(B,)."""
    e = y - q
    return torch.maximum(tau * e, (tau - 1.0) * e).mean()


def train_quantile(data, device, epochs, ch, lr, train_idx, taus, w_density, w_count, w_pin, bs):
    """Train QuantileUNet: density MSE + w_count*|μ−GT| + w_pin*mean_k pinball(q_k, GT, tau_k)."""
    taus_t = torch.tensor(taus, device=device, dtype=torch.float32)
    model = QuantileUNet(ch, n_taus=len(taus)).to(device)
    print(f"[quantile] params={sum(p.numel() for p in model.parameters())/1e6:.3f}M n_taus={len(taus)}")
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train()
    idx = list(train_idx)
    for ep in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            imgs, dT, gt = _batch(data, idx[i:i + bs], device)
            dens, q = model(imgs); mu = count_from_density(dens)
            L_pin = torch.stack([pinball(q[:, k], gt, taus_t[k]) for k in range(len(taus))]).mean()
            loss = w_density * F.mse_loss(dens, dT) + w_count * (mu - gt).abs().mean() + w_pin * L_pin
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# ============================ Predict ============================
def _enable_dropout(model):
    """MC-Dropout lúc test: BN giữ running-stats (eval), CHỈ Dropout về train (ngẫu nhiên)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def predict_mcdropout(model, data, device, T):
    _enable_dropout(model)
    preds, gts, organs = [], [], []
    for d in data:
        img = torch.from_numpy(d["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        counts = np.array([float(count_from_density(model(img))[0]) for _ in range(T)])
        preds.append({"mu": float(counts.mean()), "sigma": float(max(counts.std(), EPS))})
        gts.append([d["gt"]]); organs.append(d["organ"])
    return {"preds": preds, "gts": gts, "organs": organs}


def predict_ensemble(models, data, device):
    """Mixture Lakshminarayanan: μ*=mean μ_m; σ*²=mean(σ_m²)+mean(μ_m²)−μ*²."""
    per = [predict_r2(m, data, device) for m in models]  # mỗi cái {preds:[{mu,sigma}],...}
    N = len(data); preds = []
    for i in range(N):
        mus = np.array([per[m]["preds"][i]["mu"] for m in range(len(models))])
        sgs = np.array([per[m]["preds"][i]["sigma"] for m in range(len(models))])
        mu = float(mus.mean())
        var = float((sgs ** 2).mean() + (mus ** 2).mean() - mu ** 2)
        preds.append({"mu": mu, "sigma": float(np.sqrt(max(var, EPS)))})
    return {"preds": preds, "gts": per[0]["gts"], "organs": per[0]["organs"]}


@torch.no_grad()
def predict_quantile(model, data, device, taus, method, alpha):
    """CQR: taus=[α/2,1−α/2] -> (q_lo,q_hi)=(q[0],q[1]). CHDQR: taus=lưới -> chọn CẶP khối lượng
    ≥1−α có q_hi−q_lo NHỎ NHẤT (highest-density). μ=Σdensity cho MAE. Trả {mu,q_lo,q_hi}."""
    model.eval()
    taus = np.asarray(taus, float)
    preds, gts, organs = [], [], []
    for d in data:
        img = torch.from_numpy(d["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        dens, q = model(img)
        mu = float(count_from_density(dens)[0]); qv = q[0].cpu().numpy()  # (T,)
        if method == "cqr":
            q_lo, q_hi = float(qv[0]), float(qv[-1])
        else:  # chdqr: cặp ngắn nhất với khối lượng >= 1-alpha
            best = None
            for a in range(len(taus)):
                for b in range(a + 1, len(taus)):
                    if taus[b] - taus[a] >= (1 - alpha) - 1e-9:
                        w = qv[b] - qv[a]
                        if best is None or w < best[0]:
                            best = (w, qv[a], qv[b])
            if best is None:  # lưới không phủ 1−α -> dùng cực biên
                best = (qv[-1] - qv[0], qv[0], qv[-1])
            q_lo, q_hi = float(best[1]), float(best[2])
        preds.append({"mu": mu, "q_lo": q_lo, "q_hi": q_hi})
        gts.append([d["gt"]]); organs.append(d["organ"])
    return {"preds": preds, "gts": gts, "organs": organs}


# ============================ Data + folds (leak-free, mirror distill_student_r2) ============================
def load_data(args, device):
    tag = "gt" if args.use_gt_density else "teacher"
    if args.dataset == "pannuke":
        folds = [int(x) for x in args.pannuke_folds.split(",")]
        fstr = "".join(str(x) for x in sorted(folds))
        cache = args.cache or f"{REPO}/work/{tag}_density_pannuke_f{fstr}.pkl"
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        data = build_pannuke_density(args.pannuke_root, folds, device, cache, use_gt=args.use_gt_density)
    else:
        cache = args.cache or f"{REPO}/work/{tag}_density_nuinsseg.pkl"
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        if os.path.exists(cache):
            samples = None; print("[A] cache có sẵn -> bỏ qua build_index")
        else:
            samples = build_index(find_root()); print(f"indexed {len(samples)} pairs")
        data = build_teacher_density(samples, device, cache, use_gt=args.use_gt_density)
    if args.exclude_tissue:
        ex = [t.strip().lower() for t in args.exclude_tissue.split(",") if t.strip()]
        before = len(data)
        data = [d for d in data if not any(e in str(d["organ"]).lower() for e in ex)]
        print(f"[EXCLUDE] bỏ tissue chứa {ex}: {before} -> {len(data)} ảnh")
    return data


def prediction_folds(data, args):
    """Trả list (train_idx, test_idx) LEAK-FREE: NuInsSeg cross-fit K; PanNuke train khác fold ->
    predict test_fold; else train+predict all (debug)."""
    N = len(data)
    if args.kfold and args.kfold > 1:
        fold_of = assign_kfold([d["organ"] for d in data], args.kfold, args.seed)
        out = []
        for f in range(args.kfold):
            tr = [i for i in range(N) if fold_of[i] != f]
            te = [i for i in range(N) if fold_of[i] == f]
            out.append((tr, te))
        print(f"[FOLDS] NuInsSeg cross-fit {args.kfold} fold (leak-free)")
        return out
    if args.dataset == "pannuke" and args.test_fold is not None:
        tr = [i for i, d in enumerate(data) if d["fold"] != args.test_fold]
        te = [i for i, d in enumerate(data) if d["fold"] == args.test_fold]
        assert tr and te, f"test_fold={args.test_fold} không tách được"
        print(f"[FOLDS] PanNuke train {len(tr)} | TEST fold {args.test_fold} ({len(te)}) — leak-free")
        return [(tr, te)]
    print("[FOLDS] WARN: train+predict TOÀN BỘ (LEAK, chỉ debug)")
    return [(list(range(N)), list(range(N)))]


def run_method(method, data, folds, args, device):
    """Với mỗi (train,test): fit baseline trên train, predict test, ghép leak-free."""
    N = len(data)
    all_p = [None] * N; all_g = [None] * N; all_o = [None] * N
    if method == "chdqr":
        taus = list(np.round(np.linspace(0.02, 0.98, args.n_taus), 4))
    elif method == "cqr":
        taus = [args.alpha / 2.0, 1.0 - args.alpha / 2.0]
    for fi, (tr, te) in enumerate(folds):
        print(f"[{method}] fold {fi+1}/{len(folds)}: train {len(tr)} -> predict {len(te)}")
        te_data = [data[i] for i in te]
        if method == "mcdropout":
            m = train_pointwise(data, device, args.epochs, args.student_ch, args.lr, tr,
                                 args.w_density, args.w_count, args.bs, args.p_drop)
            of = predict_mcdropout(m, te_data, device, args.T)
        elif method == "ensemble":
            models = []
            for s in range(args.M):
                np.random.seed(args.seed + 100 * s); torch.manual_seed(args.seed + 100 * s)
                models.append(train_r2(data, device, args.epochs, args.student_ch, args.lr, list(tr),
                                       args.w_density, args.w_count, args.w_nll, args.beta, args.bs,
                                       args.detach_mu, args.sigma_mode))
            of = predict_ensemble(models, te_data, device)
        elif method in ("cqr", "chdqr"):
            m = train_quantile(data, device, args.epochs, args.student_ch, args.lr, tr, taus,
                               args.w_density, args.w_count, args.w_pin, args.bs)
            of = predict_quantile(m, te_data, device, taus, method, args.alpha)
        else:
            raise ValueError(method)
        for k, i in enumerate(te):
            all_p[i] = of["preds"][k]; all_g[i] = of["gts"][k]; all_o[i] = of["organs"][k]
    # nếu debug train-all (test=all) mọi phần tử đã set; nếu leak-free ghép hết -> lọc None phòng hờ
    keep = [i for i in range(N) if all_p[i] is not None]
    return {"preds": [all_p[i] for i in keep], "gts": [all_g[i] for i in keep],
            "organs": [all_o[i] for i in keep]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["mcdropout", "ensemble", "cqr", "chdqr"])
    # data / protocol (mirror distill_student_r2)
    ap.add_argument("--dataset", choices=["nuinsseg", "pannuke"], default="nuinsseg")
    ap.add_argument("--pannuke_root", default=f"{REPO}/data/pannuke")
    ap.add_argument("--pannuke_folds", default="1,2,3")
    ap.add_argument("--test_fold", type=int, default=None)
    ap.add_argument("--kfold", type=int, default=None)
    ap.add_argument("--exclude_tissue", default=None)
    ap.add_argument("--use_gt_density", action="store_true")
    ap.add_argument("--cache", default=None)
    # train chung
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--student_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--w_density", type=float, default=1.0)
    ap.add_argument("--w_count", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.1)
    # mcdropout
    ap.add_argument("--T", type=int, default=30, help="số forward MC-Dropout")
    ap.add_argument("--p_drop", type=float, default=0.2)
    # ensemble (dùng lại recipe R2)
    ap.add_argument("--M", type=int, default=5, help="số thành viên Deep Ensembles")
    ap.add_argument("--w_nll", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--detach_mu", action="store_true", default=True)
    ap.add_argument("--no_detach_mu", dest="detach_mu", action="store_false")
    ap.add_argument("--sigma_mode", choices=["poisson", "raw"], default="poisson")
    # cqr/chdqr
    ap.add_argument("--w_pin", type=float, default=0.1, help="trọng số pinball")
    ap.add_argument("--n_taus", type=int, default=21, help="số mức quantile (chỉ CHDQR)")
    ap.add_argument("--out", default=f"{REPO}/work/uq_out.pkl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} method={args.method} dataset={args.dataset}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    data = load_data(args, device)
    folds = prediction_folds(data, args)
    out = run_method(args.method, data, folds, args, device)
    pickle.dump(out, open(args.out, "wb"))

    mu = np.array([p["mu"] for p in out["preds"]])
    gt = np.array([g[0] for g in out["gts"]])
    extra = ""
    if "sigma" in out["preds"][0]:
        sg = np.array([p["sigma"] for p in out["preds"]])
        extra = f"| sigma mean={sg.mean():.2f} std={sg.std():.2f}"
    else:
        w = np.array([p["q_hi"] - p["q_lo"] for p in out["preds"]])
        extra = f"| pre-conformal width mean={w.mean():.2f}"
    print(f"[OUT] saved {args.out} (N={len(mu)}) | MAE={np.abs(mu-gt).mean():.2f} {extra}")
    print("  -> chấm: " + ("eval_r2_grouped.py" if "sigma" in out["preds"][0] else "eval_cqr_grouped.py"))


if __name__ == "__main__":
    main()

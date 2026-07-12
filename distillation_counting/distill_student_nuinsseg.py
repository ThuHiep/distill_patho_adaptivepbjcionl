"""
distill_student_nuinsseg.py — Distill PathoSAM (teacher) -> U-Net nhẹ (student) trên NuInsSeg.

CHẠY Ở ĐÂU: vast.ai (cần GPU + PathoSAM + ảnh NuInsSeg). KHÔNG chạy được trên máy local
không có PathoSAM. Các điểm chạm PathoSAM/NuInsSeg dưới đây được COPY nguyên từ script bạn
ĐÃ chạy thành công: kaggle/vast/run_pathosam_nuinsseg.py + kaggle/vast/pathosam_lib.py.

Ý TƯỞNG (response-based KD, đúng paradigm CellGenNet 2025: StarDist->U-Net qua pseudo-label):
  - Teacher signal = PathoSAM foreground prob map dày đặc `segmenter._foreground` (THẬT, đã
    verify tồn tại trong pathosam_lib.pathosam_instances). Đây là "soft target".
  - Student = U-Net nhẹ hồi quy lại bản đồ foreground đó.
  - KHI SUY LUẬN: student fg map -> ngưỡng -> connected components -> mỗi component là 1
    instance, s_i = mean(student fg prob) trên component. Đây ĐÚNG cách teacher tính s_i
    (mean foreground prob over each mask) => scores teacher/student cùng thang => phép đo
    coverage-transfer (eval_coverage_transfer.py) sạch, so được trực tiếp.
  - Output: work/student_nuinsseg_preds.pkl schema `{preds:[{scores,probs,K:1}],gts,organs}`
    CÙNG THỨ TỰ ẢNH với data/pathosam_nuinsseg_preds.pkl (dùng chung build_index).

HAI LỰA CHỌN MÔ HÌNH được ghi RÕ (không giấu — bạn có thể thay và ablation):
  (C1) Tín hiệu KD = foreground prob map của teacher (không phải logits per-class, vì
       NuInsSeg K=1). Với dataset đa lớp (MoNuSAC/PanNuke) cần thêm distill type-head.
  (C2) Điểm per-instance của student = mean prob trên connected component. Đây là lựa chọn
       để KHỚP định nghĩa s_i của teacher, không phải trọng số chế tùy tiện.

--lambda_kd điều khiển trộn KD (teacher fg) vs supervised (GT mask NuInsSeg), giống alpha
của Khan 2025. lambda_kd=1.0 => KD thuần (chỉ học teacher). lambda_kd=0.0 => supervised thuần
(baseline "student train from scratch" — MỘT trong các baseline bắt buộc ở FEASIBILITY mục 5.1).

Cách chạy trên vast:
  micromamba run -p /workspace/penv python distill_student_nuinsseg.py \
      --lambda_kd 1.0 --epochs 60 --student_ch 32 --out work/student_nuinsseg_preds.pkl
Rồi đo:
  python eval_coverage_transfer.py --teacher ../data/pathosam_nuinsseg_preds.pkl \
      --student work/student_nuinsseg_preds.pkl --seeds 20 --out coverage_kd.json
"""
from __future__ import annotations
import argparse, glob, os, pickle, sys, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- điểm chạm repo (copy pattern từ run_pathosam_nuinsseg.py — đã verify) ----
REPO = os.environ.get("REPO", "/workspace/sam3_research")
for p in (f"{REPO}/kaggle/vast", f"{REPO}/kaggle/lib", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
NUINSSEG_CANDS = [f"{REPO}/data/nuinsseg", f"{REPO}/data/nuinsseg/NuInsSeg",
                  "/kaggle/input/datasets/ipateam/nuinsseg"]
IMG_SIZE = 256  # ảnh NuInsSeg resize về vuông để train U-Net; teacher fg cũng resize theo


# ===================== NuInsSeg indexer (COPY từ run_pathosam_nuinsseg.py) =====================
def _find_mask_dir(organ_dir):
    for name in os.listdir(organ_dir):
        full = os.path.join(organ_dir, name); low = name.lower()
        if os.path.isdir(full) and "label" in low and "mask" in low and "modif" not in low:
            return full
    for name in os.listdir(organ_dir):
        full = os.path.join(organ_dir, name)
        if os.path.isdir(full) and "label" in name.lower():
            return full
    return None


def _load_mask(path):
    try:
        import tifffile
        if path.lower().endswith((".tif", ".tiff")):
            return np.asarray(tifffile.imread(path))
    except Exception:
        pass
    return np.asarray(Image.open(path))


def build_index(root):
    tissue_dirs = glob.glob(os.path.join(root, "**", "tissue images"), recursive=True)
    samples = []
    for tdir in tissue_dirs:
        organ_dir = os.path.dirname(tdir); organ = os.path.basename(organ_dir)
        mdir = _find_mask_dir(organ_dir)
        if mdir is None:
            continue
        masks = {os.path.splitext(f)[0]: os.path.join(mdir, f) for f in os.listdir(mdir)}
        for f in sorted(os.listdir(tdir)):
            if not f.lower().endswith(IMG_EXT):
                continue
            stem = os.path.splitext(f)[0]
            if stem in masks:
                samples.append({"organ": organ, "image": os.path.join(tdir, f),
                                "mask": masks[stem]})
    return samples


def find_root():
    root = next((c for c in NUINSSEG_CANDS if os.path.isdir(c)), None)
    if root is None:
        td = glob.glob(f"{REPO}/data/**/tissue images", recursive=True)
        root = os.path.dirname(os.path.dirname(td[0])) if td else None
    assert root, ("NuInsSeg not found. kaggle datasets download -d ipateam/nuinsseg "
                  "--unzip -p data/nuinsseg/")
    return root


# ===================== Student U-Net nhẹ =====================
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    def forward(self, x): return self.net(x)


class TinyUNet(nn.Module):
    """U-Net nhẹ (student). student_ch=32 => ~1.9M params; =16 => ~0.5M. In ra để báo cáo."""
    def __init__(self, ch=32):
        super().__init__()
        self.d1 = DoubleConv(3, ch);      self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(ch, ch * 2); self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(ch * 2, ch * 4); self.p3 = nn.MaxPool2d(2)
        self.bott = DoubleConv(ch * 4, ch * 8)
        self.u3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2); self.c3 = DoubleConv(ch * 8, ch * 4)
        self.u2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2); self.c2 = DoubleConv(ch * 4, ch * 2)
        self.u1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2);     self.c1 = DoubleConv(ch * 2, ch)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        x1 = self.d1(x); x2 = self.d2(self.p1(x1)); x3 = self.d3(self.p2(x2))
        xb = self.bott(self.p3(x3))
        y = self.c3(torch.cat([self.u3(xb), x3], 1))
        y = self.c2(torch.cat([self.u2(y), x2], 1))
        y = self.c1(torch.cat([self.u1(y), x1], 1))
        return self.out(y)  # logits (B,1,H,W)


def dice_bce_loss(logits, target):
    """target in [0,1] (soft foreground). BCE + soft-Dice — chuẩn segmentation, không heuristic."""
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy(p, target)
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    return bce + dice.mean()


# ===================== Phase A: teacher foreground targets =====================
@torch.no_grad()
def build_teacher_targets(samples, device, cache):
    """Chạy PathoSAM 1 lần, lưu (ảnh resize, teacher fg map, gt count, organ). Cache lại."""
    from pathosam_lib import load_pathosam  # verified import

    if os.path.exists(cache):
        print(f"[A] load cache {cache}")
        return pickle.load(open(cache, "rb"))
    predictor, segmenter = load_pathosam(device)
    data = []
    t0 = time.time()
    for k, s in enumerate(samples):
        img = np.asarray(Image.open(s["image"]).convert("RGB"))
        H, W = img.shape[:2]
        segmenter.initialize(img)
        _ = segmenter.generate()
        fg = np.asarray(segmenter._foreground, dtype=np.float32)  # teacher soft target THẬT
        # resize ảnh + fg về IMG_SIZE để train
        img_r = np.asarray(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        fg_t = torch.from_numpy(fg)[None, None]
        fg_r = F.interpolate(fg_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                             align_corners=False)[0, 0].numpy()
        fg_r = np.clip(fg_r, 0.0, 1.0)
        m = _load_mask(s["mask"])
        gt = int(len(np.unique(m)) - (1 if (m == 0).any() else 0))
        # GT binary foreground (dùng khi lambda_kd<1) — resize nearest
        gtbin = (m > 0).astype(np.float32)
        gtbin_r = np.asarray(Image.fromarray((gtbin * 255).astype(np.uint8))
                             .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) / 255.0
        data.append({"img": img_r.astype(np.uint8), "fg": fg_r.astype(np.float32),
                     "gtbin": gtbin_r.astype(np.float32), "gt": float(gt),
                     "organ": s["organ"]})
        if (k + 1) % 100 == 0:
            print(f"[A] {k+1}/{len(samples)} {(time.time()-t0)/(k+1):.2f}s/img")
    pickle.dump(data, open(cache, "wb"))
    print(f"[A] saved cache {cache}")
    return data


@torch.no_grad()
def build_pannuke_targets(root, folds, device, cache):
    """KD baseline trên PanNuke (K=1 tổng nhân). Teacher foreground = PathoSAM. organ=tissue."""
    from pathosam_lib import load_pathosam
    from pannuke_loader import PanNukeFold
    if os.path.exists(cache):
        print(f"[A] load cache {cache}")
        return pickle.load(open(cache, "rb"))
    predictor, segmenter = load_pathosam(device)
    data = []
    t0 = time.time(); n_done = 0
    for fold in folds:
        pf = PanNukeFold(root, fold)
        for i in range(len(pf)):
            s = pf[i]
            img = s["image"]
            segmenter.initialize(img); _ = segmenter.generate()
            fg = np.asarray(segmenter._foreground, dtype=np.float32)
            img_r = np.asarray(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
            fg_t = torch.from_numpy(fg)[None, None]
            fg_r = F.interpolate(fg_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                                 align_corners=False)[0, 0].numpy()
            fg_r = np.clip(fg_r, 0.0, 1.0)
            gtbin = (s["masks"].sum(0) > 0).astype(np.float32)   # union 5 kênh
            gtbin_r = np.asarray(Image.fromarray((gtbin * 255).astype(np.uint8))
                                 .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) / 255.0
            data.append({"img": img_r.astype(np.uint8), "fg": fg_r.astype(np.float32),
                         "gtbin": gtbin_r.astype(np.float32), "gt": float(int(s["counts"].sum())),
                         "organ": s["tissue"], "fold": int(fold)})
            n_done += 1
            if n_done % 200 == 0:
                print(f"[A] {n_done} imgs {(time.time()-t0)/n_done:.2f}s/img")
    pickle.dump(data, open(cache, "wb"))
    print(f"[A] saved {cache} ({len(data)} imgs)")
    return data


# ===================== Phase B: train student =====================
def train_student(data, device, epochs, ch, lambda_kd, lr, train_idx):
    model = TinyUNet(ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[B] student TinyUNet ch={ch} params={n_params/1e6:.3f}M lambda_kd={lambda_kd}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def batch(idxs):
        imgs = np.stack([data[i]["img"] for i in idxs]).astype(np.float32) / 255.0
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
        fg = torch.from_numpy(np.stack([data[i]["fg"] for i in idxs]))[:, None].to(device)
        gtb = torch.from_numpy(np.stack([data[i]["gtbin"] for i in idxs]))[:, None].to(device)
        return imgs, fg, gtb

    model.train()
    bs = 8
    for ep in range(epochs):
        np.random.shuffle(train_idx)
        losses = []
        for i in range(0, len(train_idx), bs):
            idxs = train_idx[i:i + bs]
            imgs, fg, gtb = batch(idxs)
            logits = model(imgs)
            # KD (học teacher fg) trộn supervised (học GT mask) — trọng số minh bạch
            loss = lambda_kd * dice_bce_loss(logits, fg) + (1 - lambda_kd) * dice_bce_loss(logits, gtb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[B] epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}")
    return model


# ===================== Phase C: student inference -> preds schema =====================
@torch.no_grad()
def student_predict(model, data, device, thresh, min_area):
    from scipy import ndimage
    model.eval()
    preds, gts, organs = [], [], []
    for d in data:
        img = torch.from_numpy(d["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        prob = torch.sigmoid(model(img))[0, 0].cpu().numpy()  # (H,W) student fg prob
        binm = prob >= thresh
        lab, n = ndimage.label(binm)
        scores = []
        for sid in range(1, n + 1):
            comp = lab == sid
            if comp.sum() < min_area:
                continue
            scores.append(float(prob[comp].mean()))  # s_i = mean prob (KHỚP teacher)
        scores = np.asarray(scores, dtype=np.float32)
        preds.append({"scores": scores, "probs": np.ones((len(scores), 1), np.float32), "K": 1})
        gts.append([d["gt"]])
        organs.append(d["organ"])
    return {"preds": preds, "gts": gts, "organs": organs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_kd", type=float, default=1.0,
                    help="1.0=KD thuần (học teacher); 0.0=supervised thuần (baseline scratch)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--student_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--min_area", type=int, default=5)
    ap.add_argument("--dataset", choices=["nuinsseg", "pannuke"], default="nuinsseg")
    ap.add_argument("--pannuke_root", default=f"{REPO}/data/pannuke")
    ap.add_argument("--pannuke_folds", default="1,2,3")
    ap.add_argument("--test_fold", type=int, default=None,
                    help="PanNuke: HELD-OUT fold để test (leak-free). Train fold còn lại, predict CHỈ test_fold.")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=f"{REPO}/work/student_nuinsseg_preds.pkl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} dataset={args.dataset}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.dataset == "pannuke":
        folds = [int(x) for x in args.pannuke_folds.split(",")]
        fstr = "".join(str(x) for x in sorted(folds))   # cache theo tập fold -> không đụng cache cũ
        cache = args.cache or f"{REPO}/work/teacher_targets_pannuke_f{fstr}.pkl"
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        data = build_pannuke_targets(args.pannuke_root, folds, device, cache)
    else:
        cache = args.cache or f"{REPO}/work/teacher_targets_nuinsseg.pkl"
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        samples = build_index(find_root())
        print(f"indexed {len(samples)} pairs, {len(set(s['organ'] for s in samples))} organs")
        data = build_teacher_targets(samples, device, cache)

    # --- tách train/test theo fold (leak-free, đúng protocol PanNuke; PHẢI khớp R2 để so công bằng) ---
    if args.dataset == "pannuke" and args.test_fold is not None:
        train_idx = [i for i, d in enumerate(data) if d["fold"] != args.test_fold]
        test_data = [d for d in data if d["fold"] == args.test_fold]
        assert train_idx and test_data, f"test_fold={args.test_fold} không tách được"
        n_tr_folds = sorted({d["fold"] for d in data if d["fold"] != args.test_fold})
        print(f"[SPLIT] train folds={n_tr_folds} ({len(train_idx)} imgs) | "
              f"TEST fold={args.test_fold} ({len(test_data)} imgs) — leak-free")
    else:
        train_idx = list(range(len(data)))
        test_data = data
        if args.dataset == "pannuke":
            print("[SPLIT] WARN: không có --test_fold -> train+predict TOÀN BỘ (LEAK, chỉ debug)")
    model = train_student(data, device, args.epochs, args.student_ch,
                          args.lambda_kd, args.lr, train_idx)

    out = student_predict(model, test_data, device, args.thresh, args.min_area)
    pickle.dump(out, open(args.out, "wb"))
    est = np.array([p["scores"].sum() for p in out["preds"]])
    gtv = np.array([g[0] for g in out["gts"]])
    print(f"\n[C] saved {args.out} | {len(out['preds'])} imgs | "
          f"student total-count MAE={np.abs(est-gtv).mean():.2f}")
    print("Tiếp theo: python eval_coverage_transfer.py "
          f"--teacher {REPO}/data/pathosam_nuinsseg_preds.pkl --student {args.out} --seeds 20")


if __name__ == "__main__":
    main()

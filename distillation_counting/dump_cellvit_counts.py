"""
dump_cellvit_counts.py — Chạy CellViT (hoặc LKCell, cùng harness) trên folder ảnh → dump count/ảnh.
Đây là "điểm tích hợp" Phần B: KHÔNG sửa thuật toán, chỉ gọi code OFFICIAL + đếm instance dự đoán.

Công thức (verified từ cell_segmentation/inference/cell_detection.py, CellSegmentationInference):
  inf = CellSegmentationInference(model_path=ckpt, gpu=0)
  x = inf.inference_transforms(image)                 # (3,H,W), norm mean/std của config
  preds = inf.model.forward(x[None].cuda(), retrieve_tokens=True)   # dict nuclei_binary/type_map + tokens
  instance_types, _ = inf.get_cell_predictions_with_tokens(preds, magnification=mag)  # tự softmax + calc_instance_map
  count = len(instance_types[0])                      # #nhân detect ảnh này

⚠️ CHƯA CHẠY THẬT (không GPU trên Mac). 2 điểm cần xác nhận nhanh trên vast lần đầu (in traceback → sửa 1 phút):
  (a) input size: CellViT-SAM-H kỳ vọng patch 1024 (SAM), CellViT-256 kỳ vọng 256. --infer_size để ép.
      Ảnh prep là 256 → thử --infer_size 0 (giữ 256) trước; nếu forward lỗi shape → --infer_size 1024.
  (b) chữ ký forward: nếu `retrieve_tokens=True` không có ở version repo → bỏ (--no_tokens).
LKCell: cùng class/tên hàm (fork CellViT) → chỉ đổi --cellvit_dir sang ./LKCell + --ckpt LKCell-L.

Chạy (trong env cvenv, có GPU):
  python dump_cellvit_counts.py --cellvit_dir /workspace/CellViT \
     --ckpt /workspace/ckpt/cellvit_sam_h/CellViT-SAM-H-x40-AMP.pth \
     --images_dir ../work/nuinsseg_png/images --out_csv cellvit_preds.csv --gpu 0 --mag 40
"""
from __future__ import annotations
import argparse, csv, glob, os, sys
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cellvit_dir", required=True, help="repo CellViT (hoặc LKCell) đã clone")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pth (CellViT-SAM-H / LKCell-L)")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mag", type=int, default=40)
    ap.add_argument("--infer_size", type=int, default=0, help="0=giữ nguyên; >0 resize vuông (thử 1024 nếu SAM-H lỗi shape)")
    ap.add_argument("--no_tokens", action="store_true", help="bỏ retrieve_tokens nếu version repo không có")
    ap.add_argument("--lkcell", action="store_true",
                    help="LKCell: __get_model gốc build sai (ViT signature) cho model unireplknet -> "
                         "patch build đúng bằng chính class CellViT (=UniRepLKNet) + config của LKCell.")
    ap.add_argument("--nulite", action="store_true",
                    help="NuLite (CosmoIknosLab): import CellSegmentationInference từ "
                         "nuclei_detection.inference.nuclei_detection (repo fork CellViT). __load_model TỰ build "
                         "NuLite(...) + reparameterize_encoder -> KHÔNG cần monkeypatch. Dùng nhánh retrieve_tokens "
                         "(KHÔNG --no_tokens). --cellvit_dir trỏ vào repo NuLite đã clone.")
    ap.add_argument("--limit", type=int, default=0, help=">0: chỉ N ảnh đầu (validate nhanh mode resize/tile)")
    ap.add_argument("--measure_cost", action="store_true",
                    help="CHỈ đo params + GMACs của model (thop) rồi thoát — đo NuLite/CellViT/LKCell bằng "
                         "CÙNG script (nhất quán với count_student_cost.py). Dùng --infer_size để đặt độ phân giải.")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.cellvit_dir))
    import torch
    # Checkpoint OFFICIAL (LKCell/CellViT) chứa numpy globals -> torch>=2.6 mặc định weights_only=True fail.
    # Ta TIN checkpoint (repo chính thức) -> ép weights_only=False. Chỉ đổi CÁCH NẠP, không đổi thuật toán.
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
    if args.nulite:
        # NuLite: fork CellViT nhưng module inference ở nuclei_detection.inference.nuclei_detection.
        # __load_model TỰ build NuLite(...) + reparameterize_encoder -> KHÔNG cần monkeypatch __get_model.
        import nuclei_detection.inference.nuclei_detection as _cd
        from nuclei_detection.inference.nuclei_detection import CellSegmentationInference
    else:
        import cell_segmentation.inference.cell_detection as _cd
        from cell_segmentation.inference.cell_detection import CellSegmentationInference

    if args.lkcell:
        # LKCell: build model ĐÚNG bằng class CellViT(=UniRepLKNet) + config, y hệt
        # inference_cellvit_experiment_pannuke.py của họ (KHÔNG tái hiện, dùng code họ).
        from models.segmentation.cell_segmentation.cellvit import CellViT as _LKCellViT

        def _get_model_lk(self, model_type):
            rc = self.run_conf
            enc = rc["model"].get("pretrained_encoder")
            if enc and not os.path.exists(str(enc)):
                enc = None   # path train-time không có trên máy này; state_dict nạp full weights sau
            return _LKCellViT(
                model256_path=enc,
                num_nuclei_classes=rc["data"]["num_nuclei_classes"],
                num_tissue_classes=rc["data"]["num_tissue_classes"],
                in_channels=rc["model"].get("input_chanels", 3),
            )
        _cd.CellSegmentationInference._CellSegmentationInference__get_model = _get_model_lk

    inf = CellSegmentationInference(model_path=args.ckpt, gpu=args.gpu)  # tự load model + transforms
    dev = next(inf.model.parameters()).device

    if args.measure_cost:
        nparam = sum(p.numel() for p in inf.model.parameters())
        sz = args.infer_size or 256
        print(f"[COST] ckpt={os.path.basename(args.ckpt)} | params={nparam/1e6:.3f}M", end="")
        try:
            from thop import profile
            dummy = torch.randn(1, 3, sz, sz).to(dev)
            macs, _ = profile(inf.model, inputs=(dummy,), verbose=False)
            print(f" | GMACs@{sz}={macs/1e9:.2f}")
        except Exception as e:
            print(f" | GMACs FAIL ({type(e).__name__}) -> chỉ báo params (đo GMACs tay nếu cần)")
        return

    paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
    if args.limit:
        paths = paths[:args.limit]
    print(f"{len(paths)} ảnh | ckpt={os.path.basename(args.ckpt)} | infer_size={args.infer_size} mag={args.mag}")

    rows = []
    inf.model.eval()
    for k, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        if args.infer_size:
            img = img.resize((args.infer_size, args.infer_size), Image.BILINEAR)
        x = inf.inference_transforms(np.array(img))            # (3,H,W)
        with torch.no_grad():
            if args.no_tokens:
                # forward không trả tokens -> gọi THẲNG calculate_instance_map như experiment
                # script của họ (softmax 2 map rồi tính instance). Dùng cho LKCell.
                fwd = inf.model.forward(x[None].to(dev))
                fwd["nuclei_binary_map"] = torch.softmax(fwd["nuclei_binary_map"], dim=1)
                fwd["nuclei_type_map"] = torch.softmax(fwd["nuclei_type_map"], dim=1)
                _, inst = inf.model.calculate_instance_map(fwd, magnification=args.mag)
            else:
                fwd = inf.model.forward(x[None].to(dev), retrieve_tokens=True)
                inst, _ = inf.get_cell_predictions_with_tokens(fwd, magnification=args.mag)
        count = len(inst[0]) if isinstance(inst, (list, tuple)) else len(inst)
        rows.append([os.path.splitext(os.path.basename(p))[0], count])
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(paths)}  (vd {rows[-1][0]} -> {rows[-1][1]} nhân)")

    with open(args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows([["image", "pred_count"], *rows])
    tot = sum(r[1] for r in rows)
    print(f"XONG -> {args.out_csv}  ({len(rows)} ảnh, tổng {tot} nhân, TB {tot/max(len(rows),1):.1f}/ảnh)")
    print("Chấm: python eval_heavy_count.py --gt <prep>/gt_counts.csv --preds", args.out_csv,
          "--label CellViT-SAM-H --student_pkl ../work/student_r2_nuinsseg_cv5_poisson_feat.pkl")


if __name__ == "__main__":
    main()

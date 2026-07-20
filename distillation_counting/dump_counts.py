#!/usr/bin/env python3
"""dump_counts.py — bản SẠCH của dump_cellvit_counts.py.

Chạy CellViT / LKCell / NuLite (code OFFICIAL, KHÔNG sửa thuật toán) trên folder ảnh → count/ảnh.
Giữ NGUYÊN phần load-model + inference đã debug (torch.load, monkeypatch LKCell, 2 nhánh forward).
Thêm độ bền để không bao giờ mất công / dừng thiếu như csv 5-ảnh cũ:
  • LƯU TĂNG DẦN (flush mỗi ảnh) → crash vẫn giữ phần đã chạy.
  • RESUME: out_csv có sẵn → bỏ qua ảnh đã xong, chạy tiếp.
  • try/except MỖI ảnh → 1 ảnh lỗi ghi vào .errors, KHÔNG giết cả 665.
  • Cuối in rõ done/total + liệt kê ảnh lỗi; CẢNH BÁO to nếu thiếu.
  • KHÔNG có --limit mặc định; muốn thử nhanh phải --smoke N (in cảnh báo).

Ví dụ (Kaggle, GPU):
  python dump_counts.py --lkcell --no_tokens \
    --cellvit_dir /kaggle/working/LKCell \
    --ckpt /kaggle/input/datasets/hipinhththu/checkpoint-nlt-lkl/checkpoint_NL_LK/model_best.pth \
    --images_dir ../work/nuinsseg_png/images --out_csv lkcell_preds_full.csv
"""
from __future__ import annotations
import argparse, csv, glob, os, sys, traceback
import numpy as np
from PIL import Image


def build_inference(args):
    """Trả object CellSegmentationInference đã load model. Logic GIỮ NGUYÊN bản cũ."""
    sys.path.insert(0, os.path.abspath(args.cellvit_dir))
    import torch
    # Checkpoint OFFICIAL chứa numpy globals -> torch>=2.6 weights_only=True fail. Tin checkpoint chính thức.
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

    if args.nulite:
        import nuclei_detection.inference.nuclei_detection as _cd  # noqa: F841
        from nuclei_detection.inference.nuclei_detection import CellSegmentationInference
    else:
        import cell_segmentation.inference.cell_detection as _cd
        from cell_segmentation.inference.cell_detection import CellSegmentationInference

    if args.lkcell:
        # LKCell: build ĐÚNG bằng class CellViT(=UniRepLKNet) + config (y hệt script họ, KHÔNG tái hiện).
        from models.segmentation.cell_segmentation.cellvit import CellViT as _LKCellViT

        def _get_model_lk(self, model_type):
            rc = self.run_conf
            enc = rc["model"].get("pretrained_encoder")
            if enc and not os.path.exists(str(enc)):
                enc = None
            return _LKCellViT(
                model256_path=enc,
                num_nuclei_classes=rc["data"]["num_nuclei_classes"],
                num_tissue_classes=rc["data"]["num_tissue_classes"],
                in_channels=rc["model"].get("input_chanels", 3),
            )
        _cd.CellSegmentationInference._CellSegmentationInference__get_model = _get_model_lk

    inf = CellSegmentationInference(model_path=args.ckpt, gpu=args.gpu)
    inf.model.eval()
    return inf


def count_one(inf, img_np, args):
    """1 ảnh (np.uint8 HxWx3) -> số nhân. Logic forward GIỮ NGUYÊN bản cũ."""
    import torch
    dev = next(inf.model.parameters()).device
    x = inf.inference_transforms(img_np)                 # (3,H,W)
    with torch.no_grad():
        if args.no_tokens:
            fwd = inf.model.forward(x[None].to(dev))
            fwd["nuclei_binary_map"] = torch.softmax(fwd["nuclei_binary_map"], dim=1)
            fwd["nuclei_type_map"] = torch.softmax(fwd["nuclei_type_map"], dim=1)
            _, inst = inf.model.calculate_instance_map(fwd, magnification=args.mag)
        else:
            fwd = inf.model.forward(x[None].to(dev), retrieve_tokens=True)
            inst, _ = inf.get_cell_predictions_with_tokens(fwd, magnification=args.mag)
    return len(inst[0]) if isinstance(inst, (list, tuple)) else len(inst)


def load_done(out_csv):
    """image đã có trong out_csv (để resume)."""
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv) as f:
        return {r["image"] for r in csv.DictReader(f) if r.get("pred_count", "") != ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cellvit_dir", required=True, help="repo CellViT/LKCell/NuLite đã clone")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mag", type=int, default=40)
    ap.add_argument("--infer_size", type=int, default=0, help="0=giữ nguyên; >0 resize vuông")
    ap.add_argument("--no_tokens", action="store_true", help="LKCell: forward không retrieve_tokens")
    ap.add_argument("--lkcell", action="store_true")
    ap.add_argument("--nulite", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help=">0: CHỈ N ảnh đầu để thử nhanh (KHÔNG dùng cho số cuối!)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
    assert paths, f"không thấy .png trong {args.images_dir}"
    if args.smoke:
        print(f"⚠️  SMOKE MODE: chỉ {args.smoke}/{len(paths)} ảnh — KẾT QUẢ KHÔNG DÙNG cho bảng!")
        paths = paths[:args.smoke]

    done = load_done(args.out_csv)
    todo = [p for p in paths if os.path.splitext(os.path.basename(p))[0] not in done]
    print(f"{len(paths)} ảnh | đã xong {len(done)} | còn {len(todo)} | ckpt={os.path.basename(args.ckpt)}")
    if not todo:
        print("Tất cả đã xong.")
        return

    inf = build_inference(args)

    err_log = args.out_csv + ".errors"
    new_file = not os.path.exists(args.out_csv)
    fout = open(args.out_csv, "a", newline="")
    w = csv.writer(fout)
    if new_file:
        w.writerow(["image", "pred_count"]); fout.flush()

    n_ok, n_err, errors = 0, 0, []
    for k, p in enumerate(todo):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            img = Image.open(p).convert("RGB")
            if args.infer_size:
                img = img.resize((args.infer_size, args.infer_size), Image.BILINEAR)
            count = count_one(inf, np.array(img), args)
            w.writerow([name, count]); fout.flush()          # LƯU NGAY từng ảnh
            n_ok += 1
        except Exception as e:                                # 1 ảnh lỗi KHÔNG giết cả run
            n_err += 1; errors.append(name)
            with open(err_log, "a") as ef:
                ef.write(f"{name}\t{type(e).__name__}: {e}\n{traceback.format_exc()}\n")
            print(f"  ⚠️ LỖI {name}: {type(e).__name__}: {e} (ghi {err_log}, chạy tiếp)")
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(todo)}  (ok={n_ok} err={n_err})")
    fout.close()

    total = len(load_done(args.out_csv))
    print(f"\nXONG -> {args.out_csv} | ghi thêm ok={n_ok} err={n_err} | tổng trong csv={total}/{len(paths)}")
    if total < len(paths):
        print(f"⚠️  THIẾU {len(paths)-total} ảnh — xem {err_log}; chạy LẠI lệnh này để resume phần còn lại.")
    if errors:
        print(f"   ảnh lỗi ({len(errors)}): {errors[:10]}{' ...' if len(errors)>10 else ''}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""measure_teacher_vram.py — đo peak VRAM của BACKBONE teacher (SAM ViT-H = kiến trúc PathoSAM).

VRAM do KIẾN TRÚC quyết định (không phải weights fine-tune) → dựng SAM ViT-H rồi đo forward
image_encoder ở 1024² (input native SAM). Số này đại diện đúng cho PathoSAM. Ghi rõ nguồn.
KHÔNG đo latency teacher (SAM prompt-based, khác paradigm PACT → so ms không fair; H-Optimus cũng chỉ đo VRAM).

Cài: pip install segment-anything
Chạy: python measure_teacher_vram.py --vit vit_h            # PathoSAM ~640M = ViT-H
      python measure_teacher_vram.py --vit vit_h --ckpt <patho_sam.pth>   # nếu muốn nạp weights thật
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit", default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
                    help="PathoSAM ~640M -> vit_h (636M). Đổi nếu PathoSAM dùng size khác.")
    ap.add_argument("--ckpt", default=None, help="(tùy chọn) checkpoint PathoSAM; None = chỉ dựng kiến trúc (VRAM y hệt)")
    ap.add_argument("--size", type=int, default=1024, help="input SAM native = 1024")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "cần GPU (Kaggle bật Accelerator=GPU)"
    from segment_anything import sam_model_registry
    dev = "cuda"; gpu = torch.cuda.get_device_name(0)

    sam = sam_model_registry[args.vit](checkpoint=args.ckpt).to(dev).eval()
    nparam = sum(p.numel() for p in sam.parameters())
    enc = sam.image_encoder

    x = torch.randn(1, 3, args.size, args.size, device=dev)
    with torch.no_grad():
        for _ in range(5):                      # warm-up
            enc(x)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    import time
    t = time.time()
    with torch.no_grad():
        for _ in range(args.iters):
            enc(x)
    torch.cuda.synchronize()
    dt = (time.time() - t) / args.iters
    peak = torch.cuda.max_memory_allocated() / 1e6

    print(f"GPU={gpu} | SAM {args.vit} (kiến trúc PathoSAM) | params={nparam/1e6:.1f}M | input={args.size}²")
    print(f"[image_encoder forward bs=1] peak VRAM = {peak:.1f} MB ({peak/1024:.2f} GB) | {dt*1000:.1f} ms")
    print("GHI CHÚ: chỉ lấy PEAK VRAM để so với PACT (kiểu H-Optimus). ms KHÔNG dùng so (SAM prompt-based, khác paradigm).")
    print(f"So sánh: PACT ch32 peak VRAM 70.7 MB → teacher/PACT ≈ {peak/70.7:.0f}× bộ nhớ.")


if __name__ == "__main__":
    main()

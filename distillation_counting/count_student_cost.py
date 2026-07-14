"""
count_student_cost.py — In params + GMACs của student DensitySigmaUNet (cho BẢNG A efficiency, Bước 2).

- Params: đếm trực tiếp từ model THẬT (import DensitySigmaUNet, không copy → không lệch số).
- GMACs@256: dùng `thop` (pip install thop). Nhiều paper (LKCell/NuLite) ghi cột "GFLOPs" nhưng thực ra
  là MACs (thop) → BÁO student bằng GMACs để ĐỒNG ĐƠN VỊ, đừng nhân 2.
- bPQ/mPQ: KHÔNG áp dụng cho student (student xuất density-count, không phải segmentation) → N/A;
  các số bPQ heavy net là CITE từ paper gốc (xem md mục Bước 2 bảng A).

Chạy:
  pip install thop
  python count_student_cost.py --ch 32
"""
from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=32, help="student_ch (32 = cấu hình chốt paper)")
    ap.add_argument("--size", type=int, default=256, help="cạnh ảnh vuông cho MACs")
    args = ap.parse_args()

    import torch
    from distill_student_r2 import DensitySigmaUNet  # model THẬT (single source of truth)

    m = DensitySigmaUNet(ch=args.ch).eval()
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"DensitySigmaUNet(ch={args.ch}): params total = {tot:,} ({tot/1e6:.3f} M), trainable = {tr:,}")

    try:
        from thop import profile
        x = torch.randn(1, 3, args.size, args.size)
        macs, _ = profile(m, inputs=(x,), verbose=False)
        print(f"MACs @{args.size} = {macs/1e9:.3f} G   (= {macs/1e9:.2f} GMACs; báo cột GFLOPs của paper = MACs)")
        print(f"(nếu paper thật sự dùng FLOPs=2xMACs thì = {2*macs/1e9:.3f} G — kiểm quy ước trước khi in)")
    except ModuleNotFoundError:
        print("thop chưa cài -> `pip install thop` để lấy GMACs. (params ở trên vẫn đúng)")

    print("\nBẢNG A (cite heavy net từ paper, student đo ở đây):")
    print(f"  Student R2 (ours) | params {tot/1e6:.3f} M | ~GMACs@256 (dòng trên) | bPQ = N/A (density, +UQ) ")
    print("  heavy net: CellViT-SAM-H 699.74M/214.33 | LKCell-L 163.84M/47.86 | NuLite-T 17.12M/26.16 (cite)")


if __name__ == "__main__":
    main()

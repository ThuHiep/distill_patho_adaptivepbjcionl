# Astro Counting — mở rộng scope (đếm sao/thiên hà dưới cross-survey shift)

Domain thứ 2 cho luận văn (sau histopathology cell counting = Bài 1). Mục tiêu: chứng minh
phương pháp **Adaptive PB-JCI Online** (conformal counting có bảo chứng coverage dưới shift)
chuyển giao sang **lĩnh vực khác hẳn** — từ khoa học sự sống sang vật lý/thiên văn.

## Vì sao thiên văn (khớp cơ chế thắng Bài 1)
Đóng góp headline của Bài 1 = **coverage recovery / conditional validity dưới shift** (từ
cửa sổ thích nghi online), KHÔNG phải PB-σ (chỉ lo độ rộng — đã chứng minh bằng ablation).
Thiên văn thỏa profile:
1. **soft-count** = Σ p_detect·p_class mỗi nguồn.
2. shift **cross-survey** (khác kính/độ sâu/seeing) → static calibration under-cover mạnh.
3. **K>1**: sao / thiên hà (/ QSO) → joint coverage qua max-statistic.
4. completeness cut kiểm soát bias.

`cal` = survey SÂU (residual nhỏ) → `test` = survey NÔNG (residual lớn) = HARDENING đúng chiều
(song ánh PanNuke→NuInsSeg của Bài 1).

## Cấu trúc thư mục
```
astro_counting_paper/
├── kaggle/
│   ├── astro_pbjci_diagnostic.py   # notebook chính: diagnostic-first + crosstable
│   └── lib/                        # (chỗ để tách method dùng chung nếu cần)
├── data/                           # cache SDSS (nếu tải)
├── figures/                        # hình cho paper
├── paper/                          # draft chương mở rộng scope
└── results/                        # *.json output
```

## Dataset + Backbone (đầy đủ)
- **Dataset:** SDSS DR17 `PhotoObj` — nguồn thật (vị trí, `type` sao/thiên hà, `psfFlux`), vùng
  RA 150–152, Dec 0–2 (~4 deg², lọc `mode=1, clean=1`). GT = đếm nguồn thật/lớp trên completeness cut.
- **Backbone:** **SDSS Photometric Pipeline (Photo)**. THẬT: `probPSF_r` (phân loại sao) + SNR (`psfFlux·√ivar`).
  **SEMI-SYNTHETIC (mô tả trung thực, KHÔNG "fully real"):** `p_star=probPSF` thật; `p_detect=completeness
  5σ = logistic((SNR−5)/1.5)` trên SNR thật (mô hình vật lý chuẩn — catalog chỉ chứa nguồn đã detect).
  σ dùng completeness(SNR) **liên tục → không thoái hóa** (Mức 2 σ-chỉ-từ-probPSF bị σ≈0 đầu sáng, đã loại).
- **Shift cross-survey (INDUCED):** `SNR_shallow = SNR_deep/DEPTH_FACTOR` + phân loại mờ đi. **Shift có
  kiểm soát trên dữ liệu thật** (chuẩn distribution-shift; KHÔNG phải survey thật thứ 2 — ghi rõ "induced").
  `DEPTH_FACTOR` = núm điều khiển → DEPTH sweep. Limitation: fully-observational Stripe82 = future work.
- **Method:** giữ NGUYÊN công thức PB-σ + max-statistic K=2 y hệt Bài 1 (không đổi method).
- **Baselines (9):** Static + ACI + NexCP + FACI + SAOCP + COP + Rolling-Origin + PB-JCI-Fixed + Adaptive.
- **Metrics:** joint coverage, per-class coverage (sao/thiên hà), conditional validity (min-local cov + max miss-run), width & Winkler (mean + median).

## Chạy
**Toy sim (offline, chỉ test harness, KHÔNG backbone):** `python kaggle/astro_pbjci_diagnostic.py`
(mặc định `USE_SDSS=False`).

**SDSS thật (Kaggle — đầy đủ backbone + dataset):**
1. Settings → Internet = On.
2. Cell 1: `!pip install -q astroquery`
3. Sửa `USE_SDSS = True` ở đầu file.
4. Run → query SDSS thật, lấy `probPSF`/`psfFlux`/`ivar` từ pipeline, forward-model độ sâu, chạy
   diagnostic sweep + crosstable. Nếu shift quá nhẹ → tăng `DEPTH_FACTOR`.

## Gate đúng (bài học từ trees)
KHÔNG gate theo σ-gain (đó chỉ là câu chuyện width). **Gate = SHIFT đủ mạnh** (static under-cover)
→ chạy crosstable xem Adaptive có hồi phục ~nominal **độc nhất** trong khi baseline under-cover.

## Kết quả toy-sim (2026-07-05, đã validate harness)
mag_cut=20.5, static sập 90→21.5%:

| Method | Joint cov % | Winkler |
|---|---|---|
| Static split-CP | 21.5 | 72.98 |
| ACI | 87.5 | 27.75 |
| Rolling-Origin | 80.6 | 29.54 |
| PB-JCI Fixed | 75.0 | 31.59 |
| **Adaptive PB-JCI** | **89.7** | 27.79 |

→ **Adaptive hồi phục ~nominal độc nhất**, mọi baseline under-cover (PB-Fixed dùng PB-σ mà tệ
nhất → xác nhận PB-σ=width). Pattern Bài 1, mạnh hơn trees. **Cần xác nhận trên SDSS thật.**

## Trung thực (chưa được spin)
- Toy sim ≠ astro thật; cần SDSS xác nhận.
- σ-gain≈0 → **không claim interval hẹp hơn**, chỉ claim coverage recovery + joint validity.
- Nếu SDSS shift quá nhẹ → chỉnh `SHAL_MLIM`/`SHAL_SKY` (tham số thí nghiệm hợp lệ, ghi rõ trong bài).

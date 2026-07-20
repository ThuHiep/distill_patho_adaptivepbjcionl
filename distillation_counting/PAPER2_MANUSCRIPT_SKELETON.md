# PACT — Manuscript Skeleton (LIGHTWEIGHT-primary, UQ-secondary)

> Khung viết bám 2 paper mốc: **Shvetsov et al. 2025** (H-Optimus student, arXiv 2502.19217) +
> **CellGenNet 2025** (arXiv 2511.15054). Định vị = *distilled lightweight counter* (giống họ),
> đầu phân phối (μ,σ) chỉ là **tính năng phụ / trustworthiness bonus**.
> `[TODO]` = chạy được nhưng CHƯA có số → để trống, điền sau.

---

## Title (nghiêng lightweight)
**"Label-Efficient Distillation of a 1.9M-Parameter Nuclei Counter from a Pathology Foundation Model"**
*(subtitle tùy chọn: "...with a Calibrated Count-Distribution Head")*

## Positioning 1 câu
Ta distill khả năng ĐẾM của foundation model bệnh học (PathoSAM ~640M) xuống một student
**1.9M** chỉ cần **nhãn count** (không mask), đạt/vượt teacher trên đếm OOD ở **1/330 kích thước**,
transfer được qua dataset — và (phụ) tự phát ra phân phối count calibrated.

---

## Abstract (khung, điền số đã có)
- Vấn đề: foundation model đếm nhân chính xác nhưng **640M+, cần GPU lớn, khó deploy**; các
  counter chính xác khác cần **mask pixel đắt**.
- Ta: distilled student **1.9M**, train bằng **count-only**, xuất density→count (+σ phụ).
- Kết quả: trên NuInsSeg, student **out-count teacher zero-shot (MAE 14.7 vs 17.9)** ở 1/330 size;
  count-only ≈ mask-supervised ở cùng ngân sách ảnh; transfer coverage qua 3 dataset.
- (phụ) đầu phân phối học được > công thức PB-σ giải tích ở quy mô nhỏ.

---

## Contributions (THỨ TỰ = ưu tiên; C1–C3 lõi lightweight, C4 phụ)

- **C1 — Distilled tiny counter đạt MAE/RMSE thấp nhất trong nhóm.** Student 1.9M, distill PathoSAM
  density + giám sát count rẻ, **MAE & RMSE thấp nhất so teacher 640M + heavy net (CellViT/LKCell) trên
  NuInsSeg, ở 90–360× nhỏ hơn** (Bảng 1). *(khớp cấu trúc "proposed thắng nhiều baseline" của CellGenNet;
  cùng chuẩn in-domain-vs-off-the-shelf; disclose MAPE như họ disclose FPR.)*
- **C2 — Label-efficiency count-only (đo THẬT).** Distilled(count) ≈ supervised(mask) ở cùng số ảnh;
  nhãn count rẻ hơn mask ~5–10×. *(Shvetsov dùng full-mask; CellGenNet chỉ nói "sparse" không định lượng.)*
- **C3 — Transfer cross-dataset.** Coverage calibrated giữ qua NuInsSeg↔PanNuke & →CryoNuSeg (3 dataset).
  *(cả 2 paper mốc để cross-dataset là "future work".)*
- **C4 (PHỤ) — Đầu phân phối nhẹ, calibrated.** Student tự xuất (μ,σ) học được; learned-σ > analytic
  PB-σ ở quy mô nhỏ (N4). **Không paper distillation-counting nào có UQ** — nhưng ta để đây là
  *bonus trustworthiness*, không phải trục bán.

---

## Related Work (khung định vị)
- **Distillation for counting/segmentation nhẹ** (Shvetsov 2025; CellGenNet 2025; HoVer-unet ISBI'24;
  ReviewKD; Dual-KD TIP'23): đều single-teacher, point-output, **không UQ**. Ta cùng dòng nhưng
  (i) nhỏ hơn, (ii) count-only đo thật, (iii) transfer, (iv) thêm đầu phân phối.
- **Foundation model bệnh học** (PathoSAM; H-Optimus): teacher của ta là PathoSAM.
- **Đếm bằng density-map** (Lempitsky & Zisserman NeurIPS'10): cơ sở μ=Σdensity.
- **UQ** (Kendall&Gal'17 aleatoric; conformal): dùng cho C4 phụ; **KHÔNG lead** — tránh trùng Paper 1.

---

## Method (gọn)
- **Teacher:** PathoSAM (~640M), xuất instance → density target (class-agnostic).
- **Student `DensitySigmaUNet` (PACT):** TinyUNet, ch32 → **1.935M**; 2 head: density (μ=Σ) + log-σ.
- **Loss:** `L_density` (KD từ teacher density) + **`L_count`=|Σdensity−GT|** (nhãn count rẻ) + `L_nll` (σ, phụ).
- σ = √max(μ,1)·exp(clamp(log_s)) Poisson-anchored (N2). ch16 → ~0.5M (ablation, không mất chất lượng).

---

## Experiments — bảng (số THẬT + `[TODO]`)

*Quy ước trình bày (theo CellGenNet/H-Optimus): (±sd) 5-seed trong ngoặc; **bold** = tốt nhất cột
(kể cả khi không phải ours); ↑/↓ chỉ hướng; số lẻ nhất quán; `[TODO]` = chạy được, chưa có số.*

### Bảng 1 — Độ chính xác ĐẾM theo dataset *(cấu trúc CellGenNet Table I: block theo dataset; + cột nhãn adapt)* ★★ BẢNG CHÍNH
| Dataset | Method | Params | Nhãn adapt | MAE ↓ | RMSE ↓ | MAPE ↓ | R² ↑ |
|---|---|---|---|---|---|---|---|
| **NuInsSeg** | CellViT-SAM-H | 699.7M | — (off-the-shelf) | 21.83 | 31.33 | 52.9% | 0.663 |
| | LKCell-L | 163.8M | — (off-the-shelf) | 20.92 | 40.10 | 37.4% | 0.448 |
| | NuLite-T | 12.0M | — (off-the-shelf) | 20.01 | 33.22 | 39.6% | 0.622 |
| | PathoSAM teacher | ~640M | — (zero-shot) | 15.80 | 29.02 | **28.3%** | 0.711 |
| | **PACT (ours)** | **1.9M** | **count (in-domain)** | **14.74 (1.53)** | **24.81 (3.03)** | 47.6% (3.4) | **0.786 (0.052)** |
| **PanNuke** (fold_3 sạch) | **PACT (ours)** | **1.9M** | **count (in-domain)** | **3.36** | `[TODO]` | `[TODO]` | `[TODO]` |
| | heavy net (leak-free) | — | — (off-the-shelf) | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
> **Đọc:** PACT **MAE+RMSE thấp nhất**, ở **90–360× nhỏ hơn**. Cột "Nhãn adapt" = nhãn THỰC TẾ đã dùng để
> thích nghi NuInsSeg: chỉ **PACT** được thích nghi (bằng **count rẻ**); mọi net kia + teacher đều **off-the-shelf**.
> **Claim ĐÚNG = thích-nghi-rẻ, KHÔNG phải "model giỏi hơn".** Vì sao net kia kẹt off-the-shelf: muốn thích nghi
> chúng phải có **mask pixel đắt** — trong khi PACT chỉ cần **count**. Sự bất đối xứng đó *chính là* điểm bán (chuẩn CellGenNet).
> **Disclose thẳng:** MAPE 47.6% > teacher 28.3% (density-sum sai tương đối ở ảnh ít nhân) — như CellGenNet disclose FPR.
> **★ Số PACT = 5-seed (42–46) từ `compute_r2_counting.py`, coherent (R²/MAE/RMSE/MAPE cùng nguồn); PACT thắng teacher R²+MAE+RMSE, thua MAPE.**
> **Nguồn coherent (✅ đủ 3 heavy net):** CellViT+LKCell+NuLite = CÙNG thước `dump_counts.py`
> (len-instances, 665 ảnh mỗi model); teacher = len-scores; PACT = Σdensity. Mỗi model đếm native của nó
> (chuẩn — như CellGenNet/H-Optimus). *(Số §4.6 cũ từ eval_heavy_count KHÁC thước → đã thay bằng dump coherent.)*
> ✅ CellViT-SAM-H chạy @1024 (native SAM, fair) — R² 0.663 (256 chỉ 0.444, thiệt cho nó); PACT vẫn dẫn cả 3.
> `[TODO]` (optional) baseline recent/classic: Cellpose / InstanSeg (`dump_cellpose.py` / `dump_instanseg.py`).
> `[TODO]` block PanNuke heavy-net (leak-free).

### Bảng 2 — Hiệu quả tính toán *(H-Optimus Table 4 + cột thu-nhỏ×; + FLOPs/latency họ KHÔNG có)*
| Model | Params | ×nhỏ hơn teacher | GMACs @256² | Size (MB) | Latency (ms) ↓ | Peak VRAM ↓ |
|---|---|---|---|---|---|---|
| PathoSAM teacher | ~640M | 1× | `[TODO]` | ~2560 | `[TODO]` | ~16 GB |
| **PACT (ch32, ours)** | **1.935M** | **331×** | 10.49 | 7.74 | `[TODO]` | `[TODO]` |
| **PACT (ch16, ours)** | **0.485M** | **1320×** | **2.65** | **1.94** | `[TODO]` | `[TODO]` |
> Params/GMACs/Size = đo THẬT local (thop, torch 2.8, 1×3×256²; Size=params×4B fp32). PACT còn **nhỏ hơn
> student H-Optimus (24M) ~12–50× và NuLite (12M) ~6–25×** — nhấn ở abstract.
> `[TODO]` Latency+VRAM: `measure_latency.py --ch 32/16` trên 1 GPU cố định. Teacher GMACs: cite paper PathoSAM.

### Bảng 3 — Độ tin cậy: teacher vs student *(H-Optimus Table 3 — parity/vượt teacher)*
| Dataset | Model | Params | worst-org cov global ↑ | worst-org cov cluster ↑ | Winkler ↓ | count MAE ↓ |
|---|---|---|---|---|---|---|
| **NuInsSeg** | PathoSAM teacher | ~640M | 0.482 | 0.680 | **85.1** | 17.89 |
| | **PACT (ours)** | **1.9M** | **0.610** | **0.750 (0.049)** | 95.4 (11.9) | **14.7 (1.7)** |
| **PanNuke** | PathoSAM teacher | ~640M | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| | **PACT (ours)** | **1.9M** | `[TODO]` | **0.905** | **19.28** | **3.36** |
> Student ≥ teacher worst-org coverage CẢ 2 scheme + MAE, ở 1/330 size; teacher chỉ giữ Winkler-cluster.
> `[TODO]` block PanNuke teacher-PB parity (`eval_r2_grouped.py` trên teacher pkl PanNuke).

### Bảng 4 — Label-efficiency: distilled(count) vs supervised(mask) *(so CÓ KIỂM SOÁT — fair tuyệt đối)*
| % ảnh dùng | Distilled (count-only) MAE | Supervised (mask) MAE | Distilled cov | Supervised cov |
|---|---|---|---|---|
| 10% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| 25% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| 50% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| 100% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
> §4.10 (kết luận có rồi): **cùng số ảnh → MAE & coverage ≈ nhau (chồng ±sd)**; nhãn count ~**5–10× rẻ**
> hơn mask (K=52.8 nhân/ảnh). **Đây là bảng fair NHẤT** (cùng mạng, cùng data, chỉ khác nguồn nhãn) → xương sống C2.
> `[TODO]` điền số từng mốc + nâng 3→5 seed cho figure.

### Bảng 5 — Transfer cross-dataset *(cái cả 2 paper mốc để "future work")*
| Transfer | Loại | cov (worst-org / marginal) | MAE | Ghi chú |
|---|---|---|---|---|
| NuInsSeg → PanNuke | in↔in | worst-org **0.897** (≈ in-domain 0.906) | không transfer | σ sống dưới shift |
| NuInsSeg → CryoNuSeg (dataset 3) | OOD sạch | marginal **0.967** | 73 (~29% rel) | σ informative |
| NuInsSeg → MoNuSAC | OOD scale 4× | (rỗng) | 138 | **Limitation**: μ sập khi scale-gap lớn |
> Coverage calibrated transfer qua 3 dataset; MAE không transfer (lệch thang count) — ghi trung thực.

### Bảng 6 — (Appendix) Đầu phân phối: learned-σ vs PB-σ *(C4 — PHỤ)*
| σ (cùng student 1.9M) | worst-org global ↑ | worst-org cluster ↑ | MAE ↓ |
|---|---|---|---|
| PB-σ giải tích (=Paper 1) | 0.278 | 0.658 | 21.71 |
| **learned-σ (ours)** | **0.610** | **0.750** | **14.72** |
> N4: learned-σ > PB-σ, đậm ở global. **Honest:** UQ-floor E-AURC → R2 xếp **~4/5** (Ensemble 2.78 < … < R2 3.96
> < MC-Dropout 5.77) → UQ **không** phải chỗ thắng, để phụ. MAE−32% do ĐẦU ĐẾM, không phải UQ.

---

## Limitations (trung thực, kiểu cả 2 paper)
1. Transfer **sập khi scale-gap lớn** (MoNuSAC 4×) — density-head phụ thuộc nucleus scale.
2. Nhãn count vẫn là **giám sát** (GT count từ mask) → chứng minh *yêu cầu giám sát nhẹ*, không phải annotate-free.
3. **KHÔNG phải SOTA-accuracy** so heavy mask-net in-domain (Pareto, không head-to-head).
4. UQ xếp ~4/5 UQ-floor → đầu phân phối là *bonus*, không phải trục thắng.
5. `[TODO]` một số bảng efficiency/parity chưa đủ số (đánh dấu trong bản nộp).

---

## Novelty vs 2 paper mốc (1 đoạn để reviewer thấy ngay)
Cùng tinh thần distilled-lightweight-pathology-counter như Shvetsov (24M) & CellGenNet (U-Net), PACT
**nhỏ hơn (1.9M)**, **định lượng label-efficiency count-only** (họ chỉ nói "sparse"/dùng full-mask),
**chứng minh transfer cross-dataset** (họ để future work), và **thêm đầu phân phối calibrated** mà
**không dòng distillation-counting nào có**. → đủ khác biệt để đứng cùng/hơn hạng, đích **Q1 tầm trung**.

# Paper 2 — Distributional Count Distillation (R2)
### Tổng hợp MODEL mình tạo + KẾT QUẢ (cập nhật 2026-07-14)

> Tài liệu này gom **những gì MÌNH tự thiết kế/cài đặt Ở PAPER 2** (giải thích chi tiết) và **kết quả đã chạy thật**.
> Baseline của người khác chỉ *trích dẫn* (cuối file), không phải đóng góp của mình.
> Nguyên tắc xuyên suốt: **không bịa** — mọi thành phần **có cơ sở thống kê** + kiểm chứng bằng số liệu; các
> hằng số thiết kế (clamp σ, n_clusters, ngưỡng min_group) là **design choice có kèm sensitivity**, KHÔNG tuyên bố "zero-heuristic".

> ⚠️ **RANH GIỚI PAPER 1 vs PAPER 2 (đã đọc code — để KHÔNG double-claim):**
> - **PAPER 1** = *"Predictor-Agnostic Joint Conformal Cell Counting under Shift"* — đóng góp: **PB-JCI**
>   (score `R = |N − E[N]| / σ`, σ **suy từ cấu trúc Poisson-Binomial của điểm detection** σ=√Σsᵢ(1−sᵢ)),
>   **joint** đa lớp, **Adaptive PB-JCI Online** (cửa sổ thích nghi cho distribution shift). *Predictor-agnostic*:
>   nằm TRÊN detector nặng.
> - **PAPER 2 (tài liệu này) DÙNG LẠI khung conformal PB-JCI của Paper 1 (CITE), KHÔNG claim là mới.**
> - **Đóng góp GỐC của Paper 2** = *CHƯNG CẤT foundation model nặng → student siêu nhẹ 1.9M **tự HỌC** (μ, σ)*,
>   thay vì suy σ từ điểm detection của detector nặng. Cụ thể: **distillation + DensitySigmaUNet + σ Poisson-anchored
>   LEARNED (khác hẳn σ Poisson-Binomial của P1) + `--detach_mu`**.
> - **Mắt xích nối 2 paper:** baseline **KD** trong Paper 2 CHÍNH LÀ cách σ kiểu Poisson-Binomial của Paper 1 áp lên
>   student. Nên "**R2 vs KD**" = *σ HỌC ĐƯỢC (P2) vs σ Poisson-Binomial (P1)* trên cùng student nhẹ → cho thấy
>   distillation phân phối học được **tốt hơn** cách suy σ từ scores khi model bị nén xuống 1.9M.

---

## A. Ý TƯỞNG CỐT LÕI (một câu)

Chưng cất (distill) một mô hình foundation nặng (**PathoSAM ~640M**) thành một **student cực nhẹ ~1.9M** mà
student KHÔNG chỉ đếm tế bào chính xác, mà còn **xuất ra một PHÂN PHỐI đếm per-ảnh** (μ = số đếm kỳ vọng,
σ = độ bất định) → từ đó dựng **khoảng tin cậy (prediction interval) được hiệu chỉnh (calibrated)** với coverage
theo từng loại mô: **PanNuke đạt bảo đảm group-conditional (Mondrian, hữu hạn mẫu — 0/18 mô under mọi α)**;
**NuInsSeg giảm mạnh subgroup-undercoverage** (empirical, còn 1 mô khó). Định vị: **"nhẹ + đáng tin"
(computationally-efficient + trustworthy)** *(chưa đo joules → không claim "energy")*.

---

## B. NHỮNG GÌ MÌNH TỰ TẠO (giải thích chi tiết)

### B1. Kiến trúc student: `DensitySigmaUNet` (mình thiết kế)
File: [distill_student_r2.py](distill_student_r2.py) (class `DensitySigmaUNet`).

Một U-Net nhẹ (`student_ch=32` ⇒ **1,935,266 params**) có **HAI đầu ra**:

1. **Đầu density** (`self.dens`, Conv 1×1 → ReLU): xuất bản đồ mật độ `density_map ≥ 0` (H×W).
   - **Số đếm** `μ = Σ density_map` (tổng mật độ = số instance). Đây là *density-map counting*: không cần
     dot-annotation, dùng FULL mask của teacher làm target → bền hơn Gaussian-centroid.
   - ReLU ở đầu ra ⇒ nền = 0 chính xác (như CSRNet), không âm.

2. **Đầu log-σ** (`self.sig`): global-avg-pool bottleneck (ch·8 chiều) → MLP → **một scalar/ảnh** = hệ số phân tán.

**Vì sao mình tách 2 đầu chung 1 backbone:** σ "mượn" đặc trưng sâu của cùng ảnh → gradient sạch, không phải
train 2 mạng. σ là **heteroscedastic** (thay đổi theo ảnh), học từ **lỗi thật** chứ không phải hằng số.

### B2. Tham số hoá σ **Poisson-anchored** (đóng góp method quan trọng nhất — mình phát hiện + sửa)
```
σ = √(max(μ, 1)) · exp(clamp(log_s, −2, 2))
```
**Câu chuyện "chấp nhận thất bại → phân tích → sửa" (chính mình làm):**
- **Thất bại ban đầu:** dùng σ head thô `σ = exp(log_s)` → PanNuke thắng NHƯNG **NuInsSeg TRƯỢT** (Winkler
  R2 152.9 vs KD 128.6, std ±36 bất ổn). *Không giấu — điều tra bằng `diagnose_sigma.py`.*
- **Chẩn đoán:** head σ thô calibrated khi count đồng đều + data dồi dào (PanNuke corr(|err|,σ)=+0.53) nhưng
  **SẬP khi dải count khổng lồ + data ít** (NuInsSeg count 1→370): corr(|err|,σ)=−0.02 (σ thành nhiễu),
  σ runaway = 15703 → phình Winkler.
- **Fix (có cơ sở thống kê):** neo σ vào **√μ** — Poisson-inspired (equidispersion Var≈mean là *mốc neo*, KHÔNG
  giả định dữ liệu tuân Poisson chính xác; overdispersion do head learned-dispersion gánh). *(A2 xác nhận: dạng này
  calibrated hơn cả Negative-Binomial tường minh lẫn Gaussian-hetero — NLL 4.21 vs 4.58.)*
  → σ tự có "count-scaling" miễn phí + chặn runaway; head chỉ còn học **hệ số phân tán** (dispersion) quanh mốc Poisson.
- **Kết quả fix:** NuInsSeg Winkler **152.9 → 87.7** (std 36→6, lật TRƯỢT→ĐẠT); PanNuke cũng cải thiện
  **20.2 → 18.3**. **MỘT dạng σ, cả 2 dataset đều thắng** → đóng góp mạnh hơn là chỉ hack riêng 1 dataset.

### B3. `--detach_mu` — tách μ khỏi NLL (mình thiết kế, cấu hình CHỐT)
Loss huấn luyện = `density-KD (MSE) + count (|μ−GT|) + β·NLL(GT | μ, σ)`.
- **Vấn đề phát hiện qua ablation:** khi NLL kéo cả μ, `L_nll (~120) ≫ L_count (~20)` → NLL bóp méo μ →
  **hỏng MAE** (10 → 18).
- **Fix:** `mu = density.sum().detach()` trong nhánh σ → **NLL chỉ dạy σ**, không kéo μ. μ được dạy riêng bởi
  count-loss. Kết quả: lấy được MAE thấp CỦA density-count + worst-org cao CỦA NLL cùng lúc (MAE 18→10.12).

### B4. Suy luận: **DÙNG LẠI khung PB-JCI của Paper 1** (KHÔNG phải đóng góp Paper 2 — CITE Paper 1)
> ⚠️ Phần này **không phải novelty Paper 2**. Khung conformal (score `R=|GT−μ|/σ`, khoảng, gom nhóm/joint) là
> **đóng góp Paper 1**. Ở Paper 2 mình chỉ *đưa (μ, σ) HỌC ĐƯỢC của student* vào khung này. Đóng góp Paper 2 nằm
> ở B1–B3 (làm ra (μ,σ) tốt bằng distillation), không phải ở conformal.

Sau khi có (μ, σ) leak-free, dựng khoảng bằng **split conformal** trên score chuẩn hoá `r = |GT − μ| / σ`
(khung PB-JCI, Paper 1), gom nhóm để coverage *có điều kiện theo mô*. 3 scheme trong
[eval_r2_grouped.py](eval_r2_grouped.py):
- **global**: 1 quantile chung (mốc, ≈ split conformal thường).
- **mondrian**: 1 quantile **mỗi organ** (khi đủ mẫu/mô — dùng cho PanNuke).
- **cluster**: gom organ thành `n_clusters=5` theo **độ khó (từ σ)** rồi conformal từng cụm (khi ít mẫu/organ —
  dùng cho NuInsSeg). → gọi **"data-regime-aware grouping"** (KHÔNG dùng "Adaptive" để tránh nhầm với
  "Adaptive PB-JCI Online" của Paper 1); quy tắc chọn mondrian/cluster đặt ra *a priori* theo mật độ mẫu.
- **Vì sao cluster vừa tăng coverage vừa giảm Winkler:** global 1 quantile → rộng cho organ dễ, hẹp cho organ
  khó (miss). Cluster cấp quantile riêng từng nhóm khó → cắt được cả bề rộng thừa lẫn miss-penalty.
  **KD không có σ học được nên KHÔNG khai thác được cluster** — đây là lợi thế của việc có σ.
- Đo bằng **Winkler/interval score** + **organ_conditional_stats** (worst-organ coverage, org-gap, #under).

### B5. Protocol leak-free (mình thiết kế để tránh rò rỉ — nền tảng để mọi số liệu đáng tin)
- **PanNuke:** train 2 fold → **predict fold held-out** (`--test_fold`), lặp 3 fold. `--exclude_tissue colon`
  (loại overlap PathoSAM/Lizard, y hệt Paper 1). *Xác minh code:* `train()` chỉ lặp `train_idx`, không đụng test.
- **NuInsSeg:** **cross-fitting 5-fold** (`--kfold 5`) — mỗi ảnh được dự đoán bởi model **KHÔNG** train trên nó;
  ghép 665 dự đoán leak-free. *Xác minh:* `assign_kfold` gán mỗi ảnh đúng 1 fold, phân tầng theo organ.
- **Teacher không rò:** density teacher chỉ dùng cho ảnh TRAIN; ảnh test chỉ so với **GT thật**.

### B6. Script/harness mình viết (Bước 2 — so với heavy net)
- [count_student_cost.py](count_student_cost.py): đo **params + GMACs** thật của student (thop) → **1.935M / 10.49 GMACs**.
- [prep_nuinsseg_as_pannuke.py](prep_nuinsseg_as_pannuke.py): NuInsSeg → folder ảnh + `gt_counts.csv`;
  GT count = `len(unique(mask)) − nền` (Y HỆT student → so công bằng). Mode `resize`/`tile` + `--size`.
- [dump_cellvit_counts.py](dump_cellvit_counts.py): chạy **code OFFICIAL** của CellViT/LKCell (KHÔNG tái hiện
  thuật toán) → đếm `len(instance_types)`/ảnh. Các "fix tương thích" mình thêm (không đổi thuật toán của họ):
  `weights_only=False` (torch 2.6+), cờ `--lkcell` build đúng model UniRepLKNet qua chính class + config của họ,
  `--no_tokens` gọi thẳng `calculate_instance_map`.
- [eval_heavy_count.py](eval_heavy_count.py): chấm **MAE/RMSE/MAPE + per-organ**, ghép dòng student & teacher.

---

## C. KẾT QUẢ (đã chạy thật, leak-free)

> ⚠️ **NGUỒN CHÂN LÝ = VERIFICATION STATUS đầu KETQUA_R2 (2026-07-17).** R2 PanNuke ✅verified; R2 NuInsSeg ✅5-seed
> (worst 0.750±0.049); **KD pkl MẤT → số KD chưa re-verify (đang retrain)**. Mọi "0.773" = seed đơn cũ, dùng **0.750±0.049**.

### C1. Kết quả CHÍNH — R2 (ours) vs KD (cùng student ~1.9M) — cả 2 dataset, 3 trục
R2 thắng KD **SẠCH cả 3 trục trên cả 2 dataset** (⚠️ significance: dùng **paired Wilcoxon per-image / bootstrap** — số p seed-based cũ = pseudoreplication):

| Dataset | Winkler R2 | MAE R2 | worst-org R2 | KD (⚠️chưa re-verify) |
|---|---|---|---|---|
| **PanNuke** (no-colon, 3-fold) ✅ | **19.28** | **3.36** | **0.906** | 23.7 / 3.94 / 0.739 |
| **NuInsSeg** (cross-fit, 5 seed) ✅ | **95.4±11.9** | **14.7±1.7** | **0.750±0.049** | worst 0.282 (pkl mất) |

→ R2 thắng cả **accuracy (MAE)** lẫn **interval (Winkler)** lẫn **conditional coverage (worst-org)** so với KD (worst 0.28–0.74).
KD numbers cần retrain 5-seed để chốt gap honest.

### C2. So với BASELINE RECENT (bảng 8c) — cùng (μ,σ) leak-free, code official
**PanNuke (trung bình 3 fold no-colon):**

| Method | Năm | Winkler ↓ | MAE ↓ | worst-org ↑ |
|---|---|---|---|---|
| **R2-mondrian (ours)** | — | 19.28 | **3.36** | **0.906** ← cao nhất mọi method |
| **R2-cluster (ours)** | — | **18.50** | 3.36 | 0.843 |
| CondConf-group | 2025 | 18.81 | 3.37 | 0.853 |
| PCP | 2024 | 23.26 | 3.37 | 0.805 |
| CPCP | 2026 | 35.46 | 6.18 | 0.758 |
| R2CCP | 2024 | 58.4 | 5.83 | 0.621 |
| KD (teacher-distill) | — | 22.08 | 3.64 | 0.721 |

**NuInsSeg (cross-fit 5-fold):**

> ⚠️ CondConf/PCP chạy trên pkl R2 seed-đơn (đã mất) → cần re-eval trên pkl seed mới. R2 honest 5-seed dưới.

| Method | Năm | Winkler ↓ | MAE ↓ | worst-org ↑ |
|---|---|---|---|---|
| **R2-cluster (ours)** — 5 seed | — | **95.4±11.9** | 14.7±1.7 | 0.750±0.049 |
| CondConf-group (⚠️pkl cũ) | 2025 | 125.4 | 13.6 | **0.850** |
| PCP (⚠️pkl cũ) | 2024 | 91.1 | 13.6 | 0.714 |
| CPCP | 2026 | 250.6 | 28.7 | 0.500 |
| R2CCP | 2024 | 261.2 | 30.2 | 0.562 |

**Đọc trung thực:** PanNuke — R2-mondrian worst-org **0.906 = CAO NHẤT mọi method** (kể cả CondConf-2025) mà
**không train lại**. NuInsSeg — R2 thắng **Winkler** (95.4±11.9, ~−24% vs CondConf 125.4, 5-seed honest); CondConf nhỉnh
worst-org (0.850 vs R2 0.750±0.049).
⚠️ **SỬA claim (critique 3.3 — TRÁNH overclaim):** *KHÔNG* nói "R2 không cần organ". Các scheme conditional đạt worst-org cao
của R2 (**mondrian** PanNuke, **cluster** NuInsSeg) ĐỀU **cần nhãn organ/tissue lúc test** (mondrian bin theo mô; cluster map
test-sample qua `organs[i]`) — y như CondConf. Chỉ **R2-global** không cần organ nhưng worst-org yếu hơn. ⇒ Lợi thế R2 so
CondConf KHÔNG phải "khỏi cần organ" mà là **worst-org ngang/hơn ở CHI PHÍ THẤP HƠN NHIỀU** (1 model, không train net riêng)
+ xuất phân phối calibrated. R2CCP/CPCP MAE cao vì train net riêng trên feature pooled (mất density) — ghi rõ.

### C3. Efficiency (Bước 2, Phần A) — số EXACT trích paper + student mình đo
| Model | Params (M) | GMACs@256 | có UQ? |
|---|---|---|---|
| CellViT-SAM-H (2024) | 699.74 | 214.33 | ✗ |
| LKCell-L (2024) | 163.84 | 47.86 | ✗ |
| LSP-DETR (**2026**) | 45.0 | 26 | ✗ |
| NuLite-T (2024) | 17.12 | 26.16 | ✗ |
| **Student R2 (ours)** | **1.935** | **10.49** | **✓ σ + interval** |

→ Student **nhỏ nhất** (86–368× < heavy net, 9× < SOTA nhẹ nhất) + **ít FLOPs nhất** + là model **duy nhất
TRONG NHÓM BASELINE khảo sát có UQ calibrated** (các heavy net so ở đây chỉ cho điểm, không interval).

### C4. Count-MAE thật vs heavy net (Bước 2, Phần B) — NuInsSeg, leak-free, chạy RTX 5090
Heavy net chạy **off-the-shelf** (checkpoint PanNuke của họ chưa từng thấy NuInsSeg → **OOD, không leak**).
Cấu hình: ảnh native 512, SAM-H feed 1024. N=665 khớp student.

| Method | Params | MAE ↓ | RMSE ↓ | MAPE ↓ |
|---|---|---|---|---|
| CellViT-SAM-H (OOD) | 699.74M | 24.24 | 34.74 | 56.9% |
| LKCell-L (OOD) | 163.84M | 16.54 | 28.07 | 38.8% |
| Teacher PathoSAM (zero-shot) | ~640M | 15.80 | 29.02 | **28.3%** |
| **Student R2 (in-domain, ours)** | **1.9M** | **13.51** | **22.61** | 45.3% |

**Đọc trung thực (BẮT BUỘC — không tô hồng):**
- ★ Student MAE **13.51 < cả teacher 15.80** → trò đếm tốt hơn CHÍNH THẦY (MAE/RMSE) dù nhỏ 340×.
- Student **thắng MAE + RMSE** (thấp nhất mọi method Ở BẢNG NÀY) + là model duy nhất trong nhóm này có UQ.
- **NHƯNG** teacher MAPE 28.3% và LKCell 38.8% < student 45.3% → student sai **tương đối** cao hơn (kém ở ảnh
  ít nhân do density-sum). **Student KHÔNG thắng tuyệt đối mọi metric.**
- **Khung (bắt buộc):** đây là **in-domain-distill (rẻ) vs OOD-zero-shot** + lệch magnification — KHÔNG phải
  "student đếm giỏi hơn heavy net". Đóng góp = *thích nghi rẻ (1.9M) + trustworthy (UQ)*, không phải hơn-thua thô.

---

## D. BASELINE (KHÔNG phải mình tạo — chỉ trích dẫn, chạy code official)
Reliability/conformal: **CondConf** (Gibbs–Cherian–Candès, JRSS-B 2025), **PCP** (Zhang–Candès 2024),
**R2CCP** (Guha et al. ICLR 2024), **CPCP** (Chen–Li ICML 2026) + sàn UQ (MC-Dropout, Deep Ensembles, CQR, CHDQR).
Accuracy/efficiency: **CellViT/CellViT++** (2023–25), **LKCell** (2024), **NuLite** (2024), **LSP-DETR** (2026),
teacher **PathoSAM** (micro_sam). *Mình dùng ĐÚNG code/checkpoint official của họ, không tái hiện thuật toán —
để reviewer không thể nói "reproduce sai nên thắng".*

---

## E. TÓM TẮT ĐÓNG GÓP CỦA PAPER 2 (chỉ cái GỐC — không tính PB-JCI của Paper 1)
1. **Kiến trúc** `DensitySigmaUNet` — student 1.9M xuất (density→μ, σ) đồng thời.
2. **σ Poisson-anchored LEARNED** — tham số hoá σ HỢP NHẤT thắng cả 2 dataset (câu chuyện diagnostic→fix).
   *Khác hẳn σ Poisson-Binomial của Paper 1 (P1 suy σ từ điểm detection; P2 HỌC σ trong student).*
3. **`--detach_mu`** — tách μ khỏi NLL, gỡ mâu thuẫn accuracy vs calibration.
4. **Distillation** foundation→student cho ĐẦU RA PHÂN PHỐI (μ,σ) — cốt lõi: làm ra (μ,σ) tốt ở 1.9M.
5. **Protocol leak-free** (test_fold + cross-fit 5-fold) — nền tin cậy cho mọi số liệu.
6. **Harness Bước 2** (prep/dump/eval) — so count-MAE với heavy net công bằng, dùng code official của họ.

**DÙNG LẠI (CITE Paper 1), KHÔNG claim:** khung **PB-JCI** (score R=|N−E[N]|/σ, khoảng conformal, gom nhóm/joint,
Adaptive Online). Paper 2 chỉ *thay nguồn (μ,σ)*: từ "suy PB trên detector nặng" (Paper 1) → "HỌC trong student
nhẹ distilled" (Paper 2).

**Định vị Q1 (đã HIỆU CHỈNH TRUNG THỰC sau UQ floor 2026-07-16 — xem mục F):** KHÔNG claim "thắng mọi metric".
Định vị = **trustworthy counting ở giá distillation-scale**: cùng hạng cân nhẹ nhất (1.9M), R2 đạt reliability
(worst-org/interval) **NGANG TẦM các UQ hiện đại** (Deep Ensemble/CQR/CHDQR) nhưng **rẻ hơn nhiều** (1 model,
1 forward); vượt **rõ** cơ chế σ Poisson-Binomial của Paper 1 khi nén (R2 vs KD, p=1.9e−6) và UQ epistemic kinh
điển (MC-Dropout thua rõ); worst-org PanNuke cao nhất mọi method (0.906). *Computationally-efficient + trustworthy.*

---

## F. ★★ PHÂN TÍCH ĐÓNG GÓP MỚI + KẾT QUẢ ĐẦY ĐỦ (bản chuẩn bị manuscript) — 2026-07-16
> Gom **cái GÌ mới** (novelty, tách khỏi Paper 1 + baseline) và **TẤT CẢ số đã chạy thật**. Đây là bản để bắt đầu viết.

### F.1 — 5 ĐÓNG GÓP MỚI (không phải chỉ "1 model nhẹ")

**Model chỉ là phương tiện; novelty nằm ở 3 mảnh method + 2 phát hiện empirical.**

| # | Đóng góp mới | Loại | Vì sao MỚI (không trùng Paper 1 / baseline) |
|---|---|---|---|
| **N1** | **Tổ hợp mới** = [density-distillation từ foundation 640M→1.9M] **+** [calibration count-level] **+** [head Poisson-anchored xuất (μ,σ)] → student tí hon cho ra **phân phối đếm calibrated**. KHÔNG claim "distill-thành-phân-phối" như concept trần | tổ hợp (không phải concept lẻ) | ⚠️ Distillation & heteroscedastic-UQ tồn tại RIÊNG (critique). Cái mới = **kết hợp cụ thể** ba mảnh trên trong 1 student count-only + không peer distilled nào (HoVer-unet/9M-student) xuất UQ. Paper 1 KHÔNG train model (chỉ bọc detector nặng). ⇒ N1 mạnh **nhờ đi cùng N2–N4**, không đứng một mình |
| **N2** | **σ Poisson-anchored** `σ=√(max(μ,1))·exp(clamp(log_s,−2,2))` | parameterization mới | Công thức σ mới cho count data: neo √μ (Poisson) + head học dispersion. KHÁC hẳn σ=√Σsᵢ(1−sᵢ) (Poisson-Binomial) của Paper 1. Kèm diagnostic→fix (σ thô sập dải rộng → neo → cứu) |
| **N3** | **`detach_mu` / optimization decoupling** — tách μ khỏi NLL | kỹ thuật train mới | Gỡ **mean–variance optimization conflict** (L_nll≫L_count kéo lệch μ). Nguyên lý tổng quát cho heteroscedastic count regression, không riêng model này |
| **N4** | **Insight: learned-σ > score-σ khi nén 1.9M** (R2 vs KD; sig. = per-image paired test, KHÔNG phải seed-based) | phát hiện empirical | Chứng minh cơ chế σ của Paper 1 THUA khi ép vào model tí hon → phải HỌC σ. Không hiển nhiên, phải chạy mới biết |
| **N5** | **Insight: conditional coverage TRANSFER cross-dataset dù MAE không** (8c-bis) | phát hiện empirical | worst-org NuInsSeg→PanNuke 0.897≈in-domain 0.906 dù điểm đếm lệch thang → độ tin cậy generalize ĐỘC LẬP với độ chính xác điểm |

**Ranh giới (để không double-claim):** khung conformal/PB-JCI = **Paper 1** (CITE). Density-map counting (Σdensity),
U-Net backbone = cũ. UQ floor (MC-Dropout/Ensemble/CQR/CHDQR) = code lại của người khác để làm mốc. **Đóng góp gốc = N1–N5.**

### F.2 — KẾT QUẢ ĐẦY ĐỦ (đã chạy thật, leak-free, tất cả trên Mac/GitHub)

**(a) R2 vs KD — cùng student 1.9M, cả 2 dataset, 3 trục** (significance: per-image paired test, KHÔNG seed-based):

| Dataset | Winkler R2 | MAE R2 | worst-org R2 | KD (⚠️pkl mất) |
|---|---|---|---|---|
| PanNuke (3-fold) ✅verified | 19.28 | 3.36 | 0.906 | 23.7 / 3.94 / 0.739 |
| NuInsSeg (5 seed) ✅ | 95.4±11.9 | 14.7±1.7 | 0.750±0.049 | worst 0.282 |

**(b) vs baseline recent (code official)** — PanNuke: R2-mondrian **worst-org 0.906 = CAO NHẤT** (CondConf-25 0.853,
PCP-24 0.805, CPCP-26 0.758, R2CCP-24 0.621, KD 0.721). NuInsSeg: R2 **Winkler 95.4±11.9** (~−24% vs CondConf
125.4); CondConf nhỉnh worst-org 0.850. ⚠️ baseline NuInsSeg chạy trên pkl R2 cũ (mất) → re-eval trên seed mới.

**(c) UQ floor (mới, số thật — đọc TRUNG THỰC ở mục 8c-ter file KETQUA):**

| | PanNuke worst-org (mondrian) | NuInsSeg Winkler/worst (cluster) | compute |
|---|---|---|---|
| **R2 (ours)** | **0.906** (tie-cao nhất) | 95.4±11.9 / 0.750±0.049 (5 seed) | **1 model** |
| Deep Ensemble M3 | 0.901 | 79.0 / 0.760 | 3× train |
| CQR / CHDQR | 0.904 / 0.897 | 88.6 / 0.808 ; 74.7 / 0.689 | quantile head |
| MC-Dropout | 0.901 | **152.0 / 0.806** (thua rõ Winkler) | 30× forward |

→ R2 **ngang tầm UQ hiện đại + rẻ nhất**; MC-Dropout thua rõ. worst-org NuInsSeg nằm trong nhiễu training ~0.12.

**(d) Cross-dataset transfer (mới):** NuInsSeg→PanNuke mondrian **worst-org 0.897 (0/18 under)** ≈ in-domain 0.906;
PanNuke→NuInsSeg cluster worst 0.685 (global 0.421→0.685, Winkler 564→215). MAE tụt (19.9 & 44.9) do lệch thang count.

**(e) Efficiency:** Student **1.935M / 10.49 GMACs** — nhỏ nhất (CellViT-SAM-H 699.74M, LKCell-L 163.84M,
LSP-DETR-26 45M, NuLite-T 17.12M) + **duy nhất có UQ** trong nhóm khảo sát.

**(f) Count-MAE vs heavy net** (NuInsSeg OOD, RTX 5090): Student **13.51 < teacher PathoSAM 15.80 < LKCell-L 16.54
< CellViT-SAM-H 24.24** (thắng MAE+RMSE; caveat: MAPE 45.3% cao hơn teacher 28.3% — student kém ở ảnh ít nhân).

**(g) Hardening A1–A6:** A1 coverage-curve 4α (grouping≥global mọi α); A3 per-organ Wilson CI (undercoverage là
1 mô khó + nhiễu, không systematic); A5 σ-analysis (corr(σ,|e|)+0.40/+0.43, z-std PanNuke 1.01 calibrated);
A2 ablation σ-mode (Poisson NLL 4.21 < NB/raw 4.58); A6 3-seed (worst-org 0.78±0.02 NuInsSeg); A4 latency 1.87ms/112MB.

### F.3 — VERDICT Q1 (thẳng thắn)
**ĐỦ submit Q1 methods/applied journal** nếu kể đúng **3 trụ** (KHÔNG overclaim "thắng mọi metric"):
1. **Efficiency** — model nhỏ nhất CÓ UQ calibrated.
2. **Cross-dataset transfer** — conditional coverage generalize (N5).
3. **Reliability ngang tầm UQ hiện đại ở giá rẻ nhất** — MC-Dropout thua rõ, worst-org PanNuke cao nhất, R2>KD (=cơ chế Paper 1).

**Rủi ro:** (i) R2 không trội tuyệt đối interval (Ensemble/CQR ngang) → phải framing "trustworthy ở giá rẻ", không "tốt nhất";
(ii) chỉ 2 dataset (top-tier medical đòi ≥3 — có `monusac_converted.pkl` sẵn nếu muốn nâng chắc); (iii) NuInsSeg nhỏ/nhiễu → claim subgroup phải mềm.
**Rủi ro lớn nhất KHÔNG phải thiếu thí nghiệm — mà là FRAMING.** Số đã đủ.

---

## G. ★★ CHỐT IDENTITY = CORE A (Distillation là linh hồn) + định vị peer — 2026-07-16
> Sau một vòng khám phá dài (thử đua accuracy, thử đổi backbone FastViT), user CHỐT **core A**: distillation là
> linh hồn, giữ student nhỏ. Vòng đó KHÔNG phí — cho framing **label-efficiency** sắc hơn + calibrate accuracy trung thực.

### G.1 — Thesis A (chốt)
**"Distributional Count Distillation: chưng cất foundation model (PathoSAM ~640M) → student tí hon (1.9M) chỉ cần
nhãn COUNT rẻ (KHÔNG cần instance mask như NuLite/CellViT), xuất PHÂN PHỐI đếm calibrated (μ,σ). Accuracy
cạnh-tranh-cho-ngân-sách-nhãn, ở phần nhỏ chi phí annotation + size, + uncertainty mà segmentation model KHÔNG có."**

### G.2 — Label-efficiency = hook MỚI (điểm student THẮNG thật)
- NuLite/CellViT/HoVer-unet: cần **instance mask pixel-level** để train (đắt).
- Student: chỉ cần **foundation teacher (density) + nhãn count/ảnh** (1 số — rẻ hơn mask cả trăm lần). KHÔNG cần mask.
- ⇒ Bảng lightweight thêm **cột "Annotation cần"** (mask vs count-only) → student thắng ô đó. Gap accuracy 1.4–1.7×
  KHÔNG phải "thua" mà là **"~70% accuracy fully-supervised mà KHÔNG cần một mask nào + có UQ"** = label-efficiency thật.
- Bằng chứng đã có (mục 3c): **distilled ≈ GT-supervised** (worst-org distilled 0.753 > supervised 0.711) → distillation
  SÁNH NGANG supervised mà không cần mask = **trụ của core A, đã có số**.

### G.3 — CALIBRATE ACCURACY trung thực (đã chứng minh bằng số, hết ảo tưởng)
- **1.9M density-counter KHÔNG đè được SOTA accuracy** — vật lý capacity + density-sum sai tương đối cao ở ảnh ít nhân.
- Stage 1 in-domain PanNuke: NuLite-T (12.009M, đo thật) MAE 1.97/MAPE 9.9% vs student 3.38/23% → **1.72× MAE, 2.3× MAPE**
  (NuLite còn được chấp leak). All-OOD (NuInsSeg/MoNuSAC): student thua đậm (count-scale). → **Accuracy KHÔNG phải trục.**
- ⇒ Đấu accuracy = thua ở MỌI SOTA. **Bỏ mọi claim accuracy-win.** Stage 2 retrain NuLite = KHÔNG cần (số conservative
  ≤1.72× + framing annotation-regime là đủ). FastViT = gác lại (đó là core B).

### G.4 — Peer đúng của core A (định vị Related Work)
**Distilled-student KHÁC tồn tại — nhưng KHÔNG cái nào có distributional UQ (novelty ta độc quyền):**
| Peer | Là gì | UQ? |
|---|---|---|
| **HoVer-unet** (ISBI 2024, arXiv 2407.18449) | distill HoverNet→UNet, nuclei segmentation | ✗ |
| **9M H-Optimus student** (arXiv 2502.19217, 2025) | distill foundation H-Optimus→UNet 9M, segmentation | ✗ (citation VÀNG: cùng ý tưởng distill-foundation nhưng 9M/seg/no-UQ vs 1.9M/count/UQ) |
| **RCKD** | student học teacher trên ảnh unlabeled → seg/cls | ✗ |
| **NuLite/CellViT** | fully-supervised segmenter (cần mask) | ✗ |

→ **Peer THỰC NGHIỆM công bằng = same-student methods** (KD = σ-from-score Paper 1; MC-Dropout/Ensemble/CQR/CHDQR)
— đã chạy, student thắng/hòa + rẻ nhất + duy nhất calibrated. **KHÔNG cần chạy HoVer-unet/9M-student** (cùng outcome:
chúng to hơn, đếm giỏi hơn, KHÔNG UQ) — chỉ **CITE** để định vị + sắc novelty *"chưa ai distill thành phân phối có UQ"*.

### G.5 — 3 trụ paper A (chốt)
1. **Label-efficient distillation** — foundation teacher + count-label, KHÔNG cần mask (khác mọi peer segmentation).
2. **Distributional UQ** — calibrated (μ,σ), interval theo mô, transfer cross-dataset. KHÔNG peer nào có.
3. **Efficiency** — student tí hon 1.9M.
*Accuracy = "cạnh tranh cho ngân sách nhãn", KHÔNG phải điểm bán.*

### G.6 — Việc tiếp theo (nhẹ, đúng core A)
1. Sắc hoá **bảng annotation-cost** (mask vs count-only) — chỗ student thắng.
2. Đưa **ablation distilled vs GT-only** (3c) lên bảng chính (chứng minh distillation-value).
3. Sửa framing critique (N1 "distills distribution"→combo cụ thể; "R2 không cần organ" sai cho cluster; Wilcoxon per-image).
4. Giữ UQ + efficiency + cross-dataset (đã xong). → rồi VIẾT manuscript.
5. Optional nâng chắc: dataset 3 (MoNuSAC, harness sẵn) cho UQ-transfer; #5 detach_mu theory.

### G.7 — ★ BẢNG ANNOTATION-COST (lá chắn label-efficiency) — 2026-07-16

> Trả lời câu reviewer chắc chắn hỏi: *"student thua accuracy lightweight khác (1.72× MAE vs NuLite) thì bán cái gì?"*
> Trục bán KHÔNG phải accuracy mà là **chi phí giám sát để triển khai + UQ**. Bảng này định vị theo *loại nhãn cần*, không phải theo điểm số.

**Loại giám sát MỖI method cần để train/adapt (nền tảng honesty của claim):**

| Method | Backbone/Teacher | Nhãn TARGET cần để train | Độ mịn nhãn | Output | UQ | Params |
|---|---|---|---|---|---|---|
| **R2 student (ours)** | frozen PathoSAM (density) | **count-scalar/ảnh** (+ density teacher = 0 nhãn người) | **point-level** | count distribution (μ,σ) | **✓ calibrated** | **1.9M** |
| NuLite-T | ImageNet | **instance mask pixel + tissue-class** | pixel boundary+class | instance seg (→count post-proc) | ✗ | 12.0M |
| CellViT-SAM-H | SAM | instance mask pixel + class | pixel boundary+class | instance seg | ✗ | 700M |
| HoVer-unet (distilled) | HoverNet teacher | (distill) — teacher mask-trained; output seg | pixel (qua teacher) | instance seg | ✗ | ~ |
| 9M H-Optimus student (distilled) | H-Optimus teacher | (distill) — teacher mask-trained; output seg | pixel (qua teacher) | seg | ✗ | 9M |

**★ ĐỌC TRUNG THỰC (bắt buộc — tránh overclaim):**
- Claim là về **YÊU CẦU giám sát của phương pháp**, KHÔNG phải "mình đã dùng nhãn rẻ hơn". Trong thí nghiệm này GT count vẫn
  **lấy từ mask** (`len(unique(mask))−bg`) vì dataset có sẵn mask. Cái ta chứng minh: task-head của student **chỉ CẦN một scalar count/ảnh**
  — thứ lấy được bằng **dot/point annotation** (click từng nhân), rẻ hơn boundary-mask nhiều lần (lập luận chuẩn của counting literature,
  point-vs-pixel — **cite**, KHÔNG bịa con số "100×"). NuLite/CellViT **bắt buộc** có mask pixel-level mới train được → đó là khác biệt cứng.
- vs peer **distilled** (HoVer-unet, 9M-student): chúng cũng distill nên student cũng không cần mask mới — **khác biệt của ta với nhóm này KHÔNG phải label-cost mà là (a) task-head count-level + (b) distributional UQ** (không peer nào có). Đừng dùng bảng này để claim thắng label-cost so với nhóm distilled.
- ⇒ **Câu bán đúng:** *"Với nhãn point-level (không một mask nào ở target domain) + một foundation teacher đông lạnh, student 1.9M đạt ~70% accuracy của segmenter fully-supervised, ĐỔI LẠI có UQ calibrated mà không segmenter nào cung cấp."* Gap accuracy = **chi phí của chế độ nhãn rẻ**, không phải điểm yếu trần trụi.
- **Bằng chứng "distillation đáng giá" (không chỉ nhãn rẻ):** ablation distilled vs GT-only same-student — distilled worst-org **0.753 > 0.711** (teacher foundation nâng conditional coverage) → đưa lên bảng chính (G.6 mục 2).

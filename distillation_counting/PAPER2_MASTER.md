# Paper 2 — Distributional Count Distillation (R2) — MASTER

> **FILE DUY NHẤT** (gộp MODEL + KETQUA cũ, 2026-07-17). Từ giờ chỉ cập nhật ở đây.
> Số cũ/leaky (single-split, seed-đơn 0.773, runbook đã xong) ĐÃ XOÁ để tránh rối.
> Nguyên tắc: **không bịa** — mọi thành phần có cơ sở thống kê + số liệu; hằng số thiết kế = design-choice có sensitivity, KHÔNG claim "zero-heuristic".

---

## 0. ★ VERIFICATION STATUS (NGUỒN CHÂN LÝ — mọi số phải khớp bảng này)

| Thành phần | Trạng thái | Số honest |
|---|---|---|
| **R2 PanNuke** | ✅ VERIFIED (tái tạo khớp từng số, 3 fold) | mondrian worst-org **0.906**, MAE **3.36**, Winkler **19.28** |
| **R2 NuInsSeg** | ✅ RE-ESTABLISHED 5-seed (2026-07-17) | cluster worst-org **0.750±0.049**, MAE **14.7±1.7**, Winkler **95.4±11.9** |
| **CondConf/PCP NuInsSeg** | ✅ RE-EVAL 5-seed matched (2026-07-17) | CondConf 0.898±0.013 / 146.8±24.9 (over-cover); PCP 0.708±0.069 / 105.9±12.1 |
| **KD NuInsSeg** | ✅ VERIFIED 5-seed (2026-07-18) | same-scheme §4.1: global 0.278/132/21.7, cluster 0.658/98.7/21.7; per-image sig MAE p<0.05 mọi seed, Winkler 1 seed n.s. **N4 reframe: PB-σ chỉ sập global.** pkl backup kaggle `sam3-paper2-uqkd` |
| **UQ-floor NuInsSeg** (Ensemble/CQR/CHDQR/MC-Dropout) | ✅ VERIFIED 5-seed (2026-07-18) | cluster n=5: Ens 69.6/0.767, CQR 80.7/0.806, CHDQR 78.7/0.722, MCD 173/0.774 — **R2 KHÔNG dẫn đầu (~4/5)**, xem §4.3 |
| **Cross-dataset N5 (3 dataset)** | ✅ CryoNuSeg thêm (2026-07-18) | NuInsSeg→CryoNuSeg marg.cov **0.967**/Winkler 503/MAE 73 (σ transfer sống); NuInsSeg↔PanNuke §4.4. MoNuSAC=boundary (scale 4× → μ sập) |
| **Baseline PanNuke (8c)** | ✅ chạy trên pkl R2 còn sống | giữ nguyên |
| pkl R2 5-seed | ✅ backup kaggle `hipinhththu/sam3-r2-nuinsseg-seeds` | — |

**Provenance (2026-07-17):** pkl NuInsSeg canonical (`student_r2_nuinsseg_cv5_poisson.pkl` + KD) đã MẤT (không ở vast/kaggle).
Bản `_feat` còn sót thiếu `--detach_mu` → số sai. Đã retrain R2 5-seed đúng config. **Mọi "0.773" cũ = seed đơn** (nằm trong dải
0.70–0.82); số chính thức = **0.750±0.049**. PanNuke KHÔNG ảnh hưởng (nhiều ảnh/mô → worst-org ổn định; NuInsSeg ít ảnh/organ → nhạy seed = lý do phải multi-seed).

**★ SPLIT PanNuke của PathoSAM (CHỐT CỨNG, verify từ CODE patho-sam — [[pathosam-training-data]]):**
**PathoSAM train PanNuke FOLD_1 + FOLD_2, TEST trên FOLD_3** (`get_generalist_datasets.py:80` train fold_1+2; `dataloaders.py:149` eval fold_3).
⟹ **Đánh giá PanNuke trên FOLD_3 = SẠCH (leak-free)** — dùng dataset KHÔNG bị leak. **Paper 2 lấy fold_3 làm số PanNuke chính (worst-org 0.905).**
**NuInsSeg + MoNuSAC không nằm trong training PathoSAM → clean OOD; NuInsSeg = anchor generalization.** Paper 1 cũng test fold_3 → sạch, loại colon đúng.

---

## 1. IDENTITY & RANH GIỚI

### 1.1 Paper 1 vs Paper 2 (KHÔNG double-claim)
- **PAPER 1** = *"Predictor-Agnostic Joint Conformal Cell Counting under Shift"*: **PB-JCI** (score `R=|N−E[N]|/σ`, σ suy từ cấu trúc
  Poisson-Binomial của điểm detection `σ=√Σsᵢ(1−sᵢ)`), joint đa lớp, Adaptive PB-JCI Online. Nằm TRÊN detector nặng.
- **PAPER 2 DÙNG LẠI khung conformal PB-JCI của P1 (CITE, KHÔNG claim mới).** Đóng góp gốc = **chưng cất foundation → student 1.9M
  tự HỌC (μ,σ)** thay vì suy σ từ detector nặng.
- **Mắt xích:** baseline **KD** trong P2 = cách σ Poisson-Binomial của P1 áp lên student. "R2 vs KD" = *σ HỌC ĐƯỢC (P2) vs σ Poisson-Binomial (P1)* cùng student nhẹ.

### 1.2 Core A (chốt 2026-07-16): distillation là linh hồn
Sau vòng thử đua accuracy + đổi backbone FastViT, chốt **core A**: distillation là linh hồn, giữ student nhỏ, **accuracy KHÔNG phải trục bán**.
Hook = **label-efficiency** (count-only vs mask). FastViT = gác lại (core B).

**Thesis:** *"Distributional Count Distillation: chưng cất PathoSAM ~640M → student 1.9M chỉ cần nhãn COUNT rẻ (KHÔNG cần instance mask
như NuLite/CellViT), xuất PHÂN PHỐI đếm calibrated (μ,σ) — trustworthy ở giá distillation-scale."*

---

## 2. METHOD (những gì mình tự thiết kế)

### 2.1 `DensitySigmaUNet` ([distill_student_r2.py](distill_student_r2.py))
U-Net nhẹ (`student_ch=32` ⇒ **1,935,266 params / 10.49 GMACs@256**), HAI đầu ra:
1. **density** (Conv1×1→ReLU): `density_map≥0`; **μ = Σ density_map** (density-map counting, dùng FULL mask teacher làm target, không cần dot-annotation).
2. **log-σ** (global-avg-pool bottleneck → MLP → 1 scalar/ảnh = hệ số phân tán).
Chung 1 backbone → σ mượn đặc trưng sâu, gradient sạch, σ **heteroscedastic** học từ lỗi thật.

### 2.2 σ **Poisson-anchored** (đóng góp method quan trọng nhất — diagnostic→fix)
```
σ = √(max(μ, 1)) · exp(clamp(log_s, −2, 2))
```
- **Thất bại ban đầu:** σ head thô `exp(log_s)` → PanNuke thắng nhưng **NuInsSeg TRƯỢT** (Winkler bất ổn, std ±36). Không giấu — điều tra `diagnose_sigma.py`.
- **Chẩn đoán:** head thô calibrated khi count đồng đều + data dồi dào (PanNuke corr(|err|,σ)=+0.53) nhưng **SẬP khi dải count lớn + data ít**
  (NuInsSeg count 1→370: corr=−0.02, σ runaway=15703).
- **Fix (cơ sở thống kê):** neo σ vào **√μ** (Poisson equidispersion Var≈mean là *mốc neo*, KHÔNG giả định data tuân Poisson; overdispersion do
  head learned-dispersion gánh). → count-scaling miễn phí + chặn runaway; head chỉ học hệ số phân tán.
- **Kết quả:** NuInsSeg lật TRƯỢT→ĐẠT; PanNuke cũng cải thiện. **MỘT dạng σ, cả 2 dataset thắng.** (A2: Poisson NLL 4.21 < NB/Gaussian-raw 4.58.)

### 2.3 `--detach_mu` (cấu hình CHỐT)
Loss = `density-KD (MSE) + count·|μ−GT| + β·NLL(GT|μ,σ)` (β-NLL Seitzer 2022, β=0.5; w_count=w_nll=0.01).
- **Vấn đề (qua ablation):** NLL kéo cả μ, `L_nll≫L_count` → bóp méo μ → hỏng MAE.
- **Fix:** `mu = density.sum().detach()` trong nhánh σ → NLL CHỈ dạy σ. Gỡ **mean–variance optimization conflict**.

### 2.4 Suy luận: DÙNG LẠI PB-JCI của Paper 1 (CITE, không claim)
Split conformal trên score `r=|GT−μ|/σ`; 3 scheme ([eval_r2_grouped.py](eval_r2_grouped.py)):
- **global** (1 quantile), **mondrian** (1 quantile/organ — PanNuke, đủ mẫu/mô), **cluster** (gom organ thành n_clusters=5 theo độ khó từ σ — NuInsSeg, ít mẫu/organ).
- Gọi **"data-regime-aware grouping"** (quy tắc mondrian/cluster đặt *a priori* theo mật độ mẫu). KD không có σ học được → KHÔNG khai thác được cluster.
- ⚠️ **mondrian & cluster ĐỀU cần nhãn organ lúc test** (critique 3.3) — KHÔNG claim "R2 không cần organ". Chỉ global mới không cần (worst-org yếu hơn).

### 2.5 Protocol leak-free
- **PanNuke:** train 2 fold → predict fold held-out (`--test_fold`), lặp 3 fold, `--exclude_tissue colon` (Lizard-overlap, y hệt P1).
- **NuInsSeg:** cross-fitting 5-fold (`--kfold 5`), ghép 665 dự đoán leak-free.
- Teacher density chỉ dùng cho ảnh TRAIN; test chỉ so GT thật.

> ★ **SPLIT PanNuke của PathoSAM (CHỐT CỨNG — verify từ CODE patho-sam, xem [[pathosam-training-data]]):**
> **PathoSAM train PanNuke FOLD_1 + FOLD_2, TEST trên FOLD_3** (`get_generalist_datasets.py:80` = `folds=["fold_1","fold_2"]`; `dataloaders.py:149` = `folds=["fold_3"]`).
> ⟹ **Đánh giá PanNuke trên FOLD_3 = SẠCH** (teacher chưa thấy fold_3). **Paper 2 lấy fold_3 làm số PanNuke chính = worst-org 0.905** (MAE/Winkler theo fold_3).
> f1/f2 chỉ ghi kèm cho robustness (là in-domain của teacher; số ~bằng f3 nên nhất quán). **KHÔNG dùng f1/f2 làm headline.**
> **NuInsSeg = out-of-domain (KHÔNG train) = CLEAN OOD = anchor generalization thật.** Mọi claim generalization của Paper 2 tựa NuInsSeg.
> Paper 1 cũng test fold_3 → sạch, colon-exclusion đúng.

### 2.6 Harness Bước 2 (so heavy net công bằng)
`count_student_cost.py` (params+GMACs thop), `prep_nuinsseg_as_pannuke.py` (GT count = len(unique(mask))−nền, y hệt student),
`dump_cellvit_counts.py` (chạy code OFFICIAL CellViT/LKCell/NuLite), `eval_heavy_count.py` (MAE/RMSE/MAPE+Bias).

---

## 3. NOVELTY (N1–N5) — đóng góp gốc, tách khỏi P1/baseline

| # | Đóng góp | Loại | Vì sao mới |
|---|---|---|---|
| **N1** | **Tổ hợp** [density-distill 640M→1.9M] + [calibration count-level] + [head Poisson-anchored (μ,σ)] → student tí hon xuất phân phối calibrated | tổ hợp (không phải concept lẻ) | Distillation & heteroscedastic-UQ tồn tại RIÊNG; cái mới = kết hợp cụ thể + không peer distilled nào xuất UQ. Mạnh **nhờ đi cùng N2–N4** |
| **N2** | **σ Poisson-anchored** `√(max(μ,1))·exp(clamp(log_s,−2,2))` | parameterization | Công thức σ mới cho count; khác hẳn σ Poisson-Binomial của P1; kèm diagnostic→fix |
| **N3** | **`detach_mu`** — tách μ khỏi NLL | kỹ thuật train | Gỡ mean–variance optimization conflict; nguyên lý tổng quát cho heteroscedastic count regression |
| **N4** | **learned-σ > PB-σ ở student 1.9M — ĐẬM dưới global, thu hẹp dưới cluster** (R2 vs KD, ✅ VERIFIED 5-seed) | phát hiện empirical | Same-scheme (§4.1): worst-org global R2 0.610 vs KD 0.278; cluster 0.750 vs 0.658. PB-σ **chỉ sập dưới global**; cluster bù lại. R2 còn đè teacher-PB 640M worst-org cả 2 scheme + MAE −32% (đầu đếm). Sig per-image: MAE p<0.05 mọi seed, Winkler p<0.05 hầu hết (1 seed n.s.). **KHÔNG claim "PB vỡ 0.282" (cross-scheme, đã bỏ).** |
| **N5** | **conditional coverage TRANSFER cross-dataset dù MAE không** | phát hiện empirical | worst-org NuInsSeg→PanNuke 0.897≈in-domain 0.906 → tin cậy generalize độc lập với accuracy điểm |

**Ranh giới:** khung conformal/PB-JCI = P1 (cite). Density-counting, U-Net = cũ. UQ floor = code người khác làm mốc. **Gốc = N1–N5.**

---

## 4. RESULTS (honest, current — 2026-07-18)

### 4.1 R2 vs KD (cùng student 1.9M) — ✅ KD VERIFIED 5-seed (2026-07-18)

| Dataset | Winkler R2 ↓ | MAE R2 ↓ | worst-org R2 ↑ | KD (verified 5-seed) |
|---|---|---|---|---|
| **PanNuke** (no-colon, headline fold_3) ✅ | **19.28** | **3.36** | **0.905** (fold_3, cluster) | 23.7 / 3.94 / 0.739 |
| **NuInsSeg** (cross-fit, 5-seed) ✅ | **95.4±11.9** | **14.7±1.7** | **0.750±0.049** (cluster) | xem bảng same-scheme ↓ |

**★ N4 SAME-SCHEME (NuInsSeg, 5-seed, cùng student 1.9M — NGUỒN CHÂN LÝ, thay số cross-scheme "0.282" cũ đã SAI):**
Số cũ so R2-**cluster** (0.750) với KD-**global** (0.282) = **lệch scheme, KHÔNG fair**. Same-scheme thật:

| scheme | metric | KD (PB-σ) | R2 (learned-σ) |
|---|---|---|---|
| **global** | worst-org ↑ | 0.278±0.048 | **0.610±0.076** |
| **global** | Winkler ↓ | 131.99±1.99 | **94.17±9.83** |
| **cluster** | worst-org ↑ | 0.658±0.026 | **0.750±0.044** |
| **cluster** | Winkler ↓ | 98.67±1.72 | 95.37±10.66 (~hòa) |
| **cả hai** | MAE ↓ | 21.71 | **14.72** |

**Đọc honest (KHÔNG overclaim "PB-σ vỡ"):**

- **MAE −32% (14.7 vs 21.7) bền cả 2 scheme** — nhưng do **ĐẦU ĐẾM** (density-sum vs segment-threshold), KHÔNG phải UQ. R2 đếm giỏi hơn KD độc lập scheme.
- **worst-org:** learned-σ > PB-σ **cả 2 scheme**, dramatic ở **global** (0.61 vs 0.28) nhưng **thu hẹp ở cluster** (0.75 vs 0.66 — grouping bù phần σ miscalibrated). ⟹ "PB-σ collapse" **CHỈ đúng dưới global**; dưới cluster PB-σ nén tốt (teacher 640M 0.680 → student 1.9M 0.658, gần như không rớt).
- **Winkler:** R2 thắng lớn global, **~hòa cluster**.

**★ vs Teacher-PB 640M (VERIFIED LOCAL `data/pathosam_nuinsseg_preds.pkl` — `eval_r2_grouped.py`):** teacher-PB cluster **0.680/85.07/17.89**; global **0.482/111.64**.
So R2 1.9M: **R2 đè teacher worst-org CẢ 2 scheme** (cluster 0.750>0.680; global 0.610>0.482) **+ MAE** (14.7<17.9); teacher CHỈ nhỉnh **Winkler-cluster** (85<95). ⟹ **Distilled 1.9M ≥ teacher 640M ở reliability+accuracy, tại 1/330 size** — teacher chỉ giữ 1 góc Winkler-cluster. *(json: `work/teacher_pb_grouped.json`.)*

**Significance per-image (đơn vị=ảnh N=333, thay p=1.9e−6 pseudoreplication cũ):** R2 vs KD **MAE p<0.05 MỌI seed** (Δ≈−4→−10 count, CI không chứa 0); **Winkler p<0.05 HẦU HẾT seed** (vd p=2e−8) nhưng **1 seed n.s.** (p=0.109, CI chứa 0) → ghi trung thực "significant in most, not all seeds".

PanNuke per-fold worst-org: f1 0.908 / f2 0.906 / f3 0.905 (0/18 mỗi fold). NuInsSeg R2-cluster 5-seed: [0.701,0.764,0.701,0.817,0.767]. KD-cluster: [0.641,0.631,0.68,0.699,0.641].

### 4.2 vs baseline recent (code official, cùng μ,σ leak-free)
**PanNuke (avg 3 fold no-colon)** ✅:

| Method | Winkler ↓ | MAE ↓ | worst-org ↑ |
|---|---|---|---|
| **R2-mondrian (ours)** | 19.28 | **3.36** | **0.906** ← cao nhất mọi method |
| **R2-cluster (ours)** | **18.50** | 3.36 | 0.843 |
| CondConf-25 | 18.81 | 3.37 | 0.853 |
| PCP-24 | 23.26 | 3.37 | 0.805 |
| CPCP-26 | 35.46 | 6.18 | 0.758 |
| R2CCP-24 | 58.4 | 5.83 | 0.621 |
| KD | 22.08 | 3.64 | 0.721 |

**NuInsSeg (5-seed matched, 2026-07-17)** ✅:

| Method | marg.cov | Winkler ↓ | worst-org ↑ | Đọc |
|---|---|---|---|---|
| **R2-cluster (ours)** | ~0.91 (sát target) | **95.4±11.9** | 0.750±0.049 | cân bằng: Winkler tốt nhất nhóm calibrated + worst-org mạnh |
| CondConf-25 | 0.95 (over-cover) | 146.8±24.9 | **0.898±0.013** | worst-org cao nhất NHƯNG nới khoảng gấp đôi → Winkler tệ +54% |
| PCP-24 | ~0.92 | 105.9±12.1 | 0.708±0.069 | **R2 đè cả 2 trục** |
| CPCP-26 / R2CCP-24 (feat) | — | 250.6 / 261.2 | 0.500 / 0.562 | không re-run (cần feature, seed pkl không có) |

MAE mọi method = 14.7±1.7 (cùng μ). **Kết luận:** R2 = **interval hiệu quả nhất** trong nhóm calibrated đúng target; CondConf đổi worst-org cao lấy over-coverage + khoảng rộng gấp đôi; R2 thắng PCP cả 2 trục.

### 4.3 UQ-floor (cùng student 1.9M, trục reliability) — ✅ VERIFIED 5-seed cluster n=5 (2026-07-18)

**PanNuke worst-org:** R2 **0.906** (tie cao nhất), Ensemble 0.901, CQR 0.904, CHDQR 0.897, MC-Dropout 0.901.
**NuInsSeg (cluster n=5, 5-seed matched — pkl `work/uq_{method}_nuinsseg_s{42..46}.pkl`, script `aggregate_uqfloor.py`):**

| Method (cùng student 1.9M) | compute | marg.cov | Winkler ↓ | worst-org ↑ |
|---|---|---|---|---|
| Ensemble | **5× model** | 0.915 | **69.61±3.14** | 0.767±0.024 |
| CQR | 1 model | 0.928 (over) | 80.71±13.20 | **0.806±0.014** |
| CHDQR | 1 model | 0.920 | 78.72±7.89 | 0.722±0.061 |
| **R2 (ours)** | **1 model, 1 pass** | ~0.91 | 95.4±11.9 | 0.750±0.049 |
| MC-Dropout | N-pass | 0.911 | 173.03±52.98 | 0.774±0.051 |

**★ ĐỌC TRUNG THỰC (setback — KHÔNG giấu):** trên NuInsSeg **R2 KHÔNG dẫn đầu UQ-floor** — xếp ~4/5 cả Winkler lẫn worst-org.
CQR (cùng 1 model) nhỉnh R2 **cả 2 trục**; Ensemble Winkler tốt nhất nhưng **tốn 5× compute**. **R2 chỉ thắng rõ MC-Dropout** (95.4 vs 173).
→ **KHÔNG được bán pillar-UQ là "reliability tốt nhất/rẻ nhất"** (claim cũ đó SAI với số này). Claim honest thu hẹp còn:
R2 = **phân phối tham số (μ,σ) 1-forward** — (a) đè MC-Dropout (UQ 1-forward ngây thơ), (b) ngang Ensemble ở **1/5 compute**,
(c) σ này cho phép **transfer cross-dataset (N5)** + tích hợp vào distillation (UQ "miễn phí", không cần cơ chế riêng).
CQR/CHDQR chỉ cho **interval cố định-α**, KHÔNG cho phân phối tái dùng/transfer. **Interval-efficiency thuần: CQR cạnh tranh — phải thừa nhận trong manuscript.**
*(Hệ quả: UQ KHÔNG phải trục bán — xem §5; lead bằng distillation/label-efficiency/N4, UQ = "distributional calibrated, integrated, competitive-not-best".)*

### 4.4 Cross-dataset transfer (N5) — ✅ 3 dataset (CryoNuSeg thêm 2026-07-18)
| Transfer | scheme | MAE | Winkler | cov (worst-org/marg) | #under |
|---|---|---|---|---|---|
| NuInsSeg → PanNuke | mondrian | 19.90 | 97.21 | **0.897** worst-org | **0/18** |
| PanNuke → NuInsSeg | cluster | 44.88 | 214.83 | 0.685 worst-org | 4/27 |
| **NuInsSeg → CryoNuSeg** (OOD→OOD, dataset 3) | global | 73.2 | 503.3 | **0.967** marg | 0/1 |

**Conditional coverage TRANSFER**: NuInsSeg→PanNuke worst-org 0.897 ≈ in-domain 0.906. σ distilled vẫn informative dưới shift (chiều khó: cluster kéo worst 0.42→0.685, Winkler 564→215). MAE KHÔNG transfer (lệch thang count) — ghi trung thực.

**★ CryoNuSeg = dataset 3 (2026-07-18):** train NuInsSeg → predict CryoNuSeg (n=30, count mean **253**, range 85–638; clean OOD, không trong PathoSAM/Lizard training). σ **informative** (mean 32.9, std 18.3) → conformal đạt marginal cov **0.967** với interval KHÔNG vacuous (width 454 ≈ ±0.9× count); MAE 73 ≈ **29% rel** degrade graceful (count-scale khác → MAE không transfer, ghi trung thực). **⟹ σ calibrated TRANSFER sang dataset thứ 3.** (cov 0.967 hơi over target 0.90 — một phần do n=30 nhỏ → split-conformal quantile bảo thủ.)

**★ Ranh giới vận hành (honest — MoNuSAC KHÔNG dùng làm số):** MoNuSAC native **1024** → resize 256 **co nhân 4×** (ngoài scale train) → μ **SẬP** (MAE 138≈mean, σ 6.5, interval vacuous ±5×; cov 0.934 rỗng). ⟹ transfer **sống khi scale-gap vừa phải** (CryoNuSeg native 512 = 2×) **, sập khi quá lớn** (MoNuSAC 4×). Đây là **điều kiện áp dụng của phương pháp** (density-head phụ thuộc nucleus scale) — ghi trung thực trong Limitations. *(pkl `work/xfer_nuinsseg2cryonuseg.pkl`, prep `prep_cryonuseg_counts.py`, backup kaggle `sam3-paper2-uqkd`.)*

### 4.5 Efficiency
| Model | Params (M) | GMACs@256 | UQ? |
|---|---|---|---|
| CellViT-SAM-H (2024) | 699.74 | 214.33 | ✗ |
| LKCell-L (2024) | 163.84 | 47.86 | ✗ |
| LSP-DETR (2026) | 45.0 | 26 | ✗ |
| NuLite-T (2024, đo thật) | 12.009 | 19.80 | ✗ |
| **Student R2 (ours)** | **1.935** | **10.49** | **✓ σ + interval** |

Student **nhỏ nhất** + ít FLOPs nhất + **duy nhất trong nhóm khảo sát có UQ calibrated**.

### 4.6 Count-MAE vs heavy net (NuInsSeg, RTX 5090)
Heavy net off-the-shelf (checkpoint PanNuke → OOD trên NuInsSeg, không leak); student in-domain distill.

| Method | Params | MAE ↓ | RMSE ↓ | MAPE ↓ |
|---|---|---|---|---|
| CellViT-SAM-H (OOD) | 699.74M | 24.24 | 34.74 | 56.9% |
| LKCell-L (OOD) | 163.84M | 16.54 | 28.07 | 38.8% |
| Teacher PathoSAM (zero-shot) | ~640M | 15.80 | 29.02 | **28.3%** |
| **Student R2 (in-domain, ours)** | **1.9M** | **~13.5–14.7** | 22.61 | 45.3% |

**Đọc trung thực:** student MAE/RMSE thấp nhất bảng này + duy nhất có UQ; **NHƯNG** MAPE 45.3% > teacher 28.3% (sai tương đối cao ở ảnh ít nhân do density-sum). Khung = *in-domain-distill (rẻ) vs OOD-zero-shot* — KHÔNG phải "student giỏi hơn heavy net". Đóng góp = thích nghi rẻ + trustworthy.

### 4.7 Annotation-cost (lá chắn label-efficiency)
| Method | Teacher | Nhãn TARGET cần | Độ mịn | Output | UQ | Params |
|---|---|---|---|---|---|---|
| **R2 (ours)** | frozen PathoSAM | **count-scalar/ảnh** (+density teacher=0 nhãn người) | **point** | count dist (μ,σ) | **✓** | **1.9M** |
| NuLite-T / CellViT-SAM-H | ImageNet / SAM | instance mask pixel + class | pixel | seg | ✗ | 12M / 700M |
| HoVer-unet / 9M-H-Optimus (distilled) | mask-trained teacher | (distill) output seg | pixel | seg | ✗ | ~/9M |

**★ Honesty caveat:** claim là về **YÊU CẦU giám sát của phương pháp** (task-head chỉ cần count-scalar, lấy được bằng point/dot-annotation),
KHÔNG phải "mình đã dùng nhãn rẻ hơn" (thí nghiệm này GT count vẫn lấy TỪ mask vì dataset có sẵn). Point-vs-pixel rẻ hơn nhiều lần = **cite** counting literature, KHÔNG bịa "100×".
vs peer **distilled** (HoVer-unet/9M): khác biệt KHÔNG phải label-cost mà là **task-head count-level + distributional UQ**.
**Câu bán:** *"nhãn point-level (không mask nào ở target) + teacher đông lạnh → student 1.9M đạt ~70% accuracy segmenter fully-supervised, ĐỔI LẠI có UQ calibrated."*

### 4.8 Ablations — ✅ RE-RUN LEAK-FREE 5-seed cross-fit (2026-07-18) — số single-split cũ ĐÃ LẬT
NuInsSeg 5-seed, cluster n=5, cùng eval bảng chính (`aggregate_ablations.py`; pkl `student_r2_nuinsseg_cv5_{tag}_s*.pkl`):

| Config | worst-org ↑ | Winkler ↓ | MAE ↓ |
|---|---|---|---|
| **MAIN** ch32 + detach + teacher-distill | 0.750±0.044 | 95.4 | 14.72 |
| detach OFF (coupled NLL) | 0.767±0.024 | 99.0 | 15.62 |
| ch16 (~0.5M) | **0.797±0.016** | **86.7** | **13.79** |
| ch64 (~7.7M) | 0.781±0.032 | 169.2 | 17.13 |
| GT-supervised (density từ GT mask) | 0.789±0.028 | 92.8 | 14.23 |

**★ 3 KẾT LUẬN LẬT (honest — số single-split cũ = leaky artifact, ĐÃ BỎ):**
1. **detach_mu (N3) YẾU hơn claim cũ:** detach-OFF MAE **15.62** (KHÔNG phải ~18), worst-org còn **nhỉnh** (0.767>0.750). ⟹ detach cho MAE tốt hơn CHÚT (14.72 vs 15.62) nhưng đánh đổi worst-org. **KHÔNG claim "detach cứu MAE"**; giữ detach vì MAE (trục chính) + đơn giản.
2. **ch32 KHÔNG phải sweet spot:** **ch16 (~0.5M) thắng CẢ 3 trục** NuInsSeg (0.797/86.7/13.79); ch64 tệ nhất. ⟹ chỉ dám claim sweet-spot nếu có ablation PanNuke (**CHƯA có**). Honest: ch16 đủ tốt trên NuInsSeg; ch32 headline (cân PanNuke — cần verify). *(Efficiency story CÓ THỂ mạnh hơn: 0.5M vẫn tốt.)*
3. **★ Distilled vs GT-supervised — LẬT:** supervised **0.789 > distilled 0.750** (+ Winkler 92.8<95.4 + MAE 14.23<14.72) → supervised đè distilled MỌI trục. Số cũ "distilled 0.753 > supervised 0.711" = **leaky, BỎ**.
   ⟹ **Distillation KHÔNG nâng reliability so với dùng GT mask.** Giá trị distillation = **LABEL-EFFICIENCY** (reliability *cạnh tranh* mà KHÔNG cần mask), khớp core A. §4.8 = "competitive without masks", KHÔNG phải "better than masks". **KHÔNG đưa distilled>supervised lên bảng chính.**

### 4.9 Hardening A1–A6
A1 coverage-curve 4α (grouping≥global mọi α); A3 per-organ Wilson CI (undercoverage = 1 mô khó + nhiễu mẫu nhỏ, không systematic);
A5 σ-analysis (corr(σ,|e|)+0.40/+0.43, z-std PanNuke 1.01 calibrated); A2 σ-mode ablation (Poisson NLL 4.21 < NB/raw 4.58);
A4 latency 1.87ms/112MB VRAM. *(A6 3-seed worst-org 0.78±0.02 → superseded bởi 5-seed leak-free 0.750±0.049.)*

---

## 5. VERDICT Q1 + 3 TRỤ

**ĐỦ submit Q1 methods/applied** nếu kể đúng 3 trụ (KHÔNG overclaim "thắng mọi metric"):
1. **Label-efficient distillation** — foundation teacher + count-label, KHÔNG cần mask (khác mọi peer segmentation). **Distilled đạt reliability CẠNH TRANH (hơi dưới) supervised MÀ không cần mask** (§4.8: distilled 0.750 vs supervised 0.789) — value = label-efficiency, KHÔNG phải "hơn supervised".
2. **Distributional UQ** — calibrated (μ,σ), interval theo mô, transfer cross-dataset. Không peer distilled nào có.
3. **Efficiency** — 1.935M, nhỏ nhất có UQ.
*Accuracy = "cạnh tranh cho ngân sách nhãn", KHÔNG phải điểm bán (1.9M density-counter không đè SOTA: in-domain PanNuke 1.72× MAE vs NuLite-12M).*

**Peer (Related Work, chỉ CITE):** HoVer-unet (ISBI24), 9M H-Optimus student (2502.19217, citation vàng), RCKD — đều distilled/lightweight nhưng **KHÔNG UQ**.

**Rủi ro:** (i) ★ UQ-floor 5-seed VERIFIED: **R2 xếp ~4/5, CQR (cùng 1 model) nhỉnh cả worst-org lẫn Winkler** (§4.3) → **KHÔNG lead bằng UQ**; UQ = "distributional (μ,σ) 1-forward, đè MC-Dropout, ngang Ensemble ở 1/5 compute, cho phép transfer N5" — competitive-not-best; (ii) ✅ ĐÃ 3 dataset (NuInsSeg + PanNuke + CryoNuSeg transfer, §4.4); MoNuSAC = boundary honest (scale-gap 4×); (iii) NuInsSeg nhỏ/nhiễu → claim subgroup mềm. **Rủi ro lớn nhất = FRAMING, không phải thiếu thí nghiệm.**

---

## 6. NEXT STEPS (thứ tự)
1. ✅ R2 NuInsSeg 5-seed + backup kaggle.
2. ✅ Re-eval CondConf/PCP 5-seed matched.
3. ✅ **UQ-floor regen 5-seed** (2026-07-18) — 20 pkl `work/uq_{method}_nuinsseg_s{42..46}.pkl`, `aggregate_uqfloor.py`. Số §4.3. **R2 ~4/5 → UQ không phải trục bán.** Backup kaggle `sam3-paper2-uqkd`.
4. ✅ **KD 5-seed + per-image significance** (2026-07-18) — 5 pkl `work/student_kd_nuinsseg_cv5_s{42..46}.pkl`, `aggregate_kd.py`/`aggregate_r2.py`. Số same-scheme §4.1. **N4 CLOSED (reframe honest: PB-σ chỉ sập global).** Backup kaggle `sam3-paper2-uqkd`.
5. ✅ **Dataset 3 = CryoNuSeg transfer** (2026-07-18) — NuInsSeg→CryoNuSeg: marg.cov 0.967, Winkler 503, MAE 73 (σ transfer sống). MoNuSAC bỏ (co nhân 4× → μ sập; ghi boundary). `prep_cryonuseg_counts.py` + `xfer_nuinsseg2cryonuseg.pkl`, backup kaggle.
6. ✅ **Ablations §4.8 re-run leak-free 5-seed** (2026-07-18) — `aggregate_ablations.py`. **3 claim cũ LẬT:** supervised>distilled (0.789>0.750), ch16>ch32 NuInsSeg, detach yếu. §4.8 reframe honest (distillation=label-efficiency KHÔNG phải hơn-supervised). Backup kaggle.
7. → **VIẾT manuscript**: đóng gói "Distributional Count Distillation under mean-variance optimization conflict", KHÔNG claim PB-JCI (=P1); hình ~4–5.

**Baseline = code official (không tự chế):** CondConf (Gibbs–Cherian–Candès JRSS-B 2025), PCP (Zhang–Candès 2024), R2CCP (Guha ICLR 2024),
CPCP (ICML 2026), UQ floor (MC-Dropout/Ensemble/CQR/CHDQR); CellViT/LKCell/NuLite/LSP-DETR/PathoSAM. Dùng đúng checkpoint official.
Runbook chi tiết vast: xem [RUN_LIST_STRENGTHENING.md](RUN_LIST_STRENGTHENING.md).

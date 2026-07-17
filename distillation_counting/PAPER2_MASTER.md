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
| **KD (cả 2 dataset)** | ❌ pkl MẤT, số cũ CHƯA verify | cần retrain (raw NuInsSeg + PathoSAM) — **đừng cite tới khi có** |
| **UQ-floor NuInsSeg** (Ensemble/CQR/CHDQR/MC-Dropout) | ⚠️ pkl cũ mất | cần regen 5-seed (unblocked, density cache) |
| **Baseline PanNuke (8c)** | ✅ chạy trên pkl R2 còn sống | giữ nguyên |
| pkl R2 5-seed | ✅ backup kaggle `hipinhththu/sam3-r2-nuinsseg-seeds` | — |

**Provenance (2026-07-17):** pkl NuInsSeg canonical (`student_r2_nuinsseg_cv5_poisson.pkl` + KD) đã MẤT (không ở vast/kaggle).
Bản `_feat` còn sót thiếu `--detach_mu` → số sai. Đã retrain R2 5-seed đúng config. **Mọi "0.773" cũ = seed đơn** (nằm trong dải
0.70–0.82); số chính thức = **0.750±0.049**. PanNuke KHÔNG ảnh hưởng (nhiều ảnh/mô → worst-org ổn định; NuInsSeg ít ảnh/organ → nhạy seed = lý do phải multi-seed).

**★ TEACHER-LEAK — split chính xác (verify từ CODE patho-sam, 2026-07-17 — [[pathosam-training-data]]):** PathoSAM train PanNuke
**fold_1+2, chừa fold_3** → **fold_3 SẠCH (teacher held-out); fold_1/2 teacher-in-domain.** Paper 2 nên **headline PanNuke fold_3 (0.905)**;
f1/f2 (~0.906-0.908, ~bằng → leak không thổi phồng) ghi kèm. **NuInsSeg = OOD (không train) = anchor generalization sạch.**
Paper 1 test fold_3 → sạch, colon-exclusion đúng. Xem §2.5. (README cũ "not PanNuke" SAI, đã sửa.)

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

> ⚠️ **TEACHER-LEAK — split PanNuke CHÍNH XÁC (verify từ CODE patho-sam, 2026-07-17 — xem [[pathosam-training-data]]):**
> PathoSAM train PanNuke **fold_1+fold_2**, chừa **fold_3** (`get_generalist_datasets.py:80` = `folds=["fold_1","fold_2"]`; `dataloaders.py:149` = `folds=["fold_3"]`).
> → **PanNuke fold_3 = SẠCH (teacher held-out); fold_1/fold_2 = teacher-in-domain (leaky).**
> Protocol Paper 2 xoay cả 3 fold → **chỉ vòng test-fold_3 sạch**. Số f1 0.908/f2 0.906/**f3 0.905** ~bằng nhau → leak KHÔNG thổi phồng,
> **NHƯNG headline nên lấy fold_3 (0.905) làm số SẠCH**, f1/f2 ghi kèm (robustness/in-domain).
> **NuInsSeg = eval out-of-domain (KHÔNG train) = CLEAN OOD = anchor generalization thật.** Mọi claim generalization của Paper 2 tựa NuInsSeg.
> (Paper 1 test đúng fold_3 → sạch, colon-exclusion đúng. Đính chính: "cả PanNuke leaky" là quá tay — chỉ fold_1/2 in-domain.)

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
| **N4** | **learned-σ > score-σ khi nén 1.9M** (R2 vs KD) | phát hiện empirical | Cơ chế σ của P1 THUA khi ép vào student tí hon → phải HỌC σ. *(significance = per-image paired test, CHƯA chạy — cần KD)* |
| **N5** | **conditional coverage TRANSFER cross-dataset dù MAE không** | phát hiện empirical | worst-org NuInsSeg→PanNuke 0.897≈in-domain 0.906 → tin cậy generalize độc lập với accuracy điểm |

**Ranh giới:** khung conformal/PB-JCI = P1 (cite). Density-counting, U-Net = cũ. UQ floor = code người khác làm mốc. **Gốc = N1–N5.**

---

## 4. RESULTS (honest, current — 2026-07-17)

### 4.1 R2 vs KD (cùng student 1.9M, 3 trục)
> ⚠️ KD pkl MẤT → cột KD = số cũ chưa re-verify. Significance ĐÚNG = **per-image paired Wilcoxon / bootstrap** (số p=1.9e−6 cũ = seed-based pseudoreplication, BỎ).

| Dataset | Winkler R2 | MAE R2 | worst-org R2 | KD (chưa re-verify) |
|---|---|---|---|---|
| **PanNuke** (no-colon, 3-fold) ✅ | **19.28** | **3.36** | **0.906** (mondrian) | 23.7 / 3.94 / 0.739 |
| **NuInsSeg** (cross-fit, 5-seed) ✅ | **95.4±11.9** | **14.7±1.7** | **0.750±0.049** (cluster) | worst 0.282 |

PanNuke per-fold worst-org: f1 0.908 / f2 0.906 / f3 0.905 (0/18 mỗi fold). NuInsSeg 5-seed worst-org: [0.701,0.764,0.701,0.817,0.767].

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

### 4.3 UQ-floor (cùng student, trục reliability) — ⚠️ NuInsSeg cần regen 5-seed
PanNuke worst-org: R2 **0.906** (tie cao nhất), Ensemble 0.901, CQR 0.904, CHDQR 0.897, MC-Dropout 0.901.
NuInsSeg (⚠️ số pkl cũ, chờ regen): Ensemble 79.0/0.760, CQR 88.6/0.808, CHDQR 74.7/0.689, MC-Dropout **152.0/0.806** (thua rõ Winkler).
→ R2 **ngang tầm UQ hiện đại + rẻ nhất (1 model)**; MC-Dropout thua rõ. *(KHÔNG claim "best UQ" — Ensemble/CQR cạnh tranh; claim "comparable reliability at lowest compute".)*

### 4.4 Cross-dataset transfer (N5)
| Transfer | scheme | MAE | Winkler | worst-org | #under |
|---|---|---|---|---|---|
| NuInsSeg → PanNuke | mondrian | 19.90 | 97.21 | **0.897** | **0/18** |
| PanNuke → NuInsSeg | cluster | 44.88 | 214.83 | 0.685 | 4/27 |

**Conditional coverage TRANSFER**: NuInsSeg→PanNuke worst-org 0.897 ≈ in-domain 0.906. σ distilled vẫn informative dưới shift (chiều khó: cluster kéo worst 0.42→0.685, Winkler 564→215). MAE KHÔNG transfer (lệch thang count) — ghi trung thực.

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

### 4.8 Ablations — ⚠️ số single-split (leaky), finding định tính giữ, RE-RUN leak-free cho manuscript
- **detach_mu:** coupled NLL hỏng MAE (~18) → detach lấy lại MAE thấp + worst-org cao đồng thời (N3).
- **Compression sweep:** ch=16(~0.5M)/32(~1.9M)/64(~7.7M) → **ch=32 sweet spot**; ngay ch=16 vẫn thắng KD.
- **★ Distilled vs GT-supervised (same student):** distilled worst-org **0.753 > supervised 0.711** → teacher foundation nâng *conditional reliability*, không chỉ bắt chước nhãn (+ dùng được nơi không nhãn). **Bằng chứng "distillation đáng giá"** → đưa lên bảng chính.

### 4.9 Hardening A1–A6
A1 coverage-curve 4α (grouping≥global mọi α); A3 per-organ Wilson CI (undercoverage = 1 mô khó + nhiễu mẫu nhỏ, không systematic);
A5 σ-analysis (corr(σ,|e|)+0.40/+0.43, z-std PanNuke 1.01 calibrated); A2 σ-mode ablation (Poisson NLL 4.21 < NB/raw 4.58);
A4 latency 1.87ms/112MB VRAM. *(A6 3-seed worst-org 0.78±0.02 → superseded bởi 5-seed leak-free 0.750±0.049.)*

---

## 5. VERDICT Q1 + 3 TRỤ

**ĐỦ submit Q1 methods/applied** nếu kể đúng 3 trụ (KHÔNG overclaim "thắng mọi metric"):
1. **Label-efficient distillation** — foundation teacher + count-label, KHÔNG cần mask (khác mọi peer segmentation). Distilled≈supervised (4.8).
2. **Distributional UQ** — calibrated (μ,σ), interval theo mô, transfer cross-dataset. Không peer distilled nào có.
3. **Efficiency** — 1.935M, nhỏ nhất có UQ.
*Accuracy = "cạnh tranh cho ngân sách nhãn", KHÔNG phải điểm bán (1.9M density-counter không đè SOTA: in-domain PanNuke 1.72× MAE vs NuLite-12M).*

**Peer (Related Work, chỉ CITE):** HoVer-unet (ISBI24), 9M H-Optimus student (2502.19217, citation vàng), RCKD — đều distilled/lightweight nhưng **KHÔNG UQ**.

**Rủi ro:** (i) R2 không trội tuyệt đối interval (Ensemble/CQR ngang) → framing "trustworthy ở giá rẻ"; (ii) mới 2 dataset (top-tier đòi ≥3 → MoNuSAC sẵn); (iii) NuInsSeg nhỏ/nhiễu → claim subgroup mềm. **Rủi ro lớn nhất = FRAMING, không phải thiếu thí nghiệm.**

---

## 6. NEXT STEPS (thứ tự)
1. ✅ R2 NuInsSeg 5-seed + backup kaggle.
2. ✅ Re-eval CondConf/PCP 5-seed matched.
3. ⬜ **UQ-floor regen** (Ensemble/CQR/CHDQR/MC-Dropout) 5-seed — UNBLOCKED (density cache):
   `baselines_uq.py --method $M --dataset nuinsseg --kfold 5 --seed $S`.
4. ⬜ **KD 5-seed** — cần raw NuInsSeg + rebuild PathoSAM → `distill_student_nuinsseg.py --kfold 5 --lambda_kd 1.0 --seed $S` → **per-image significance** `eval_r2_grouped.py --per_image_test` (test code mới).
5. ⬜ **MoNuSAC UQ-transfer** (dataset 3) — `eval_cross_dataset.py --train_dataset pannuke → predict MoNuSAC`.
6. ⬜ Ablations 4.8 re-run leak-free (thay số single-split).
7. → **VIẾT manuscript**: đóng gói "Distributional Count Distillation under mean-variance optimization conflict", KHÔNG claim PB-JCI (=P1); hình ~4–5.

**Baseline = code official (không tự chế):** CondConf (Gibbs–Cherian–Candès JRSS-B 2025), PCP (Zhang–Candès 2024), R2CCP (Guha ICLR 2024),
CPCP (ICML 2026), UQ floor (MC-Dropout/Ensemble/CQR/CHDQR); CellViT/LKCell/NuLite/LSP-DETR/PathoSAM. Dùng đúng checkpoint official.
Runbook chi tiết vast: xem [RUN_LIST_STRENGTHENING.md](RUN_LIST_STRENGTHENING.md).

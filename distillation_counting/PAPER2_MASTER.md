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
| **Label-efficiency frontier** | ✅ EMPIRICAL (2026-07-19, §4.10) | distilled↔supervised head-to-head: **ngang coverage+MAE ở cùng số ảnh, distilled chỉ cần count ≈ 5–10× rẻ hơn mask** (K=52.8 nhân/ảnh). Đóng lỗ "tại sao distill". `label_efficiency{,_both}.py` |
| **Multi-teacher committee (Hướng A)** | ⏳ PROBE xong, thí nghiệm đầy đủ CHƯA (2026-07-19, §4.11) | 4 teacher (PathoSAM+SAM3+NuLite+LKCell): consensus KHÔNG hạ accuracy (CNN teacher đếm thiếu OOD); **disagreement-σ +0.65 ngang learned-σ** (dao động 0.49→0.41→0.65, chưa chắc). CÒN: train σ-target=disagreement, so coverage. `probe_multiteacher_full.py` |
| **Baseline PanNuke (8c)** | ✅ chạy trên pkl R2 còn sống | giữ nguyên |
| pkl R2 5-seed | ✅ backup kaggle `hipinhththu/sam3-r2-nuinsseg-seeds` | — |

**Provenance (2026-07-17):** pkl NuInsSeg canonical (`student_r2_nuinsseg_cv5_poisson.pkl` + KD) đã MẤT (không ở vast/kaggle).
Bản `_feat` còn sót thiếu `--detach_mu` → số sai. Đã retrain R2 5-seed đúng config. **Mọi "0.773" cũ = seed đơn** (nằm trong dải
0.70–0.82); số chính thức = **0.750±0.049**. PanNuke KHÔNG ảnh hưởng (nhiều ảnh/mô → worst-org ổn định; NuInsSeg ít ảnh/organ → nhạy seed = lý do phải multi-seed).

**★ BACKUP INVENTORY (2026-07-18 — mọi artifact đắt, đừng rebuild/mất):**
| kaggle dataset | chứa |
|---|---|
| `hipinhththu/sam3-r2-nuinsseg-seeds` | R2 NuInsSeg 5-seed pkl (headline 0.750) |
| `hipinhththu/sam3-paper2-uqkd` | 20 UQ-floor + 5 KD + 3 ch16-PanNuke pkl + teacher_targets_nuinsseg + teacher_density_nuinsseg + xfer_cryonuseg + prep_cryonuseg |
| `hipinhththu/sam3-pannuke-density-cache` | teacher_density_pannuke_f123.pkl (3.6GB, PathoSAM output — rebuild rất tốn) |
| `hipinhththu/monusac` | raw MoNuSAC (bỏ dùng, giữ tham khảo) |
| git repo | PAPER2_MASTER.md + mọi script (`aggregate_*.py`, `prep_cryonuseg_counts.py`) + `data/pathosam_nuinsseg_preds.pkl` (teacher-PB) |

**★ SPLIT PanNuke của PathoSAM (CHỐT CỨNG, verify từ CODE patho-sam — [[pathosam-training-data]]):**
**PathoSAM train PanNuke FOLD_1 + FOLD_2, TEST trên FOLD_3** (`get_generalist_datasets.py:80` train fold_1+2; `dataloaders.py:149` eval fold_3).
⟹ **Đánh giá PanNuke trên FOLD_3 = SẠCH (leak-free)** — dùng dataset KHÔNG bị leak. **Paper 2 lấy fold_3 làm số PanNuke chính (worst-org 0.905).**
**NuInsSeg + MoNuSAC không nằm trong training PathoSAM → clean OOD; NuInsSeg = anchor generalization.** Paper 1 cũng test fold_3 → sạch, loại colon đúng.

---

## 0.5 ★★★ PIVOTAL (2026-07-22) — DISTILLATION KHÔNG GIÚP CHO ĐẾM → REPOSITION

**Bối cảnh:** chẩn đoán vì sao PACT (TinyUNet, teacher-distill) accuracy chỉ ~0.78. Kết quả **lật trục "distillation = linh hồn"**.

**Bảng chẩn đoán (leak-free, global R² qua `compute_r2_counting.py`; efflite0 = EfficientNet-Lite0 pretrained ~3.6M, backbone hỗ trợ sẵn trong `distill_student_r2.py --backbone`):**

| Dataset | Backbone | count-only (w_d=0,w_c=1) | teacher-distill (w_d=1,w_c=0.01) | GT-density (target hoàn hảo) |
|---|---|---|---|---|
| NuInsSeg | TinyUNet (5-seed) | 0.767 | 0.786 | 0.810±0.023 |
| NuInsSeg | efflite0 (**1 seed**) | **0.925** | **0.512** ⬇️ | 0.881 |
| PanNuke f3 | efflite0 (**1 seed**) | 0.928 | 0.935 | — |
| *(teacher PathoSAM zero-shot NuInsSeg = 0.711)* | | | | |

**Kết luận cứng (logic bịt đường cãi):**
1. **GT-density hoàn hảo (0.881) < count-only (0.925)** → bắt chước density map, DÙ target hoàn hảo, **không thể bằng** dạy thẳng count. Ceiling của distill = chất lượng density, mà ceiling đó vẫn dưới count-only.
2. Teacher thật ≤ GT: **in-domain** (PanNuke, density≈GT) → distill **HÒA** count-only (0.935≈0.928); **OOD** (NuInsSeg, density rác) → distill **SẬP** (0.512≪0.925).
3. ⟹ **Distill giỏi nhất = hòa, dở nhất = hại, KHÔNG BAO GIỜ thắng count-only.** Nguyên nhân 0.512 = **teacher tồi (OOD)**, KHÔNG phải student yếu (efflite0 đạt 0.881 với GT, 0.925 với count trực tiếp).
4. **"Distill foundation → tiny counter" chết ở trục accuracy.** Phản biện trực tiếp tiền đề CellGenNet/H-Optimus (đều distill).

**⚠️ IN-DOMAIN caveat (lỗi đã mắc, phải nhớ):** efflite0 0.925 là **in-domain** (train NuInsSeg count). KHÔNG được claim "đè foundation 640M" (teacher zero-shot) — bất đối xứng = critique A. So fair phải là **các counter nhẹ khác cũng in-domain count**, hoặc efflite0 eval **zero-shot cross-dataset**.

**→ REPOSITION (chốt hướng với user 2026-07-22):** xây **MODEL MỚI count-only sạch** (backbone nhẹ pretrained + đầu đếm trực tiếp + đầu σ calibrated, CHỈ nhãn count, KHÔNG teacher/mask). "Distill" **rời khỏi method**, chỉ còn là **ablation/động-cơ** ("thử distill foundation — không giúp/hại → nên thiết kế gọn không teacher"). Novelty gánh bởi: **(i) negative-result distill-là-thừa** (cái mới nhất, ngược field) + **(ii) đầu UQ calibrated/transfer** + (iii) label-efficiency + chặt chẽ đa dataset. Model một mình = incremental → PHẢI đi kèm (i).

### ⏳ PHẢI CHẠY để firm-up claim "distill không giúp" lên chuẩn Q1 (chưa làm)
- **(a) [BẮT BUỘC] efflite0 5-seed** cho cả 3 điều kiện (count-only / teacher-distill / GT-density) trên NuInsSeg + PanNuke → hiện mới **1 seed** (±0.000), cần mean±sd loại may rủi. Backbone efflite0, `--kfold 5` (NuInsSeg) / `--test_fold 3` (PanNuke), lặp `--seed {42..46}`.
- **(b) [NÊN] CryoNuSeg teacher-OOD** — điểm thứ 3 cho trục "teacher OOD → distill hại" (hiện chỉ 2 điểm: NuInsSeg hại, PanNuke hòa). Teacher PathoSAM cũng OOD trên CryoNuSeg → kỳ vọng distill < count-only, xác nhận xu hướng. Cache/prep: `prep_cryonuseg_counts.py`.
- **(c) [GHI PHẠM VI]** chỉ 1 teacher (PathoSAM) → scope claim "một foundation pathology mạnh", hoặc thử teacher thứ 2.

**Artifacts đã tạo (chẩn đoán, chưa backup):** `work/baseline_countonly_efflite0.pkl` (0.925), `pact_efflite0_distill.pkl` (0.512), `efflite0_gtdensity.pkl` (0.881), `baseline_countonly_tinyunet.pkl` (0.767), `efflite0_pan_countonly_f3.pkl` (0.928), `efflite0_pan_distill_f3.pkl` (0.935). **→ backup Kaggle khi chạy 5-seed.** TinyUNet GT-density 5-seed = pkl `student_r2_nuinsseg_cv5_supervised_s*` (dataset `sam3-paper2-uqkd`).

### 0.6 ★ REORIENT (2026-07-22) — hướng FEATURE-DISTILL "phễu" (gate DƯƠNG đầu tiên)

**Chuỗi chẩn đoán loại sạch:** distill-OUTPUT bác; crowding-module tiền đề bác (`premise_tiling_test`: tiles TỆ hơn whole, whole đã gần đúng ở bin dày); scale-robustness = nghiệm tầm thường (MPP). ⟹ **accuracy in-domain ĐÃ bão hoà**, không còn khe method.

**Insight cứu distill (trả lời "student không thừa hưởng gì từ teacher"):** mình đã distill **OUTPUT (density)** của teacher OOD (lỗi) — phải distill **REPRESENTATION (feature)**. Teacher đếm dở nhưng **đặc trưng thị giác pathology giàu**.

**★ GATE DƯƠNG (2026-07-22, `gate_feature_surplus.py`, linear-probe leave-organ-out, Phikon ViT-B vs ImageNet ViT-B, CÙNG kiến trúc):**
| nhãn | ImageNet R² | Phikon R² | Δ |
|---|---|---|---|
| 100% | 0.801 | **0.875** | +0.074 |
| 50% | 0.738 | 0.868 | +0.130 |
| 25% | 0.688 | 0.838 | **+0.150** |
| 10% | 0.670 | 0.788 | +0.118 |

**Feature pathology CÓ thặng dư** — lớn nhất ở **low-label + shift (mô chưa thấy)**, thu hẹp khi đủ nhãn. ⟹ hướng **feature-distill "phễu"** (distill feature, cổng lọc lỗi teacher = confidence × count-agreement) **SỐNG**.

**⚠️ Honest:** (1) cải thiện là **R²/generalization, KHÔNG phải MAE** (MAE ~ngang/hơi tệ) → claim = generalize/label-efficiency, không phải sai-số-tuyệt-đối. (2) đây là **upper-bound** (probe trên Phikon 86M đông lạnh, CHƯA phải student tí hon). (3) **bẫy capacity-gap** khi nhồi 86M→efflite0 3.6M.

**⏳ Bước tiếp (chưa làm):** transferability — efflite0 + feature-distill-Phikon có thắng efflite0-ImageNet ở low-label/shift không? (`transfer_feature_distill.py`). Confidence teacher cho phễu = **đã cache** `teacher_targets_nuinsseg.pkl`. Phikon truy cập HF `owkin/phikon` (Internet ON; ⚠️ `pip install` làm hỏng torch-CUDA → factory-reset, đừng pip).

### 0.7 PHỄU feature-distill (2026-07-22, `transfer_feature_distill.py`, 25% nhãn, leave-organ-out) — 8-SEED LẬT 3-SEED: KHÔNG SIGNIFICANT

Student = efflite0 (DensitySigmaUNet, count-only density). **Phễu 2 cổng (chỉ nhãn count):** density-gate spatial (density student detach → dồn distill vào vùng nhân) + reliability-gate per-image (Phikon-probe-count vs GT, 2-fold cross-fit → hạ ảnh teacher kém tin).

**★ KẾT QUẢ FIRM 8-seed + paired-Wilcoxon (w_feat=30, ablate 5 mode):**

| mode | R² mean±sd | Δ vs count | #thắng | p(Wilcoxon) |
|---|---|---|---|---|
| count-only | **+0.688±0.204** | — | — | — |
| naive | +0.614±0.154 | **−0.074** | 1/8 | 0.20 |
| density-only | +0.708±0.216 | +0.020 | 4/8 | 0.84 |
| reliab-only | +0.643±0.191 | **−0.046** | 1/8 | 0.20 |
| gated-both | +0.744±0.174 | +0.056 | 5/8 | **0.74** |

**⚠️ 3-seed (gated 0.812 > count 0.751) là LẠC QUAN MẪU-NHỎ — 8 seed dội lại: gated-both 5/8, p=0.74 → KHÔNG có ý nghĩa thống kê.** Cùng loại dương-tính-giả multi-seed từng bắt ở naive.

**Cái GÌ vững (tái lập):**
- **naive feature-distill HẠI** (−0.074, 1/8) — nhồi feature teacher thô vào student 3.6M làm tệ. Vững.
- **reliab-gate một mình cũng HẠI** (−0.046, 1/8) — cổng tin-cậy per-image hạ nhầm ảnh tốt (w_feat=30 ép quá?).
- **density-gate GỠ được cái hại** (−0.074→+0.020) nhưng **không thêm giá trị** vs không-distill.
- gated-both +0.056 nhưng ±0.20 nuốt trọn → không phân biệt được với 0.

**Nghĩa khoa học (finding đáng giá):** thặng dư feature pathology-FM **CÓ THẬT** (gate probe §0.6: Phikon>ImageNet) **nhưng KHÔNG chuyển được vào student 3.6M**. Cổng lọc chỉ *khử hại* của distill, **không thu hồi surplus**. Root cause = **capacity gap** ViT-86M (probe đọc surplus) ↔ student 3.6M. = negative **giải thích được**, khớp arc rigor cả paper (distillation KHÔNG phải trục accuracy — xem [[paper2-core-A-decision]], [[paper2-q1-hardening]]).

**Quét w_feat∈{5,10,30} × 4 seed (2026-07-22):** gated-both Δ = +0.062 / −0.092 / −0.149 (ĐỔI DẤU quanh hyperparam) → không w_feat nào cứu → **negative robust**. Cũng lộ bug: count-only (không dùng w_feat) vẫn nhảy 0.658/0.703/0.736 giữa các lần chạy → `np.random.shuffle` global chưa seed → sàn nhiễu cao (đã fix ở script pseudo). Artifacts: script git; chưa pkl backup (train nhanh, tái tạo được).

### 0.7b PIVOT pseudo-label semi-supervised (2026-07-23, `pseudo_label_semisup.py`) — TREND DƯƠNG NHƯNG CHƯA SIGNIFICANT (NuInsSeg thiếu power)

Sau khi funnel chết: **giữ tốt/thay xấu** — probe Phikon làm **teacher OUTPUT** (teacher THẬT mạnh hơn student, điều kiện distill hoạt động) thay vì mimic feature (đấu capacity); confidence-gate đổi việc = **chọn pseudo-label** thay vì hạ nhãn thật. Count-only, 25% nhãn, leave-organ-out, 8 seed.

| version | teacher−student | pseudo vùng-shift | **Δ student** | p | #thắng |
|---|---|---|---|---|---|
| v1 (teacher ensemble yếu, pool no-shift) | +0.042 | — | +0.051 | 0.74 | 4/8 |
| **v2 (teacher full-data mạnh 0.755, transductive pool phủ shift)** | **+0.116** | **+0.530** | **+0.079** | **0.38** | 5/8 |

**Đọc honest:** sửa 2 lỗi thiết kế (teacher yếu + pool bỏ đói vùng shift) → effect TĂNG ĐỀU (+0.05→+0.08), hướng nhất quán, teacher giờ mạnh hơn thật, pseudo phủ được mô-mới. **NHƯNG p=0.38 — chưa significant.** Rào cản = **phương sai NuInsSeg** (labeled-only ±0.309, pseudo-shift ±0.506; ít ảnh/mô → leave-organ-out nhảy) **nuốt** effect +0.08. = **thiếu power của dataset, KHÔNG phải lỗi method.**

**confidence-gate: VÔ DỤNG** (v1: pseudo-all == pseudo-confident = +0.051; SANITY confident không chọn pseudo tốt hơn) → **bỏ**, ghi làm ablation honest.

**Stop-rule pre-register:** p<0.05 → method thật; không → pseudo = mục phụ (ổn định label-efficiency), DỪNG. **v2 p=0.38 → theo rule = DỪNG trên NuInsSeg.** Đề xuất mở: 1 power-check trên **PanNuke** (nhiều ảnh/mô → variance nhỏ, dataset headline) — KHÔNG phải chase (sửa power cho vấn đề đã chẩn đoán là power), dứt điểm dương/âm. Chờ cô chọn (A) fold ngay / (B) 1 lần PanNuke.

**Ghi nhận positive thật (dù chưa sig):** pseudo-label **giảm phương sai student** (v1: 0.279→0.136) = ổn định hoá — property có thật, dùng được cho mục label-efficiency phụ.

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

**★ FRAMING KHOA HỌC (chốt 2026-07-18 — ĐÃ SỬA 2 lần, đây là bản đúng):**
> ⚠️ **BỎ framing "cái gì sống sót qua nén — đếm hay bất định?"** (từng ghi, SAI): **PathoSAM KHÔNG có bất định nội tại** để "sống sót". PathoSAM là segmentation → xuất mask + điểm detection sᵢ; **PB-σ (√Σsᵢ(1−sᵢ)) là CẤU TRÚC của Paper 1** dựng từ điểm detection, KHÔNG phải output teacher. Và **cả R2 lẫn KD đều KHÔNG distill bất định của teacher**: R2 **HỌC** σ từ GT; KD áp công thức PB lên điểm của **CHÍNH student**. → Không có "teacher uncertainty" trong pipeline.
>
> **Framing ĐÚNG:** *"Chưng cất khả năng ĐẾM của pathology foundation model vào student 1.9M, trang bị đầu **phân phối đếm HỌC được** (μ,σ) calibrated. Bất định KHÔNG đến từ teacher — nó được HỌC (N2 Poisson-anchored); learned-σ đáng tin hơn công thức analytic PB-σ (Paper 1) khi model NHỎ (N4 — vì điểm detection của model nhỏ không đủ calibrated cho cấu trúc PB), và learned-σ này TRANSFER cross-dataset (N5)."*
> **Câu bán sạch (không overclaim):** *"a lightweight distilled counter that provides **learned, calibrated, transferable** count uncertainty — where the analytic Poisson-Binomial uncertainty degrades at small scale."*

**⚠️ CẢNH BÁO THUẬT NGỮ (tránh reject):** KHÔNG gọi phương pháp là *"distribution distillation"* — khái niệm ĐÃ CÓ TÊN (Malinin et al., *Ensemble Distribution Distillation*, ICLR 2020). **σ HỌC từ GT (§2.3), KHÔNG distill từ teacher**; baseline KD (áp PB) THUA learned-σ (N4). Phát biểu: *"distilled counter + **learned** calibrated distributional head"*.

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

**★ Triage / selective-prediction (2026-07-18, `analysis_triage.py`, 0-compute trên pkl 5-seed) — cược top-tier THẤT BẠI, ghi honest:**
Đo E-AURC (chất lượng σ-ranking, tách base-MAE): Ensemble **2.78** (5× compute) < CHDQR 3.49 < CQR 3.89 < **R2 3.96** < MC-Dropout 5.77.
**R2 xếp 4/5** (cả AURC 8.60, MAE@80% 11.24). CQR/CHDQR cùng-1-model lọc lỗi tốt hơn R2; R2 chỉ hơn rõ MC-Dropout.
⟹ **σ của R2 KHÔNG enable triage tốt nhất** — lần thứ 3 xác nhận UQ competitive-not-best (sau UQ-floor + N4). **KHÔNG có hook top-tier qua UQ**; chốt Q1 tầm-trung, framing = distillation/label-efficiency/compression.

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

**★ 4.6b — BẢNG CHÍNH COHERENT (2026-07-20/21, `dump_counts.py` + `compute_r2_counting.py`, 665 ảnh, len-instances CÙNG THƯỚC; teacher=len-scores, PACT=Σdensity):** thay bảng cũ (§4.6 là `eval_heavy_count` KHÁC thước → đừng trộn).

| Method | Params | R² ↑ | MAE ↓ | RMSE ↓ | MAPE ↓ |
|---|---|---|---|---|---|
| CellViT-SAM-H (off-the-shelf, @1024 native) | 699.7M | 0.663 | 21.83 | 31.33 | 52.9% |
| LKCell-L (off-the-shelf) | 163.8M | 0.448 | 20.92 | 40.10 | 37.4% |
| NuLite-T (off-the-shelf) | 12.0M | 0.622 | 20.01 | 33.22 | 39.6% |
| PathoSAM teacher (zero-shot) | ~640M | 0.711 | 15.80 | 29.02 | **28.3%** |
| **PACT (ours, in-domain, 5-seed)** | **1.9M** | **0.786±0.052** | **14.74±1.53** | **24.81±3.03** | 47.6±3.4% |

✅ CellViT-SAM-H chạy @1024 native (fair): R² 0.663 (ở 256 chỉ 0.444 → 256 thiệt cho nó, đã bỏ). csv `cellvit_preds_1024.csv`.

**PACT dẫn R²+MAE+RMSE, nhỏ nhất 6–368×; teacher giữ MAPE.** LKCell (164M) thua cả NuLite (12M) OOD; không model dùng-sẵn nào (kể cả CellViT 699M) gần PACT → foundation nặng off-the-shelf không transfer tốt tương ứng kích thước. Đóng khung = **thích-nghi-rẻ (nhãn count) vs off-the-shelf**, KHÔNG "model giỏi hơn".

**★ 4.6c — Reviewer-hardening (2026-07-21, `stratified_error.py` + `significance_counting.py`, Kaggle):**
- **[B] Significance Bảng 1:** PACT 5-seed **ensemble** MAE 12.28 vs teacher 15.80 → **ΔMAE −3.52** (95%CI [−4.85,−2.24]), **paired-Wilcoxon per-ẢNH p=3.5e-4 \*\*\*** (n=665, KHÔNG per-seed). Single-model 14.74 cũng <15.80. → critique B CLOSED.
- **[C1] Lỗi theo tầng mật độ (PACT 5-seed, mean±sd):** Thấp(1-20,n167,GT̄10.6) MAE 6.78±0.28/MAPE **110.1±8.5%** · TB(21-50,n261) 9.82±0.71/28.4±2.0% · Cao(>50,n237) 25.78±3.74/**24.8±3.1%** · toàn cục 14.74±1.53/47.6±3.4%. **2 kết luận honest:** (1) MAPE-47.6% bị kéo bởi vùng ít nhân (mẫu số nhỏ) — đúng cơ chế C1; (2) **limitation THẬT**: bin Thấp PACT (6.78/110%) thua teacher (2.95/37.2% local) — density-sum over-count ảnh gần-trống; PACT thắng nhờ vùng đông (R²_cao 0.63). ⚠️ R² per-bin âm = range-restriction, KHÔNG báo. → critique C1 CLOSED + Limitations có nội dung.
- **[A-1] Baseline khác-kiến-trúc count-only** (efficientnet_lite0, w_density=0): ⏳ CHỜ CHẠY (`RUNBOOK_reviewer_ABC.md`). Kỳ vọng ≈ PACT (khớp §4.8 supervised≈distilled) → xác nhận lợi thế = in-domain count-label, KHÔNG phải kiến trúc. TUYỆT ĐỐI không claim "PACT arch superior".

**PROVENANCE csv (backup Kaggle dataset `hipinhththu/lkcell-nulite-preds`):**
- `nulite_preds.csv` (665) · `lkcell_preds_full.csv` (665) · CellViT `cellvit_preds_full.csv`@256 + `cellvit_preds_1024.csv`@1024 (đang chạy) · gt `gt_counts.csv`.
- ⚠️ csv cũ `lkcell_preds.csv` = **5 ảnh HỎNG** (dấu vết `--limit 5`), ĐỪNG dùng — đã thay bằng `_full`.
- Script SẠCH resume/flush/try-except: `dump_counts.py` (thay `dump_cellvit_counts.py`); baseline recent: `dump_instanseg.py`/`dump_cellpose.py`.
- Manuscript skeleton (lightweight-primary, UQ-phụ): `PAPER2_MANUSCRIPT_SKELETON.md`.

### 4.7 Annotation-cost (lá chắn label-efficiency)
| Method | Teacher | Nhãn TARGET cần | Độ mịn | Output | UQ | Params |
|---|---|---|---|---|---|---|
| **R2 (ours)** | frozen PathoSAM | **count-scalar/ảnh** (+density teacher=0 nhãn người) | **point** | count dist (μ,σ) | **✓** | **1.9M** |
| NuLite-T / CellViT-SAM-H | ImageNet / SAM | instance mask pixel + class | pixel | seg | ✗ | 12M / 700M |
| HoVer-unet / 9M-H-Optimus (distilled) | mask-trained teacher | (distill) output seg | pixel | seg | ✗ | ~/9M |

**★ Honesty caveat:** claim là về **YÊU CẦU giám sát của phương pháp** (task-head chỉ cần count-scalar, lấy được bằng point/dot-annotation),
KHÔNG phải "mình đã dùng nhãn rẻ hơn" (thí nghiệm này GT count vẫn lấy TỪ mask vì dataset có sẵn). Point-vs-pixel rẻ hơn nhiều lần = **cite** counting literature, KHÔNG bịa "100×".
vs peer **distilled** (HoVer-unet/9M): khác biệt KHÔNG phải label-cost mà là **task-head count-level + distributional UQ**.
**★ Bằng chứng thực nghiệm (2026-07-19): xem §4.10** — budget-frontier + head-to-head distilled↔supervised cho thấy **ngang coverage+MAE ở cùng số ảnh, distilled chỉ cần count-scalar (≈5–10× rẻ hơn mask)** → luận điểm này giờ CÓ số, không còn suông.
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
2. **Capacity 0.5M–1.9M ĐỀU tốt (ch32 KHÔNG phải "sweet spot" — ch16 ≥ ch32):** ✅ VERIFIED cả 2 dataset (2026-07-18):
   - NuInsSeg: **ch16 (~0.5M) thắng ch32** cả 3 trục (0.797/86.7/13.79 vs 0.750/95.4/14.72); ch64 tệ nhất (0.781/169/17.1).
   - PanNuke (mondrian avg 3 fold, `student_r2_pannuke_ch16_f{1,2,3}`): **ch16 HÒA ch32** — 0.906/18.18/3.35 vs 0.906/19.28/3.36 (ch16 nhỉnh Winkler; per-fold wo [0.907,0.903,0.909]).
   ⟹ **Method robust theo capacity; chạy tốt ở 0.5M = nén 1280× teacher 640M.** Giữ **ch32 làm primary** (mọi thí nghiệm khác chạy ở đó — không đổi headline vì chi phí re-run), báo ch16 như "scales down to 0.5M no loss" (efficiency + robustness). ch64 overfit (nhỏ-data) → capacity lớn KHÔNG giúp.
3. **★ Distilled vs GT-supervised — LẬT:** supervised **0.789 > distilled 0.750** (+ Winkler 92.8<95.4 + MAE 14.23<14.72) → supervised đè distilled MỌI trục. Số cũ "distilled 0.753 > supervised 0.711" = **leaky, BỎ**.
   ⟹ **Distillation KHÔNG nâng reliability so với dùng GT mask.** Giá trị distillation = **LABEL-EFFICIENCY** (reliability *cạnh tranh* mà KHÔNG cần mask), khớp core A. §4.8 = "competitive without masks", KHÔNG phải "better than masks". **KHÔNG đưa distilled>supervised lên bảng chính.**

### 4.9 Hardening A1–A6
A1 coverage-curve 4α (grouping≥global mọi α); A3 per-organ Wilson CI (undercoverage = 1 mô khó + nhiễu mẫu nhỏ, không systematic);
A5 σ-analysis (corr(σ,|e|)+0.40/+0.43, z-std PanNuke 1.01 calibrated); A2 σ-mode ablation (Poisson NLL 4.21 < NB/raw 4.58);
A4 latency 1.87ms/112MB VRAM. *(A6 3-seed worst-org 0.78±0.02 → superseded bởi 5-seed leak-free 0.750±0.049.)*

**A7 — Probe stain-perturbation uncertainty (2026-07-18, `probe_stain_uncertainty.py`, Kaggle GPU):** thử stain-TTA-variance (HED-jitter, Tellez 2019) làm **nguồn σ thay thế/bổ sung** → **variance LỚN (45.4% gt) nhưng KHÔNG informative**: corr(σ_stain,|err|)=**+0.17** << learned-σ corr **+0.65** (cùng test split). ⟹ **stain-sensitivity ≠ count-uncertainty**; distill teacher-stain-uncertainty KHÔNG motivated. **Design-justification cho learned heteroscedastic σ (N2)** + preempt câu hỏi reviewer *"sao không distill uncertainty của teacher / dùng stain-uncertainty?"*. Kết thúc mọi hướng add-on (UQ-floor/triage/stain-uncertainty đều âm) → paper = solid mid-Q1, viết.

### 4.10 ★ Label-efficiency frontier (EMPIRICAL — đóng lỗ "tại sao distill", 2026-07-19)
`label_efficiency.py` (piece 1 count-only) + `label_efficiency_both.py` (piece 2 distilled↔supervised head-to-head). NuInsSeg (665 ảnh, K=**52.8** nhân/ảnh mean, median 38, tổng 35,138), 80/20 single-split × 3 seed {0,1,2}, cluster n=5, ch32, 60 epoch, **grad-clip norm 5** (chặn β-NLL phân kỳ hiếm). Caches align qua **img-hash** (teacher & GT cache khác thứ tự build_index). **⚠️ Protocol khác bảng chính (single-split, calib-set lớn hơn → số tuyệt đối cao hơn §4.1/4.8, vd full-budget worst-org 0.84–0.89) → CHỈ đọc TƯƠNG ĐỐI giữa budget/nhánh, KHÔNG cross-compare với bảng chính.**

**Piece 1 — count-only frontier (chỉ distilled):**
| frac | n_lab | worst-org ↑ | MAE ↓ |
|---|---|---|---|
| 0.10 | 53 | 0.888±0.044 | 26.34±4.45 |
| 0.25 | 133 | 0.819±0.092 | 31.11±16.2 *(1 seed xui, n=133 nhỏ)* |
| 0.50 | 266 | 0.866±0.066 | 18.33±1.94 |
| 1.00 | 532 | 0.891±0.027 | 12.77±0.68 |

⟹ **worst-org PHẲNG mọi budget (kể cả 10% nhãn)** — conformal cho conditional coverage hợp lệ **gần-như-miễn-phí-nhãn**; MAE giảm đơn điệu theo nhãn (accuracy CẦN nhãn, coverage thì KHÔNG).

**Piece 2 — DISTILLED (teacher-density) vs SUPERVISED (GT-density) cùng ngân sách ẢNH:**
| budget | DISTILL worst / MAE | SUPERV worst / MAE | đọc |
|---|---|---|---|
| 10% (53) | 0.865 / **24.26** | 0.879 / 26.09 | distill nhỉnh MAE |
| 25% (133) | 0.858 / 21.85 | 0.824 / **18.49** | superv nhỉnh |
| 50% (266) | 0.840 / 20.34 | 0.830 / **17.73** | superv nhỉnh |
| 100% (532) | 0.843 / **14.12** | 0.897 / 14.61 | hòa (distill nhỉnh) |

⟹ **coverage KHÔNG phân biệt được** (chồng ±SD, ~0.82–0.90); **MAE supervised nhỉnh ~2-3 ở giữa (25-50%), HÒA ở 10%+100%** (full-budget distill 14.12<14.61). **≈ tương đương, KHÔNG winner nhất quán.**

**★ Luận điểm annotation-cost:** cùng số ảnh & ngang accuracy+coverage, nhưng nhãn TARGET/ảnh khác hẳn:
- **DISTILL:** teacher-density (0 nhãn người, PathoSAM unsup) + **1 count-scalar** ≈ dot-count 52.8×2.4s ≈ **127s/ảnh** (point cost, Bearman ECCV'16 "What's the Point").
- **SUPERV:** **mask 52.8 nhân** ≈ 52.8×(5–10× point) ≈ **640–1270s/ảnh**.
- Cả hai tỉ lệ K ⟹ **ratio = t_mask/t_point ≈ 5–10× bất kể K** → *distilled đạt coverage+MAE ngang supervised ở **1/5–1/10 chi phí annotation**, chỉ cần nhãn đếm mức-ảnh thay vì mask dày.* = **câu trả lời trực tiếp cho "tại sao distill".**

**★ Honesty:** (1) protocol single-split ≠ bảng chính → đọc TƯƠNG ĐỐI. (2) GT count ở đây vẫn lấy từ mask (dataset có sẵn) → chứng minh **YÊU CẦU giám sát của method** (count-level), KHÔNG phải "đã annotate rẻ hơn thật". (3) ratio **5–10×** dùng khoảng an toàn — point 2.4s/instance có Bearman; t_mask/nhân KHÔNG có citation chuẩn → **KHÔNG bịa 1 con số**. (4) 3 seed hơi mỏng + 0.25 MAE nhiễu 1 seed → nếu lên figure paper nên nâng 5 seed. Script: `label_efficiency_p1_result.txt` / `label_efficiency_p2_result.txt`. GT cache dựng lại 35s từ `ipateam/nuinsseg` (rẻ, không cần backup gấp).

### 4.11 ⏳ Multi-teacher committee (Hướng A — PROBE xong 2026-07-19, thí nghiệm đầy đủ CHƯA)
**Động cơ:** user muốn **model là đóng góp chính**. Ý tưởng: distill **hội đồng foundation model** (không chỉ 1 teacher), lấy **bất đồng giữa teacher làm σ epistemic distill được** — vá lỗ "PathoSAM không có bất định nội tại để distill" (với *committee* thì có). Script probe: `probe_multiteacher.py` (2-teacher local) + `probe_multiteacher_full.py` (3-4 teacher) + notebook `multiteacher_probe.ipynb`.

**4 teacher trên NuInsSeg 665 ảnh (count = #instance):** PathoSAM (MAE 15.80) + SAM3+A2-LoRA (15.45) [2 pkl sẵn có, `data/pathosam_nuinsseg_preds.pkl` + `weights/phase_E_nuinsseg_preds.pkl`] + **NuLite-T (20.01)** + **LKCell-L (20.92)** [dump off-the-shelf, `dump_cellvit_counts.py --nulite / --lkcell --no_tokens`, csv].

| # teacher | consensus-MAE | corr(std-disagreement, |lỗi|) |
|---|---|---|
| 2 (SAM×2) | **14.97** (<15.45, hạ nhẹ) | +0.49 |
| 3 (+NuLite) | 16.08 (tệ hơn) | +0.41 |
| 4 (+LKCell) | 17.07 (tệ hơn) | **+0.65** |

**★ 2 KẾT LUẬN (honest):**
1. **Consensus KHÔNG cải thiện accuracy** — 2 model UniRepLKNet (NuLite/LKCell) train PanNuke → đếm THIẾU trên OOD NuInsSeg (~32-35 vs gt 52.8) → kéo naive-mean xuống. ⟹ **μ phải lấy từ teacher tốt (SAM), KHÔNG trung bình cả 4.** Accuracy **KHÔNG phải trục** của hướng này.
2. **Disagreement-σ = tín hiệu epistemic THẬT, mạnh khi committee đủ đa dạng:** corr nhảy +0.49→+0.41→**+0.65** (2→3→4 teacher); ở 4 teacher (2 SAM + 2 CNN) **ngang learned-σ (+0.65)**. ⟹ σ = bất đồng hội đồng distill được, *khác loại* learned-σ (epistemic vs aleatoric) → **có thể trộn**.

**⚠️ CẢNH BÁO trung thực:** (a) corr **dao động không đơn điệu** (0.49→0.41→0.65) trên 665 mẫu + align xấp xỉ (organ,gt cho 2 pkl cũ) → **+0.65 CHƯA nên tin tuyệt đối**; số quyết định là **coverage/Winkler sau conformal**, KHÔNG phải corr. (b) corr đo so *consensus-error* (17.07, hơi tệ) → có thể thuận lợi. (c) accuracy-win KHÔNG có — giá trị (nếu có) nằm ở **chất lượng/novelty của uncertainty**, không phải đếm giỏi hơn.

**→ HƯỚNG TIẾP (thí nghiệm đầy đủ, chưa làm):** train student **μ distill từ teacher tốt**, **σ-target = committee disagreement (4 teacher)**; so **coverage/Winkler** của σ-disagreement vs learned-σ (§4.3) vs PB-σ (§4.1); test **blend** epistemic+aleatoric. Nếu blend/disagreement **đè** trên coverage → **headline model contribution** (distilled committee-disagreement = principled epistemic UQ). Nếu KHÔNG đè → ghi honest, quay về label-efficiency (§4.10) làm trục chính. Backup: `nulite_preds.csv` + `lkcell_preds.csv` (dump lại được nhưng setup đau — nên giữ). NuLite-T/LKCell-L kiêm **heavy-baseline** (peer cite).

---

## 5. VERDICT Q1 + 3 TRỤ

**ĐỦ submit Q1 methods/applied** nếu kể đúng 3 trụ (KHÔNG overclaim "thắng mọi metric"):
1. **Label-efficient distillation** — foundation teacher + count-label, KHÔNG cần mask (khác mọi peer segmentation). **Distilled đạt reliability CẠNH TRANH (hơi dưới) supervised MÀ không cần mask** (§4.8: distilled 0.750 vs supervised 0.789) — value = label-efficiency, KHÔNG phải "hơn supervised". **★ NAY CÓ BẰNG CHỨNG BUDGET-FRONTIER (§4.10): ngang coverage+MAE ở cùng số ảnh, distilled chỉ cần count-scalar ≈ 5–10× rẻ hơn mask (K=52.8 nhân/ảnh).** = lỗ reviewer "tại sao distill / không có thí nghiệm tiết kiệm nhãn" ĐÃ ĐÓNG.
2. **Distributional UQ** — calibrated (μ,σ), interval theo mô, transfer cross-dataset. Không peer distilled nào có.
3. **Efficiency** — 1.935M, nhỏ nhất có UQ.
*Accuracy = "cạnh tranh cho ngân sách nhãn", KHÔNG phải điểm bán (1.9M density-counter không đè SOTA: in-domain PanNuke 1.72× MAE vs NuLite-12M).*

**Peer (Related Work, chỉ CITE):** HoVer-unet (ISBI24), 9M H-Optimus student (2502.19217, citation vàng), RCKD — đều distilled/lightweight nhưng **KHÔNG UQ**.

**Rủi ro:** (i) ★ UQ-floor 5-seed VERIFIED: **R2 xếp ~4/5, CQR (cùng 1 model) nhỉnh cả worst-org lẫn Winkler** (§4.3) → **KHÔNG lead bằng UQ**; UQ = "distributional (μ,σ) 1-forward, đè MC-Dropout, ngang Ensemble ở 1/5 compute, cho phép transfer N5" — competitive-not-best; (ii) ✅ ĐÃ 3 dataset (NuInsSeg + PanNuke + CryoNuSeg transfer, §4.4); MoNuSAC = boundary honest (scale-gap 4×); (iii) NuInsSeg nhỏ/nhiễu → claim subgroup mềm. **Rủi ro lớn nhất = FRAMING, không phải thiếu thí nghiệm.**

**★ VERDICT TẦNG VENUE (chốt 2026-07-18, đúng framing §1.2):**
- **As-is (số đóng băng): solid Q1 tầm-trung** (Comput. Biol. Med. / Comput. Med. Imaging Graph. / Artif. Intell. Med.). Đóng góp = gói distillation-UQ hội tụ + rigor + honest. Đủ bar Q1 thật.
- **⚠️ Cửa "cái gì sống sót qua nén — đếm hay bất định?" ĐÃ ĐÓNG** (RETRACTED §1.2: PathoSAM là segmentation, KHÔNG có bất định nội tại để "sống sót" — PB-σ là cấu trúc của P1). Đừng dựng top-tier trên framing này.
- **Cửa top-tier còn lại (nếu muốn tiếp): (a) đào sâu cơ chế N4** (vì sao PB-σ structural vỡ dưới global, learned-σ sống — làm hết scheme-dependent); hoặc **(b) validation lâm sàng.** Cả hai ngoài scope hiện tại. **As-is đã đủ solid mid-Q1; label-efficiency (§4.10) là trụ mạnh nhất giờ có số.**
- **Cửa ĐÃ ĐÓNG (đừng lặp):** UQ-superiority (thử 3× — UQ-floor/N4/triage đều ~4/5, thua); grab-bag stain/scale/DA (nồi lẩu, top-tier phạt). Chi tiết triage §4.3.

---

## 6. NEXT STEPS (thứ tự)
1. ✅ R2 NuInsSeg 5-seed + backup kaggle.
2. ✅ Re-eval CondConf/PCP 5-seed matched.
3. ✅ **UQ-floor regen 5-seed** (2026-07-18) — 20 pkl `work/uq_{method}_nuinsseg_s{42..46}.pkl`, `aggregate_uqfloor.py`. Số §4.3. **R2 ~4/5 → UQ không phải trục bán.** Backup kaggle `sam3-paper2-uqkd`.
4. ✅ **KD 5-seed + per-image significance** (2026-07-18) — 5 pkl `work/student_kd_nuinsseg_cv5_s{42..46}.pkl`, `aggregate_kd.py`/`aggregate_r2.py`. Số same-scheme §4.1. **N4 CLOSED (reframe honest: PB-σ chỉ sập global).** Backup kaggle `sam3-paper2-uqkd`.
5. ✅ **Dataset 3 = CryoNuSeg transfer** (2026-07-18) — NuInsSeg→CryoNuSeg: marg.cov 0.967, Winkler 503, MAE 73 (σ transfer sống). MoNuSAC bỏ (co nhân 4× → μ sập; ghi boundary). `prep_cryonuseg_counts.py` + `xfer_nuinsseg2cryonuseg.pkl`, backup kaggle.
6. ✅ **Ablations §4.8 re-run leak-free 5-seed** (2026-07-18) — `aggregate_ablations.py`. **3 claim cũ LẬT:** supervised>distilled (0.789>0.750), ch16>ch32 NuInsSeg, detach yếu. §4.8 reframe honest (distillation=label-efficiency KHÔNG phải hơn-supervised). Backup kaggle.
7. ✅ **Label-efficiency frontier EMPIRICAL** (2026-07-19, §4.10) — `label_efficiency.py` (count-only) + `label_efficiency_both.py` (distilled↔supervised head-to-head, align img-hash, grad-clip). **Đóng lỗ "tại sao distill": ngang coverage+MAE ở cùng số ảnh, distilled chỉ cần count (≈5–10× rẻ hơn mask, K=52.8 nhân/ảnh).** GT cache dựng-lại 35s (rẻ). *(Tùy chọn hardening: nâng 3→5 seed nếu lên figure.)*
8. ⏳ **Multi-teacher committee (Hướng A, §4.11)** — PROBE 4-teacher XONG (gate qua: disagreement-σ +0.65 ngang learned-σ; consensus KHÔNG hạ accuracy). **CÒN thí nghiệm đầy đủ:** train student σ-target = committee disagreement, so coverage/Winkler vs learned-σ/PB-σ + test blend. Đè → headline model contribution; không đè → về label-efficiency. `probe_multiteacher_full.py` + `multiteacher_probe.ipynb`.
9. → **VIẾT manuscript**: đóng gói "Distributional Count Distillation under mean-variance optimization conflict", KHÔNG claim PB-JCI (=P1); hình ~4–5. **Lead = distillation + label-efficiency (§4.10) + N2-Poisson-σ + PanNuke comparative-win (§4.2); KHÔNG lead UQ; KHÔNG dùng "distribution distillation".** (nếu §4.11 đè → thêm trục committee-epistemic-σ.)

**Baseline = code official (không tự chế):** CondConf (Gibbs–Cherian–Candès JRSS-B 2025), PCP (Zhang–Candès 2024), R2CCP (Guha ICLR 2024),
CPCP (ICML 2026), UQ floor (MC-Dropout/Ensemble/CQR/CHDQR); CellViT/LKCell/NuLite/LSP-DETR/PathoSAM. Dùng đúng checkpoint official.
Runbook chi tiết vast: xem [RUN_LIST_STRENGTHENING.md](RUN_LIST_STRENGTHENING.md).

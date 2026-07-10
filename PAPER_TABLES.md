# Bảng kết quả — Predictor-Agnostic Joint Conformal Cell Counting under Shift

> File số liệu gốc, tập trung mọi bảng. α = 0.1 (target coverage 90%) cho mọi bảng conformal.
> Hai backbone: **SAM3+LoRA+TypeHead** (Table 1–7) và **PathoSAM** (Table 8). Table 9 = so baseline hiện đại + ablation.
> Thước đo hiệu quả: **Winkler / Interval score** (gộp width + phạt miss, thấp = tốt).

---

## Table 1 — SAM3 zero-shot trên PanNuke

**Test FULL Fold 3, N=2722. Macro mIoU qua 5 class.**

| Chiến lược prompt | Kong mIoU | Ours mIoU | Order |
|---|---:|---:|:--:|
| Medical (full sentence) | 0.26% | 4.81% | rank 1 |
| LLM-gen (synonyms avg) | 4.08% | 7.50% | rank 2 |
| Generic ("cell") | 6.22% | 13.77% | rank 3 |

*Order Medical < LLM < Generic tái hiện đúng 3/3. Số tuyệt đối cao ~2× Kong do single-fold (Fold 3) vs Kong 3-fold CV.*

### Table 1b — Per-class mIoU (zero-shot, Fold 3 N=2722)

| Class | Medical | LLM-avg | Generic |
|---|---:|---:|---:|
| Neoplastic | 12.43% | 14.07% | 37.32% |
| Inflammatory | 1.23% | 7.91% | 6.67% |
| Connective | 5.41% | 11.43% | 14.45% |
| Dead | 0.17% | 0.62% | 0.31% |
| Epithelial | 4.82% | 3.46% | 10.08% |

*Dead sụp ~0 mọi chiến lược. Neoplastic ổn định nhất (~52% dataset).*

---

## Table 2 — Fine-tune A1 → A2 (LoRA, mIoU)

**Macro mIoU Fold 3. A2 multi-seed [42,100,200] mean±std.**

| Chiến lược | A1 zero-shot | A2 LoRA | Δ |
|---|---:|---:|---:|
| Medical | 4.81% | 12.78±0.44% | +7.97pp |
| LLM-gen | 7.50% | 15.52±0.04% | +8.02pp |
| Generic | 13.77% | 15.03±0.04% | +1.26pp |

*LoRA rank-16, FFN-only, 442,368 params (0.05%). Hai prompt fail (Medical, LLM) tăng ~+8pp → LoRA đưa các prompt thất bại về ngang prompt tốt nhất (ba prompt hội tụ ~13–15.5%). Vai trò LoRA = predictor đáp ứng ổn định với prompt, không phải tối ưu mIoU.*

### Table 2b — TypeHead phân loại 5 loại (Phase A3)

| Class | Type Acc % | Counting MAE |
|---|---:|---:|
| Neoplastic | 89.44±0.67 | 7.69±0.83 |
| Inflammatory | 75.30±0.99 | 2.18±0.14 |
| Connective | 76.38±0.99 | 5.14±0.30 |
| Dead | 10.08±4.30 | 0.47±0.01 |
| Epithelial | 73.13±1.55 | 1.96±0.07 |
| **Macro** | **80.48±0.05** | **3.49** |

*TypeHead 33,664 params. Macro acc 80.48%. Dead yếu (class hiếm 0.4%).*

---

## Table 3 — Distribution shift detection

**δ giữa reference (Fold 1, N=200) và mỗi điều kiện. Mean±std qua ×5 seed.**

| Condition | MMD² | Wasserstein-1 | Energy |
|---|---:|---:|---:|
| fold2 | 0.0000±0.0000 | 0.0120±0.0005 | 0.0286±0.0008 |
| fold3 | 0.0001±0.0002 | 0.0122±0.0008 | 0.0292±0.0019 |
| hed_mild | 0.0253±0.0031 | 0.0382±0.0029 | 0.0914±0.0060 |
| hed_moderate | 0.0593±0.0024 | 0.0601±0.0019 | 0.1397±0.0034 |
| hed_severe | 0.1177±0.0088 | 0.0889±0.0032 | 0.2014±0.0072 |
| blur_mild | 0.0026±0.0007 | 0.0151±0.0004 | 0.0390±0.0016 |
| blur_moderate | 0.0394±0.0009 | 0.0422±0.0006 | 0.1017±0.0004 |
| blur_severe | 0.1855±0.0066 | 0.0942±0.0014 | 0.2234±0.0041 |
| hsv_mild | 0.0000±0.0000 | 0.0065±0.0003 | 0.0170±0.0009 |
| hsv_moderate | 0.0056±0.0008 | 0.0206±0.0007 | 0.0513±0.0018 |
| hsv_severe | 0.0342±0.0049 | 0.0440±0.0027 | 0.1060±0.0067 |

*Severity ordering đơn điệu 3/3 augmentation. Severe >> within-PanNuke (fold2/3 ≈ 0). δ_max = 0.1855 (blur_severe).*

---

## Table 4 — Joint Conformal Counting: BẢNG CHÍNH (SAM3)

**Joint Coverage (%) / Width. α=0.1, N_test=1361. Model-seed CI: 3 model × 5 cal = 15 run.**

| Phương pháp | in-dist | mild | severe | drift |
|---|---|---|---|---|
| Marginal Split | 62.7±1.6 / 9.63±0.57 | 54.8±1.9 / 10.01±0.62 | 61.8±2.0 / 9.19±0.41 | 59.9±1.5 / 9.60±0.54 |
| ACI | 89.9±0.1 / 14.62±0.79 | 89.7±0.1 / 18.70±1.41 | 89.3±0.2 / 24.54±1.42 | 89.4±0.1 / 19.45±1.24 |
| PB-JCI split | 90.0±1.3 / 13.54±0.74 | **82.9±1.2** / 13.93±0.80 | **81.1±2.2** / 12.79±0.53 | 84.8±1.0 / 13.41±0.70 |
| **PB-JCI Online** ★ | **90.5±0.3 / 13.76±0.74** | **89.9±0.3 / 16.18±0.88** | **89.4±0.4 / 17.77±0.34** | **89.3±0.4 / 15.56±0.59** |
| Class-Strat Bonferroni | 94.4±1.1 / 18.59±0.69 | 89.2±1.5 / 19.05±0.78 | 85.3±1.8 / 16.19±0.41 | 89.9±1.1 / 17.91±0.62 |

*Static sụp: Marginal ~55–63%, PB-JCI split 90→81% dưới shift. ACI giữ coverage nhưng width nổ (severe 24.5). PB-JCI Online duy nhất giữ ≥89% mọi setting với width ổn định.*

---

## Table 5 — NuInsSeg in-domain (total-count K=1)

**Calibrate NuInsSeg → test NuInsSeg (665 ảnh). MAE = 15.70. 5 cal seed, LoRA seed42, window=300.**

| Phương pháp | Coverage (%) | Width |
|---|---:|---:|
| Marginal Split | 90.2±2.2 | 50.56±4.31 |
| ACI | 89.9±0.3 | 57.89±8.63 |
| PB-JCI split | 90.2±2.2 | 50.56±4.31 |
| **PB-JCI Online** | **90.2±1.5** | **51.22±3.75** |

*K=1 nên Marginal Split ≡ PB-JCI split. Mọi method ~90%; PB-JCI Online ổn định nhất (std width 3.75 vs ACI 8.63). Framework generalize sang dataset thứ 2 + task total-count.*

> **Table 6 (SAM3 cross-dataset)** đã **gộp** vào **Table 8c** (bảng cross-dataset cả hai backbone) để tránh lặp số liệu.

---

## Table 7 — Phụ lục: protocol Kong 5:1:3

| Chiến lược | Kong-subset (N≈907) | Full Fold 3 (N=2722) | Chênh |
|---|---:|---:|---:|
| Medical | 4.98% | 4.81% | 0.17pp |
| LLM-gen | 7.37% | 7.50% | 0.13pp |
| Generic | 13.81% | 13.77% | 0.04pp |

*Chênh <0.2pp → kết quả không nhạy với protocol.*

---

## Table 8 — Backbone thứ hai: PathoSAM (predictor-agnostic)

> Table 8a–8b chạy trên PanNuke **Fold-3**. PathoSAM generalist chỉ train trên Fold 1+2 (`get_generalist_datasets.py`: `folds=["fold_1","fold_2"]`), nên Fold-3 là **tập kiểm thử held-out hợp lệ** — giống chuẩn của SAM3, không cần coi là OOD hay leaky. Đây là kiểm chứng predictor-agnostic (online vs tĩnh lặp lại trên backbone khác).
>
> **Headline OOD = cross-dataset PathoSAM → NuInsSeg** (Mahbod 2024, eval-only): **Table 8c (cross) / 9a (Winkler chính)**.

### Table 8a — Counting MAE: hai backbone tương đương (per-class)

| Class | PathoSAM | SAM3 | Tốt hơn |
|---|---:|---:|:--:|
| Neoplastic | 5.07 | 7.69 | PathoSAM |
| Inflammatory | 2.68 | 2.18 | SAM3 |
| Connective | 3.36 | 5.14 | PathoSAM |
| Dead | 0.89 | 0.47 | SAM3 |
| Epithelial | 3.54 | 1.96 | SAM3 |
| **Macro** | **3.11** | 3.49 | PathoSAM (~11%) |

*Hai backbone đếm tương đương (macro 3.11 vs 3.49), SAM3 thắng 3/5 class. PathoSAM segment giỏi hơn nhiều nhưng đếm chỉ ngang: vì đếm chỉ cần phát hiện nhân, không cần mask đẹp.*

### Table 8b — PB-JCI trên PathoSAM (joint, 5 cal seed, N_test=1114)

| Phương pháp | in-dist | mild | severe | drift |
|---|---|---|---|---|
| Marginal Split | 65.1 / 8.84 | 57.6 / 8.93 | **17.3** / 6.98 | 46.4 / 8.28 |
| ACI | 89.9 / 14.79 | 89.6 / 18.05 | 87.4 / 68.70 | 87.2 / 32.76 |
| PB-JCI split | 89.7 / 13.67 | 83.8 / 13.75 | **35.7** / 11.38 | 70.1 / 12.98 |
| **PB-JCI Online** ★ | **90.4 / 13.94** | **89.2 / 15.87** | **85.7 / 54.02** | **85.6 / 23.87** |
| Class-Strat Bonferroni | 96.6 / 20.17 | 93.7 / 20.20 | 48.5 / 16.54 | 79.6 / 19.05 |

*Lặp đúng câu chuyện Table 4 trên backbone khác: static sụp (split 90→36%), PB-JCI Online giữ 85–90%, hẹp hơn ACI 6–27%. Đóng góp ở tầng conformal, độc lập detector.*

### Table 8c — Cross-dataset total-count (K=1): SAM3 vs PathoSAM — shift nhẹ vs cực mạnh

**Hiệu chỉnh trên nguồn (PanNuke), test trên đích KHÔNG có nhãn. *Oracle* = hiệu chỉnh bằng nhãn đích (cận trên, không dùng trong cross). PathoSAM CHỈ test NuInsSeg vì CoNSeP/CoNIC nằm trong train PathoSAM (Lizard) → leaky; SAM3 không bị nên test cả CoNSeP.**

| Backbone → Đích | Oracle *(có nhãn đích)* | Cross no-adapt | Cross ACI | **Cross PB-JCI Online (ours)** |
|---|---:|---:|---:|---:|
| SAM3 → NuInsSeg | 90.2±2.2 / 50.56 | 86.8 / 38.48 | 89.1±0.3 / 52.93 | **89.2±0.3 / 47.07** |
| SAM3 → CoNSeP | 91.1±1.9 / 38.70 | 93.0 / 41.34 | 90.2±0.3 / 38.30 | **90.7±0.7 / 38.81** |
| **PathoSAM → NuInsSeg** | 88.2±3.7 / 61.06 | **41.5** / 18.00 | 84.0±0.2 / 63.58 | 81.8±1.0 / 53.11 |

*Mỗi ô = Coverage(%) / Width. Cột **no-adapt** = áp thẳng quantile nguồn, không chỉnh (= mốc zero-feedback).*

- **SAM3 — shift nhẹ & hai chiều:** no-adapt lệch (NuInsSeg under 86.8%, CoNSeP over 93.0%); **adaptive kéo cả hai về ~90%**, gần chạm oracle, NuInsSeg còn hẹp hơn oracle (47.07 < 50.56).
- **PathoSAM — shift CỰC MẠNH:** no-adapt **sụp 41.5%** (PathoSAM tự tin, s_i≈0.9 → σ nhỏ → R nổ); online chỉ bò lên **81.8% — CHƯA đủ** → đây là động lực cho **Adaptive PB-JCI Online** (cửa sổ co/giãn), đạt **90.0%** ở Table 9a (bảng Winkler chính). Online giả định feedback GT lúc test.

### Table 8d — PathoSAM feedback trễ / thưa / nhiễu (severe shift, 5 seed)

| Điều kiện feedback | Coverage (%) | Width |
|---|---|---|
| Full feedback (baseline) | 86.2±0.6 | 55.23 |
| Noisy ×1 / ×2 / ×3 | 86.1 / 86.5 / **86.9** | 55.7 / 56.6 / 57.9 |
| Delayed 10 / 50 / 100 mẫu | 85.7 / 83.2 / 81.1 | 54.8 / 53.0 / 50.9 |
| Sparse 50% / 25% / 10% | 82.6 / 73.3 / **54.2** | 50.5 / 39.4 / 20.9 |
| *Sparse 0% (= static, Table 8c)* | *41.5* | *18.00* |

*Nhiễu: miễn nhiễm (86.2→86.9%). Trễ: suy giảm êm (≤50 mẫu vẫn ~83%). Thưa: cần ≥~50% mật độ nhãn (10% sụp 54%, 0% = static 41.5%). Nhiễu feedback luôn lệch về phía an toàn (over-cover), không gây under-cover.*

**(ii) SAM3 feedback robustness (severe shift, 5 seed):**

| Feedback | Coverage (%) | Width |
|---|---:|---:|
| Full | 89.6±0.3 | 17.48 |
| Delayed 50 | 89.3±0.3 | 17.32 |
| Sparse 50% / 25% | 88.5 / 87.4 | 17.0 / 16.5 |
| Noisy ×2 | 97.9 | 58.77 |

*SAM3 bền hơn PathoSAM (sparse 25% = 87.4% vs 73.3%, vì shift nhẹ hơn). Noisy ×2 → over-cover 97.9% (SAM3 count nhỏ → σ nhỏ → nhiễu bị max-statistic khuếch đại). Nhiễu không bao giờ gây under-cover.*

### Table 8e — Đa lớp K=4 (MoNuSAC, within-taxonomy, 5 seed): tổng quát theo K

**MoNuSAC = eval-only của PathoSAM. Cal/test cùng 4 lớp (Epithelial/Lymphocyte/Macrophage/Neutrophil), tách theo patient (0 overlap). 96 ảnh test, target joint 90%.**

| Phương pháp | Joint coverage (%) | Macro width | Per-class cov (Epi/Lym/Mac/Neu) |
|---|---:|---:|---|
| Marginal (no corr.) | 82.9±5.2 | 131.87 | 96 / 92 / 97 / 96 |
| Class-Strat Bonferroni | 87.1±6.4 | 170.33 | 96 / 96 / 97 / 96 |
| **PB-JCI (joint max-stat)** ★ | **91.7±1.9** | 140.19 | 97 / 93 / 100 / 100 |

*Lặp đúng câu chuyện PanNuke K=5 (Table 8b) trên dataset K=4 độc lập → tổng quát theo số lớp K, không phụ thuộc riêng K=5. PB-JCI đạt joint 91.7% (≥90%) với width hẹp hơn Bonferroni 18% (140 vs 170) mà coverage CAO hơn (91.7 vs 87.1%) → max-stat vừa hợp lệ vừa hiệu quả hơn Bonferroni. Bonferroni under-cover + bất ổn (±6.4%) do lớp hiếm (Macrophage/Neutrophil ~600 nhân/toàn bộ) ít điểm calibration; PB-JCI ổn định hơn (±1.9%). Width tuyệt đối lớn (counts ~80/lớp + PathoSAM đếm OOD yếu MAE~40) → conformal đúng-đắn nới khoảng giữ coverage với predictor yếu, KHÔNG phải lỗi conformal.*

---

## Table 9 — So baseline hiện đại + ablation

### Table 9a — BẢNG CHÍNH: So baseline hiện đại + Winkler (PathoSAM → NuInsSeg, 5 seed)

**Mục tiêu 90%. Winkler / Interval score (thấp = tốt; gộp width + phạt miss). Bảng Winkler đầy đủ duy nhất — dùng cho paper.**

| Phương pháp | Nguồn / tài liệu tham khảo | Coverage (%) | Width | Winkler ↓ |
|---|---|---:|---:|---:|
| Weighted Conformal | Tibshirani, Barber, Candès & Ramdas, NeurIPS 2019 | 40.8 | 17.56 | 228.87 |
| ACI | Gibbs & Candès, NeurIPS 2021 | 84.0±0.2 | 63.58 | 129.55 |
| NexCP | Barber, Candès, Ramdas & Tibshirani, Annals of Statistics 2023 | 84.7±0.6 | 55.52 | 119.56 |
| SAOCP | Bhatnagar, Wang, Xiong & Bai, ICML 2023 (PMLR 202) — code gốc | 87.8±0.5 | 61.32 | 113.63 |
| COP | Hu, Wu, Xia & Zou, ICLR 2026 | 87.9±0.2 | 60.60 | 113.13 |
| Rolling-Origin CP | Halkiewicz, arXiv:2605.08422 (2026) | 89.5±0.7 | 64.39 | 110.24 |
| PB-JCI Online (ours) | — | 81.8±1.0 | 53.11 | 125.96 |
| **Adaptive PB-JCI Online (ours)** ★ | — (đề tài) | **90.0±0.6** | 66.50 | **108.67** |

*Weighted Conformal sụp **40.8%** (Winkler **228.87** — tệ nhất nhóm conformal) → reweighting không cứu được kiểu shift này, cần online recalibration (bỏ conformal hoàn toàn còn tệ hơn → 10.4% / Winkler 319, xem Ablation 9b). COP (SOTA online 2026) 87.9%, SAOCP (mã nguồn gốc Salesforce, port verbatim) 87.8% / 113.63, Rolling-Origin 89.5% — **tất cả vẫn <90%**. Adaptive PB-JCI Online đạt **90.0% và Winkler thấp nhất 108.67**, dưới cả SAOCP/Rolling-Origin/COP/NexCP/ACI.*

***Đóng góp = conditional validity, KHÔNG phải width:*** *Adaptive có width thô **lớn nhất** (66.50) mà Winkler **thấp nhất** → bác bỏ "chỉ đổi width lấy coverage". Ba biến thể nội bộ (không vào bảng) khẳng định điều này: **Detector-flush** 88.7% / Winkler 110.07 (biến thể cho shift-từ-đầu); **Fallback-multiplier** over-cover 91.5% nhưng 111.85 do width nổ (79.62); **Hybrid max(PB,ACI)** 119.55 — đều kém Adaptive. Cơ chế orthogonal → bọc được lên COP/NexCP; SAOCP điều khiển trong cùng score-space PB-σ, chỉ khác cơ chế online → cô lập đúng đóng góp adaptive.*

### Table 9b — Ablation từng thành phần

| Bỏ thành phần | Kết quả |
|---|---|
| **Bỏ conformal hoàn toàn** (→ Naive PB-σ, không calibration) | coverage sụp **90→10.4%**, Winkler **319** (σ model quá tự tin, không có hệ số q hiệu chỉnh) |
| Bỏ PB-σ (sai số thô) | width +28–30% ở regime thường (14.0→19.9, 16.0→22.2) |
| Bỏ joint max-statistic (→ Bonferroni) | width +31% (13.94→20.17) |
| Bỏ online window (→ split) | coverage sụp 90→36% dưới shift |
| Bỏ adaptive recalib | extreme-shift coverage 82% vs 90% (Table 9a) |

### Table 9c — Window size sensitivity (lý do chọn W=300, PathoSAM→NuInsSeg)

| Window | Coverage (%) | Width |
|---|---:|---:|
| 100 | 87.5±0.5 | 60.50 |
| 200 | 84.6±0.7 | 56.34 |
| 300 ★ | 81.8±1.0 | 53.11 |
| 500 | 77.1±0.8 | 47.40 |

*Tradeoff: window nhỏ → thích nghi nhanh, coverage cao nhưng rộng; lớn → ngược lại. W=300 = cân bằng. Chính tradeoff này là động lực cho cửa sổ thích nghi (tự co khi coverage tụt).*

---

## Tóm tắt trạng thái

| Bảng | Nội dung | Backbone |
|---|---|---|
| 1 / 1b | Zero-shot mIoU (macro + per-class) | SAM3 |
| 2 / 2b | LoRA fine-tune + TypeHead | SAM3 |
| 3 | Shift detection (×5 seed) | SAM3 |
| 4 | Conformal chính (15 run) | SAM3 |
| 5 | NuInsSeg in-domain (K=1) | SAM3 |
| 7 | Kong-subset appendix | SAM3 |
| 8a | Counting MAE 2 backbone | PathoSAM / SAM3 |
| 8b | PB-JCI joint (predictor-agnostic) | PathoSAM |
| 8c | Cross-dataset (gộp 2 backbone) | SAM3 + PathoSAM |
| 8d | Feedback trễ/thưa/nhiễu | cả hai |
| 8e | Đa lớp K=4 MoNuSAC | PathoSAM |
| **9a** | **BẢNG CHÍNH: Winkler + baseline hiện đại** | PathoSAM |
| 9b / 9c | Ablation + window-size | PathoSAM |

*Mọi số tái lập: SAM3 trong `kaggle/sam3_*`; PathoSAM trong `kaggle/vast/` (đặc biệt `pathosam_winkler_table3.py` cho Winkler 9a, `pathosam_risk2_feedback.py` cho 8d, `run_pathosam_crossdataset.py` cho 8c).*

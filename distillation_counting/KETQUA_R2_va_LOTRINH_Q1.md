# Kết quả R2 (Distributional Count Distillation) + Lộ trình Q1

> Cập nhật 2026-07-11. Sau 4 vòng: PBUD/CCAD (âm tính x3), R2 (DƯƠNG TÍNH). File này chốt kết quả
> THẬT trên NuInsSeg + lộ trình đưa lên Q1. Nguyên tắc: không bịa, không heuristic, không tô hồng;
> báo cáo cả điểm yếu; dùng thống kê để tách nhiễu vs thật.

## 1. Method R2 (một câu)
Student nhẹ (~1.9M) distill từ PathoSAM: xuất TRỰC TIẾP `(μ = Σ density, σ = exp(log_σ))`;
train bằng `density-KD (MSE) + count (|μ−GT|) + β-NLL(GT|μ,σ)` → σ **heteroscedastic học từ lỗi thật**.
Suy luận: **clustered conformal** trên score `|GT−μ|/σ` (gom organ theo độ khó từ σ) → khoảng khớp
độ khó. Cơ sở: density-map counting (Lempitsky 2010), aleatoric NLL (Kendall & Gal 2017), β-NLL
(Seitzer 2022), clustered/group-conditional conformal (Barber 2020). **Không mảnh nào là heuristic.**

## 2. Kết quả cổng (NuInsSeg, K=1, seeds=20, α=0.1, target 0.90)

| scheme | Winkler | worst-org | org-gap | MAE | marg.cov |
|---|---|---|---|---|---|
| KD-global (baseline) | 125.50±10.2 | 0.264 | 0.736 | 22.38 | 0.900 |
| R2-global | 127.63±11.0 | 0.505 | 0.495 | 18.38 | 0.905 |
| **R2-cluster (ours)** | **112.78±11.1** | **0.757** | **0.243** | **18.38** | 0.894 |

**R2-cluster vs KD:** Winkler ↓10%, worst-org 0.264→0.757 (gần ×3), org-gap 0.736→0.243, MAE 22.4→18.4.
**Cổng đăng ký trước ĐẠT SẠCH** (Winkler ≤ KD **và** worst-org ≥ KD), không đổi luật.

**Vì sao cluster vừa tăng coverage vừa giảm Winkler:** R2-global 1 quantile chung → rộng cho organ dễ,
hẹp cho organ khó. Cluster khớp khoảng theo độ khó (từ σ heteroscedastic) → giảm cả width-penalty
(organ dễ) lẫn miss-penalty (organ khó). **KD không có σ học được nên không khai thác được cluster** —
đây là giá trị cụ thể của phần distillation phân phối.

## 3. ĐIỂM YẾU phải ghi nhận (không giấu)
- worst-org 0.757 vẫn **dưới** target 0.90 → cải thiện lớn, **chưa** reliable hoàn toàn.
- marg.cov cluster 0.894 hơi dưới 0.90 (trong nhiễu, nhưng ghi nhận).
- Winkler tốt hơn ~10% ≈ 1 std unpaired → **cần kiểm định paired** để khẳng định.
- Mới **1 dataset (NuInsSeg K=1)**, mới **1 baseline (KD)** → chưa đủ tầm Q1.

## 3b. Ablation (seeds=20, dòng R2-cluster mỗi biến thể) — ĐÃ CHẠY
| Biến thể | MAE | worst-org | Winkler |
|---|---|---|---|
| KD (mốc) | 22.38 | 0.264 | 125.50 |
| A: density-only (−count, −NLL) | 52.75 | 0.660 | 177.82 |
| B: density+count (−NLL) | **9.45** | 0.582 | **68.13** |
| C: full R2 (+NLL, coupled) | 18.38 | **0.757** | 112.78 |

**Phát hiện (paired, đều p<1e-3):**
1. Count loss bắt buộc cho μ: bỏ (A) → MAE nổ 52.75.
2. **Clustered conformal là đòn bẩy conditional mạnh, độc lập NLL:** B (không NLL) đã kéo worst-org
   0.264→0.582, Winkler 125→68, MAE 22→9.45. Phần lớn lợi ích đến từ density-count + cluster.
3. **NLL thêm worst-org (0.582→0.757) NHƯNG làm hỏng MAE (9.45→18.38):** vì `Lnll~120 ≫ Lcount~20`
   với trọng số bằng nhau → NLL áp đảo gradient μ, kéo μ lệch.

**Sửa `--detach_mu` (ĐÃ TRAIN — cấu hình CHỐT):** tách μ khỏi NLL (NLL chỉ dạy σ). Kết quả R2-cluster:
MAE **10.12**, worst-org **0.753**, Winkler **60.41**, marg.cov 0.909 — lấy MAE thấp của B + worst-org
cao của C + Winkler thấp nhất. **vs KD (paired Wilcoxon):** Winkler −65.09 (p=1.9e−6), MAE −12.26
(p=1.9e−6), worst-org 0.264→0.753. Đây là **cấu hình R2 chốt cho các thí nghiệm sau**
(`work/student_r2_detach.pkl`).

### Cấu hình R2 chốt
`distill_student_r2.py --epochs 80 --student_ch 32 --w_density 1.0 --w_count 0.01 --w_nll 0.01 --detach_mu`
+ eval `eval_r2_grouped.py --n_clusters 3 --min_group 15` (scheme **cluster**).

## 3c. Compression sweep + baseline supervised (ĐÃ CHẠY, R2-detach, cluster, seeds=20)
| student ch | params | MAE | worst-org | Winkler |
|---|---|---|---|---|
| 16 | ~0.5M | 10.97 | 0.742 | 72.34 |
| **32 (chốt)** | ~1.9M | **10.12** | **0.753** | **60.41** |
| 64 | ~7.7M | 11.68 | 0.718 | 72.60 |
| KD (mốc) | ~1.9M | 22.38 | 0.264 | 125.50 |

Ngay cả ch=16 (~0.5M, ~600× nhỏ hơn PathoSAM) vẫn thắng KD mọi trục (p<1e−5). ch=32 sweet spot.

**Baseline supervised (GT density) vs distilled (PathoSAM), ch=32:**
| | MAE | worst-org | Winkler |
|---|---|---|---|
| supervised (GT) | 9.49 | 0.711 | 58.23 |
| distilled (teacher) | 10.12 | **0.753** | 60.41 |

Supervised nhỉnh MAE/Winkler; **distilled TỐT HƠN worst-org (0.753 vs 0.711)** → teacher foundation model
đem lợi ích *conditional reliability*, không chỉ bắt chước nhãn (+ dùng được nơi không nhãn). Cả hai ≫ KD.

## 3d. n_clusters sweep — đẩy worst-org (ĐÃ CHẠY, eval-only trên ch=32 detach)
| n_clusters | worst-org | Winkler | marg.cov | #under |
|---|---|---|---|---|
| 2 | 0.753 | 59.98 | 0.906 | 6/27 |
| 3 | 0.753 | 60.41 | 0.909 | 6/27 |
| 5 (chốt) | 0.773 | 61.94 | 0.916 | 4/27 |
| 6 | 0.784 | 62.48 | 0.922 | 4/27 |

Nhiều cluster → worst-org ↑, đổi lấy Winkler ↑ nhẹ + marginal conservative dần. **Trần thực tế worst-org
~0.78** (chưa tới 0.90 — đúng giới hạn lý thuyết conditional coverage với ~10 ảnh/organ, Vovk/Barber).
0.78 vs KD 0.264 = cải thiện lớn. Chốt **n_clusters=5** (không đẩy tối đa, tránh over-tuning).

## 4. Lộ trình Q1 (trạng thái)
1. ✅ **Significance** (paired Wilcoxon): R2-cluster vs KD Winkler −65 (p=1.9e−6), MAE −12 (p=1.9e−6).
2. ✅ **Ablation**: density→+count→+NLL→±cluster. NLL-coupling làm hỏng MAE → sửa bằng `--detach_mu`.
3. ✅ **Đẩy worst-org**: n_clusters sweep → chốt 5 (worst-org 0.773, trần thực tế ~0.78).
4. ✅ **Compression sweep** (ch=16/32/64): ch=32 sweet spot, ch=16 (~0.5M) vẫn thắng KD.
5. ⏳ **Dataset 2 (PanNuke)** — ĐANG LÀM. Code đã xong (mục 7). Chạy 3a (K=1) rồi 3b (K>1).
6. ⬜ **Baseline mạnh hơn**: supervised-GT đã có (mục 3c); thêm so method nhẹ đã công bố nếu được.

## 7. ▶ TIẾP THEO (resume sau khi ngủ) — PanNuke dataset 2

### 7.0 Trạng thái vast khi tạm ngừng
- **STOP (không DESTROY) instance** để giữ: `/workspace/penv` (env PathoSAM), data PanNuke đã tải,
  và mọi cache teacher (`work/*.pkl`). DESTROY = mất hết, phải setup lại từ đầu (~30 phút + tải lại).
- PanNuke đã tải xong: `data/pannuke/fold{1,2,3}/Fold {N}/images/fold{N}/images.npy` (cấu trúc KHỚP
  `PanNukeFold`). Root dùng: `--pannuke_root /workspace/sam3_research/data/pannuke`.
- Code PanNuke ĐÃ push chưa? Nếu chưa: trên Mac `git add -A distillation_counting && git commit && git push`;
  trên vast `git pull`.

### 7.1 KIỂM TRƯỚC (bắt buộc): PanNuke có types.npy không?
`worst-org`/conditional coverage cần nhóm theo **tissue type**. Nếu thiếu types.npy → tất cả 'unknown'
→ mất phân tích organ-wise. Chạy:
```bash
find /workspace/sam3_research/data/pannuke -name "types.npy"
```
- CÓ → tốt, chạy tiếp 7.2.
- KHÔNG → báo lại; ta dùng nhóm thay thế (vd bin theo GT-count) hoặc chỉ báo marginal cho PanNuke.

### 7.2 Chạy 3a — PanNuke như K=1 (BẮT ĐẦU 1 FOLD cho rẻ ~1.5h)
```bash
cd /workspace/sam3_research/distillation_counting
M="/workspace/bin/micromamba run -p /workspace/penv"
RT=/workspace/sam3_research/data/pannuke

# R2-detach ch=32 (teacher density PathoSAM tự build cache lần đầu ~45' cho fold3)
REPO=/workspace/sam3_research $M python distill_student_r2.py --dataset pannuke \
  --pannuke_root $RT --pannuke_folds 3 --epochs 80 --student_ch 32 \
  --w_density 1.0 --w_count 0.01 --w_nll 0.01 --detach_mu --out work/student_r2_pannuke_f3.pkl

# KD baseline ch=32 (teacher foreground, lượt PathoSAM thứ 2 ~45')
REPO=/workspace/sam3_research $M python distill_student_nuinsseg.py --dataset pannuke \
  --pannuke_root $RT --pannuke_folds 3 --lambda_kd 1.0 --epochs 60 --student_ch 32 \
  --out work/student_kd_pannuke_f3.pkl

# CỔNG generalization: R2 có thắng KD giống NuInsSeg không?
REPO=/workspace/sam3_research $M python eval_r2_grouped.py \
  --preds work/student_r2_pannuke_f3.pkl --kd work/student_kd_pannuke_f3.pkl \
  --seeds 20 --n_clusters 5 --min_group 15 --out grouped_pannuke_f3.json
```
**Đọc cổng:** R2-cluster vs KD — Winkler thấp hơn + worst-org cao hơn (paired p<0.05)? Nếu CÓ →
core generalize sang dataset 2 → mở rộng cả 3 fold (`--pannuke_folds 1,2,3`) rồi làm 3b. Nếu KHÔNG →
phân tích vì sao (PanNuke ảnh nhỏ 256, mật độ khác NuInsSeg) trước khi đi tiếp.

### 7.3 Chạy 3b — PanNuke K>1 (SAU khi 3a dương tính) — CHƯA code
Thiết kế (hybrid, xem mục "Thách thức K>1"): student 5 density-head + 5 log-σ; total density distill từ
PathoSAM (class-agnostic); per-class density supervised từ GT type (PanNuke masks 5 kênh); NLL per-class;
clustered conformal per-class. TODO code: (a) `DensitySigmaUNet` → K kênh out; (b) `r2_loss` cộng per-class;
(c) build per-class GT density; (d) eval per-class Winkler/coverage (macro). Tái dùng được `type_head`/
`pannuke_loader` của Paper 1.

## 5. Đóng góp dự kiến (định vị paper)

## 5. Đóng góp dự kiến (định vị paper)
- **Method:** Distributional Count Distillation — nén foundation model đếm (PathoSAM/SAM3) → student nhẹ
  xuất phân phối đếm (μ,σ), σ học từ lỗi thật; + clustered conformal khai thác σ để cân bằng conditional
  coverage. **Model mới + loss mới.**
- **Trục đóng góp:** reliability (conditional coverage) + accuracy khi NÉN — trục chưa ai báo cáo cho
  counting + distillation (đã verify novelty, xem `DISTILLATION_COUNTING_GAP_ANALYSIS.md`).
- **Không phải** đua PQ segmentation (bão hoà); là "energy-efficient + trustworthy counting".

## 6. Trạng thái file/artefact
- `distill_student_r2.py` — trainer R2 (có --w_density/--w_count/--w_nll để ablation).
- `eval_r2_conformal.py` — conformal global (cổng KD vs R2).
- `eval_r2_grouped.py` — global/mondrian/cluster + (đang thêm) paired significance.
- `work/student_kd.pkl`, `work/student_r2.pkl` — đã train trên vast (KD MAE 22.34, R2 MAE 18.20).
- `gate_r2.json`, `grouped_r2.json` — kết quả cổng đã lưu.

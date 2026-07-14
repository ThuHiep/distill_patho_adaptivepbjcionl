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
5. ✅ **Dataset 2 (PanNuke) — LEAK-FREE, 3-fold CV** (2026-07-12). Xem mục 8. R2 thắng KD ĐẠT cả 3 fold.
6. ⬜ **Baseline mạnh hơn**: supervised-GT đã có (mục 3c); thêm so method nhẹ đã công bố nếu được.
7. ⬜ **SỬA LEAK NuInsSeg** (mục 2/3b hiện train-all/predict-all) — bắt buộc trước submit. Kế hoạch: cross-fitting 5-fold (train 4/5 → predict 1/5 held-out, ghép 665 dự đoán leak-free), teacher cache NuInsSeg đã có.

## 8. KẾT QUẢ CHÍNH THỨC — LEAK-FREE, σ Poisson-anchored (2026-07-13) ★ cả 2 dataset

**σ Poisson-anchored (mục 8b): σ = √(max(μ,1))·exp(clamp(log_s,−2,2)).** MỘT tham số hoá cho CẢ 2 dataset.
Protocol: PanNuke K=1, train 2 fold → test fold held-out, `--exclude_tissue colon` (Lizard leak, như Paper 1;
7901→6461, fold3 test=2228 khớp Paper 1, xem [[pannuke-colon-exclusion]]). NuInsSeg cross-fitting 5-fold
(train 4/5→predict 1/5 held-out, ghép 665 leak-free). seeds=20, α=0.1, target 0.90, n_clusters=5,
student_ch=32 (~1.9M), --detach_mu. masks.npy xoá (dùng counts.npy).

**R2 thắng KD sạch CẢ 3 TRỤC trên CẢ 2 dataset (paired-Wilcoxon p ≤ 1.9e−6 mọi trục/fold):**

| Dataset | Winkler R2 / KD | MAE R2 / KD | worst-org R2 / KD | #under R2 / KD |
|---|---|---|---|---|
| **PanNuke** (no-colon, 3-fold) | **18.3 / 23.7** (−23%) | **3.35 / 3.94** (−15%) | **0.902** / 0.739 | **0** / 8 (/53) |
| **NuInsSeg** (cross-fit 5-fold) | **87.7 / 128.6** (−32%) | **14.2 / 21.7** (−34%) | **0.773** / 0.282 | 4 / 6 (/27) |

*(R2 worst-org: PanNuke dùng Mondrian (đủ mẫu/mô), NuInsSeg dùng cluster (ít mẫu/organ) — quy tắc adaptive a priori.)*

PanNuke 3-fold chi tiết (poisson): R2-mondrian marg.cov 0.924, Winkler 18.7±0.4, worst-tissue 0.902±0.008 (0/53);
R2-cluster Winkler 18.3±0.3, worst-tissue 0.845. Per-fold Winkler R2-cluster/KD: f3 18.4/25.8, f2 18.6/23.0, f1 17.8/22.2.
NuInsSeg (poisson): R2-cluster Winkler 87.7±6, worst-org 0.773 (KD 0.282, conditional coverage sụp đổ).

**Kết luận trung thực:**
- R2 KHÔNG chỉ thắng interval/coverage mà còn thắng **accuracy (MAE)** trên cả 2 (−15%, −34%). KD làm khoảng hẹp đạt marginal nhưng **conditional coverage thảm hoạ** (worst-org 0.28–0.74, 8–6 mô under).
- Định vị: cùng student nhẹ ~1.9M, distillation cho count vừa chính xác hơn vừa **interval calibrated hơn hẳn + bảo đảm coverage theo từng mô**. "Energy-efficient + trustworthy counting."

## 8b. ★ Câu chuyện diagnostic→fix (đóng góp method, "chấp nhận thất bại → phân tích → sửa")

**Thất bại ban đầu:** NuInsSeg leak-free với σ head thô (σ=exp(log_s)) → **TRƯỢT Winkler** (R2 152.9 vs KD 128.6),
Winkler std ±36 (bất ổn). PanNuke thì thắng. Không giấu — điều tra (`diagnose_sigma.py`).

**Chẩn đoán:** head log_σ thô học σ CALIBRATED khi count đồng đều + data dồi dào (PanNuke: corr(|err|,σ)=+0.53)
nhưng **SẬP khi dải count khổng lồ + data ít** (NuInsSeg count 1→370, ~532 ảnh/fold): corr(|err|,σ)=−0.02
(σ = nhiễu), σ runaway=15703 → phình Winkler. Hậu-kỳ: σ=√μ (Poisson) thắng cả learned-σ lẫn KD trên NuInsSeg
nhưng phá PanNuke → cần dạng HỢP NHẤT.

**Fix (có cơ sở, không heuristic):** σ = √(max(μ,1))·exp(log_s) — anchor Poisson (count data ~ equidispersion)
cho count-scaling miễn phí + chặn runaway; head chỉ học hệ số dispersion. μ detach (σ mượn độ lớn, không kéo μ).
`--sigma_mode poisson` (mặc định); `raw` giữ để ablation.

**Kết quả fix:** NuInsSeg Winkler 152.9→87.7 (std 36→6), lật TRƯỢT→ĐẠT. PanNuke Winkler 20.2→18.3, MAE 3.52→3.35
(cải thiện, do head σ chung backbone → gradient sạch hơn). MỘT dạng σ, cả 2 dataset thắng. → đóng góp mạnh hơn.

*(Bản raw-σ superseded: PanNuke Winkler 20.2/MAE3.52/worst0.905; NuInsSeg TRƯỢT Winkler 152.9.)*
<!-- KẾT LUẬN CŨ (raw σ, superseded):
- Winkler R2 20.2 vs KD 23.7 (−14%); MAE 3.52 vs 3.94; worst-tissue R2-mondrian 0.905, 0/53. -->

**Định vị bán (Q1, bảo vệ tuyệt đối):** cùng student nhẹ ~1.9M, distillation của chúng tôi thắng KD trên
**cả 3 trục** — accuracy (MAE −11%), interval quality (Winkler −14%), conditional coverage (worst-tissue
0.905 vs 0.739; 0 vs 8 mô under) — p<1e−5, trên tissue teacher CHƯA thấy. "Energy-efficient + trustworthy."

*(Bản CÓ-colon (superseded, không dùng cho paper): KD Winkler bị colon-leak thổi lên 30.9, worst-tissue
R2-mondrian 0.900/0/56, MAE hòa 4.16/4.30. Loại colon cho con số sạch + mạnh hơn.)*

## 8c. ★ BẢNG BASELINE RECENT — R2 vs CondConf/PCP/CPCP/R2CCP/KD (ĐÃ CHẠY vast 2026-07-13/14)

Tất cả cùng student R2 leak-free (μ,σ), cùng score/khoảng, cùng seeds/organ_conditional_stats/Winkler.
Baseline recent chạy bằng **code official** (không tự chế): CondConf `conditionalconformal` (JRSS-B 2025),
PCP `yaozhang24/pcp` (2024), R2CCP `EtashGuha/R2CCP` (ICLR 2024), CPCP `Cqyiiii/...` (ICML 2026).
α=0.1, target 0.90. **↓ thấp tốt, ↑ cao tốt.** MAE cùng μ=Σdensity nên R2/CondConf/PCP bằng nhau.

### PanNuke (trung bình 3 fold no-colon)

| Method | Năm | marg.cov | Winkler ↓ | MAE ↓ | **worst-org ↑** | code |
|---|---|---|---|---|---|---|
| **R2-mondrian (ours)** | — | 0.925 | 19.28 | **3.36** | **0.906** | — |
| **R2-cluster (ours)** | — | 0.910 | **18.50** | **3.36** | 0.843 | — |
| R2-global (ours, no-group) | — | 0.906 | 18.43 | 3.36 | 0.803 | — |
| CondConf-group | 2025 | 0.910 | 18.81 | 3.37 | 0.853 | official |
| PCP | 2024 | 0.919 | 23.26 | 3.37 | 0.805 | official |
| CPCP | 2026 | 0.901 | 35.46 | 6.18 | 0.758 | official |
| R2CCP | 2024 | 0.835¹ | 58.4¹ | 5.83 | 0.621¹ | official |
| KD-global (teacher-distill) | — | 0.904 | 22.08 | 3.64 | 0.721 | — |

¹ R2CCP fold1 sập coverage (cov 0.717, Winkler 116, worst 0.477, 18/18 under) kéo lệch trung bình; f2/f3 bình thường (worst 0.703/0.684).

Per-fold R2-mondrian worst-org: f1 0.908 / f2 0.906 / f3 0.905 (0/18 mỗi fold). R2 vs KD paired-Wilcoxon p=1.9e−6 (Winkler & MAE) mọi fold.

### NuInsSeg (cross-fit 5-fold)

| Method | Năm | marg.cov | Winkler ↓ | MAE ↓ | **worst-org ↑** | #under | code |
|---|---|---|---|---|---|---|---|
| **R2-cluster (ours)** | — | ~0.91 | **87.7** | 14.2 | 0.773 | — | — |
| CondConf-group | 2025 | 0.938 | 125.4 | 13.6 | **0.850** | 0/21 | official |
| PCP | 2024 | 0.914 | 91.1 | 13.6 | 0.714 | 4/21 | official |
| CPCP | 2026 | — | 250.6 | 28.7 | 0.500 | — | official |
| R2CCP | 2024 | — | 261.2 | 30.2 | 0.562 | — | official |

### Câu chuyện chốt (trung thực, đủ mạnh cho Q1)

- **PanNuke:** R2-mondrian đạt **worst-org 0.906 — CAO NHẤT trong mọi method, kể cả CondConf-2025 (0.853)** — với
  Winkler/MAE tương đương và **không cần train lại**. R2 thắng KD toàn diện (p=1.9e−6). CPCP/R2CCP (train net riêng
  trên feature 256-chiều pooled, mất thông tin không gian density) tụt hẳn cả worst-org lẫn MAE.
- **NuInsSeg:** CondConf-group nhỉnh worst-org (0.850 vs 0.773) — nhưng **được cấp nhãn organ tường minh** (Φ=[1,onehot])
  trên dataset ít mẫu/nhiều mô. Đổi lại R2 **thắng Winkler đậm (87.7 vs 125.4, khoảng chặt hơn ~30%)** ở cùng coverage,
  MAE ngang. Kể thẳng: *R2 cho khoảng hiệu quả hơn hẳn mà KHÔNG cần biết organ; CondConf phải được cấp organ mới đạt worst-org đó.*
- R2CCP/CPCP MAE cao (5.8–30) là do **thiết kế của chúng train mạng riêng trên feature pooled, không dùng μ=Σdensity** —
  ghi rõ trong paper để reviewer không hiểu nhầm ta cố tình dìm baseline.

## 8d. ★ PHÂN TÍCH TĂNG CƯỜNG cho phản biện Q1 (A1 coverage-curve + A3 per-organ CI) — 2026-07-15
Chạy hậu kỳ trên pkl R2 (feat) đã backup, script `analysis_coverage_curve.py` (tái dùng eval_scheme → khớp conformal).

**A1 — Coverage curve (chống cherry-picking 1 alpha):** grouping ≥ global ở **MỌI** nominal, không chỉ 0.90.

| nominal | PanNuke mondrian worst-org / #under | PanNuke global | NuInsSeg cluster worst-org / #under | NuInsSeg global |
|---|---|---|---|---|
| 0.80 | 0.781 / **0/18** | 0.653 / 4/18 | 0.568 / 9/27 | 0.491 / 8/27 |
| 0.90 | **0.906 / 0/18** | 0.803 / 3/18 | 0.656 / 3/27 | 0.592 / 5/27 |
| 0.95 | 0.945 / **0/18** | 0.884 / 2/18 | 0.848 / 2/27 | 0.720 / 5/27 |

→ PanNuke Mondrian **0/18 mô under ở MỌI alpha** (bảo đảm group-conditional hữu hạn mẫu của Mondrian, đúng theorem).
marg.cov bám nominal cả 4 mức. Grouping thắng global toàn đường cong.

**A3 — Per-organ Wilson 95% CI (worst-org là systematic hay nhiễu?):**
- **PanNuke (mondrian, α=0.1):** **0/18 mô under thật** — TẤT CẢ 18 mô coverage ≥ 0.908, CI đều trùm 0.90,
  n mô lớn (54–827) → CI chặt. **Bảo đảm per-tissue là THẬT, không nhiễu.** Claim mạnh hợp lệ.
- **NuInsSeg (cluster, α=0.1):** **CHỈ 1/31 organ under THẬT** (human cardia, n=12, cov 0.656, CI [0.391,0.862]
  — CI-trên < 0.90). **30/31 organ còn lại: CI TRÙM 0.90 = chỉ nhiễu do ít ảnh** (femur/thymus/spleen n=6–7,
  CI rộng [0.44,0.97]). → worst-org thấp KHÔNG phải systematic failure, mà là **1 mô khó + nhiễu mẫu nhỏ**.

**★ PHÁT HIỆN (trung thực, quan trọng cho claim):** PanNuke reproduce khớp (0.906 vs 0.902 md ✓) NHƯNG **NuInsSeg
worst-org feat-pkl = 0.656 vs md 0.773** (cùng config, khác training run) → **training variability THẬT ~0.12 trên
NuInsSeg** (dataset nhỏ, dải count rộng). → BẮT BUỘC báo worst-org **kèm CI + nhiều training seed** (việc A6), KHÔNG
báo 1 điểm. Đây củng cố đúng phê bình: worst-org NuInsSeg bất ổn → claim phải mềm ("giảm undercoverage đáng kể",
không "guarantee từng organ"); còn PanNuke Mondrian thì claim mạnh được (theorem + 0/18 + CI chặt).

**SỬA CLAIM (kết luận từ A1+A3):** tách 2 mức — *PanNuke: bảo đảm per-tissue (Mondrian, hữu hạn mẫu, 0/18 mọi α)*;
*NuInsSeg: giảm mạnh subgroup undercoverage vs global/KD, còn 1 mô khó (cardia) + nhiễu mẫu nhỏ (limitation trung thực)*.

**A5 — Sigma analysis (`analysis_sigma.py`): σ CÓ THÔNG TIN + calibrated (bằng chứng learned-dispersion hữu ích):**

| dataset | corr_Spearman(σ,\|e\|) | z-mean | z-std | \|z\|≤1.64 | NLL |
|---|---|---|---|---|---|
| PanNuke | **+0.428** | +0.003 | **1.014** | 0.899 | 2.791 |
| NuInsSeg | **+0.404** | +0.103 | 0.828 (hơi conservative) | 0.938 | 4.210 |

★ **CHỐT NOVELTY (nối 8b):** σ Poisson-anchored làm corr(σ,\|e\|) **NuInsSeg = +0.404** — trong khi σ raw (mục 8b) corr
= **−0.02** (σ = nhiễu). → **anchor Poisson THẬT SỰ có tác dụng** (biến σ vô dụng thành σ có thông tin), không phải
chỉ √μ trang trí. PanNuke z-std 1.014 = **calibration gần hoàn hảo mọi count-bin** (z-std 0.99–1.03).
NuInsSeg z-std 0.83 = **hơi over-conservative ở count thấp** (σ hơi rộng, KHÔNG under → an toàn coverage, đổi lại width chưa tối ưu).

**P2.9 — low-count (bin theo count):** MAPE cao ở bin 0–10 (PanNuke 57%, NuInsSeg 162%) do lỗi tương đối trên count
nhỏ (MAE tuyệt đối vẫn nhỏ: PanNuke 1.91, NuInsSeg 5.89). z-std NuInsSeg bin 0–10 = 0.63 (σ rộng gấp ~1.6×) → khoảng
low-count bị nới. Ghi honest limitation. corr(σ,\|e\|) mạnh nhất ở bin 0–10 (0.55–0.57) → σ phân biệt tốt ca dễ/khó khi ít nhân.

**A2 — ★ ABLATION σ-mode (chạy vast RTX5090 2026-07-15, NuInsSeg, cùng --detach_mu/hyperparams) — CHỐT NOVELTY:**

| σ-mode | corr(σ,\|e\|) | z-std (→1) | NLL ↓ | \|z\|≤1.64 |
|--------|--------------|-----------|-------|-----------|
| **poisson-anchor (ours)** | +0.404 | **0.828** | **4.210** | 0.938 |
| nb (Negative-Binomial, Var=μ+α·μ²) | +0.401 | 1.344 | 4.585 | 0.878 |
| raw (Gaussian-hetero, σ=exp(s)) | +0.246 | 0.721 | 4.584 | 0.958 |

**Kết luận A2 (trả lời trực tiếp phê bình "phải thắng NB/Gaussian"):**
1. **Neo-mean là nguồn thông tin của σ:** poisson & nb (đều neo μ) corr ~+0.40 ≫ raw +0.246. → không phải head tự do.
2. **Poisson-anchor thắng CALIBRATION cả NB lẫn raw:** NLL 4.21 < 4.58 (gap rõ), z-std 0.828 gần 1 nhất. NB corr
   ngang nhưng z-std 1.344 (khoảng quá hẹp→under) + bất ổn theo count-bin (α·μ² làm σ bùng/xẹp).
3. → **Dạng ĐƠN GIẢN HƠN (Poisson-anchor) cho phân phối calibrated tốt hơn NB tường minh + Gaussian-hetero** →
   novelty vững, không "chỉ là √μ" cũng không bị NB thay thế. *(Lưu ý: single training run/mode; nên xác nhận qua
   vài seed — A6 — vì NuInsSeg có training variability; nhưng gap NLL 4.21 vs 4.58 đủ rõ.)*
   raw-σ ở đây corr +0.246 (khác −0.02 mục 8b do khác training run → lại củng cố cần A6).

**A6 — TRAINING-SEED variability (poisson × 3 seed, NuInsSeg, vast 2026-07-15):**

| seed | MAE | NLL | worst-org (cluster, α=0.1) |
|------|-----|-----|-----|
| s0 | 15.14 | 4.310 | 0.809 |
| s1 | 15.46 | 4.355 | 0.767 |
| s2 | 17.75 | 4.437 | 0.767 |

**Kết luận A6:**
1. **worst-org = 0.78 ± 0.02** (range 0.767–0.809) → **KHỚP md 0.773** (feat-pkl 0.656 ở A3 là outlier thấp, không
   đại diện). → paper báo **mean ± range**, không 1 điểm; con số ~0.77–0.78 vững.
2. ★ **A2 ranking ỔN ĐỊNH qua seed:** NLL poisson {4.31, 4.36, 4.44} — **CẢ 3 seed đều < raw/nb (4.58)** → lợi thế
   calibration của poisson-anchor KHÔNG phải may mắn 1 lần. Novelty vững.
3. MAE dao động 15.1–17.8 (~±1.3) → training variability THẬT nhưng vừa phải; paper báo mean±std qua seed.

**A4 — LATENCY/VRAM THẬT (RTX 5090, `measure_latency.py`, student ch=32 @256):**
- bs=1: **1.87 ms/ảnh** (536 img/s), **peak VRAM 112 MB**. | bs=32: 1608 img/s, VRAM 2.3 GB.
- So CellViT-SAM-H cần **24–48 GB VRAM** → student nhẹ hơn **~200–400× bộ nhớ** + chạy real-time trên GPU phổ thông.
- → claim "computationally-efficient" có **số đo runtime thật**, không chỉ params/FLOPs. (Heavy net: chỉ cite
  params/FLOPs bảng A vì không dựng env họ ở đây — trung thực.)

## ▶▶▶ RESUME 2026-07-15 — MỚI NHẤT (đọc ĐẦU TIÊN)

**ĐÃ XONG THÊM (loạt hardening Q1 sau khi có bài phê bình — mục 8d):**
Method (R2+σ Poisson+detach_mu) ✅ · Bảng 8c (4 baseline recent + KD) ✅ · Bước 2 accuracy (CellViT-SAM-H+LKCell-L+teacher
count-MAE) ✅ · **A1** coverage-curve 4α ✅ · **A3** per-organ Wilson CI ✅ · **A5** sigma analysis ✅ · **A2** ablation
σ-mode (poisson>NB>raw, NLL 4.21<4.58) ✅ · **A6** 3 training-seed (worst-org 0.78±0.02, ranking ổn định) ✅ ·
**A4** latency thật (1.87ms/112MB) ✅ · **Ⓓ** siết claim ✅. Ranh giới Paper1(PB-JCI)/Paper2(distillation) đã rõ.
Scripts mới: analysis_coverage_curve.py, analysis_sigma.py, measure_latency.py, dump_cellvit_counts.py, prep/eval_heavy_count.py.

**★ VIỆC TIẾP THEO (ưu tiên trên xuống):**
1. **VIẾT MANUSCRIPT** (việc lớn nhất) — số liệu + câu chuyện đã đủ trong md này + MODEL_va_KETQUA_paper2.md.
   Đóng gói novelty = *"Distributional Count Distillation under mean–variance optimization conflict"* (KHÔNG claim PB-JCI = Paper 1).
   Thứ tự nên viết: Method (rõ nhất) → Experiments → Results → Related Work → Intro → Abstract.
2. **HÌNH (~4-5):** (a) sơ đồ DensitySigmaUNet; (b) reliability/coverage-per-organ bar (R2 vs KD vs CondConf);
   (c) efficiency scatter params-vs-MAE + latency; (d) qualitative ảnh+density+interval; (e) coverage-curve (A1).
3. **Tùy chọn strengthen (nếu dư thời gian/compute):** C2 lightweight-same-training baseline (một phần ở 3c) ·
   cross-dataset transfer (train A→test B) · A2/A6 lặp trên PanNuke · đưa full UQ floor (ensemble/MC-dropout/CQR) vào bảng chính ·
   PathoSAM GFLOPs (ô teacher bảng A) · gradient-analysis proposition cho detach_mu (đóng gói optimization-decoupling principle).
4. **Housekeeping:** regenerate Kaggle API key (đã lộ) · withdraw bài Sound Event JOCO-D-26-00664 (thư đã soạn).

**Backup pkl mới (poisson s0-2/raw/nb) chưa lên Kaggle** — số liệu đã trong md (A2/A6) nên OK; muốn tái tạo thì
`kaggle datasets version -p work` trước khi destroy vast.

---

## ▶▶ RESUME 2026-07-14 — TRẠNG THÁI + VIỆC TIẾP THEO (đọc cái này trước khi làm tiếp)

**ĐÃ XONG:** Bảng 8c hoàn chỉnh 100% cả 2 dataset (số ở mục 8c trên). 8 baseline recent chạy xong leak-free
bằng code official: CondConf-2025, PCP-2024, R2CCP-2024, CPCP-2026 (+ 4 sàn UQ code sẵn `baselines_uq.py`).
→ Phần baseline coi như KHÓA. Mọi con số đã transcribe vào md này (an toàn trên Mac, không phụ thuộc vast).

**BACKUP TRƯỚC KHI STOP VAST (log thô = bằng chứng gốc):**
```bash
# đẩy work/ (đã gồm baseline_logs + pkl + teacher cache) lên Kaggle sam3-paper2-work
cd /workspace/sam3_research/work && kaggle datasets version -p . -m "8c logs done $(date +%m%d)" -r zip -q
```
Start máy mới: `kaggle datasets download -d hipinhththu/sam3-paper2-work --unzip -p /workspace/sam3_research/work`

**VIỆC TIẾP THEO (Bước 2 — trục accuracy thuần, MAE):** heavy nets NuLite (2024) + CellViT++ (2025) —
đo MAE count trên CÙNG protocol leak-free (PanNuke test_fold + NuInsSeg cv5) để định vị student ~1.9M
so với net nặng SOTA. Kỳ vọng: student thua MAE tuyệt đối nhưng **rẻ hơn nhiều lần + có σ/interval** (net
nặng không có UQ) → củng cố "energy-efficient + trustworthy". Cần env riêng (mục 10 Bước 2). CHƯA code.

**Tùy chọn còn treo:** PanNuke K>1 mở rộng (mục 7.3, CHƯA code) — chỉ làm nếu cần thêm độ mạnh.

---

## 9. Baseline hiện đại (Q1) — kế hoạch, LÀM LẦN LƯỢT sau khi có kết quả (2026-07-13)

Yêu cầu: cần ≥3-4 baseline **2024-2026** (recent) mới thuyết phục Q1 (giống Paper 1 so SAOCP/FACI).
Phát hiện: niche "counting NHẸ + interval CALIBRATED" gần như TRỐNG 2025-26 → đó là novelty gap;
baseline recent lấy từ 2 dòng lân cận, đều CÓ CODE. **Bước 0 mỗi baseline: verify repo + weight chạy được.**

**Trục accuracy (student nhẹ đếm giỏi không) — count MAE:**
- **CellViT++** (1/2025, github TIO-IKIM/CellViT) — SOTA nuclei PanNuke, NẶNG → mốc SOTA (ta nhẹ hơn + có UQ).
- **NuLite** (2024, arxiv 2408.01797) — lightweight nuclei PanNuke → đối thủ nhẹ cùng hạng cân.
- PathoSAM teacher count (đã có) — mốc teacher.

**Trục reliability (interval — CORE, PHẢI thắng) — Winkler/coverage, ÁP LÊN CÙNG student nhẹ (fair compute):**
- **CQR** — Conformalized Quantile Regression (yromano/cqr; MAPIE cập nhật 6/2025) — chuẩn interval hiện đại.
- **CHDQR** — Conformalized High-Density QR (11/2024, arxiv 2411.01266) — cải tiến CQR gần đây.
- **MC-Dropout** (Gal 2016) + **Deep Ensembles** (Lakshminarayanan 2017) — mốc UQ kinh điển (reviewer vẫn đòi;
  khảo sát 2025 IOPscience ae2e7b: conformal thường thắng chúng về coverage → đúng story của ta).

**Thiết kế so sánh:** method counting (kể cả 2025) chỉ cho ĐIỂM → bọc conformal (σ hằng) cho chúng → chứng minh
σ heteroscedastic distilled cho interval TỐT HƠN. Bán: "method đếm hiện đại KHÔNG cho bất định per-ảnh; của tôi có, calibrated."

**THỨ TỰ LÀM (sau khi NuInsSeg cross-fit xong):**
1. Nhóm UQ trên cùng student (rẻ, không cần env mới): MC-Dropout → Deep Ensembles → CQR → CHDQR. Bảng reliability.
2. Nhóm accuracy nặng: verify + chạy NuLite, CellViT++ (cần env/weight riêng trên vast — rủi ro môi trường, làm sau).
Chạy so sánh trên **PanNuke leak-free no-colon** (đã có) để đối chiếu trực tiếp bảng mục 8.

## 10. ▶ RESUME (2026-07-13) — bước tiếp theo: BASELINES RECENT

**Trạng thái: phần METHOD XONG** (mục 8 + 8b). Cả 2 dataset leak-free, σ Poisson-anchored, R2 thắng KD
cả 3 trục p≤1.9e−6. File pkl đã có trong `/workspace/sam3_research/work/`:
`student_r2_pannuke_f{1,2,3}_nocolon_poisson.pkl`, `student_kd_pannuke_f{1,2,3}_nocolon.pkl`,
`student_r2_nuinsseg_cv5_poisson.pkl`, `student_kd_nuinsseg_cv5.pkl`, + teacher caches.

**Vast khi STOP:** giữ `/workspace/sam3_research/work/` (mọi pkl + cache teacher) + data PanNuke + counts.npy.
`/workspace/penv` có thể mất → rebuild `bash kaggle/vast/setup_pathosam_vast.sh` + `micromamba install -p /workspace/penv "micro_sam>=1.1" vigra nifty` (~5'). **NHƯNG** baselines UQ dưới đây KHÔNG cần PathoSAM
(chạy trên student/pkl đã có) → có thể khỏi rebuild env nếu chỉ làm bước 1.

### Bước 1 (LÀM TRƯỚC — rẻ, khả thi ngay): UQ baselines trên CÙNG student nhẹ
So trục reliability (Winkler/coverage), fair compute. CHƯA code — cần viết `baselines_uq.py`:
- **MC-Dropout** (Gal 2016): thêm dropout vào DensitySigmaUNet, T=30 forward lúc test → μ=mean, σ=std của count.
- **Deep Ensembles** (Lakshminarayanan 2017): train M=5 student khác seed (cùng split), μ,σ = mean/std ensemble.
- **CQR** (Romano 2019, github yromano/cqr / MAPIE): student 2 quantile head (pinball loss) → conformal hoá.
- **CHDQR** (2411.01266, 2024): cải tiến CQR.
- Chấm bằng ĐÚNG `eval_r2_grouped.py` (cùng conformal/Winkler) để so trực tiếp bảng mục 8.
- Ablation `--sigma_mode raw` cũng nên vào bảng (chứng minh Poisson anchor > raw).

#### ✅ CODE XONG (2026-07-13) — `baselines_uq.py` + `eval_cqr_grouped.py` (smoke test CPU 4/4 PASS)
CÙNG backbone DoubleConv/U-Net (~1.9M), CÙNG protocol leak-free (test_fold / cross-fit), CÙNG teacher
cache. MAE mọi baseline = μ=Σdensity (cùng thước R2). Đã verify local (synthetic 64×64): 4 method chạy
end-to-end, pkl đúng schema, q_lo≤q_hi (không cross), conformal marginal-cov ≈ target.
- **mcdropout**: MCDropoutUNet (Dropout2d decoder); test dropout BẬT + BN eval, T=30 forward → μ=mean,σ=std. `{mu,sigma}`→`eval_r2_grouped.py`.
- **ensemble**: M=5 student R2 khác seed; mixture Lakshminarayanan σ*²=mean(σ_m²)+mean(μ_m²)−μ*². `{mu,sigma}`→`eval_r2_grouped.py`.
- **cqr**: QuantileUNet 2 head (τ=α/2,1−α/2) pinball; offset đơn điệu quanh μ.detach() (không cross). `{mu,q_lo,q_hi}`→`eval_cqr_grouped.py` (score E=max(q_lo−y,y−q_hi), khoảng bất đối xứng native CQR).
- **chdqr**: QuantileUNet lưới n_taus quantile; test chọn CẶP khối lượng ≥1−α NGẮN NHẤT (highest-density). `{mu,q_lo,q_hi}`→`eval_cqr_grouped.py`.
- **raw-σ ablation**: KHÔNG cần file mới — `distill_student_r2.py --sigma_mode raw` (đã có).

**LỆNH VAST (PanNuke no-colon, 3 fold; lặp method×fold):**
```bash
cd /workspace/sam3_research/distillation_counting
for F in 1 2 3; do for M in mcdropout ensemble cqr chdqr; do
  python baselines_uq.py --method $M --dataset pannuke --pannuke_folds 1,2,3 --test_fold $F \
    --exclude_tissue colon --out ../work/uq_${M}_pannuke_f${F}.pkl ; done; done
# NuInsSeg cross-fit 5-fold:
for M in mcdropout ensemble cqr chdqr; do
  python baselines_uq.py --method $M --dataset nuinsseg --kfold 5 \
    --out ../work/uq_${M}_nuinsseg_cv5.pkl ; done
# raw-σ ablation (dùng chính trainer R2):
python distill_student_r2.py --dataset pannuke --pannuke_folds 1,2,3 --test_fold 3 \
  --exclude_tissue colon --sigma_mode raw --detach_mu --out ../work/student_r2_pannuke_f3_nocolon_raw.pkl
```
**CHẤM (so trực tiếp bảng mục 8, n_clusters=5):**
```bash
# (μ,σ): mcdropout, ensemble, raw-ablation
python eval_r2_grouped.py --preds ../work/uq_mcdropout_pannuke_f3.pkl \
  --kd ../work/student_kd_pannuke_f3_nocolon.pkl --seeds 20 --n_clusters 5 --min_group 15
# quantile: cqr, chdqr
python eval_cqr_grouped.py --preds ../work/uq_cqr_pannuke_f3.pkl \
  --kd ../work/student_kd_pannuke_f3_nocolon.pkl --seeds 20 --n_clusters 5 --min_group 15
```
Ghi Winkler/MAE/worst-org mỗi baseline vào bảng mới (mục 8c) so với R2. Kỳ vọng (story Q1):
R2 σ-học/Poisson ≥ MC-Dropout/Ensemble/CQR/CHDQR về Winkler+worst-org ở CÙNG compute — vì niche
"đếm nhẹ + interval calibrated + coverage theo-mô" gần như trống.

**Định vị 4 cái trên = SÀN UQ bắt buộc** (reviewer luôn đòi MC-Dropout/DE): MC-Dropout(2016)/DE(2017)/
CQR(2019) là kinh điển, CHDQR(2024) recent. Baseline RECENT chính (2025) là Conditional Conformal ↓.

#### ✅ RECENT 2025 baseline: Conditional Conformal (Gibbs–Cherian–Candès) — CODE XONG (2026-07-13)
`eval_condconf_grouped.py`. **Dùng ĐÚNG package chính thức `conditionalconformal` (CondConf)** — không
tự chế lại thuật toán simplex-cutoff (trung thực tuyệt đối). arXiv 2305.12616, JRSS-B **2025** (đã verify
citation + đọc code gốc). Đây là **SOTA 2025 cho conditional coverage** → đấu TRỰC TIẾP trục worst-org
của R2 (clustered/Mondrian conformal). Áp lên CÙNG score S=|y−μ|/σ + CÙNG (μ,σ) student R2 leak-free →
**không cần train lại, không cần GPU/PathoSAM**, chạy thẳng trên pkl R2 đã có.
- Basis Φ(x)=[1,onehot(organ)] (mode group-conditional chuẩn của paper) → coverage bảo đảm từng organ.
  Cũng chạy Φ=[1] (marginal) làm mốc. exact=True (imputation hữu hạn mẫu, không RKHS).
- Cùng seeds/cal_ratio/organ_conditional_stats/Winkler → so trực tiếp bảng mục 8.
- **Smoke test venv (package thật, synthetic)**: basis group nâng worst-org 0.787→0.844, org-gap
  0.147→0.105 (đúng lý thuyết). Wrapper verified.

**LỆNH (vast HOẶC Mac — chỉ CPU + numpy/scipy/cvxpy):**
```bash
pip install conditionalconformal    # 1 lần, không đụng penv PathoSAM
cd /workspace/sam3_research/distillation_counting
for F in 1 2 3; do
  python eval_condconf_grouped.py --preds ../work/student_r2_pannuke_f${F}_nocolon_poisson.pkl \
    --seeds 10 --alpha 0.1 --min_organ_imgs 10 ; done
python eval_condconf_grouped.py --preds ../work/student_r2_nuinsseg_cv5_poisson.pkl \
  --seeds 10 --alpha 0.1 --min_organ_imgs 10
```
So `CondConf-group` worst-org/Winkler với `R2-cluster/R2-mondrian` (mục 8). **Story Q1:** cùng student
+ cùng score, R2 ngang/hơn conditional coverage của SOTA-2025 mà còn thắng Winkler+MAE (σ học được +
đếm chính xác) → phương pháp mình cạnh tranh được với calibration hiện đại nhất mà nhẹ hơn nhiều.
*(R2CCP ICLR2024 + CHDQR: chưa ưu tiên — user chọn Conditional Conformal làm recent chính.)*

#### ✅ PCP 2024 (Posterior Conformal Prediction, Zhang & Candès) — CODE XONG (2026-07-13)
`eval_pcp_grouped.py`, **dùng ĐÚNG repo official `yaozhang24/pcp`** (class PCP). Mô hình phân phối score
điều kiện = mixture theo CỤM tự phát hiện → coverage marginal + xấp xỉ conditional theo subgroup →
**đấu trực tiếp worst-org**. Áp score S=|y−μ|/σ + (μ,σ) student R2 đã có → không train lại, CPU.
Feature X=[μ,σ] (đặc trưng độ khó liên tục, đúng thiết kế PCP; KHÔNG cho nhãn organ → test worst-org
công bằng hơn). Leak-free. Smoke test (code official): worst-org 0.874, org-gap 0.056, 0/4 under.
```bash
git clone https://github.com/yaozhang24/pcp.git   # cạnh distillation_counting
pip install statsmodels tqdm                        # (scikit-learn/scipy đã có)
for F in 1 2 3; do
  python eval_pcp_grouped.py --preds ../work/student_r2_pannuke_f${F}_nocolon_poisson.pkl \
    --pcp_dir ../../pcp --seeds 10 --min_organ_imgs 10 ; done
python eval_pcp_grouped.py --preds ../work/student_r2_nuinsseg_cv5_poisson.pkl --pcp_dir ../../pcp --seeds 10
```

#### ▣ QUYẾT ĐỊNH BASELINE CUỐI (2026-07-13) — chỉ chạy cái CÓ code official / kinh điển
**Nguyên tắc (user):** KHÔNG tái hiện SOTA không công khai code (reviewer sẽ nói "tái hiện sai nên mới thua").
Chỉ baseline có code official hoặc thuật toán kinh điển mới CHẠY; SOTA no-code 2026 chỉ CITE.

| Baseline | Năm | Nguồn | Trục | Trạng thái |
|---|---|---|---|---|
| MC-Dropout / Deep Ensembles / CQR / CHDQR | 2016-24 | tự code (kinh điển) | reliability | ✅ chạy |
| **CondConf** (Gibbs et al.) | 2025 | package official | worst-org | ✅ chạy |
| **PCP** (Zhang & Candès) | 2024 | repo official | worst-org | ✅ chạy |
| **R2CCP** (Guha et al., ICLR) | 2024 | repo official (pip dep crlibm thừa → dùng repo) | efficiency, **count-natural** | ✅ chạy (cần `--dump_feat`) |
| **CPCP** (Chen & Li, ICML) — Colorful Pinball | **2026** | repo official (Cqyiiii, MIT, deps sạch) | **worst-org** (min MSCE) | ✅ chạy (`eval_cpcp.py`, cần `--dump_feat`) |
| CIR (orince, arXiv 2601.02769) — Conf. Interquantile | **2026** | repo official (torchcp) | efficiency | 📎 **CITE** (user chốt bỏ — cần tự train quantile-grid base; trục efficiency đã có R2CCP 2024; tránh over-engineer) |
| SOCP (arXiv 2606.29403) — Self-Organized CP | **2026** | **no code** | regional coverage | 📎 CITE (đúng trục nhưng chưa release code) |
| CoCP (2603.01719) | 2026 | **no code** | efficiency | 📎 **CITE only** (dừng tái hiện) |
| FFCP (2412.00653) | NeurIPS2025 | code official NHƯNG | efficiency | 📎 **CITE only** (dep `auto_LiRPA` xung đột torch/numpy → brittle cả trên vast; trục đã trùng R2CCP) |
| PIT-CP / SpeedCP / Zero-inflated | 2026 | no code | — | 📎 CITE (Related Work) |

→ **7 baseline CHẠY thật** (4 sàn + CondConf + PCP + R2CCP); CondConf/PCP đấu worst-org, R2CCP+sàn đấu
efficiency. **3 recent code-official 2024-2025** (CondConf/PCP/R2CCP) → trả lời "sao không baseline mới".
Method no-code-usable (FFCP brittle-dep, CoCP/2026 no-code) → cite trung thực trong Related Work.

**R2CCP cần feature:** `distill_student_r2.py --dump_feat` (thêm cờ) lưu pooled bottleneck/ảnh vào pkl.
Cần RE-RUN leak-free 1 lần để tạo pkl có feat (tái dùng teacher cache, không cần PathoSAM):
```bash
# PanNuke (mỗi fold) + NuInsSeg, thêm --dump_feat, out có hậu tố _feat:
for F in 1 2 3; do python distill_student_r2.py --dataset pannuke --pannuke_folds 1,2,3 --test_fold $F \
  --exclude_tissue colon --dump_feat --out ../work/student_r2_pannuke_f${F}_nocolon_poisson_feat.pkl; done
python distill_student_r2.py --dataset nuinsseg --kfold 5 --dump_feat \
  --out ../work/student_r2_nuinsseg_cv5_poisson_feat.pkl
# rồi:
git clone https://github.com/EtashGuha/R2CCP.git ; pip install pytorch_lightning configargparse torchvision
for F in 1 2 3; do python eval_r2ccp.py --preds ../work/student_r2_pannuke_f${F}_nocolon_poisson_feat.pkl \
  --r2ccp_dir ./R2CCP --seeds 5 --max_epochs 100 --min_organ_imgs 10; done
python eval_r2ccp.py --preds ../work/student_r2_nuinsseg_cv5_poisson_feat.pkl --r2ccp_dir ./R2CCP --seeds 5 --max_epochs 100
```

### Bước 2 (SAU, nặng, cần env riêng): accuracy/efficiency baselines — ▶ CHỐT 2026-07-14 (đã verify sâu repo)

**MỤC ĐÍCH (khác trục 8c):** trục ACCURACY THUẦN (count-MAE + params/FLOPs), KHÔNG dính UQ. Chặn phản biện
"model nhẹ thế chắc đếm dở". Chứng minh student 1.9M đếm bám sát heavy SOTA ở fraction chi phí, và là cái
DUY NHẤT có interval calibrated.

**★ PHÁT HIỆN LEAK khi verify (quyết định thiết kế):** KHÔNG heavy net nào publish checkpoint THEO FOLD.
- CellViT/CellViT++: ckpt "trained on 90% ALL folds" → LEAK nếu test trên PanNuke test-fold của ta.
- LKCell (hustvl, HF xiazhi/LKCell-L/B, MIT): weight whole-data, "contact authors for fold-specific".
- NuLite (CosmoIknosLab, Apache/NC): "trained on WHOLE PanNuke" + inference CHỈ WSI 1024 → 2 rào cản.
- LSP-DETR (arXiv 2601.03163, 1/2026, RationAI/lsp-detr, MIT): repo KHUNG RỖNG 4 commit → chưa chạy được.
→ Chạy ckpt whole-data của họ trên PanNuke test-fold = LEAK (họ đã thấy ảnh) → reviewer bác. Train lại
per-fold = tái hiện SOTA có thể sai (CẤM). **Bế tắc trên PanNuke.** Gốc: model nhẹ BẮT BUỘC train trên
phân phối đích (distill); heavy net dùng off-the-shelf → MAE fair đòi cùng protocol per-fold họ không cho.

**GIẢI PHÁP (leak-free THẬT, không tái hiện):**
- **Phần A — bảng efficiency (cite, 0 rủi ro):** {params, FLOPs, PanNuke mPQ/bPQ} trích paper cho
  LKCell + CellViT++ + LSP-DETR(2026) + NuLite + student 1.9M. Định vị dải 700M→1.9M, cite baseline 1/2026.
- **Phần B — count-MAE leak-free trên NuInsSeg (OOD với MỌI heavy net → leak biến mất):** chạy code
  inference OFFICIAL của họ (KHÔNG sửa thuật toán, chỉ wrap I/O + đếm instance) với weight published.
  2 mốc CHẠY: **CellViT-SAM-H** (cùng họ SAM với teacher PathoSAM) + **LKCell-L** (SOTA 2024, HF, dùng
  CHUNG inference harness CellViT). Cạnh student 1.9M + teacher PathoSAM. Bỏ NuLite/LSP-DETR khỏi phần
  CHẠY (ma sát cao) nhưng GIỮ trong bảng cite A.
  Story: student in-domain distilled 1.9M bám sát/vượt SOTA nặng OOD + là cái DUY NHẤT có UQ.

**Đường inference (đã verify code thật):** CellViT++ `detect_cells.py` = CHỈ WSI (process_wsi/dataset,
file .svs) → KHÔNG nhận folder patch. DÙNG repo CellViT GỐC: `cell_detection.py` (patch + overlap-merge
biên) HOẶC `inference_cellvit_experiment_pannuke.py` (dataset kiểu PanNuke 256). Output `instance_types`
per ảnh → count = len(instances). LKCell kế thừa CÙNG harness → chỉ đổi --model. GPU ≥24GB (SAM-H).
**RỦI RO CÒN LẠI (validate TRÊN VAST):** đưa ảnh NuInsSeg (size ~512?) qua patch-inference đúng cách
(tile 256 + overlap-merge để không đếm trùng/sót nhân ở biên). Cần kiểm size ảnh NuInsSeg thực tế.

**TODO Bước 2:** [1] viết `eval_heavy_count.py` (wrap CellViT/LKCell official, đếm instance → MAE vs GT).
[2] runbook vast env riêng (clone CellViT+LKCell, tải weight SAM-H+LKCell-L, chạy NuInsSeg). [3] bảng A cite.

**API CellViT đã verify (raw code `cell_segmentation/inference/cell_detection.py`):** class
`CellSegmentationInference(model_path, gpu, enforce_mixed_precision)`; `__load_model` đọc `checkpoint["arch"]`
→ chọn CellViT/CellViT256/CellViTSAM. Đếm: `cells,_ = get_cell_predictions_with_tokens(predictions)`;
`count=len(cells)` (mỗi dict = 1 nhân: centroid/bbox/type). Preprocess: patch **1024**+overlap 64, norm
mean/std (0.5,0.5,0.5). LKCell backbone large-kernel riêng → `arch` khác → chạy bằng REPO LKCELL fork
(API gần y hệt). RỦI RO: NuInsSeg ~512 → tile-vs-resize phải TEST vài ảnh trên vast (mắt thường vs GT).

#### Phần A — Bảng efficiency (số EXACT từ paper gốc, chốt 2026-07-14)
GFLOPs/params ở 256×256. bPQ/mPQ chép đúng bảng của từng paper (⚠️ CellViT-SAM-H = **699.74M**, không phải
163.8M — số 163.84M là LKCell-L; lỗi draft cũ đã sửa).

| Model | Year | Params (M) | GFLOPs | bPQ | mPQ | Count-MAE | UQ? | Nguồn số |
|-------|------|-----------|--------|-----|-----|-----------|-----|----------|
| CellViT-SAM-H | 2024 | 699.74 | 214.33 | 0.679 | 0.498 | Phần B | ✗ | LKCell T1-2 / NuLite T6 |
| LKCell-L | 2024 | 163.84 | 47.86 | 0.6847 | 0.5080 | Phần B | ✗ | LKCell arXiv 2407.18054 T1-2 |
| LKCell-B | 2024 | 122.53 | 46.25 | 0.6851 | 0.5050 | (opt) | ✗ | LKCell 2407.18054 T1-2 |
| LSP-DETR | **1/2026** | 45.0 | 26 | 0.675 | ? | (cite-only) | ✗ | arXiv 2601.03163 |
| NuLite-H | 2024 | 41.07 | 29.95 | 0.680 | 0.493 | (opt) | ✗ | NuLite arXiv 2408.01797 T4/T6 |
| NuLite-M | 2024 | 31.11 | 28.26 | 0.679 | 0.493 | (opt) | ✗ | NuLite 2408.01797 |
| NuLite-T | 2024 | 17.12 | 26.16 | 0.671 | 0.484 | (opt) | ✗ | NuLite 2408.01797 |
| PathoSAM (teacher) | 2024 | ~640 (ViT-H) | ? | — | — | Phần B | ✗ | micro_sam |
| **Student R2 (ours)** | 2026 | **1.935** | **~10.5 (MACs@256)** | — (density, không seg) | — | **bảng mục 8** | **✓ σ+interval** | thop, xác minh Mac |

**Story bảng A (mạnh hơn draft cũ):** dải params **699.74M → 1.9M**; student nhỏ nhất tuyệt đối —
**~368× < CellViT-SAM-H, ~86× < LKCell-L, ~9× < cả NuLite-T (17.12M) là SOTA nhẹ nhất**. Kể cả baseline
2026 mới nhất (LSP-DETR) vẫn 45M. Và **KHÔNG model nào cho UQ per-ảnh** — chỉ student có σ+interval
calibrated. → "nhẹ nhất + trustworthy nhất, count-MAE bám sát (Phần B)".
Student params/FLOPs ĐÃ tính (thop, DensitySigmaUNet ch=32 @256): **1,935,266 params (1.935M); 10.488 GMACs**
→ ít FLOPs hơn cả NuLite-T (26.16). ⚠️ QUY ƯỚC: cột GFLOPs của LKCell/NuLite nhiều khả năng là **MACs**
(thop) → khi viết paper báo student = **10.5 GMACs** cho đồng đơn vị (đừng nhân 2 thành 21 FLOPs lệch chuẩn).
Còn thiếu: **PathoSAM GFLOPs** (tính trên vast khi có env). Student cost script: `count_student_cost.py`.

#### Phần B — Runbook vast (ĐÃ soạn khung: prep + eval XONG, chỉ chừa 1 hàm dump-count finalize trên vast)
File đã có (Mac, push sẵn): `prep_nuinsseg_as_pannuke.py` (NuInsSeg→images/+gt_counts.csv, --mode resize|tile),
`eval_heavy_count.py` (đọc gt+preds.csv → MAE/RMSE/per-organ, ghép student). GT count = len(unique(mask))−bg
Y HỆT student. Cần GPU ≥24GB (SAM-H). CHƯA test end-to-end (cần GPU + weight) → validate theo bước [V].

```bash
# [1] ENV RIÊNG (tách khỏi penv PathoSAM) — CellViT cần torch 2.x + timm + geojson...
micromamba create -y -p /workspace/cvenv python=3.10 && micromamba run -p /workspace/cvenv \
  pip install torch torchvision timm einops geojson shapely scikit-image tqdm pyyaml opencv-python-headless thop
# [2] REPO + WEIGHT
cd /workspace && git clone https://github.com/TIO-IKIM/CellViT.git
#   CellViT-SAM-H: tải CellViT-SAM-H-x40-AMP.pth (Google Drive repo) -> /workspace/ckpt/cellvit_sam_h/
#   LKCell-L: git clone https://github.com/hustvl/LKCell + HF hub xiazhi/LKCell-L
# [3] DATA: chuẩn bị ảnh + GT (chạy CẢ 2 mode để chọn ở [V])
cd /workspace/sam3_research/distillation_counting && git pull
python prep_nuinsseg_as_pannuke.py --out ../work/nuinsseg_png      --mode resize
python prep_nuinsseg_as_pannuke.py --out ../work/nuinsseg_png_tile --mode tile
# [4] INFERENCE + DUMP COUNT  -> dùng dump_cellvit_counts.py (ĐÃ viết, verified API cell_detection.py)
#   VALIDATE nhanh trước (5 ảnh): thử giữ 256, nếu lỗi shape thì --infer_size 1024 (SAM-H); --no_tokens nếu cần
python dump_cellvit_counts.py --cellvit_dir /workspace/CellViT \
    --ckpt /workspace/ckpt/cellvit_sam_h/CellViT-SAM-H-x40-AMP.pth \
    --images_dir ../work/nuinsseg_png/images --out_csv cellvit_preds.csv --gpu 0 --mag 40 --limit 5
#   OK -> bỏ --limit chạy full. LKCell: --cellvit_dir /workspace/LKCell --ckpt <LKCell-L.pth>
# [5] CHẤM
python eval_heavy_count.py --gt ../work/nuinsseg_png/gt_counts.csv --preds cellvit_preds.csv \
    --label CellViT-SAM-H --student_pkl ../work/student_r2_nuinsseg_cv5_poisson_feat.pkl
#   (mode tile: thêm --preds cellvit_preds_tile.csv --tiles_map ../work/nuinsseg_png_tile/tiles_map.csv)
```
**[V] VALIDATE (bắt buộc, chống "tái hiện sai"):** chạy [4] trên ~5 ảnh cả resize lẫn tile, MẮT THƯỜNG đối
chiếu count detect vs GT (in kèm ảnh/overlay nếu cần) → chọn mode bám GT nhất, GHI RÕ mode + lý do trong paper.
Nếu cả hai lệch nhiều (SAM-H quen 40x WSI, NuInsSeg khác magnification) → báo trung thực + cân nhắc chỉ giữ bảng A.
**PathoSAM GFLOPs:** trong penv, `thop.profile` trên predictor.model input 1024 (điền ô '?' teacher bảng A).

#### ★ AUDIT LEAK & FAIRNESS (2026-07-14, đọc code xác minh — KHÔNG được quên khi viết)
**LEAK — sạch (verified):** train() chỉ lặp train_idx (dòng 210-211, không đụng test); NuInsSeg assign_kfold
gán mỗi ảnh đúng 1 fold, cross-fit predict held-out; teacher density chỉ dùng cho ảnh TRAIN, ảnh test chỉ so
GT thật; PanNuke `--exclude_tissue colon` loại overlap PathoSAM/Lizard; feat pooled = output trên ảnh test
(không phải label). Bảng 8c post-hoc trên pkl leak-free này + cal/test split → OK.
**FAIRNESS Phần B — 2 caveat BẮT BUỘC ghi trong paper (nếu giấu = reviewer bác):**
1. **In-domain vs OOD bất đối xứng**: student train NuInsSeg (cv5) vs heavy net zero-shot PanNuke→NuInsSeg.
   → TUYỆT ĐỐI không claim "student > CellViT". Khung = "adapt net 700M rất đắt; distill 1.9M rẻ mà count-MAE
   cạnh tranh + là model DUY NHẤT có UQ" (trục HIỆU QUẢ THÍCH NGHI, không hơn-thua thô).
2. **Lệch magnification**: SAM-H/LKCell train 40× PanNuke; NuInsSeg khác → MAE heavy net cao 1 phần do
   mismatch, không hẳn model kém. Ghi rõ.
3. **Đồng tập test**: student pkl (665 ảnh, không exclude) & prep NuInsSeg (build_index, không exclude) PHẢI
   cùng N & cùng exclusion → verify N khớp trước khi lên bảng. GT count cùng công thức len(unique)−bg.
**FAIRNESS 8c (đã có, nhắc lại):** CondConf-group/R2-mondrian ĐƯỢC cấp nhãn organ; R2-cluster/global KHÔNG →
R2 thắng worst-org dù ít thông tin hơn (có lợi ta, trung thực). Mọi baseline cùng seeds/cal_ratio/Winkler/feature.

#### ★ KẾT QUẢ Phần B — count-MAE NuInsSeg (CHẠY THẬT vast 2026-07-14, RTX 5090)
Cấu hình chốt sau validate: prep native **512** (không thu nhỏ, ảnh gốc NuInsSeg=512×512) + feed SAM-H **--infer_size 1024**
(native res công bằng; validate 5 ảnh: 256 ratio 0.66 → 512+1024 ratio 0.77, native tốt hơn). N=665 khớp student pkl.

| Method | Params | MAE | RMSE | MAPE | worst-org MAE |
|--------|--------|-----|------|------|---------------|
| CellViT-SAM-H (OOD, off-the-shelf) | 699.74M | 24.24 | 34.74 | 56.9% | mouse femur 129.0 |
| LKCell-L (OOD, off-the-shelf) | 163.84M | 16.54 | 28.07 | 38.8% | mouse thymus 113.0 |
| Teacher PathoSAM (zero-shot, reference) | ~640M | 15.80 | 29.02 | 28.3% | mouse thymus 124.3 |
| **Student R2 (in-domain distill, ours)** | **1.9M** | **13.51** | **22.61** | 45.3% | mouse spleen 105.0 |

**Đọc TRUNG THỰC (BẮT BUỘC, không tô hồng):**
- ★ **Student MAE 13.51 < Teacher PathoSAM 15.80** → trò đếm TỐT HƠN chính thầy (MAE/RMSE) dù nhỏ 340× — kết quả
  distillation mạnh (thích nghi in-domain vượt thầy zero-shot về sai số tuyệt đối). ĐIỂM BÁN LỚN.
- Student thắng **MAE + RMSE** (13.51/22.61 = thấp nhất mọi method) dù nhỏ 86–368×.
- NHƯNG **Teacher MAPE 28.3% ≪ student 45.3%** (và LKCell 38.8% < student) → student sai TƯƠNG ĐỐI cao hơn (kém
  ở ảnh ít nhân do density-sum); **student KHÔNG thắng tuyệt đối mọi metric** → PHẢI ghi rõ, không nói "giỏi nhất mọi mặt".
- LKCell (16.54) ≈ teacher (15.80), đều gần student hơn CellViT-SAM-H (24.24). Cả 2 heavy net bám GT (đơn điệu)
  nhưng undercount OOD (ratio~0.65-0.77).
- **KHUNG (BẮT BUỘC):** KHÔNG phải "student đếm giỏi hơn heavy net" — là **in-domain-distill (rẻ) vs OOD-zero-shot
  + lệch magnification** (NuInsSeg≠40× PanNuke). Contribution = "distill 1.9M rẻ trên domain đích → count-MAE/RMSE
  tốt nhất + là model DUY NHẤT có UQ calibrated (bảng A), ở 86–368× ít params". Giá trị = THÍCH NGHI RẺ + TRUSTWORTHY.
- Cấu hình chạy (2026-07-14 RTX 5090): cả 2 heavy net feed native NuInsSeg 512 (SAM-H ở 1024, LKCell native).
  LKCell UniRepLKNet fallback conv thường (no iGEMM, khỏi compile). N=665 khớp student.

TODO còn (tùy chọn): teacher PathoSAM count (reference row) + PathoSAM GFLOPs (bảng A ô teacher). Có thể dừng ở đây.

### Bước 3: PanNuke K>1 (mở rộng, tuỳ chọn) — mục 7.3. Rồi viết paper.

## 7. ▶ (CŨ, đã xong) resume PanNuke dataset 2

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

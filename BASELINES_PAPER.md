# Danh sách baseline & related work cho bảng kết quả paper

> Mục đích: tra nhanh khi lập bảng so sánh. Ưu tiên 2023→nay (kèm vài nền tảng <2023 bắt buộc).
> **Quy ước trạng thái:**
> - ✅ **đã chạy** — có số trong `PAPER_TABLES.md`, dùng được ngay
> - ➕ **nên chạy** — đối thủ đáng thêm, CHƯA có số (phải chạy mới được đưa vào bảng)
> - 📎 **chỉ cite** — đưa vào related work, không cần chạy
>
> ⚠️ **Không có số liệu nào trong file này được bịa.** Bài ➕ chỉ là ứng viên; phải
> chạy thực nghiệm rồi mới điền số vào bảng.
> ⚠️ Một số arXiv ID / venue / tác giả ghi theo trí nhớ — **kiểm tra lại trước khi nộp**.

---

## ⚠️ NỀN LÝ THUYẾT BẮT BUỘC TRÍCH (tìm khi rà novelty 06/2026)

Các bài này **đã chứng minh** phần lý thuyết mà đề tài từng định "đóng góp" → **phải trích, KHÔNG nhận vơ**.
Phần lý thuyết của bài viết theo kiểu **vận dụng + trích dẫn** (xem [Q1_PLAN.md](Q1_PLAN.md)).

| Bài | Năm | arXiv | Đã chứng minh cái gì (trùng với em) |
|---|---|---|---|
| **Halkiewicz — Rolling-Origin Conformal Prediction** | 2026 | 2605.08422 | window tối ưu `m*≍T^(2β/(2β+1))`, cận coverage qua **sup-distance trên score CDF (KS)**, **minimax**, rolling thắng full-history 86% — **trùng nhất với "Layer 3"** |
| **Han, Huang, Wang — Distribution-Free Predictive Inference under Unknown Temporal Drift** | 2024 | 2406.06516 | **adaptive window bằng bias–variance ước lượng** + sharp drift-adaptive coverage — trùng "adaptive window" |
| **Wasserstein-Regularized Conformal Prediction under General Distribution Shift** | 2025 (ICLR) | 2501.13430 | coverage gap ≤ **Wasserstein giữa score CDF**, tách **covariate (đo được) + concept shift** |
| **Barber, Candès, Ramdas, Tibshirani — Beyond Exchangeability (NexCP)** | 2023 | 2202.13415 | cận coverage-gap dạng TV cho nonexchangeable |
| **Adapting Prediction Sets to Distribution Shifts Without Labels (ECP/EACP)** | 2024 | 2406.01416 | adapt conformal **không nhãn** — đóng luôn hướng label-free |
| **Pseudo-Calibrated Conformal Prediction under Distribution Shift** | 2026 | 2602.14913 | lower-bound coverage theo Wasserstein shift đo được |

→ **Hệ quả:** không còn theorem mới khả thi. Đóng góp = **method + ứng dụng + eval**, lý thuyết chỉ trích dẫn.

---

## A0. Baseline HIỆN ĐẠI 2025–2026 (mới rà — cho Q1)

> Để qua reviewer Q1, nên có ≥2–3 cái **online-adaptive 2025–2026** trong bảng. **Không chạy hết** —
> chọn cái mạnh & sát nhất để chạy, còn lại cite trong related work. ⚠️ arXiv ID/venue verify lại.

| Bài | Năm | arXiv | Cốt lõi | Nên |
|---|---|---|---|---|
| **AdaptNC — Adaptive Nonconformity Scores under Shift** | 2026 | 2602.01629 | **đồng thời adapt tham số score + ngưỡng** (không chỉ scale threshold) — sát nhất với "adaptive" của em | ➕ **chạy** (đối thủ mạnh nhất) |
| **COP = Distribution-Informed Online CP** (Hu, Wu, Xia, Zou, ICLR 2026) | 2026 | 2512.07770 | **= baseline "COP" ĐÃ có trong Table 9a** (Winkler 113.13, em thắng 108.67). Dùng CDF-of-score, KHÔNG metric shift tường minh, KHÔNG count/đa lớp | ✅ đã so |
| **Online CP for Non-Exchangeable Panel Data** | 2026 | 2605.17705 | similarity-weights + adaptive miscoverage khi có feedback — giống cơ chế feedback của em | 📎 cite (hoặc chạy) |
| **CP Adaptive to Unknown Subpopulation Shifts** (ICLR 2026) | 2026 | (OpenReview 0aNfWttgHd) | shift theo subpopulation | 📎 cite |
| **Online CP with Retrospective Adjustment** (Jun & Ohn) | 2025 | 2511.04275 | hồi cứu, sửa dự đoán quá khứ — phân biệt forward-only của em | 📎 cite |
| **Multi-model Ensemble CP in Dynamic Environments** | 2024/25 | 2411.03678 | ensemble online | 📎 cite |
| **Non-exchangeable CP with Optimal Transport (unlabeled)** | 2025 | 2507.10425 | OT cho shift, dữ liệu chưa nhãn | 📎 cite |
| **ECI (Wu et al. 2025) · LQT (Areces et al. 2025)** | 2025 | — | biến thể ACI mới | 📎 cite |
| **CP under Lévy–Prokhorov Shifts** | 2025 | 2502.14105 | robust local+global | 📎 cite |

→ **Tối thiểu cho Q1:** chạy **AdaptNC + (rolling-origin Halkiewicz) + SAOCP**; cite phần còn lại.
→ Bài **2512.07770 (ICLR 2026)** em **đã tải về repo** — đọc kỹ để biết có trùng cơ chế "distribution-informed" không.

---

## A. Conformal online / dưới distribution shift  (hàng so chính cho bảng shift)

| # | Bài | Năm / Venue | arXiv | Vai trò | TT |
|---|---|---|---|---|---|
| A1 | Barber, Candès, Ramdas, Tibshirani — *Conformal Prediction Beyond Exchangeability* (**NexCP**) | 2023, Ann. Statist. | 2202.13415 | weighted-CP dưới shift; cũng là nền cận lý thuyết của em | ✅ |
| A2 | Bhatnagar et al. — *Improved Online CP via Strongly Adaptive Online Learning* (**SAOCP**) | 2023, ICML | 2302.07869 | online adaptive **có regret bound** — đối thủ lý thuyết mạnh nhất | ➕ |
| A3 | Gibbs & Candès — *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts* (**DtACI**) | 2024, JMLR | 2208.08401 | online chỉnh α thích nghi nhanh | ➕ |
| A4 | Angelopoulos, Candès, Tibshirani — *Conformal PID Control for Time Series* | 2023, NeurIPS | 2307.16895 | online recalibration kiểu điều khiển | ➕ |
| A5 | **COP** (online conformal, SOTA) | 2026, ICLR | (ref nội bộ) | SOTA online em đã so (Winkler 113.1) | ✅ |
| A6 | Jun & Ohn — *Online CP with Retrospective Adjustment* | 2025 | 2511.04275 | hồi quy 1 biến; **chỉ phân biệt**, không chạy | 📎 |
| A7 | *CP under Lévy–Prokhorov Distribution Shifts* | 2025 | 2502.14105 | robust CP với perturbation cục bộ+toàn cục | 📎 |
| A8 | *Online CP via Universal Portfolio Algorithms* | 2026 | 2602.03168 | online CP hướng portfolio | 📎 |

## B. Conformal đa lớp / output có cấu trúc  (so cho phần joint coverage)

| # | Bài | Năm | arXiv | Vai trò | TT |
|---|---|---|---|---|---|
| B1 | *Probabilistic Object Detection with Conformal Prediction* | 2026 | 2605.07549 | CP cho output cấu trúc + Bonferroni — đối chứng cho **max-stat joint** | ➕ |
| B2 | *Class-conditional CP for multiple inputs by p-value aggregation* (Fermanian et al.?) | 2025 | 2507.07150 | giữ coverage theo lớp — so với joint của em | ➕ |
| B3 | *Conformal Prediction for Hierarchical Data* | 2024 | 2411.13479 | coverage nhiều đầu ra phân cấp | 📎 |
| B4 | *Multi-Scale Conformal Prediction: coverage guarantees* | 2025 | 2502.05565 | khung coverage đa thang đo | 📎 |
| B5 | Class-stratified / Bonferroni (baseline nội bộ) | — | — | chia α/K, quá bảo thủ — động lực dùng joint | ✅ |

## C. Định lượng bất định cho ĐẾM  (so ở mức bài toán)

| # | Bài | Năm / Venue | arXiv | Vai trò | TT |
|---|---|---|---|---|---|
| C1 | Eaton-Rosen, Varsavsky, Ourselin, Cardoso — *As easy as 1,2…4? Uncertainty in counting tasks for medical imaging* | 2019, MICCAI | 1907.11555 | **BÀI GẦN NHẤT**: đếm tế bào + interval, nhưng learned-interval, không guarantee/đa lớp/shift | 📎 (bắt buộc) |
| C2 | **DeepDeconUQ** — conformalized quantile regression, cell-fraction prediction interval | 2024/25, PMC | — | conformal + cell + interval (RNA-seq deconvolution) | ➕ |

## D. Foundation model đếm / segment nhân  (hàng point-prediction MAE + lớp detector)

| # | Bài | Năm | arXiv | Vai trò | TT |
|---|---|---|---|---|---|
| D1 | **PathoSAM** — *Segment Anything for Histopathology* | 2025 | 2502.00408 | backbone #2 của em + SOTA segment nhân | ✅ |
| D2 | **SAM3 readiness** — Kong et al. — *Is SAM3 ready for pathology segmentation?* | 2025 | — | đánh giá SAM3 pathology — backbone #1 | ✅ |
| D3 | **CellViT++** (Hörst et al.?) | 2024/25 | — | SOTA segment+classify nhân, fine-tune PanNuke | ➕ |
| D4 | **Cellpose-SAM** (Stringer/Pachitariu group) | 2025, bioRxiv | — | tổng quát hoá segment cell/nuclei | ➕ |
| D5 | *Revisiting foundation models for cell instance segmentation* | 2026 | 2603.17845 | khảo sát foundation model đếm/segment | 📎 |

## Nền tảng <2023 — BẮT BUỘC cite dù ưu tiên 2023+

| # | Bài | Năm / Venue | arXiv | Vai trò |
|---|---|---|---|---|
| E1 | Gibbs & Candès — *Adaptive Conformal Inference under Distribution Shift* (**ACI**) | 2021, NeurIPS | 2106.00170 | gốc mọi online conformal — ✅ đã so |
| E2 | Tibshirani, Barber, Candès, Ramdas — *Conformal Prediction Under Covariate Shift* (**Weighted CP**) | 2019, NeurIPS | 1904.06019 | gốc shift-CP — ✅ đã so |
| E3 | Lei, G'Sell, Rinaldo, Tibshirani, Wasserman — *Distribution-Free Predictive Inference for Regression* | 2018, JASA | 1604.04173 | gốc split conformal |
| E4 | Vovk, Gammerman, Shafer — *Algorithmic Learning in a Random World* | 2005, sách | — | gốc lý thuyết conformal |

---

## Bảng baseline ĐỀ XUẤT cho paper (gọn, đủ qua phản biện)

Không nhồi hết — chọn phủ đủ 4 nhóm naive / static / weighted / online-adaptive:

| Method | Nhóm | TT |
|---|---|---|
| Naive PB (model tự ước, không conformal) | naive | ✅ |
| Marginal / Split conformal | static | ✅ |
| Weighted Conformal (Tibshirani 2019) | weighted/shift | ✅ |
| ACI (Gibbs & Candès 2021) | online (chỉnh α) | ✅ |
| NexCP (Barber 2023) | weighted online | ✅ |
| **SAOCP (Bhatnagar 2023)** | online + regret bound | ➕ **nên thêm** |
| COP (ICLR 2026) | SOTA online | ✅ |
| **Adaptive PB-JCI Online (ours)** | — | ✅ |

→ Thêm được **SAOCP** là nâng giá trị nhất (đối thủ có bảo đảm lý thuyết).
Các bài còn lại để **related work**, không cần thành hàng trong bảng.

## Việc cần làm trước khi điền số vào bảng

- [ ] Chạy **SAOCP** trên đúng setup PathoSAM→NuInsSeg (5 seed) → lấy Coverage/Width/Winkler
- [ ] (Tuỳ chọn) Chạy **DtACI** và **Conformal PID** nếu muốn phủ kín họ online-chỉnh-α
- [ ] **CellViT++ / Cellpose-SAM**: chỉ cần cho hàng **point-prediction MAE** (so backbone), không cần conformal
- [ ] DeepDeconUQ: cân nhắc — khác modality (RNA-seq), có thể chỉ cite thay vì chạy

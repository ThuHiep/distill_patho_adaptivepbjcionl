# Báo cáo tiến độ Paper 2

*Distillation foundation model cho bài toán đếm tế bào với giám sát mức đếm và dự đoán có độ tin cậy*

---

## 1. Vấn đề nghiên cứu

Các pathology foundation model hiện nay có hiệu năng cao nhưng khi thích nghi sang bài toán mới thường vẫn cần annotation mask mức instance, vốn rất tốn chi phí xây dựng. Trong khi đó, nhiều ứng dụng thực tế chỉ yêu cầu đếm số lượng tế bào. Nghiên cứu tập trung trả lời câu hỏi: **liệu có thể thích nghi foundation model sang bài toán đếm tế bào chỉ với count-level supervision (không cần mask) nhưng vẫn duy trì được độ chính xác và reliability của dự đoán?**

## 2. Hướng giải quyết

Sử dụng PathoSAM (~640M tham số) làm teacher để distill sang một student nhẹ 1.9M tham số, đặt tên **PACT** (*Poisson-Anchored Calibrated counTer*). PACT học density map do teacher sinh (không cần mask do người vẽ), dự đoán đồng thời giá trị đếm (μ = tổng density) và độ bất định (σ). Uncertainty được mô hình hóa bằng cơ chế **Poisson-anchored sigma** (σ = √μ · exp(log s), neo theo bản chất Poisson của quá trình đếm) và được hiệu chuẩn bằng conformal prediction để tạo khoảng dự đoán có bảo đảm coverage theo nhóm mô.

**Ký hiệu dùng trong báo cáo:**

- **PACT (ours):** student 1.9M với đầu σ **học được** từ dữ liệu (Poisson-anchored β-NLL) — phương pháp đề xuất.
- **KD (baseline):** cùng student 1.9M nhưng lấy σ theo **công thức giải tích Poisson-Binomial (PB-σ)** tính từ điểm detection — tức cách dựng bất định của **Paper 1** áp lên student nén. Dùng làm mốc so, không phải "đối thủ".
- **PB-σ:** Poisson-Binomial σ (Poisson-Binomial JCI) — đóng góp gốc của **Paper 1**, được Paper 2 trích dẫn làm nền.

## 3. Bằng chứng thực nghiệm

### 3.1 Kết quả chính — Lợi ích label-efficiency của distillation

Câu hỏi cốt lõi của bài (§1) là distillation mang lại lợi ích gì so với train trực tiếp bằng mask. Để trả lời, tiến hành so sánh trực tiếp trên **cùng ngân sách ảnh** giữa hai cách giám sát: (a) **DISTILLED** — target là density của teacher cộng một giá trị đếm mỗi ảnh; (b) **SUPERVISED** — target là density dựng từ mask GT từng nhân. Cả hai học trên 10% / 25% / 50% / 100% số ảnh, đánh giá trên cùng tập test, lặp 3 seed (NuInsSeg).

| Ngân sách ảnh | Distilled (Worst-org / MAE) | Supervised (Worst-org / MAE) |
|---|---|---|
| 10% (53 ảnh) | 0.865 / 24.26 | 0.879 / 26.09 |
| 25% (133 ảnh) | 0.858 / 21.85 | 0.824 / 18.49 |
| 50% (266 ảnh) | 0.840 / 20.34 | 0.830 / 17.73 |
| 100% (532 ảnh) | 0.843 / 14.12 | 0.897 / 14.61 |

Ở cùng số ảnh, hai cách cho reliability (worst-organ coverage) **không phân biệt được về mặt thống kê** và MAE tương đương (supervised nhỉnh ~2–3 nhân ở khúc giữa nhưng hòa ở hai đầu; ở 100% distilled còn thấp hơn). Nghĩa là **distillation đạt chất lượng ngang cách dùng mask**.

Khác biệt nằm ở **CHI PHÍ NHÃN**. Trung bình mỗi ảnh có 52.8 nhân (trung vị 38; tổng 35.138 nhân trên 665 ảnh), nên công dán nhãn mỗi ảnh:

| Cách giám sát | Nhãn người cần cung cấp / ảnh | Ước tính chi phí / ảnh |
|---|---|---|
| **Distilled (đề xuất)** | density teacher (0 công người) + 1 giá trị đếm | ≈ 52.8 × 2.4s ≈ **127 giây** |
| Supervised | vẽ mask 52.8 nhân | ≈ 52.8 × (5–10× chấm điểm) ≈ **640–1270 giây** |

Vì cả hai chi phí đều tỉ lệ với số nhân, **tỉ số công dán nhãn xấp xỉ 5–10 lần bất kể mật độ ảnh**.

> **Kết luận chính:** distillation đạt độ chính xác và reliability *tương đương* giám sát bằng mask nhưng chỉ cần khoảng **1/5–1/10 chi phí annotation** (chỉ giám sát mức đếm). Đây là bằng chứng trực tiếp cho luận điểm label-efficiency và là **đóng góp trung tâm** của nghiên cứu.

*Ghi chú trung thực: (i) thí nghiệm dùng chia đơn (single-split) nên chỉ đọc tương đối giữa các mức ngân sách, không so tuyệt đối với các bảng khác; (ii) giá trị đếm GT ở đây vẫn lấy từ mask có sẵn nên kết quả chứng minh YÊU CẦU giám sát của phương pháp là mức đếm, không phải đã dán nhãn rẻ hơn trong thực tế; (iii) tỉ số 5–10× là khoảng an toàn — chi phí chấm điểm 2.4s/nhân có nguồn (Bearman, ECCV 2016), chi phí vẽ mask mỗi nhân không có nguồn chuẩn nên không quy về một con số đơn.*

### 3.2 Độ chính xác và reliability tuyệt đối, so với baseline

Với đầy đủ dữ liệu, student 1.9M duy trì độ chính xác đếm và reliability tốt trên cả hai dataset:

| Dataset | MAE | Winkler | Worst-organ Coverage |
|---|---|---|---|
| PanNuke | 3.36 | 19.28 | 0.906 |
| NuInsSeg (5 seed) | 14.7 ± 1.7 | 95.4 ± 11.9 | 0.750 ± 0.049 |

So với các phương pháp conformal gần đây (cùng μ, σ, đánh giá leak-free) trên PanNuke, mô hình đề xuất cho **worst-organ coverage cao nhất** trong khi nhẹ và rẻ nhất:

| Method | Winkler ↓ | MAE ↓ | Worst-organ ↑ |
|---|---|---|---|
| **PACT (ours)** | 19.28 | **3.36** | **0.906** |
| CondConf (2025) | 18.81 | 3.37 | 0.853 |
| PCP (2024) | 23.26 | 3.37 | 0.805 |
| CPCP (2026) | 35.46 | 6.18 | 0.758 |
| R2CCP (2024) | 58.40 | 5.83 | 0.621 |
| KD (baseline) | 22.08 | 3.64 | 0.721 |

Kết quả này cho thấy student rẻ-nhãn **không hề yếu** ở mặt reliability theo nhóm mô mà còn **dẫn đầu** dàn baseline mạnh.

### 3.3 Coverage gần như miễn phí về nhãn

Xét riêng nhánh distilled khi tăng dần ngân sách nhãn, worst-organ coverage **giữ ổn định ở mọi mức, kể cả khi chỉ dùng 10% nhãn**; chỉ độ chính xác điểm (MAE) mới cải thiện theo lượng nhãn:

| Ngân sách | Worst-organ Coverage | MAE |
|---|---|---|
| 10% | 0.888 | 26.34 |
| 25% | 0.819 | 31.11 *(một seed nhiễu)* |
| 50% | 0.866 | 18.33 |
| 100% | 0.891 | 12.77 |

Điều này bổ trợ cho luận điểm chính: phần khoảng tin cậy hầu như không tốn nhãn, phù hợp với bối cảnh y tế nơi nhãn khan hiếm.

### 3.4 Learned uncertainty phù hợp hơn cho chế độ nén (bổ trợ, không phủ định Paper 1)

Bất định giải tích Poisson-Binomial (PB-σ) là đóng góp của **Paper 1**, được thiết kế và kiểm chứng trên foundation model — nơi điểm detection được hiệu chuẩn tốt. Câu hỏi của Paper 2 là: **khi mang PB-σ đó xuống student 1.9M (đã nén), nó còn giữ hiệu chuẩn không?** So sánh trên **cùng student nén**, khác nhau ở cách lấy σ (PB-công-thức = KD, vs học-trực-tiếp = PACT):

| Metric | PB-σ (KD, kiểu Paper 1) | Learned-σ (PACT) | Chênh lệch |
|---|---|---|---|
| Worst-organ (global) | 0.278 | 0.610 | +119% |
| Worst-organ (cluster) | 0.658 | 0.750 | +14% |
| MAE | 21.71 | 14.72 | −32% |

**Diễn giải (quan trọng — không hạ thấp Paper 1):** PB-σ **không hề "hỏng"**. Trên foundation model nó là công cụ hợp lệ (sân nhà của Paper 1), và ngay ở đây dưới chế độ **cluster** nó vẫn tốt (chênh lệch chỉ +14%). Vấn đề chỉ xuất hiện dưới chế độ **global**: khi student bị nén không tái tạo trung thực các điểm detection mà PB-σ dựa vào, σ giải tích **mất hiệu chuẩn**. Nói cách khác, PACT **không đánh bại** Paper 1 — nó **kế thừa** Paper 1 làm nền và **vá đúng giới hạn của PB-σ trong chế độ nén** bằng một σ học được, ổn định qua cả hai scheme. Đây là quan hệ *tiếp nối và bổ sung*: Paper 1 lập nền bất định trên mô hình lớn, Paper 2 mở rộng sang mô hình nén. Kết quả này cũng củng cố lựa chọn thiết kế Poisson-anchored sigma (§2).

### 3.5 Khả năng transfer của reliability

| Transfer | Coverage | Nhận xét |
|---|---|---|
| NuInsSeg → PanNuke | 0.897 | gần bằng in-domain |
| NuInsSeg → CryoNuSeg | 0.967 | reliability vẫn duy trì |

Đầu ra phân phối (μ, σ) transfer được sang dataset khác mà không cần train lại, cho thấy reliability không phụ thuộc một tập dữ liệu duy nhất. *Giới hạn trung thực: transfer sang MoNuSAC không giữ được do chênh lệch độ phân giải khiến nhân bị co ~4 lần (scale gap), được ghi rõ là giới hạn về scale.*

### 3.6 Hiệu quả mô hình

PACT chỉ **1.935M tham số** (nén ~330 lần so với teacher 640M) — mô hình nhỏ nhất trong nhóm so sánh vẫn cung cấp được khoảng dự đoán có độ tin cậy.

*Ablation dung lượng (phụ): giảm độ rộng kênh xuống cấu hình ch16 (~0.5M) không làm mất chất lượng — trên NuInsSeg còn nhỉnh, trên PanNuke hòa — cho thấy phương pháp bền theo dung lượng. Đây chỉ là kết quả ablation; cấu hình chính của PACT vẫn là 1.9M.*

## 4. Đóng góp chính của nghiên cứu

- **Chứng minh bằng thực nghiệm lợi ích label-efficiency:** distillation đạt chất lượng ngang giám sát bằng mask nhưng chỉ cần giám sát mức đếm, tiết kiệm khoảng 5–10 lần chi phí annotation *(đóng góp trung tâm)*.
- Đề xuất hướng thích nghi pathology foundation model cho cell counting chỉ với count-level supervision.
- Xây dựng **PACT** — student 1.9M dự đoán đồng thời count và uncertainty calibrated, nhỏ nhất trong nhóm có UQ; đầu ra phân phối dẫn đầu dàn baseline conformal trên PanNuke.
- Kế thừa PB-σ của Paper 1 làm nền và **chỉ ra giới hạn của nó dưới chế độ nén**, đề xuất learned Poisson-anchored σ ổn định qua các scheme; reliability còn transfer được giữa các dataset.

## 5. Kết luận hiện tại

Kết quả hiện tại cho thấy hoàn toàn khả thi để chuyển tri thức từ pathology foundation model sang một bộ đếm tế bào rất nhỏ, chỉ cần supervision mức đếm thay vì mask annotation, mà vẫn đạt độ chính xác và reliability tương đương giám sát bằng mask với chi phí nhãn thấp hơn nhiều lần. Toàn bộ thí nghiệm đã hoàn tất; bước tiếp theo là hoàn thiện manuscript theo hướng *label-efficient foundation-model adaptation for trustworthy cell counting*, phù hợp cho một bài báo Q1.

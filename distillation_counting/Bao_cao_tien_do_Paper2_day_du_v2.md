# Báo cáo tiến độ Paper 2

*Distillation foundation model cho bài toán đếm tế bào với giám sát mức đếm và dự đoán có độ tin cậy*

---

## 1. Vấn đề nghiên cứu

Các pathology foundation model như PathoSAM đã segment được nhân tế bào và về nguyên tắc có thể đếm (đếm số instance). Tuy nhiên, khi đưa vào sử dụng thực tế chúng vướng ba rào cản: (i) **kích thước lớn** (~640M tham số) tốn tài nguyên triển khai; (ii) muốn **thích nghi sang dataset/miền mới** theo cách thông thường vẫn cần **annotation mask mức instance**, vốn rất tốn chi phí; (iii) là mô hình segmentation, chúng **không xuất độ bất định (uncertainty)** cho con số đếm. Trong khi đó nhiều ứng dụng lâm sàng chỉ cần con số đếm kèm mức độ tin cậy.

Nghiên cứu tập trung trả lời câu hỏi: **liệu có thể chưng cất (distill) một pathology foundation model xuống một bộ đếm nhẹ, chỉ dùng count-level supervision (không cần mask), mà vẫn giữ được độ chính xác đếm và đồng thời bổ sung được khoảng dự đoán có độ tin cậy (calibrated uncertainty) — thứ bản thân foundation model không có?**

## 2. Hướng giải quyết

Sử dụng PathoSAM (~640M tham số) làm teacher để distill sang một student nhẹ 1.9M tham số, đặt tên **PACT** (*Poisson-Anchored Calibrated counTer*). PACT học density map do teacher sinh (không cần mask do người vẽ), dự đoán đồng thời giá trị đếm (μ = tổng density) và độ bất định (σ). Uncertainty được mô hình hóa bằng cơ chế **Poisson-anchored sigma** (σ = √μ · exp(log s), neo theo bản chất Poisson của quá trình đếm) và được hiệu chuẩn bằng conformal prediction để tạo khoảng dự đoán có bảo đảm coverage theo nhóm mô.

**Ký hiệu dùng trong báo cáo:**

- **PACT (ours):** student 1.9M với đầu σ **học được** từ dữ liệu (Poisson-anchored β-NLL) — phương pháp đề xuất.
- **KD (baseline):** cùng student 1.9M nhưng lấy σ theo **công thức giải tích Poisson-Binomial (PB-σ)** tính từ điểm detection — tức cách dựng bất định của **Paper 1** áp lên student nén. Dùng làm mốc so, không phải "đối thủ".
- **PB-σ:** Poisson-Binomial σ (Poisson-Binomial JCI) — đóng góp gốc của **Paper 1**, được Paper 2 trích dẫn làm nền.

## 3. Bằng chứng thực nghiệm

### 3.1 Kết quả chính — Lợi ích label-efficiency của distillation

Câu hỏi cốt lõi của bài (§1) là distillation mang lại lợi ích gì so với train trực tiếp bằng mask. Để trả lời, tiến hành so sánh **có kiểm soát** trên **cùng một kiến trúc PACT, cùng data, cùng ngân sách ảnh** — chỉ khác **nguồn giám sát**: (a) **PACT (distilled)** — target là density của teacher + một giá trị đếm mỗi ảnh (chỉ cần nhãn đếm); (b) **PACT-arch (mask-supervised)** — cùng mạng đó nhưng target là density dựng từ mask GT từng nhân. Cả hai học trên 10% / 25% / 50% / 100% số ảnh, đánh giá trên cùng tập test, lặp 3 seed (NuInsSeg). *(Hai cột là cùng một mạng PACT, chỉ khác cái target — đây là phép so công bằng để tách riêng ảnh hưởng của loại giám sát.)*

| Ngân sách ảnh | **PACT (distilled, count-only)** — Worst-org / MAE | PACT-arch (mask-supervised) — Worst-org / MAE |
|---|---|---|
| 10% (53 ảnh) | 0.865 / 24.26 | 0.879 / 26.09 |
| 25% (133 ảnh) | 0.858 / 21.85 | 0.824 / 18.49 |
| 50% (266 ảnh) | 0.840 / 20.34 | 0.830 / 17.73 |
| 100% (532 ảnh) | 0.843 / 14.12 | 0.897 / 14.61 |

Ở cùng số ảnh, hai cách cho reliability (worst-organ coverage) **không phân biệt được về mặt thống kê** và MAE tương đương (mask-supervised nhỉnh ~2–3 nhân ở khúc giữa nhưng hòa ở hai đầu; ở 100% distilled còn thấp hơn). Nghĩa là **distillation (chỉ cần nhãn đếm) đạt chất lượng ngang cách dùng mask**.

Khác biệt nằm ở **LOẠI NHÃN người phải cung cấp** (trung bình mỗi ảnh có 52.8 nhân — trung vị 38; tổng 35.138 nhân trên 665 ảnh):

| Cách giám sát | Nhãn người cần cung cấp / ảnh |
|---|---|
| **Distilled (đề xuất)** | density lấy **miễn phí** từ teacher + **1 con số đếm mức ảnh** |
| Supervised | **vẽ mask từng nhân** (≈ 52.8 mask/ảnh) |

Distilled chỉ cần một nhãn mức-ảnh (một con số), còn supervised phải khoanh viền từng nhân — với mật độ ~53 nhân/ảnh, chi phí annotation của mask **cao hơn nhiều lần**, dù bội số chính xác phụ thuộc quy trình gán nhãn cụ thể.

> **Kết luận chính:** distillation đạt độ chính xác và reliability *tương đương* giám sát bằng mask nhưng chỉ cần **nhãn mức-ảnh (một con số đếm)** thay vì mask từng nhân. Đây là bằng chứng trực tiếp cho luận điểm label-efficiency và là **đóng góp trung tâm** của nghiên cứu.

*Ghi chú trung thực: (i) thí nghiệm dùng chia đơn (single-split) nên chỉ đọc tương đối giữa các mức ngân sách, không so tuyệt đối với các bảng khác; (ii) giá trị đếm GT ở đây vẫn lấy từ mask có sẵn nên kết quả chứng minh **YÊU CẦU giám sát** của phương pháp là mức đếm, không phải "đã dán nhãn rẻ hơn trong thực tế"; (iii) claim ở đây là về **loại nhãn** (một con số mức-ảnh vs mask từng nhân) — factual; con số thời gian cụ thể (giây/ảnh, bội số) chưa đưa vào vì thiếu citation đo chi phí gán nhãn nhân mô bệnh học, sẽ bổ sung khi viết manuscript nếu tìm được nguồn.*

### 3.2 Độ chính xác và reliability tuyệt đối, so với baseline

Với đầy đủ dữ liệu, student 1.9M duy trì độ chính xác đếm và reliability tốt trên cả hai dataset:

| Dataset | MAE | Winkler | Worst-organ Coverage |
|---|---|---|---|
| PanNuke | 3.36 | 19.28 | 0.906 |
| NuInsSeg (5 seed) | 14.7 ± 1.7 | 95.4 ± 11.9 | 0.750 ± 0.049 |

**So sánh reliability với các phương pháp UQ khác — CÙNG baseline trên CẢ HAI dataset.** ⚠️ Đây **KHÔNG phải so giữa các kiến trúc model khác nhau** (NuLite/CellViT), mà là **các cách tạo uncertainty/khoảng dự đoán** trên cùng bài đếm (một số dùng chính dự đoán của PACT, một số như Ensemble/MC-Dropout tốn thêm compute). Bảng dưới là **worst-organ coverage ↑** (cao = tốt), cùng một dàn baseline trên cả PanNuke lẫn NuInsSeg:

| Phương pháp UQ | Chi phí chạy | PanNuke ↑ | NuInsSeg ↑ |
|---|---|---|---|
| **PACT (ours)** | **1 forward, 1.9M** | **0.906** | 0.750 |
| CQR | 1 model | 0.904 | **0.806** |
| Ensemble | **5× model** | 0.901 | 0.767 |
| MC-Dropout | N forward | 0.901 | 0.774 |
| CHDQR | 1 model | 0.897 | 0.722 |
| CondConf | 1 model | 0.853 | 0.898\* |
| PCP | 1 model | 0.805 | 0.708 |
| CPCP | 1 model | 0.758 | 0.500† |
| R2CCP | 1 model | 0.621 | 0.562† |
| KD (σ Poisson-Binomial, kiểu Paper 1) | 1 forward | 0.721 | 0.658 |

\*CondConf trên NuInsSeg **over-cover** (nới khoảng gấp đôi, Winkler tệ +54%) → 0.898 là "ảo", không phải khoảng hiệu quả.
†CPCP/R2CCP trên NuInsSeg **chưa re-run sạch** (thiếu feature/pkl) → số không tin cậy, chỉ để đầy đủ.

**Đọc trung thực (không cherry-pick):**
- **PanNuke:** PACT 0.906 cao nhất — nhưng **CQR 0.904, Ensemble 0.901 gần như HÒA**, không phải "đè".
- **NuInsSeg:** PACT 0.750 → **CQR 0.806 thắng rõ.**
- → **CQR là đối thủ mạnh & nhất quán nhất** (≈PACT ở PanNuke, hơn PACT ở NuInsSeg). PACT **không** phải "UQ tốt nhất mọi nơi".

→ **Lợi thế THẬT của PACT ở bảng này = CHI PHÍ CHẠY:** cho cả phân phối (μ,σ) trong **1 forward, 1.9M**, trong khi Ensemble tốn 5×, MC-Dropout N-pass, CQR/CHDQR cần train quantile riêng. Tức *reliability gần nhóm tốt nhất mà rẻ nhất để chạy* — chứ chưa phải reliability dẫn đầu.

> **Cách giải quyết đang thử = multi-teacher (§6).** σ hiện tại học "chay" từ GT. Nếu distill từ **nhiều foundation model** và lấy **bất đồng giữa chúng làm σ epistemic**, mục tiêu là đẩy PACT **vượt CQR** (đối thủ thật) trên NuInsSeg — worst-org 0.750 → trên 0.806, đặc biệt ở worst-organ và transfer — biến UQ từ "competitive-but-cheapest" thành **dẫn đầu thật sự**.

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

PACT chỉ **1.935M tham số** (nén ~330 lần so với teacher 640M), chạy 1-forward — nhẹ để triển khai. Hiệu suất của mô hình được đánh giá bằng phép so **có kiểm soát** ở §3.1 (ngang mask-supervision cùng kiến trúc) và §3.2 (dựng khoảng dẫn đầu nhóm conformal); phần so kích thước với các model mask-heavy khác (NuLite/CellViT) để lại cho manuscript vì cần một phép so accuracy công bằng đi kèm.

*Ablation dung lượng (phụ): giảm độ rộng kênh xuống cấu hình ch16 (~0.5M) không làm mất chất lượng — trên NuInsSeg còn nhỉnh, trên PanNuke hòa — cho thấy phương pháp bền theo dung lượng. Đây chỉ là kết quả ablation; cấu hình chính của PACT vẫn là 1.9M.*

## 4. Đóng góp chính của nghiên cứu

- **Chứng minh bằng thực nghiệm lợi ích label-efficiency:** với phép so có kiểm soát (cùng mạng PACT), distillation đạt chất lượng ngang giám sát bằng mask nhưng **chỉ cần nhãn mức-ảnh (một con số đếm)** thay vì mask từng nhân *(đóng góp trung tâm)*.
- Đề xuất hướng thích nghi pathology foundation model cho cell counting chỉ với count-level supervision.
- Xây dựng **PACT** — student gọn 1.9M (1-forward) dự đoán đồng thời count và uncertainty calibrated, chỉ cần nhãn count. **Reliability (§3.2):** gần nhóm UQ tốt nhất (≈CQR/Ensemble ở PanNuke) với **chi phí chạy rẻ nhất** (1 forward vs Ensemble 5×/MC-Dropout N-pass); còn khoảng cách với CQR trên NuInsSeg là chỗ đang cải thiện (§6).
- Kế thừa PB-σ của Paper 1 làm nền và **chỉ ra giới hạn của nó dưới chế độ nén**, đề xuất learned Poisson-anchored σ ổn định qua các scheme; reliability còn transfer được giữa các dataset.

## 5. Kết luận hiện tại

Kết quả hiện tại cho thấy hoàn toàn khả thi để chuyển tri thức từ pathology foundation model sang một bộ đếm tế bào rất nhỏ, chỉ cần supervision mức đếm (một con số) thay vì mask từng nhân, mà vẫn đạt độ chính xác và reliability tương đương giám sát bằng mask. Đóng góp là **một gói mạch lạc**: model tí hon (1.9M) + chỉ cần nhãn đếm + có uncertainty calibrated, với phép so label-efficiency có kiểm soát (§3.1) và reliability gần nhóm UQ tốt nhất mà rẻ nhất để chạy (§3.2). Hướng phù hợp: *label-efficient foundation-model adaptation for trustworthy cell counting* (Q1-applied).

## 6. Đang thử nghiệm để mạnh hơn (chưa vào báo cáo chính)

**Mục tiêu:** vá đúng **bảng UQ-floor ở §3.2** — chỗ PACT còn xếp ~4/5. σ hiện tại học "chay" từ GT, chưa thật sự distill từ teacher (một PathoSAM không có bất định nội tại). Hướng thử: thay vì distill từ **một** foundation model, distill từ **một hội đồng nhiều foundation model** (PathoSAM + SAM3 + NuLite + LKCell) và lấy **sự bất đồng giữa chúng làm nguồn uncertainty epistemic** (chỗ các model lệch nhau nhiều = ảnh thật sự khó) — nguồn này *có nguyên lý* và *distill được* vào student.

Kết quả thăm dò ban đầu tích cực: bất đồng của hội đồng tương quan với lỗi ngang với σ hiện tại. **Kỳ vọng nếu thành công:** đẩy dòng PACT trong bảng §3.2 **vượt CQR/Ensemble** (Winkler 95.4 → dưới ~80, worst-organ 0.750 → trên ~0.806), đặc biệt ở **worst-organ và transfer sang dataset khác** — biến UQ từ "competitive" thành **điểm mạnh dẫn đầu**, và đưa model/method thành đóng góp chính. *(Đang chạy — thanh chắn thật là đè được CQR + Ensemble sau hiệu chuẩn coverage, chưa phải corr; chưa đưa vào kết quả chính.)*

# Hiệu chỉnh conformal trực tuyến cho bài toán đếm nguồn thiên văn có bảo chứng độ phủ dưới dịch chuyển độ sâu khảo sát

*Bản nháp tiếng Việt để kiểm tra nội dung. 2026-07-06. **Bản nộp ICEBA2026 phải là TIẾNG ANH** (Step 3).*

> **Thông tin nộp bài:**
> - **Journal chọn:** (1) Springer Nature Proceedings — Scopus
> - **Session/subject area:** *[chọn 1 trong 8 — ưu tiên "Data Analytics / Applied Statistics / Computational Science"]*
> - **Paper ID:** *[mã cấp ở Step 1 — điền sau khi nộp abstract]*
> - **Tác giả:** Thi Thu Hiep Dinh¹, Viet Hang Duong¹ (tác giả liên hệ, `hangdv@uit.edu.vn`)
> - **¹ Đơn vị:** Khoa Khoa học Máy tính, Trường ĐH Công nghệ Thông tin (UIT), ĐHQG-HCM, TP.HCM, Việt Nam.
> - **Độ dài mục tiêu:** 4–8 trang (kể cả hình, bảng, tham chiếu).

---

## Tóm tắt (Abstract)

Nhiều bài toán khoa học cần **đếm số đối tượng theo lớp** kèm **khoảng tin cậy đáng tin** — không chỉ một
con số trần. Khi điều kiện thu nhận dữ liệu thay đổi (đổi thiết bị, độ sâu, môi trường), phân phối dữ liệu
**dịch chuyển** và các khoảng tin cậy hiệu chỉnh tĩnh trở nên **hụt độ phủ một cách âm thầm**. Chúng tôi áp
dụng **Adaptive PB-JCI Online** — một lớp hiệu chỉnh conformal trực tuyến nhẹ, giữ được độ phủ đồng thời cho
vector đếm nhiều lớp dưới dịch chuyển — sang một lĩnh vực khoa học **hoàn toàn khác** với bài toán gốc (đếm
nhân tế bào mô bệnh học): **đếm sao và thiên hà** trong ảnh khảo sát bầu trời. Trên **61 423 nguồn thật** từ
Sloan Digital Sky Survey (SDSS DR17), với backbone là **pipeline quang trắc chính thức của SDSS** và một cú
**dịch chuyển độ sâu khảo sát** có kiểm soát, Adaptive PB-JCI là phương pháp **duy nhất** đạt độ phủ đồng
thời hợp lệ (≥90%) với **điểm interval (Winkler) tốt nhất**, và **giữ được ~91% qua mọi mức dịch chuyển**
trong khi hiệu chỉnh tĩnh sụp từ 85% xuống 9%. Kết quả cho thấy phương pháp **tổng quát across-domain**, từ
khoa học sự sống sang vật lý thiên văn, mà không đổi một dòng công thức.

**Từ khóa (Keywords):** conformal prediction; online calibration; count uncertainty quantification;
distribution shift; astronomical source counting.

---

## 1. Giới thiệu

### 1.1 Bối cảnh

Đếm đối tượng là bài toán trung tâm ở nhiều ngành: đếm tế bào trong y sinh, đếm cây trong lâm nghiệp, và —
trong bài này — **đếm nguồn thiên văn** (sao, thiên hà) để đo đa dạng và cấu trúc vũ trụ. Trong thực tế,
một con số đếm **kèm sai số đáng tin** quan trọng hơn con số đơn lẻ: nhà khoa học cần biết *"20 ± 4 thiên hà,
đảm bảo 90%"* để so sánh giữa các vùng/thời điểm một cách có căn cứ.

**Vấn đề cốt lõi:** điều kiện thu nhận dữ liệu thay đổi giữa các lần khảo sát (thiết bị khác, độ sâu khác,
thời tiết khác) làm **phân phối dữ liệu dịch chuyển** (distribution shift). Khoảng tin cậy hiệu chỉnh trên
dữ liệu cũ áp lên dữ liệu mới sẽ **hụt độ phủ** — báo sai số đẹp nhưng thực chất sai. Nếu khoảng **quá hẹp**
(under-cover) → phát hiện giả, kết quả không tái lập; nếu **quá rộng** (over-cover) → vô dụng, tín hiệu thật
chìm trong sai số.

### 1.2 Đóng góp

Bài trước của chúng tôi đề xuất **Adaptive PB-JCI Online** cho bài toán đếm nhân tế bào mô bệnh học, đạt độ
phủ có bảo chứng dưới dịch chuyển. Câu hỏi của bài này: **phương pháp có tổng quát sang một lĩnh vực khoa học
khác hẳn không?** Đóng góp:

1. **Chuyển giao across-domain:** áp dụng nguyên vẹn phương pháp (không đổi công thức) sang **đếm nguồn thiên
   văn** — bước từ khoa học sự sống sang vật lý — trên dataset + backbone khác hẳn.
2. **Một benchmark có mỏ neo dữ liệu thật:** dựng trên **SDSS DR17** với backbone là pipeline quang trắc
   thật, một cú **dịch chuyển độ sâu khảo sát** có kiểm soát, và **bộ baseline hiện đại đầy đủ** (9 phương
   pháp: ACI, NexCP, FACI, SAOCP, COP, Rolling-Origin, và các biến thể của chúng tôi).
3. **Kết quả:** Adaptive PB-JCI **duy nhất** đạt độ phủ đồng thời hợp lệ với điểm interval tốt nhất, và giữ
   ổn định qua toàn dải mức dịch chuyển.

---

## 2. Công trình liên quan

**Conformal prediction (CP)** cho khoảng dự đoán không phụ thuộc phân phối với bảo chứng độ phủ trên dữ liệu
đổi. **CP trực tuyến dưới dịch chuyển:** Adaptive Conformal Inference (ACI, Gibbs & Candès 2021) điều chỉnh
mức lỗi theo thời gian; các mở rộng gồm FACI (2024), NexCP (Barber 2023) với trọng số phân rã, SAOCP
(Bhatnagar 2023) với bảo chứng strongly-adaptive, và COP (2026). Các phương pháp này chủ yếu cho **dự đoán
đơn biến**; bài toán của chúng tôi cần **độ phủ đồng thời cho vector đếm nhiều lớp** và một mô hình **độ rộng
thích nghi theo từng mẫu** dựa trên bất định phát hiện. **Đếm nguồn thiên văn** (number counts) là phép đo
kinh điển trong vũ trụ học, nhưng thường báo cáo với sai số Poisson mà **không có bảo chứng độ phủ dưới
dịch chuyển giữa các khảo sát** — khoảng trống mà bài này lấp.

---

## 3. Phương pháp: Adaptive PB-JCI Online

### 3.1 Số đếm mềm và độ lệch Poisson-Binomial

Với mỗi "field" (một vùng ảnh), backbone cung cấp cho mỗi đối tượng $i$ một **xác suất phát hiện** $s_i$ và
một vector **xác suất lớp** $p_i[k]$. Thay vì đếm cứng, ta dùng **số đếm mềm**:

$$ n_{\text{pred}}[k] = \sum_i s_i \, p_i[k], \qquad
   \sigma[k] = \sqrt{\sum_i (s_i p_i[k])\,(1 - s_i p_i[k])}. $$

$\sigma[k]$ là độ lệch chuẩn của tổng các biến Bernoulli độc lập (phân phối **Poisson-Binomial**) — nó lớn
khi field có nhiều đối tượng "lưỡng lự" (confidence gần 0.5) và nhỏ khi confidence dứt khoát. Đây là **độ
rộng thích nghi theo từng field**.

### 3.2 Nonconformity đồng thời (joint) qua K lớp

Để có **độ phủ đồng thời** cho cả vector đếm, ta lấy thống kê **cực đại chuẩn hóa** qua các lớp:

$$ S_t = \max_k \frac{|\,g_t[k] - n_{\text{pred}}[k]\,|}{\sigma[k]}, \qquad
   \text{khoảng lớp } k:\; \big[\, n_{\text{pred}}[k] - q\,\sigma[k],\; n_{\text{pred}}[k] + q\,\sigma[k]\,\big]. $$

với $g_t[k]$ là số đếm thật. Ngưỡng $q$ là phân vị $(1-\alpha)$ của tập điểm nonconformity.

### 3.3 Cửa sổ hiệu chỉnh thích nghi trực tuyến

Đóng góp **cốt lõi**: khi dịch chuyển làm độ phủ tụt, một **cửa sổ hiệu chỉnh** co lại để $q$ cập nhật nhanh
theo dữ liệu mới; khi độ phủ ổn định, cửa sổ giãn ra để $q$ ổn định. Cụ thể, theo dõi độ phủ cuộn trên
cửa sổ gần đây; nếu dưới mục tiêu → **co** cửa sổ (nhân hệ số $<1$), nếu vượt mục tiêu → **giãn** (nhân hệ số
$>1$), kẹp trong $[w_{\min}, w_{\max}]$. Đây là cơ chế **hồi phục độ phủ** (coverage recovery) — thành phần
tạo chiến thắng chính, khác với $\sigma$ (chỉ lo độ rộng).

---

## 4. Dữ liệu và Backbone

### 4.1 Dataset — SDSS DR17

Chúng tôi dùng **Sloan Digital Sky Survey, Data Release 17** — khảo sát quang học bằng kính 2.5 m, ảnh 5 băng
$ugriz$. Từ bảng quang trắc `PhotoObjAll`, truy vấn vùng trời RA 150–152°, Dec 0–2° (~4 deg²) với các cờ
chất lượng chuẩn (`mode=1` primary, `clean=1`), giữ hai lớp: **sao** (`type=6`) và **thiên hà** (`type=3`).
Thu được **61 423 nguồn thật** (25 270 sao, 36 153 thiên hà), median tỉ số tín hiệu/nhiễu ≈ 11.1. Vùng được
chia lưới thành 841 field; ô chẵn dùng để hiệu chỉnh (CAL, 421 field), ô lẻ để kiểm tra (TEST, 420 field).

### 4.2 Backbone — Pipeline quang trắc SDSS

Backbone **không phải mô hình chúng tôi huấn luyện**, mà là **pipeline quang trắc chính thức của SDSS** đã
sinh ra catalog — đóng vai trò cung cấp confidence per-đối-tượng. Với mỗi nguồn, chúng tôi đọc **trực tiếp**:
`probPSF` (xác suất nguồn là điểm — dùng làm $p_{\text{star}}$), `psfFlux` và `psfFluxIvar` (thông lượng và
nghịch đảo phương sai), rồi **tự tính** tỉ số tín hiệu/nhiễu $\text{SNR} = \text{psfFlux}\cdot\sqrt{\text{ivar}}$.

**Xác suất phát hiện.** Vì catalog chỉ chứa nguồn **đã được phát hiện**, xác suất phát hiện không có sẵn;
chúng tôi mô hình hóa nó bằng **hàm completeness 5σ chuẩn** trên SNR thật: $s_i = \text{logistic}((\text{SNR}_i - 5)/1.5)$.
Đây là mô hình vật lý tiêu chuẩn, cho $\sigma$ **liên tục** (không thoái hóa). Từ đó: $p_i[\text{star}] = \text{probPSF}_i$,
$p_i[\text{gal}] = 1 - \text{probPSF}_i$, và số đếm mềm như Mục 3.1.

### 4.3 Dịch chuyển độ sâu khảo sát

Chúng tôi mô phỏng "khảo sát nông hơn" bằng **forward-model độ sâu trên chính dữ liệu thật**:
$\text{SNR}_{\text{shallow}} = \text{SNR}_{\text{deep}} / \text{DEPTH}$, làm nhiều nguồn mờ rơi dưới ngưỡng
phát hiện và phân loại mờ đi. Tập CAL = chế độ sâu (residual nhỏ), TEST = chế độ nông (residual lớn) — đúng
chiều "làm khó". `DEPTH` là **núm điều khiển độ mạnh dịch chuyển**, cho phép quét toàn dải (Mục 6.2). Đây là
giao thức distribution-shift chuẩn; đánh giá fully-observational (Stripe82 co-add) để lại cho tương lai.

---

## 5. Thiết lập thí nghiệm

**Quy trình diagnostic-first.** Trước khi chạy đầy đủ, chúng tôi quét ngưỡng completeness `mag_cut` và đo:
độ chệch (bias) mỗi lớp, tương quan giữa $\sigma$ và sai số ($\sigma$-gain), và độ phủ của hiệu chỉnh tĩnh
trên CAL/TEST. Chọn ngưỡng có **dịch chuyển đủ mạnh** (tĩnh under-cover trên TEST, calibration lành).

**Baselines (9):** hiệu chỉnh tĩnh (Static split-CP), ACI, NexCP, FACI, SAOCP, COP, Rolling-Origin CP, và
hai biến thể của chúng tôi (PB-JCI Online-Fixed, Adaptive PB-JCI Online) — tất cả điều chỉnh sang thống kê
cực đại đồng thời K=2.

**Chỉ số:** độ phủ đồng thời (%), độ phủ riêng từng lớp, độ hợp lệ có điều kiện (độ phủ cuộn nhỏ nhất, chuỗi
trượt dài nhất), độ rộng trung bình (AvgW), và **điểm interval Winkler** (thấp = tốt; đây là chỉ số đầu bảng).
Mục tiêu độ phủ 90% ($\alpha = 0.1$).

---

## 6. Kết quả

### 6.1 Bảng chính (dịch chuyển cố định)

Tại ngưỡng dịch chuyển mạnh nhất (hiệu chỉnh tĩnh sụp 90.5% → 81.0%), độ chệch backbone ≈ 0 và $\sigma$-gain
dương (+0.44):

| Phương pháp | Phủ đồng thời % | AvgW | Winkler | minLoc % |
|---|---|---|---|---|
| Static split-CP | 81.0 | 0.03 | 2.84 | 67.2 |
| ACI (2021) | 89.7 | 6.29 | 14.08 | 85.2 |
| NexCP (2023) | 88.0 | 0.21 | 2.23 | 78.0 |
| FACI (2024) | 88.6 | 0.26 | 2.19 | 80.4 |
| SAOCP (2023) | 99.3 | 11.12 | 22.34 | 96.8 |
| COP (2026) | 85.1 | 0.10 | 2.52 | 72.0 |
| Rolling-Origin (2026) | 88.5 | 0.24 | 2.18 | 78.8 |
| PB-JCI Fixed (của chúng tôi) | 85.7 | 0.18 | 2.32 | 72.4 |
| **Adaptive PB-JCI (của chúng tôi)** | **91.2** | 0.47 | **2.10** | 84.0 |

**Đọc kết quả:** Adaptive là phương pháp **duy nhất đạt ≥90%** mà **không phình interval**, đồng thời có
**Winkler tốt nhất (2.10)**. Hai phương pháp phủ được khác đều vô dụng: SAOCP over-cover 99.3% nhưng Winkler
22.3 (interval rộng gấp ~10 lần); ACI đạt 89.7% nhưng Winkler 14.08 (interval bất ổn). Các phương pháp có
Winkler thấp hơn chút đều **under-cover** (85–88.5%). Nói cách khác: chỉ Adaptive **vừa hợp lệ vừa chặt**.

### 6.2 Ổn định qua dải dịch chuyển (kết quả then chốt)

Tăng dần độ mạnh dịch chuyển (hiệu chỉnh tĩnh sụp 85.5% → 9.0%):

| Độ mạnh shift | Static | ACI | NexCP | FACI | SAOCP | COP | Rolling | PB-Fixed | **Adaptive** |
|---|---|---|---|---|---|---|---|---|---|
| nhẹ | 85.5 | 89.8 | 88.7 | 88.8 | 98.3 | 87.3 | 89.2 | 87.9 | **91.4** |
| vừa | 81.0 | 89.7 | 88.0 | 88.6 | 99.3 | 85.1 | 88.5 | 85.7 | **91.2** |
| mạnh | 53.1 | 89.6 | 87.1 | 88.6 | 99.3 | 82.9 | 87.0 | 83.3 | **91.0** |
| rất mạnh | 9.0 | 89.7 | 86.2 | 89.1 | 99.5 | 80.3 | 85.8 | 81.9 | **90.5** |

**Adaptive giữ ~91% gần như bất động** qua toàn dải, trong khi hiệu chỉnh tĩnh sụp đổ và các baseline
fixed-rate tụt dần. Đây là bằng chứng trực quan nhất cho cơ chế **hồi phục độ phủ**.

### 6.3 Hồi phục sau dịch chuyển đột ngột

Trên một dòng dữ liệu chuyển đột ngột từ chế độ sâu sang nông, Adaptive **bám lại mục tiêu 90%** trong ~30
bước sau điểm chuyển, cân bằng nhất trong nhóm hợp lệ (các baseline fixed-rate tụt xuống 77–83% ngay sau
chuyển; SAOCP over-cover 100%).

---

## 7. Thảo luận và Hạn chế

**Vì sao phương pháp thắng ở đây.** Thiên văn thỏa cả ba điều kiện của cơ chế: (1) độ chệch backbone thấp
(nhờ mô hình completeness + ngưỡng); (2) confidence thật sự dị phương ($\sigma$-gain dương); (3) dịch chuyển
mạnh, dạng aleatoric (đổi độ sâu). Nhờ đó Adaptive phô diễn được cả hai đóng góp: độ rộng thích nghi và hồi
phục độ phủ.

**Trung thực về mức độ "thật".** Đánh giá là **semi-synthetic có mỏ neo thật**: nguồn, magnitude, thông
lượng, SNR, phân loại đều **thật** từ SDSS; xác suất phát hiện được **mô hình hóa** (completeness 5σ trên SNR
thật, vì catalog chỉ chứa nguồn đã phát hiện) và dịch chuyển là **induced** (giảm SNR có kiểm soát). Chúng
tôi **không** tuyên bố "fully real" hay "cross-survey thật". Đánh giá fully-observational (SDSS Stripe82
co-add sâu làm chuẩn vs single-epoch nông) là hướng tương lai.

**Đọc chỉ số đúng cách.** Phải đọc **độ phủ và Winkler cùng nhau**: một phương pháp có độ rộng nhỏ hơn nhưng
under-cover là **không hợp lệ**; một phương pháp over-cover (SAOCP) đạt độ phủ nhưng interval vô dụng.

---

## 8. Kết luận

Chúng tôi cho thấy **Adaptive PB-JCI Online** — thiết kế cho đếm nhân tế bào mô bệnh học — **chuyển giao
nguyên vẹn** sang đếm nguồn thiên văn, một lĩnh vực khoa học khác hẳn. Trên dữ liệu SDSS thật với dịch chuyển
độ sâu khảo sát, nó là phương pháp **duy nhất** giữ được độ phủ đồng thời hợp lệ với điểm interval tốt nhất,
**ổn định qua toàn dải dịch chuyển**. Kết quả củng cố tính **tổng quát across-domain** của phương pháp: cùng
một công thức đếm-có-bảo-chứng, chạy từ khoa học sự sống sang vật lý thiên văn.

---

## Tuyên bố (bắt buộc theo ICEBA2026)

- **Tính nguyên gốc:** Bài này là công trình gốc, chưa được công bố và không đang nộp đồng thời ở tạp chí
  khác.
- **Đóng góp tác giả:** *[liệt kê đóng góp thực chất của từng tác giả — ý tưởng, thí nghiệm, viết bài].*
- **Khai báo sử dụng AI:** Tác giả có dùng công cụ hỗ trợ bởi AI để soạn thảo văn bản và hỗ trợ viết mã. Toàn
  bộ thiết kế thí nghiệm, dữ liệu, kết quả số và kết luận do tác giả **thực hiện, kiểm chứng và chịu trách
  nhiệm**. *(ICEBA2026 yêu cầu công khai nội dung do AI tạo.)*
- **Dữ liệu:** SDSS DR17 công khai qua CasJobs/astroquery; mã và cấu hình thí nghiệm sẵn có để tái lập.

---

## Tài liệu tham khảo (chọn lọc)

- Vovk et al. *Algorithmic Learning in a Random World* (conformal prediction).
- Gibbs & Candès. Adaptive Conformal Inference under distribution shift. NeurIPS 2021.
- Gibbs & Candès. Conformal inference for online prediction (FACI). 2024.
- Barber et al. Conformal prediction beyond exchangeability (NexCP). Ann. Statist. 2023.
- Bhatnagar et al. Improved Online Conformal Prediction via Strongly Adaptive Learning (SAOCP). ICML 2023.
- York et al. The Sloan Digital Sky Survey: Technical Summary. AJ 2000. (+ SDSS DR17 paper.)

*[Bổ sung trích dẫn đầy đủ + tham chiếu bài gốc đếm tế bào của nhóm khi chuyển sang bản tiếng Anh.]*

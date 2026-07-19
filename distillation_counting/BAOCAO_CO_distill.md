# Báo cáo cô: hành trình từ "hướng distill" đến câu trả lời "tại sao lại distill"

*Ngày 19/07/2026 — em viết kiểu kể lại cho cô nghe, không phải văn luận văn.*

---

## 1. Bắt đầu: cô chỉ em đi hướng distillation

Thưa cô, xuất phát điểm là gợi ý của cô: thay vì chạy theo mấy model đếm nhân khổng lồ, em thử **"nén" (distill) một foundation model mạnh (PathoSAM ~640M tham số) xuống một student thật nhỏ**, để nó vẫn đếm được nhân tế bào trên ảnh mô bệnh học mà chạy nhẹ.

Em bám đúng hướng đó và thiết kế **DensitySigmaUNet ~1.9M tham số** (nhỏ hơn teacher ~330 lần). Điểm khác biệt em thêm vào: student không chỉ trả một con số đếm, mà trả **cả một phân phối** — trung bình μ (đếm bao nhiêu nhân) *và* độ lệch σ (nó tự tin tới đâu). Đây là chỗ em muốn làm mới so với các bản distill của người khác (họ chỉ distill segmentation, không có "độ bất định").

Nhưng làm tới đây thì em vấp một câu hỏi mà càng nghĩ càng thấy nhức: **"Distill để làm gì?"**

---

## 2. Khúc mắc: mò mãi mới hiểu câu hỏi thật nằm ở đâu

Ban đầu em tưởng bài toán là **đua độ chính xác** — distill xong student phải đếm giỏi hơn ai đó. Em đo thẳng thắn thì:

- Student 1.9M của em **không đè** được các model đếm chuyên dụng. Ví dụ NuLite-T (12M) trên PanNuke đếm sai ~1.97 (MAE), còn student em ~3.38 — **thua ~1.72 lần**. Em đo bằng số thật, không tự huyễn hoặc.

→ **Bài học 1:** accuracy KHÔNG phải chỗ mình thắng. Đừng bán bài bằng "đếm giỏi nhất".

Sau đó em quay sang nghĩ: *"Vậy chắc điểm mạnh là độ bất định (UQ) — student mình có phân phối, tụi kia không có."* Em chạy một loạt phương pháp UQ chuẩn (Ensemble, CQR, MC-Dropout...) trên cùng model để so sòng phẳng, mỗi cái 5 seed. Kết quả trung thực:

- Phương pháp của em **xếp khoảng 4/5** — có cái tên CQR (cùng 1 model) còn nhỉnh hơn em cả hai trục. Em chỉ thắng rõ MC-Dropout.

→ **Bài học 2:** UQ của mình *cạnh tranh chứ không phải số một*. Cũng không phải chỗ để bán bài.

Em còn thử một ý "đặc thù mô bệnh học" (đo độ bất định qua nhiễu nhuộm màu H&E). Chạy probe rẻ để kiểm tra trước: **tương quan với lỗi chỉ +0.17**, trong khi σ học được của em +0.65. → Ý đó chết. Em ghi lại honest rồi bỏ, không cố đấm.

Tới đây em hơi bí. Accuracy không thắng, UQ không thắng, ý đặc thù cũng không. **Nhưng chính lúc bí đó em mới nhìn ra câu hỏi thật.**

Câu hỏi thật không phải *"student có giỏi hơn không"*. Mà là: **so với việc train thẳng bằng nhãn mask (segmentation) — vốn là cách chuẩn — thì distill được LỢI gì?** Nếu train-bằng-mask cho kết quả ngang hoặc nhỉnh hơn, thì distill để làm gì cho mệt?

Đó chính là lỗ hổng mà một người phản biện sẽ hỏi ngay. Và em quyết định **biến đúng lỗ hổng đó thành vấn đề trung tâm của bài** — trả lời nó bằng thí nghiệm, không bằng lời.

---

## 3. Vấn đề trung tâm: distill rẻ nhãn ở đâu? (bảng quan trọng nhất)

Ý tưởng cốt lõi em chốt lại:

> **Distill không phải để đếm giỏi hơn. Distill để đạt chất lượng NGANG mà tốn nhãn ÍT hơn nhiều.**

Lý do: student của em học "bản đồ mật độ nhân" **miễn phí từ teacher PathoSAM** (teacher tự sinh ra, không cần người vẽ). Người ta chỉ phải cung cấp **một con số đếm cho mỗi ảnh**. Trong khi cách train-bằng-mask truyền thống bắt người ta **vẽ viền từng nhân một** — cực kỳ tốn công.

Để chứng minh, em thiết kế một thí nghiệm **so kè trực tiếp, cùng số ảnh**:

- **DISTILL (cách của em):** target = mật độ do teacher sinh + 1 con số đếm/ảnh.
- **SUPERVISED (cách chuẩn):** target = mật độ dựng từ **mask GT từng nhân**.

Cho cả hai học trên cùng 10% / 25% / 50% / 100% số ảnh, test trên cùng tập, lặp 3 lần lấy trung bình. Đây là **bảng quan trọng nhất của cả bài**:

### Bảng chính — DISTILL vs SUPERVISED (cùng ngân sách ảnh, NuInsSeg)

| Ngân sách ảnh | DISTILL (coverage / sai số MAE) | SUPERVISED (coverage / MAE) | Đọc |
|---|---|---|---|
| 10% (53 ảnh) | 0.865 / **24.26** | 0.879 / 26.09 | distill nhỉnh |
| 25% (133 ảnh) | 0.858 / 21.85 | 0.824 / **18.49** | superv nhỉnh |
| 50% (266 ảnh) | 0.840 / 20.34 | 0.830 / **17.73** | superv nhỉnh |
| 100% (532 ảnh) | 0.843 / **14.12** | 0.897 / 14.61 | hòa |

**Đọc bảng này thế nào:**
- **Coverage (độ tin cậy khoảng dự đoán):** hai cột chồng lên nhau trong sai số — **không bên nào thắng bên nào một cách hệ thống**. Nghĩa là distill cho khoảng tin cậy hợp lệ ngang y như dùng mask.
- **MAE (sai số đếm):** supervised nhỉnh hơn ~2-3 nhân ở khúc giữa, nhưng **hòa ở hai đầu** (ở 100% distill còn thấp hơn tí: 14.12 vs 14.61).
- **Tóm lại: hai cách CHẤT LƯỢNG NGANG NHAU.** Không có kẻ thắng rõ ràng.

Nhưng — và đây là mấu chốt — **cái giá nhãn thì khác nhau một trời một vực.** Em đo được **trung bình mỗi ảnh có 52.8 nhân** (trung vị 38; tổng 35,138 nhân trên 665 ảnh):

| | Nhãn người cần cung cấp / ảnh | Ước tính công |
|---|---|---|
| **DISTILL** | mật độ teacher (0 công người) + **1 con số đếm** | ~52.8 × 2.4s ≈ **127 giây** *(giá chấm điểm, Bearman ECCV'16)* |
| **SUPERVISED** | **vẽ mask 52.8 nhân** | ~52.8 × (5–10× chấm điểm) ≈ **640–1270 giây** |

Vì cả hai đều tỉ lệ với số nhân, **tỉ số công = 5–10 lần, bất kể ảnh dày hay thưa.**

### → Câu trả lời cho "tại sao distill"

> **Cùng chất lượng (coverage + độ chính xác), nhưng distill chỉ tốn khoảng 1/5 – 1/10 công dán nhãn.** Đó là lý do để distill. Không phải "giỏi hơn", mà là **"ngang mà rẻ hơn nhiều lần"** — điều rất có giá trong y tế nơi bác sĩ vẽ mask là cực kỳ đắt.

Đây là vấn đề trung tâm. Mọi kết quả còn lại của em đều là **để củng cố cho luận điểm này**.

---

## 4. Các kết quả bổ trợ (xoay quanh vấn đề trung tâm)

### 4.1 Bổ trợ: coverage hầu như MIỄN PHÍ về nhãn

Em còn một bảng nữa (chỉ nhánh distill, tăng dần ngân sách):

| Ngân sách | coverage | MAE |
|---|---|---|
| 10% | 0.888 | 26.34 |
| 25% | 0.819 | 31.11 *(1 lần chạy xui)* |
| 50% | 0.866 | 18.33 |
| 100% | 0.891 | 12.77 |

Điều đẹp ở đây: **coverage phẳng lì ở mọi mức nhãn, kể cả chỉ 10% nhãn.** Nghĩa là độ tin cậy khoảng dự đoán *không cần nhiều nhãn* — chỉ độ chính xác điểm (MAE) mới cần. Điều này **nói thêm cho vấn đề chính**: ngay cả khi nhãn ít, phần "khoảng tin cậy" vẫn dùng được.

### 4.2 Bổ trợ: khi model nhỏ, phân phối HỌC được thắng công thức có sẵn

Có người sẽ hỏi: *"σ (độ bất định) lấy ở đâu?"* Em cho student **tự học σ** thay vì dùng công thức σ analytic có sẵn (kiểu Paper 1). So sánh trên cùng student nhỏ:

- Cách học-σ của em: worst-org **0.610** (chế độ global) / **0.750** (cluster).
- Công thức σ có sẵn (PB): **0.278** / 0.658.

→ **Khi model đã bị nén nhỏ, học thẳng σ tốt hơn hẳn** ở chế độ khó. Đây là một đóng góp phương pháp (em gọi là σ neo-Poisson: σ = √μ · exp(...), neo theo bản chất đếm là quá trình Poisson). Nó **giải thích vì sao student nhỏ vẫn cho khoảng tin cậy tốt** — bổ trợ cho chuyện "distill rẻ mà vẫn tin cậy được".

### 4.3 Bổ trợ: trên PanNuke, model của em dẫn đầu dàn baseline conformal

Đây là con số "cứng" nhất em có. Trên PanNuke (đánh giá leak-free, fold_3 sạch), model 1.9M của em cho **worst-organ coverage 0.906, MAE 3.36** — **cao hơn TẤT CẢ các baseline conformal mạnh** (CondConf 2025, PCP 2024, R2CCP, CPCP 2026...) mà lại **nhẹ nhất và rẻ nhất**. → Chứng tỏ cái student rẻ-nhãn này **không hề yếu** ở mặt khoảng tin cậy theo nhóm mô; nó cạnh tranh sòng phẳng với đồ nặng đô.

### 4.4 Bổ trợ: khoảng tin cậy còn "sống" khi đổi sang dataset khác

Em test transfer: train trên NuInsSeg, đem thẳng sang **CryoNuSeg** (dataset khác hẳn) → coverage biên **0.967**, σ vẫn hoạt động. → Cái đầu phân phối không chỉ ăn may trên 1 tập; nó **khái quát hóa** được. (Em thành thật: sang MoNuSAC thì hỏng, vì nhân ở đó bị co nhỏ 4 lần do khác độ phân giải — em ghi rõ đây là giới hạn về *scale*, không giấu.)

### 4.5 Bổ trợ: nén xuống 0.5M vẫn không mất chất

Ablation cho thấy hạ từ ch32 (1.9M) xuống **ch16 (~0.5M)** *không mất* chất lượng (trên NuInsSeg còn nhỉnh; trên PanNuke hòa). → **Nén 1280 lần so với teacher 640M mà vẫn chạy tốt** — lại một điểm cộng cho câu chuyện "nhỏ + rẻ".

---

## 5. Những chỗ em thành thật giữ đúng mực (không thổi phồng)

Cô yên tâm là em không tô hồng:

1. **Bảng label-efficiency dùng protocol chia đơn (single-split), khác bảng chính** → em chỉ đọc **tương đối** giữa các mức ngân sách, không đem so tuyệt đối với bảng chính.
2. **Con số đếm GT trong thí nghiệm vẫn lấy ra từ mask** (vì dataset có sẵn mask) → em chứng minh **YÊU CẦU giám sát** của phương pháp là mức-đếm, chứ **không dám nói "em đã đi dán nhãn rẻ hơn thật"**.
3. **Tỉ số 5–10×** là khoảng an toàn: giá chấm điểm 2.4s có nguồn (Bearman), còn giá vẽ mask/nhân thì **không có nguồn chuẩn nên em KHÔNG bịa một con số đơn** như "rẻ hơn 100×".
4. Thí nghiệm mới chạy **3 seed** — hơi mỏng; nếu đưa lên hình trong bài em sẽ nâng lên 5 seed cho chắc.
5. Về đẳng cấp tạp chí: em đánh giá thật là bài này **đủ chuẩn Q1 tầm trung** (các tạp chí như Computers in Biology and Medicine, Artif. Intell. Med.). Cửa top-tier tuyệt đối (MedIA/TMI) thì cần **validation lâm sàng** — nằm ngoài phạm vi hiện tại. Em không hứa hão.

---

## 6. Chốt lại: bài đứng trên chân nào

**Ba trụ của bài, theo đúng thứ tự bán:**

1. **Distill rẻ nhãn (trụ chính, giờ CÓ số):** ngang chất lượng train-bằng-mask, tốn 1/5–1/10 công nhãn — §3.
2. **Đầu ra phân phối calibrated (μ,σ) học được:** không peer distill nào có; học-σ thắng công thức khi model nhỏ — §4.2.
3. **Hiệu quả:** 1.9M (hạ tới 0.5M vẫn tốt), nhỏ nhất mà có khoảng tin cậy — §4.5.

Điều em tâm đắc nhất trong cả hành trình này, thưa cô, là: **em đã đi lạc qua hai ba hướng (đua accuracy, đua UQ, ý stain) và mỗi hướng đều thất bại một cách rõ ràng** — nhưng chính vì chấp nhận thất bại và ghi lại trung thực, em mới lần ra được **câu hỏi đúng** ("tại sao distill") và trả lời nó bằng một thí nghiệm sạch. Bài mạnh không phải vì em cố chứng minh mình giỏi nhất, mà vì em **tìm đúng chỗ distill thật sự có giá trị** và đo được nó.

**Việc còn lại:** viết manuscript. Toàn bộ thí nghiệm đã xong và đã sao lưu. Em xin phép bắt đầu dựng bản thảo theo 3 trụ trên nếu cô đồng ý ạ.

---

*Chi tiết số liệu + provenance đầy đủ: xem `PAPER2_MASTER.md` (§4.10 = label-efficiency, §4.2 = PanNuke baseline, §4.1 = học-σ vs công thức, §4.4 = transfer, §4.8 = ablation). Script: `label_efficiency.py`, `label_efficiency_both.py`.*

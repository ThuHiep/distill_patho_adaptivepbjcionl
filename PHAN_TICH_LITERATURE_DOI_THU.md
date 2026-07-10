# Phân tích literature & định vị đề tài — bài của tôi khác họ chỗ nào, hơn được gì

> Cập nhật: 06/2026. Mục đích: trả lời cho cô (và reviewer) ba câu —
> (1) người ta đã làm gì, (2) bài của em **khác** chỗ nào, (3) em có gì **hơn**.
> Nguyên tắc viết: trung thực. Cái gì đã có thì ghi "đã có", chỉ nhận là mới
> ở phần thật sự mới. Reviewer ghét nhất là nhận vơ.

---

## 0. Cách đọc file này

Bài của em nằm ở **giao của 4 dòng nghiên cứu** mà bình thường tách rời nhau:

```
   (A) Conformal trong         (B) Online/Adaptive
       bệnh học số                 conformal dưới shift
            \                         /
             \                       /
              >>>  ĐỀ TÀI CỦA EM  <<<
             /                       \
            /                         \
   (C) Đếm/segment nhân tế     (D) Conformal đa lớp /
       bào bằng foundation model    conformal cho count
```

Không có bài nào đứng ở đúng **giao điểm** này. Mỗi dòng có nhiều bài mạnh,
nhưng **không ai gộp cả bốn**. Đó là khe hở của em. Bên dưới phân tích từng dòng:
họ làm gì, em khác gì, em hơn/kém gì, và **chỗ dễ bị hỏi vặn**.

---

## A. Conformal prediction trong bệnh học số (digital pathology)

### Họ đã làm gì
| Bài | Nội dung | Loại bài toán |
|---|---|---|
| Olsson et al., *Nature Communications* 2022 — "Estimating diagnostic uncertainty in AI-assisted pathology using conformal prediction" | Dùng conformal để **gắn cờ ca không tin cậy** trong chẩn đoán ung thư tuyến tiền liệt. Giảm lỗi từ 2% → 0.1%, đánh dấu 22% ca khi gặp dữ liệu mới | **Phân loại** (classification) ở mức slide/ca |
| "Conformalized uncertainty-aware framework for NSCLC", arXiv 2501.00053 (2025) | Conformal trên whole-slide foundation model cho ung thư phổi | **Phân loại** WSI |
| "Pitfalls of Conformal Predictions for Medical Image Classification", arXiv 2506.18162 (2025) | Chỉ ra các cạm bẫy khi áp conformal vào ảnh y khoa | **Phân loại**, phê bình |
| Validation in cervical atypia, *Sci. Reports* 2026 | Conformal cho sàng lọc ung thư cổ tử cung | **Phân loại** |

### Em **khác** chỗ nào
- **Tất cả các bài này là CONFORMAL CHO PHÂN LOẠI** (output = nhãn lớp / tập nhãn).
  Bài của em là **conformal cho ĐẾM** (output = số nguyên đếm + khoảng `[l, u]`).
  Đây là khác biệt căn bản: phân loại cho ra *prediction set* (tập nhãn);
  đếm cho ra *prediction interval* (khoảng số). Cơ chế nonconformity khác hẳn.
- Họ làm ở **mức slide/ca**; em làm ở **mức số đếm từng loại tế bào** trong một ảnh patch.

### Em **hơn** gì / **kém** gì
- ✅ Hơn: em mang conformal xuống **tác vụ đếm có cấu trúc** (Poisson-Binomial),
  một chỗ chưa ai chạm trong pathology. Họ chưa ai mô hình hoá độ bất định của
  **số đếm** bằng cấu trúc xác suất của từng instance.
- ✅ Hơn: em xử lý **distribution shift** (đổi dataset). Olsson chỉ *gắn cờ* khi
  gặp dữ liệu lạ — **không phục hồi** bảo đảm. Em phục hồi (online/adaptive).
- ⚠️ Kém / rủi ro: các bài này đã *được đăng ở venue mạnh* và đã "cắm cờ" rằng
  "conformal hữu ích cho pathology". Reviewer sẽ hỏi *"khác Olsson 2022 chỗ nào?"*
  → trả lời: **task khác (đếm vs phân loại), có xử lý shift, có cơ chế online**.

---

## B. Online / Adaptive conformal dưới distribution shift  ⚠️ DÒNG NGUY HIỂM NHẤT

Đây là dòng **đông đúc và mạnh nhất** — và là nơi em dễ bị "đã có rồi" nhất.

### Họ đã làm gì
| Bài | Cơ chế | Điều chỉnh cái gì |
|---|---|---|
| Gibbs & Candès 2021 — **ACI** | `α_{t+1} = α_t + γ(α* − err_t)` | chỉnh **mức α** theo lỗi |
| Gibbs & Candès 2022 — **DtACI** | reweight bước học theo exp | chỉnh α, thích nghi nhanh hơn |
| Zaffran et al. 2022 | ACI cho chuỗi thời gian | mức α |
| Bhatnagar et al. 2023 — **SAOCP** | strongly-adaptive, regret bound | mức α, đảm bảo mạnh hơn |
| Barber, Candès, Ramdas, Tibshirani 2023 — **"Conformal Prediction Beyond Exchangeability"** (NexCP) | **weighted** nonconformity, **cận coverage-gap theo độ lệch khỏi exchangeable** | trọng số mẫu |
| **arXiv 2511.04275 (11/2025)** — "Online Conformal Inference with **Retrospective Adjustment**" | hồi cứu, **chỉnh lại cả dự đoán quá khứ** bằng leave-one-out regression | dự đoán quá khứ |

### Em **khác** chỗ nào (đọc kỹ — đây là phần phải nói chuẩn)
1. **ACI / DtACI / SAOCP chỉnh MỨC α; em chỉnh KÍCH THƯỚC CỬA SỔ (W_eff).**
   Đây là trục điều khiển khác. ACI giữ nguyên tập dữ liệu, lắc ngưỡng α lên xuống.
   Em giữ nguyên α = 0.1, **thay tập score** (cửa sổ trượt co/nới). Trong thực
   nghiệm của em (Bảng 5, cross-dataset) **online-window thắng ACI** ở Winkler —
   đây là bằng chứng thực nghiệm cho lựa chọn này, **phải giữ và làm nổi bật**.
2. **Barber 2023 (NexCP) đã có cận coverage-gap dưới shift.** ⚠️ Đây là điểm
   em **không được nhận là mới**. Cận `Coverage ≥ 1−α − (độ lệch khỏi exchangeable)`
   đã tồn tại. Phần lý thuyết của em phải định vị là *"vận dụng cận có sẵn,
   cụ thể hoá cho cấu trúc count/PB"* — KHÔNG phải phát minh cận.
3. **arXiv 2511.04275 — đối thủ trực tiếp & mới nhất.** ⚠️⚠️ Bài này cũng làm
   *online conformal dưới shift* và có yếu tố **hồi cứu/điều chỉnh quá khứ** —
   nghe rất giống phần "delayed feedback" của em. **PHẢI phân biệt rõ:**
   - Họ: **chỉnh lại (re-issue) dự đoán quá khứ** bằng regression leave-one-out
     — tức là *thay đổi khoảng đã phát ra*.
   - Em: **không re-issue**. Khoảng đã phát cho ảnh cũ giữ nguyên; khi nhãn về
     muộn, em chỉ **nạp score đó vào cửa sổ** để cập nhật `q̂` cho ảnh **tương lai**.
     Đây là *forward-only với nhãn trễ*, hợp với ràng buộc lâm sàng (khoảng đã
     báo cho bác sĩ thì không rút lại được).
   - Họ: bài lý thuyết tổng quát (synthetic + regression). Em: **pathology, count,
     PB, foundation-model backbone, có Winkler**.

### Em **hơn** gì / **kém** gì
- ✅ Hơn: không bài online nào ở trên gắn với **cấu trúc đếm Poisson-Binomial**
  hay **joint multi-class**. Họ làm hồi quy/chuỗi thời gian một chiều đầu ra.
- ✅ Hơn: em có **ứng dụng thật + đo shift được** (MMD²/Wasserstein/Energy ở Bảng 2)
  nối thẳng độ shift đo được vào hành vi cửa sổ.
- ❌ Kém (thành thật): **lý thuyết của họ chặt hơn em hiện tại.** SAOCP/DtACI có
  *regret bound* và *long-run coverage* chứng minh đàng hoàng. Em **mới có thực
  nghiệm**. Đây chính là lý do "kế hoạch sắp tới" của em là đi chứng minh cận cho
  cơ chế cửa sổ — để không bị bỏ lại ở mặt lý thuyết.
- ❌ Rủi ro: 2511.04275 quá mới (11/2025). Reviewer có thể bắt em **so sánh/trích**.
  → Cần thêm nó vào related work, nêu rõ khác biệt forward-only-with-delay vs
  retrospective-re-issue.

---

## C. Đếm / segment nhân tế bào bằng foundation model

### Họ đã làm gì
| Bài | Nội dung |
|---|---|
| **PathoSAM** — "Segment Anything for Histopathology", arXiv 2502.00408 (2025) | Foundation model **đầu tiên** cho instance segmentation nhân tế bào; SOTA interactive + automatic; backbone ViT-L. **Em đang dùng làm backbone thứ 2** |
| **CellViT / CellViT++** | Segment + phân loại nhân trên H&E, fine-tune PanNuke, backbone Virchow/SAM/HIPT |
| **Cellpose-SAM** (bioRxiv 2025) | Tổng quát hoá "siêu phàm", train trên 22k ảnh gồm cả PanNuke |
| **SAM3 readiness** — Kong et al. 2025 "Is SAM3 ready for pathology segmentation?" | **Đánh giá** SAM3 trên pathology (không phải model paper). Em dùng SAM3 làm backbone chính |
| HoVer-Net, StarDist | Baseline kinh điển segment+phân loại nhân |

### Em **khác** chỗ nào
- **Tất cả các bài này dừng ở SEGMENTATION / DETECTION** (ra mask + nhãn).
  **Không ai trong số họ cho ra khoảng tin cậy có bảo đảm cho SỐ ĐẾM.**
  Họ báo Dice/AJI/PQ/F1 — **không báo coverage, không báo interval, không Winkler**.
- Em **lấy output của họ làm đầu vào** (detect → instance → đếm), rồi **bọc một
  lớp định lượng bất định có bảo đảm** lên trên. Đây là tầng họ không có.

### Em **hơn** gì / **kém** gì
- ✅ Hơn rõ rệt: em là tầng **trustworthiness/uncertainty** mà các SOTA segmentation
  còn thiếu. Một bác sĩ dùng CellViT chỉ nhận được "47 tế bào"; dùng hệ của em
  nhận "47, khoảng [41, 53] với độ tin cậy 90%, và khoảng này **vẫn đúng khi đổi
  bệnh viện**". Đó là giá trị lâm sàng họ chưa cung cấp.
- ✅ Hơn: em **backbone-agnostic** — chứng minh trên CẢ SAM3 (yếu, MAE cao) lẫn
  PathoSAM (mạnh, MAE 2.88). Cho thấy phương pháp không phụ thuộc một backbone.
- ❌ Kém: về **chất lượng segmentation thuần tuý**, em không cạnh tranh với họ
  (và không nên cố). Phải nói rõ: *"em không cải tiến segmentation; em xây tầng
  định lượng bất định ở trên bất kỳ segmenter nào."* Nếu lỡ khoe "đếm chính xác hơn"
  sẽ bị đập vì đó không phải đóng góp và cũng không đúng.

---

## D. Conformal đa lớp / conformal cho output có cấu trúc

### Họ đã làm gì
| Bài | Nội dung |
|---|---|
| "Conformal Prediction for Hierarchical Data", arXiv 2411.13479 (2024) | Conformal cho dữ liệu phân cấp, nhiều đầu ra |
| "Class conditional CP for multiple inputs by p-value aggregation", arXiv 2507.07150 (2025) | Gộp p-value để giữ coverage theo lớp |
| "Probabilistic Object Detection with Conformal Prediction", arXiv 2605.07549 | Conformal cho **object detection** (multi-output có cấu trúc) |
| "Multi-Scale Conformal Prediction", arXiv 2502.05565 | Khung lý thuyết coverage đa thang đo |

### Em **khác** chỗ nào
- **max-statistic joint coverage** (`S = max_k R_k`) mà em dùng là **kỹ thuật ĐÃ CÓ**
  trong conformal đa đầu ra/đồng thời. ⚠️ **Em KHÔNG được nhận joint-coverage là
  phát minh.** (Đã thống nhất trong các buổi trước.)
- Cái mới của em **không phải** "joint coverage", mà là **ghép joint coverage với
  độ bất định Poisson-Binomial của số đếm** — tức nonconformity `R_k = |N_k−E[N_k]|/σ_k`
  với `σ_k` **lấy từ cấu trúc xác suất của các instance**, không phải residual trần.
- Conformal object detection (2605.07549) gần nhất về tinh thần (output có cấu trúc),
  nhưng họ bọc **box/nhãn**, không bọc **số đếm theo lớp với phương sai PB**.

### Em **hơn** gì / **kém** gì
- ✅ Hơn: **σ_k tự co giãn theo độ khó ảnh** nhờ PB — khoảng hẹp ở ảnh dễ, rộng ở
  ảnh khó, *trong cùng một mức bảo đảm*. Đây là điểm tinh tế ít bài count nào có.
- ❌ Kém / rủi ro: phần "joint qua max-stat" là đồ mượn. Nếu reviewer ở mảng
  conformal đọc, họ biết ngay. → Đừng dồn trọng số đóng góp vào đây; dồn vào
  **PB-structured score + online dưới shift pathology**.

---

## 1. Định vị một câu (dùng được trong báo cáo & abstract)

> "Các công trình hiện có hoặc (A) áp conformal cho **phân loại** trong bệnh học số,
> hoặc (B) phát triển conformal **online** dưới shift cho hồi quy/chuỗi thời gian
> tổng quát, hoặc (C) xây foundation model **segment** nhân tế bào nhưng **không
> định lượng bất định**, hoặc (D) làm conformal **đa lớp** trên phân loại. Đề tài
> của em đứng ở giao điểm còn trống: **định lượng bất định có bảo đảm cho bài toán
> ĐẾM nhân tế bào đa lớp, dựa trên cấu trúc Poisson-Binomial, và duy trì bảo đảm
> đó dưới distribution shift bằng cơ chế hiệu chỉnh online/adaptive** — chạy trên
> foundation-model backbone (SAM3, PathoSAM)."

---

## 2. Ba điểm mới THẬT (chốt được, không sợ bị bóc)

1. **Score conformal có cấu trúc Poisson-Binomial cho số đếm.** `σ_k` suy ra từ
   xác suất tồn tại × phân loại của từng instance, không phải residual rỗng.
   → Chưa thấy trong literature đếm tế bào.
2. **Bộ ba (đếm đa lớp joint) × (PB uncertainty) × (online dưới shift pathology)
   ghép lại.** Từng mảnh có thể có nơi khác, nhưng **tổ hợp + ứng dụng + bằng chứng
   trên 2 backbone + Winkler headline** là gói chưa ai có.
3. **(Kế hoạch) cận lý thuyết cho cơ chế cửa sổ:** chứng minh cửa sổ trượt bám
   regime hiện tại → độ lệch khỏi exchangeable nhỏ hơn → cận coverage chặt hơn
   conformal tĩnh. Đây là phần **thực sự novel** nếu làm được (xây trên Barber 2023,
   không trùng).

## 3. Ba chỗ KHÔNG được nhận là mới (để khỏi bị đập)

1. ❌ **Joint coverage qua max-statistic** — kỹ thuật đã có. Chỉ trích dẫn, dùng lại.
2. ❌ **Cận coverage-gap dưới shift** — Barber et al. 2023 đã có dạng cận này. Em
   vận dụng, không phát minh.
3. ❌ **Conformal hữu ích cho pathology** — Olsson 2022 đã cắm cờ. Em là *task mới*
   trong pathology, không phải người đầu tiên đưa conformal vào pathology.

---

## 4. Đối thủ nguy hiểm nhất & cách phòng thủ

| Đối thủ | Vì sao nguy | Câu phòng thủ |
|---|---|---|
| **arXiv 2511.04275** (online conformal retrospective, 11/2025) | Cũng online + shift + "điều chỉnh quá khứ" | "Họ **re-issue dự đoán quá khứ** bằng regression; em **không rút lại khoảng đã phát** (ràng buộc lâm sàng) — nhãn trễ chỉ cập nhật `q̂` cho ảnh tương lai. Khác cơ chế, khác giả định triển khai." |
| **Barber 2023 (NexCP)** | Đã có cận shift | "Em **vận dụng** cận của họ, cụ thể hoá cho count/PB; đóng góp là phần online làm δ nhỏ đi, không phải bản thân cận." |
| **SAOCP / DtACI** | Lý thuyết chặt hơn | "Họ chỉnh **α**; em chỉnh **cửa sổ** và thắng ở Winkler thực nghiệm. Em đang bổ sung phần lý thuyết cho cơ chế cửa sổ (kế hoạch sắp tới)." |
| **Olsson 2022** | Conformal pathology đã đăng | "Họ **phân loại + gắn cờ**; em **đếm + phủ có bảo đảm + phục hồi dưới shift**." |
| **PathoSAM / CellViT++** | SOTA segment mạnh | "Em **không thi segmentation**; em xây tầng định lượng bất định lên trên output của họ — thứ họ không cung cấp." |

---

## 5. Việc cần làm để CHẮC CHẮN hơn họ (checklist)

- [ ] **Thêm 2511.04275 vào related work** và viết rõ 2–3 câu phân biệt (forward-only
      with delayed labels vs retrospective re-issue). Không né, đối diện thẳng.
- [ ] **Trích Barber 2023 đúng chỗ** ở phần lý thuyết, định vị cận là "adapted".
- [ ] **Bảng so sánh trục điều khiển:** ACI/DtACI/SAOCP chỉnh α — của em chỉnh W.
      Kèm số Winkler để cho thấy chỉnh-W thắng chỉnh-α trên bài count/shift của em.
- [ ] **Làm nổi PB-structured score** như đóng góp #1 (chỗ này sạch nhất, ít ai đụng).
- [ ] **Đừng khoe segmentation/đếm chính xác hơn** — không phải đóng góp, dễ bị đập.
- [ ] **Hoàn thành cận lý thuyết cửa sổ** (kế hoạch) — đây là cái nâng bài từ
      "ứng dụng hay" lên "có đóng góp lý thuyết", cần cho TMI/MIA.

---

## Nguồn tham khảo

**Conformal trong pathology**
- Olsson et al. 2022 — https://www.nature.com/articles/s41467-022-34945-8
- NSCLC conformal framework 2025 — https://arxiv.org/pdf/2501.00053
- Pitfalls of CP for medical imaging 2025 — https://arxiv.org/pdf/2506.18162
- Cervical atypia validation 2026 — https://www.nature.com/articles/s41598-026-44850-5

**Online / adaptive conformal dưới shift**
- Online CP with Retrospective Adjustment 2025 — https://arxiv.org/abs/2511.04275
- Conformal Inference for Online Prediction with Arbitrary Shifts — https://arxiv.org/pdf/2208.08401
- Lévy–Prokhorov shift robustness 2025 — https://arxiv.org/html/2502.14105v2
- (Barber, Candès, Ramdas, Tibshirani 2023 — Conformal Prediction Beyond Exchangeability)
- (Gibbs & Candès 2021 ACI; 2022 DtACI; Bhatnagar et al. 2023 SAOCP)

**Foundation model segment/đếm nhân tế bào**
- Segment Anything for Histopathology (PathoSAM) — https://arxiv.org/pdf/2502.00408
- Revisiting foundation models for cell instance segmentation — https://arxiv.org/html/2603.17845v1
- Cellpose-SAM — https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1.full
- Fine-grained multiclass nuclei (All-in-SAM) — https://arxiv.org/pdf/2508.15751

**Conformal đa lớp / output có cấu trúc**
- CP for Hierarchical Data 2024 — https://arxiv.org/pdf/2411.13479
- Class-conditional CP by p-value aggregation 2025 — https://arxiv.org/pdf/2507.07150
- Probabilistic Object Detection with CP — https://arxiv.org/abs/2605.07549
- Multi-Scale Conformal Prediction 2025 — https://arxiv.org/pdf/2502.05565

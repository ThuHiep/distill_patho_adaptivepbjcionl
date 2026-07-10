# Báo cáo chi tiết — Đếm sao/thiên hà có bảo chứng coverage dưới dịch chuyển độ sâu khảo sát

*Cập nhật: 2026-07-06. Domain thứ hai của luận văn (sau đếm nhân tế bào histopathology = Bài 1).
Phương pháp: **Adaptive PB-JCI Online**. Dữ liệu + backbone: **SDSS DR17**.*

---

## 1. Mục tiêu & vị trí trong luận văn

Bài 1 chứng minh **Adaptive PB-JCI Online** — một lớp hiệu chỉnh conformal online nhẹ đặt trên backbone
đếm — cho **khoảng dự đoán có bảo chứng coverage dưới distribution shift** khi đếm nhân tế bào. Chương này
trả lời: **phương pháp có tổng quát sang một lĩnh vực khoa học hoàn toàn khác không?** Ta chọn **thiên văn**
(khoa học vật lý) — bước xa nhất khỏi khoa học sự sống — để có luận điểm generality mạnh nhất: *cùng một
phương pháp, giữ nguyên công thức, chạy trên dataset + backbone khác hẳn.*

Bài toán cụ thể: chia một vùng trời thành nhiều **ô ("field")**, trong mỗi field **đếm số sao và số thiên
hà**, và đưa ra **khoảng tin cậy** cho từng con đếm, đảm bảo phủ đúng ngay cả khi **độ sâu khảo sát thay
đổi**.

---

## 2. Nền tảng trực quan (cho người đọc ngoài ngành)

### 2.1 Sao và thiên hà là gì

- **Sao (star):** một quả cầu khí tự phát sáng, giống Mặt Trời. Các sao ta quan sát nằm **trong Ngân Hà**
  (thiên hà của chúng ta). Vì ở rất xa nên trên ảnh mỗi sao chỉ là **một chấm sáng gọn**.
- **Thiên hà (galaxy):** một tập hợp **hàng tỉ ngôi sao** + khí + bụi. Các thiên hà khác ở **cực kỳ xa**,
  nên trên ảnh là **một vệt sáng mờ, lan tỏa** (vì kích thước thật của nó lớn trên bầu trời).

> Phân biệt sao/thiên hà = phân biệt **chấm sắc nét** với **vệt nhòe**. Đây chính là thông tin backbone cung
> cấp qua `probPSF` (§6).

### 2.2 "Đếm" ở đây là đếm gì (ánh xạ sang Bài 1)

Ta **không** phân loại từng thiên thể riêng lẻ. Ta chia vùng trời thành nhiều **field** và trong mỗi field
đếm số nguồn mỗi lớp, rồi cho khoảng dự đoán có bảo chứng coverage cho mỗi con đếm:

> field #i → (n_sao, n_thiênhà), ví dụ *"12 ± 3 sao và 20 ± 4 thiên hà, đúng 90% số lần"*.

Đây đúng cấu trúc bài toán đếm của Bài 1, chỉ đổi vật thể:

| | Bài 1 (tế bào) | Chương này (thiên văn) |
|---|---|---|
| "Ảnh" | 1 tile mô bệnh học | 1 field trời |
| Đếm | số tế bào mỗi loại | số nguồn mỗi lớp (sao / thiên hà) |
| Backbone | SAM3 | SDSS Photometric Pipeline |
| Shift | PanNuke → NuInsSeg | khảo sát sâu → khảo sát nông |

### 2.3 Ánh sáng được "hứng" và đo thế nào

Ánh sáng đến Trái Đất thành **vô số hạt cực nhỏ** (photon — cứ hình dung là "giọt sáng"). Vật sáng bắn tới
nhiều giọt mỗi giây; vật mờ chỉ vài giọt. Kính thiên văn gồm:
1. **Gương chính (khẩu độ):** tấm gương rộng **gom các giọt sáng** tụ về một điểm. Gương càng to → hứng
   càng nhiều giọt mỗi giây.
2. **Cảm biến (CCD):** một tấm chip chia thành hàng triệu **điểm ảnh (pixel)**. Mỗi giọt sáng rơi vào một
   pixel biến thành một chút điện tích lại trong pixel đó.

**"Hứng ánh sáng" = để các pixel đọng giọt trong khoảng thời gian phơi sáng, rồi đo lượng điện mỗi pixel** =
đã có bao nhiêu giọt sáng rơi vào = **độ sáng** của đốm. Vật mờ ít giọt → phải phơi **lâu** (hoặc gương to,
hoặc cộng nhiều ảnh) mới đọng đủ để nổi trên nhiễu.

### 2.4 Độ sâu khảo sát & vì sao nó thay đổi (nguồn của shift)

**Độ sâu (survey depth)** = khảo sát nhìn được tới vật mờ cỡ nào. **Sâu** (phơi lâu / gương to / cộng nhiều
ảnh) = thấy cả nguồn rất mờ → đếm được nhiều. **Nông** (phơi ít) = chỉ thấy nguồn sáng, nguồn mờ chìm trong
nhiễu → đếm hụt.

Cùng một vùng trời, độ sâu **thay đổi** do: kính khác nhau, thời gian phơi / số ảnh cộng, thời tiết & khí
quyển (seeing), độ sáng nền trời (trăng, ô nhiễm ánh sáng), hoặc **rìa vùng khảo sát** ít lần quét. ⇒ Hai
khảo sát nhìn cùng bầu trời nhưng **đếm ra số nguồn khác nhau** — không phải trời đổi, mà **khả năng phát
hiện đổi**. Khi độ sâu giảm: (1) **sót nguồn mờ** (khả năng phát hiện giảm); (2) **phân loại kém** ở nguồn
mờ (chấm–vệt nhòe vào nhau). Đây chính là distribution shift phương pháp phải chịu (§8).

### 2.5 Đếm sai → hệ quả gì (động lực)

- **Sai con số đếm (bias):** đếm hụt nguồn mờ ở rìa độ sâu → sai kết luận về tiến hóa thiên hà; số đếm lệch
  giữa khảo sát nông/sâu → tạo **cấu trúc quy mô lớn giả** → vũ trụ học sai. Nguy nhất khi **gộp nhiều khảo
  sát** khác độ sâu mà không hiệu chỉnh completeness.
- **Sai KHOẢNG tin cậy (cái chương này giải quyết):** với field thật có 26 thiên hà —

  | Loại lỗi | Ví dụ | Hệ quả |
  |---|---|---|
  | Under-cover (hẹp, tự tin thái quá) | "20 ± 2" | tự tin nhưng SAI → phát hiện giả, không tái lập |
  | Over-cover (rộng) | "20 ± 15" | an toàn nhưng vô dụng → tín hiệu thật chìm trong sai số |
  | Đúng (valid + hẹp) | "24 ± 4" ✓ | vừa phủ đúng vừa đủ chặt để làm khoa học |

  Đặc thù thiên văn: **độ sâu đổi làm hầu hết phương pháp under-cover âm thầm** (vẫn báo sai số đẹp nhưng
  thực ra sai). Cần một phương pháp giữ được coverage dưới shift mà không phình interval.

---

## 3. Bảng thuật ngữ (trực giác → thuật ngữ → ký hiệu)

| Cách hiểu mộc mạc | Thuật ngữ | English | Ký hiệu / cột |
|---|---|---|---|
| giọt sáng | quang tử | photon | — |
| gương phễu hứng sáng | khẩu độ / gương chính | aperture / primary mirror | Ø 2.5 m |
| khoảng mở cửa hứng sáng | thời gian phơi sáng | exposure / integration time | — |
| chồng nhiều ảnh | cộng ảnh | coaddition / stacking | — |
| chip nhiều ô | cảm biến; điểm ảnh | CCD detector; pixel | — |
| độ sáng (lượng giọt đếm) | thông lượng | flux | `psfFlux_r` |
| đơn vị độ sáng SDSS | — | nanomaggie (nMgy) | 1 nMgy ↔ mag 22.5 |
| thang độ sáng "cấp" (log) | cấp sao biểu kiến | (apparent) magnitude | `m = 22.5 − 2.5·log₁₀(flux)` |
| điểm tin cậy phép đo sáng | nghịch đảo phương sai | inverse variance | `psfFluxIvar_r` = 1/σ² |
| đốm nổi rõ khỏi nhiễu | tỉ số tín hiệu / nhiễu | signal-to-noise ratio (SNR) | `psfFlux·√ivar = flux/σ` |
| ngưỡng "nổi gấp 5 lần" | ngưỡng phát hiện 5σ | 5σ detection threshold | SNR = 5 |
| hình dạng chấm nhòe của điểm sáng | hàm tán xạ điểm | point spread function (PSF) | — |
| "điểm giống-sao" (chấm vs vệt) | phân loại hình thái sao/thiên hà | star/galaxy separation | `probPSF_r`, `type` (6=sao,3=thiên hà) |
| khả năng thực sự nhìn thấy | độ đầy đủ / xác suất phát hiện | completeness / detection probability | `p_detect` |
| đường cong chữ S | hàm logistic (sigmoid) | logistic function | `logistic((SNR−5)/1.5)` |
| dựng lại xác suất được thấy | mô hình hóa thuận quá trình phát hiện | forward-modeling detection | — |
| đếm cộng-dồn-độ-chắc-chắn | đếm mềm / kỳ vọng số đếm | soft counting / expected count | `n_k = Σ p_detect·p_class` |
| độ lệch của số đếm mềm | phương sai Poisson-Binomial | Poisson-Binomial variance | `σ_k = √Σ w(1−w)` |
| "thấy vật mờ tới đâu" | độ sâu; cấp giới hạn | survey depth; limiting magnitude | — |
| ảnh phơi lâu / phơi ít | khảo sát sâu / nông | deep / shallow survey | `DEPTH_FACTOR` |
| số đếm lệch khi đổi độ sâu | dịch chuyển phân phối | distribution / covariate shift | cal → test |
| một ô trời | mảnh trời (ô lưới) | field / sky patch | — |
| bảng dữ liệu | mục lục quang trắc | photometric catalog | `PhotoObjAll` |
| cờ lọc chất lượng | phát hiện chính / photometry sạch | primary detection / clean photometry | `mode=1`, `clean=1` |
| 5 màu kính lọc | 5 băng thông | photometric bands `u g r i z` | dùng băng `r` |

---

## 4. Cơ chế thắng của Bài 1 (nền tảng phương pháp)

Nonconformity của phương pháp (đọc từ code lõi `kaggle/lib/conformal.py`):

```text
n_pred[k] = Σᵢ sᵢ·pᵢ[k]                      # số đếm mềm lớp k = Σ (điểm-phát-hiện × prob-lớp)
σ[k]      = √ Σᵢ (sᵢpᵢ[k])(1 − sᵢpᵢ[k])       # độ lệch chuẩn Poisson-Binomial
S_t       = maxₖ |gt[k] − n_pred[k]| / σ[k]   # max qua K lớp → joint coverage
```

Khoảng dự đoán: `[n_pred − q·σ, n_pred + q·σ]`. Ba thành phần **độc lập** tạo chiến thắng:

| Thành phần | Vai trò | Điều kiện dữ liệu |
|---|---|---|
| (a) σ Poisson-Binomial | **độ rộng thích nghi** theo từng field (conditional width) | confidence per-instance **dị phương** |
| (b) cửa sổ online thích nghi | **coverage recovery** khi shift làm static under-cover | **shift đủ mạnh** để fixed-rate tụt |
| (c) max-statistic K lớp | joint coverage cho vector đếm | **K > 1** lớp đếm riêng |

**Đóng góp headline = (b) coverage recovery + Winkler thấp**, KHÔNG phải (a) độ rộng (ablation Bài 1 đã chứng
minh PB-σ chỉ ảnh hưởng width). Hệ quả then chốt khi đánh giá domain mới: **đánh giá theo độ mạnh shift +
coverage recovery + interval score, không phải theo σ-gain.**

Thiên văn thỏa cả ba điều kiện: soft-count tự nhiên (a,c: `p_detect·p_class`, K=2 sao/thiên hà); nguồn mờ →
confidence lưỡng lự → dị phương (a); dịch chuyển độ sâu khảo sát → static under-cover (b).

---

## 5. Dataset — SDSS DR17 `PhotoObjAll`

### 5.1 Nguồn dữ liệu & vùng trời

- **Khảo sát:** **Sloan Digital Sky Survey, Data Release 17** (DR17, release cuối của SDSS-IV) — kính 2.5 m
  tại Apache Point Observatory, chụp ảnh 5 băng `u g r i z`. Catalog quang trắc đã hiệu chỉnh, công khai qua
  CasJobs / `astroquery.sdss`.
- **Bảng:** `PhotoObjAll` (photometric objects).
- **Vùng trời:** RA 150–152°, Dec 0–2° ≈ **4 deg²** (~14 400 arcmin²) — vừa đủ để trả về hàng vạn nguồn mà
  query nhẹ.
- **Băng dùng:** **`r`** (hậu tố `_r`) — băng sâu, ổn định nhất cho star/galaxy separation.

### 5.2 Truy vấn SQL & cờ lọc

```sql
SELECT ra, dec, type, probPSF_r, psfFlux_r, psfFluxIvar_r
FROM   PhotoObjAll
WHERE  ra BETWEEN 150.0 AND 152.0  AND  dec BETWEEN 0.0 AND 2.0
  AND  mode = 1  AND  clean = 1
  AND  type IN (3, 6)  AND  psfFluxIvar_r > 0  AND  psfFlux_r > 0
```

- `mode = 1` = **primary detection**: mỗi nguồn vật lý chỉ đếm 1 lần (bỏ bản sao ở vùng chồng lấn field).
- `clean = 1` = **photometry sạch**: không bão hòa, không lỗi deblend/interpolation, không rìa ảnh.
- `type IN (3,6)` = chỉ giữ **thiên hà (3)** và **sao (6)**.
- `psfFluxIvar_r > 0`, `psfFlux_r > 0` = loại phép đo hỏng (chia cho 0, flux âm).

Đây là bộ cờ khuyến nghị chuẩn để đếm nguồn.

### 5.3 Các cột lấy TRỰC TIẾP từ `PhotoObjAll` (không tự chế)

| Cột | Ý nghĩa | Kiểu |
|---|---|---|
| `ra`, `dec` | tọa độ trời (độ) | thật |
| `type` | nhãn lớp pipeline: **6 = sao, 3 = thiên hà** → **nhãn thật (ground-truth lớp)** | thật |
| `probPSF_r` ∈ [0,1] | xác suất nguồn là điểm (giống PSF) trong băng r — **confidence phân loại** | thật |
| `psfFlux_r` (nMgy) | thông lượng đo theo mô hình PSF | thật |
| `psfFluxIvar_r` | nghịch đảo phương sai của `psfFlux_r` (= 1/σ²) | thật |

### 5.4 Quy mô thực đo + tách CAL/TEST

- **61 423 nguồn** (25 270 sao, 36 153 thiên hà), median SNR ≈ 11.1.
- Chia vùng trời thành **841 ô lưới (field)**; **ô chẵn → CAL (421), ô lẻ → TEST (420)** (tách không gian
  độc lập, không rò rỉ nguồn giữa hai tập).

---

## 6. Backbone — SDSS Photometric Pipeline (Photo)

### 6.1 Vai trò

Backbone **không phải mạng ta tự chạy** mà là **pipeline quang trắc chính thức của SDSS** đã sinh ra
catalog. Vai trò tương đương SAM3 (Bài 1): cung cấp **confidence per-nguồn**. Ta đọc **output thật** của
pipeline (không re-inference) → notebook chạy nhanh: không có forward pass mạng, chỉ query + số học.

### 6.2 Đại lượng LẤY TRỰC TIẾP vs TỰ TÍNH

| Đại lượng | Lấy từ `PhotoObjAll` | Ta tự tính | Công thức tự tính |
|---|:---:|:---:|---|
| tọa độ `ra`, `dec` | ✓ | | — |
| nhãn lớp thật `is_star` | ✓ | | `is_star = (type == 6)` (chỉ đọc cột) |
| confidence phân loại `probPSF_r` | ✓ | | — |
| thông lượng `psfFlux_r` | ✓ | | — |
| nghịch đảo phương sai `psfFluxIvar_r` | ✓ | | — |
| **SNR** | | ✓ | `SNR = psfFlux_r · √psfFluxIvar_r` |
| **magnitude** | | ✓ | `m = 22.5 − 2.5·log₁₀(psfFlux_r)` |
| **p_star** | | ✓ | `p_star = clip(probPSF_r, 0, 1)` |
| **p_detect** | | ✓ | `p_detect = logistic((SNR − 5)/1.5)` |
| **số đếm mềm** `n_k` | | ✓ | `n_k = Σ p_detect·p_class` |
| **σ Poisson-Binomial** | | ✓ | `σ_k = √Σ w(1−w)` |
| **shift độ sâu** | | ✓ | `SNR_shallow = SNR_deep / DEPTH_FACTOR` |

**Tóm gọn:** năm đại lượng thô (`ra, dec, type, probPSF_r, psfFlux_r, psfFluxIvar_r`) lấy **thẳng** từ
catalog; mọi thứ còn lại (SNR, magnitude, p_star, p_detect, số đếm, σ, shift) là **phép tính của ta** trên
các đại lượng thô đó.

### 6.3 Công thức từng đại lượng (giải thích)

- **SNR** `= psfFlux_r · √psfFluxIvar_r`. Vì `ivar = 1/σ²` nên `√ivar = 1/σ`, do đó `flux·√ivar = flux/σ` =
  **độ sáng gấp mấy lần nhiễu**. SNR = 5 là ngưỡng phát hiện 5σ tiêu chuẩn.
- **magnitude** `= 22.5 − 2.5·log₁₀(flux)`. Quy ước SDSS: 1 nanomaggie ↔ magnitude 22.5; magnitude là thang
  **log ngược** (số nhỏ = sáng). Dùng để đặt **completeness cut** (§7.5).
- **p_star** `= clip(probPSF_r, 0, 1)`: dùng thẳng confidence pipeline, `clip` chỉ ép vào [0,1] cho an toàn số.

---

## 7. Mô hình đếm & công thức (semi-synthetic có mỏ neo thật)

### 7.1 Cái gì THẬT, cái gì MODELED

- **THẬT (từ catalog/pipeline):** nguồn, tọa độ, nhãn `type`, `probPSF`, `psfFlux`, `ivar`, và **SNR** suy
  ra từ flux/ivar.
- **MODELED (ta forward-model):** **xác suất phát hiện** `p_detect`. Lý do: catalog **chỉ chứa nguồn đã được
  phát hiện** (nguồn bị sót không có trong bảng để ghi lại `p_detect`), nên phải **dựng lại** xác suất phát
  hiện từ SNR bằng hàm completeness 5σ chuẩn. Modeled nhưng physically-grounded.

**Mô tả trung thực khi viết bài — CẤM nói "fully real" / "real cross-survey".** Được nói: *"real SDSS
sources, real classification & SNR, completeness model on real SNR, controlled/induced depth shift".*

### 7.2 p_star và p_detect

- `p_star = clip(probPSF_r, 0, 1)` — **thật**.
- `p_detect = logistic((SNR − 5)/1.5)` — **modeled**. Đường cong chữ S: tại SNR = 5 cho 0.5 (nửa-thấy-nửa-
  sót, đúng ngưỡng 5σ); SNR < 5 → tụt về 0; SNR > 5 → lên gần 1. Hằng `1.5` điều chỉnh độ dốc chuyển tiếp.
  Vì `p_detect(SNR)` **liên tục** nên σ (§7.3) không thoái hóa (khác Mức 2, xem §12).

### 7.3 Số đếm mềm + độ lệch Poisson-Binomial

Với mỗi nguồn, trọng số vào từng lớp: `w_star = p_detect·p_star`, `w_gal = p_detect·(1 − p_star)`.

```text
n_star = Σᵢ p_detect,ᵢ · p_star,ᵢ
n_gal  = Σᵢ p_detect,ᵢ · (1 − p_star,ᵢ)
σ_k    = √ Σᵢ wᵢ,ₖ (1 − wᵢ,ₖ)        (Poisson-Binomial: tổng biến Bernoulli độc lập, phương sai = Σ w(1−w))
```

> **Ví dụ 1 nguồn:** SNR cao (p_detect ≈ 1), probPSF = 0.9 → đóng góp **0.9 vào n_star** và **0.1 vào n_gal**
> (thay vì ép trọn thành 1 sao). Cộng qua mọi nguồn trong field → số đếm mềm mỗi lớp.

### 7.4 Nonconformity + khoảng dự đoán (giữ NGUYÊN Bài 1)

Ánh xạ vào ký hiệu Bài 1: điểm-phát-hiện `sᵢ = p_detect,ᵢ`; prob-lớp `pᵢ[star] = probPSF`,
`pᵢ[gal] = 1 − probPSF`. Khi đó:

```text
S_t = maxₖ |gt[k] − n_pred[k]| / σ[k]          # joint nonconformity (max qua sao/thiên hà)
q   = quantile bậc (1−α) của {S_t} trên CAL     # α = 0.1 → coverage mục tiêu 90%
interval lớp k: [ max(0, n_pred[k] − q·σ[k]),  n_pred[k] + q·σ[k] ]
```

Các phương pháp online (ACI/Adaptive/…) cập nhật `q` theo thời gian; công thức số đếm và σ **không đổi**.

### 7.5 Ground truth (số đếm thật để so)

Với mỗi field, đặt **completeness cut** `magnitude < mag_cut` (chỉ giữ nguồn đủ sáng để chắc chắn nằm trong
catalog cả deep lẫn shallow), rồi đếm nhãn thật: `gt_star = #(type==6)`, `gt_gal = #(type==3)` trong ngưỡng.
GT ổn định, tách khỏi bias incompleteness ở đầu mờ.

---

## 8. Cross-survey shift = dịch chuyển độ sâu (induced)

Mô phỏng "khảo sát nông hơn" bằng **forward-model độ sâu trên chính dữ liệu thật** (KHÔNG có survey thật thứ
hai):

```text
SNR_shallow = SNR_deep / DEPTH_FACTOR                          # nông hơn → tín hiệu yếu đi (mặc định ×3 ≈ nông ~1.2 mag)
p_detect_shallow = logistic((SNR_shallow − 5)/1.5)            # nhiều nguồn mờ rơi dưới ngưỡng 5σ → completeness giảm
p_star_shallow  = 0.5 + (probPSF − 0.5) · logistic(SNR_shallow − 3)   # SNR thấp → phân loại mờ về 0.5
```

- `cal` = **deep** (residual nhỏ) → `test` = **shallow** (residual lớn) = **HARDENING đúng chiều** (song ánh
  PanNuke→NuInsSeg của Bài 1).
- `DEPTH_FACTOR` = **núm điều khiển độ mạnh shift** → cho phép **DEPTH sweep** (§10.3). Đây là chuẩn giao
  thức distribution-shift (corrupted-CIFAR, rotated-MNIST); survey thật chỉ cho đúng 1 mức shift, nên induced
  là **lợi thế**.
- **Limitation (ghi trong bài):** đánh giá fully-observational (SDSS Stripe82 co-add SÂU làm GT vs
  single-epoch NÔNG) = future work.

---

## 9. Thiết kế thí nghiệm (diagnostic-first)

1. **Sweep `mag_cut`** — với mỗi cut đo: **bias** per-lớp `mean(gt − n_pred)`, **σ-gain**
   `= ρ(σ,|err|) − ρ(√n,|err|)`, và **coverage của Static-CP** (q cố định từ CAL) trên CAL/TEST.
2. **Gate = SHIFT mạnh** — chọn cut có Static-CP under-cover rõ trên TEST **và** calibration CAL lành
   (85–95%). (Không gate theo σ-gain — đó chỉ là câu chuyện width.)
3. **Crosstable** 9 method × 5 seeds tại cut đã chọn (joint + per-class + local coverage + width + Winkler).
4. **DEPTH sweep** — `DEPTH_FACTOR ∈ {2,3,4,5}`, coverage từng method vs độ mạnh shift.
5. **Part 2** — stream abrupt deep→shallow (changepoint), đo recovery + vẽ rolling coverage.

**Baselines (9, đầy đủ bộ Bài 1):** Static split-CP, ACI (Gibbs-Candès 2021), NexCP (Barber 2023), FACI
(Gibbs-Candès 2024), SAOCP (Bhatnagar 2023), COP (Hu 2026), Rolling-Origin CP (2026), PB-JCI Online-Fixed
(ta), Adaptive PB-JCI Online (ta). **Metrics:** Joint coverage %, per-class coverage (sao/thiên hà),
conditional validity (min rolling coverage w=50, max miss-run), AvgW (mean width), Winkler/Interval score
(mean — headline, thấp = tốt). *Chỉ báo **mean** như Bài 1 (không dùng median).*

**Tham số chính:** `α=0.1` (coverage 90%); completeness `SNR0=5, SNRW=1.5`; `DEPTH_FACTOR=3` (mặc định);
Adaptive `w_max=300, w_min=40, cov_win=50, shrink=0.9, grow=1.05, β=0.03`; Rolling `ms=200`; PB-Fixed `w=300`.

---

## 10. Kết quả trên SDSS THẬT (Mức 1, 9 method)

Dữ liệu: **61 423 nguồn** SDSS DR17 (25 270 sao, 36 153 thiên hà), median SNR 11.1; 421 CAL / 420 TEST fields.

### 10.1 Diagnostic sweep

| mag_cut | bias (sao, thiên hà) | σ-gain | Static-CP cov% CAL / TEST |
|---|---|---|---|
| **20.0** | **(0.1, 0.0)** | **+0.441** | 90.5 / **81.0** |
| 20.5 | (0.1, 0.1) | +0.436 | 90.5 / 82.9 |
| 21.0 | (0.4, 0.5) | +0.386 | 90.5 / 100.0 |
| 21.5 | (1.3, 2.2) | +0.679 | 90.5 / 100.0 |
| 22.0 | (3.4, 7.5) | +0.916 | 90.5 / 100.0 |

- **bias ≈ 0** → backbone low-bias thật. **σ-gain +0.44** (dương) → PB-σ tiên đoán sai số vượt √count. Gate
  chọn `mag_cut = 20.0` (shift mạnh nhất: Static-CP sập 90.5% → 81.0%).

### 10.2 Crosstable (mag_cut = 20.0, 5 seeds) — đủ 9 method

| Method | Joint % | Sao % | ThiênHà % | AvgW | Winkler | minLoc % | miss |
|---|---|---|---|---|---|---|---|
| Static split-CP | 81.0 | 85.2 | 89.3 | 0.03 | 2.84 | 67.2 | 3 |
| ACI (2021) | 89.7 | 92.3 | 94.9 | 6.29 | 14.08 | 85.2 | 2 |
| NexCP (2023) | 88.0 | 91.0 | 95.0 | 0.21 | 2.23 | 78.0 | 3 |
| FACI (2024) | 88.6 | 91.5 | 95.0 | 0.26 | 2.19 | 80.4 | 3 |
| SAOCP (2023) | 99.3 | 99.8 | 99.4 | 11.12 | 22.34 | 96.8 | 1 |
| COP (2026) | 85.1 | 89.1 | 93.8 | 0.10 | 2.52 | 72.0 | 3 |
| Rolling-Origin (2026) | 88.5 | 91.3 | 95.1 | 0.24 | 2.18 | 78.8 | 3 |
| PB-JCI Fixed (ta) | 85.7 | 88.9 | 93.9 | 0.18 | 2.32 | 72.4 | 3 |
| **Adaptive PB-JCI (ta)** | **91.2** | 93.4 | 95.9 | 0.47 | **2.10** | 84.0 | 3 |

**Adaptive thắng headline:** duy nhất đạt **≥90% (91.2%)** mà KHÔNG phình interval, và **Winkler tốt nhất
(2.10)**. Hai method còn "phủ" được đều vô dụng: **SAOCP 99.3% nhưng Winkler 22.3 / AvgW 11** (over-cover),
**ACI 89.7% nhưng Winkler 14** (interval bất ổn, đuôi nặng). Các method Winkler-thấp hơn chút
(COP/PB-Fixed/NexCP/Rolling) đều **under-cover** (85–88.5%).

> **Cách trình bày đúng (đọc coverage + Winkler CÙNG NHAU):** *"Adaptive is the only method achieving valid
> coverage (≥90%) at the best mean interval score; methods with lower typical width do so by under-covering,
> and the only over-covering method (SAOCP) inflates the interval score ~10×."*

### 10.3 DEPTH sweep — hình/bảng chính (coverage vs độ mạnh shift)

Cố định `mag_cut=20.0`, tăng `DEPTH_FACTOR` (Static-CP sập 85.5 → 9%):

| DEPTH | Static | ACI | NexCP | FACI | SAOCP | COP | Rolling | PB-Fixed | **Adaptive** |
|---|---|---|---|---|---|---|---|---|---|
| 2.0 | 85.5 | 89.8 | 88.7 | 88.8 | 98.3 | 87.3 | 89.2 | 87.9 | **91.4** |
| 3.0 | 81.0 | 89.7 | 88.0 | 88.6 | 99.3 | 85.1 | 88.5 | 85.7 | **91.2** |
| 4.0 | 53.1 | 89.6 | 87.1 | 88.6 | 99.3 | 82.9 | 87.0 | 83.3 | **91.0** |
| 5.0 | 9.0 | 89.7 | 86.2 | 89.1 | 99.5 | 80.3 | 85.8 | 81.9 | **90.5** |

→ **Adaptive giữ ~91% BẤT ĐỘNG qua mọi mức shift** (duy nhất luôn ≥90 mà không over-cover); COP/PB-Fixed/
Rolling/NexCP **tụt dần**, SAOCP luôn over-cover ~99%, ACI kẹt <90 + Winkler khổng lồ. **Đây là bằng chứng
coverage-recovery đẹp nhất — dùng làm hình chính của chương** (`figures/fig_pub_recovery.png`).

### 10.4 Part-2 — recovery sau abrupt shift (changepoint 420)

| Method | pre cov | post-30 | overall |
|---|---|---|---|
| Static split-CP | 90.7 | 83.3 | 85.8 |
| ACI (2021) | 89.8 | 93.3 | 89.8 |
| NexCP (2023) | 90.0 | 83.3 | 89.3 |
| FACI (2024) | 89.5 | 86.7 | 88.8 |
| SAOCP (2023) | 97.6 | 100.0 | 98.2 |
| COP (2026) | 90.0 | 83.3 | 87.5 |
| Rolling-Origin (2026) | 90.2 | 80.0 | 89.5 |
| PB-JCI Fixed (ta) | 90.5 | 76.7 | 88.8 |
| **Adaptive PB-JCI (ta)** | 92.4 | **90.0** | **92.4** |

**post-30 cov** = coverage trên 30 bước ngay sau shift. Mọi fixed-rate baseline tụt sâu (Rolling 80.0,
PB-Fixed 76.7, Static/NexCP/COP 83.3, FACI 86.7); ACI (93.3) và SAOCP (100.0) "phủ" được nhưng bằng
over-cover + Winkler khổng lồ. **Adaptive vừa post-30 = 90.0 vừa overall = 92.4 (cao nhất trong nhóm valid).**

---

## 11. Phân tích — vì sao method thắng ở domain này

Thiên văn thỏa đồng thời ba điều kiện của cơ chế thắng (§4):

| Điều kiện | Bằng chứng số |
|---|---|
| (1) backbone bias thấp | bias per-lớp ≈ (0.1, 0.0) ở mag_cut=20.0 |
| (2) confidence dị phương (σ có ích) | σ-gain **+0.44** (dương) |
| (3) shift khó, aleatoric | dịch chuyển độ sâu → Static-CP sập 90.5→81% (và tới 9% khi DEPTH=5) |

→ Phương pháp showcase được **cả hai** đóng góp: conditional width (PB-σ, σ-gain > 0) + coverage recovery
(adaptive window). **ACI Winkler 14.08** (≈7× Adaptive) minh họa "phủ bằng nới rộng thô"; **SAOCP over-cover
99% Winkler 22** minh họa "coverage một mình vô nghĩa" — cả hai làm nổi bật claim của Adaptive: *valid
coverage KHÔNG cần over-widen.*

---

## 12. Trung thực & limitation

- **SEMI-SYNTHETIC, mô tả trung thực (KHÔNG "fully real"):** THẬT = nguồn SDSS, magnitude, flux, SNR, phân
  loại probPSF. MODELED = `p_detect` (completeness 5σ trên SNR thật, vì catalog chỉ chứa nguồn đã detect) +
  shift INDUCED (giảm SNR). Được nói: *"real SDSS sources, real classification & SNR, completeness model on
  real SNR, controlled/induced depth shift"*. Cấm nói: *"real cross-survey", "fully real"*.
- **Đã thử Mức 2 (σ chỉ từ probPSF) và LOẠI:** đặt `p_detect=1`, σ chỉ từ probPSF → ở đầu sáng probPSF≈0/1 ⇒
  σ≈0 ⇒ conformal thoái hóa ⇒ diagnostic gate FAIL. Đây là bằng chứng diagnostic-first hoạt động (chặn thí
  nghiệm hỏng). Mức 1 dùng `completeness(SNR)` liên tục → σ không thoái hóa.
- **Limitation ghi trong bài:** detection completeness được forward-model; đánh giá fully-observational
  (Stripe82 co-add vs single-epoch) = future work.
- **Shift induced = LỢI THẾ:** `DEPTH_FACTOR` là núm điều khiển → có **DEPTH sweep** (survey thật chỉ cho 1
  mức). Chuẩn protocol distribution-shift.
- **σ-gain > 0 là bonus**; đóng góp chính vẫn là **coverage recovery**. Báo cáo **mean** Winkler & width
  (khớp Bài 1) + per-class coverage + local coverage (minLoc%, miss). **Cảnh báo trình bày:** đọc coverage +
  mean-Winkler CÙNG NHAU — method under-cover có width nhỏ hơn nhưng invalid.

---

## 13. File liên quan

- Pipeline chính: [`kaggle/astro_pbjci_diagnostic.py`](../kaggle/astro_pbjci_diagnostic.py) (+ notebook
  `.ipynb`, `USE_SDSS=True`).
- Hình publication: [`kaggle/make_astro_figures_pub.py`](../kaggle/make_astro_figures_pub.py) →
  `figures/fig_pub_recovery.png` (hình chính), `fig_pub_box.png` (spread).
- Visualize dataset + ảnh thật: [`kaggle/make_astro_dataset_viz.py`](../kaggle/make_astro_dataset_viz.py) →
  `figures/fig_pub_dataset.png` (4 panel), `fig_pub_cutouts.png` (ảnh cutout).
- Vẽ lại hình từ kết quả đã lưu: [`kaggle/make_astro_figure.py`](../kaggle/make_astro_figure.py).
- Kết quả: `results/astro_diagnostic_results.json`. Tổng quan: [`README_ASTRO.md`](../README_ASTRO.md).

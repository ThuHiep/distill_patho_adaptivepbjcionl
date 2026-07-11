# Vì sao kết quả kém — phân tích chi tiết & thiết kế lại (method, KHÔNG chứng minh)

> Cập nhật 2026-07-11. Sau 3 vòng PBUD/CCAD âm tính + test Mondrian âm tính. File này: (1) mổ xẻ ĐÚNG
> nguyên nhân kỹ thuật trong code hiện tại, (2) vì sao các bài KD+uncertainty khác kết hợp được, (3) đặc
> thù bài này cần gì, (4) thiết kế lại theo hướng METHOD (có cửa thắng), không phải chứng minh.

---

## 1. Nguyên nhân gốc: TRAIN và EVAL đang đo hai thứ KHÁC NHAU (misalignment ở mọi tầng)

Đây là lỗi lớn nhất, xuyên suốt. Loss huấn luyện một đại lượng, còn conformal eval đo một đại lượng
khác — nên tối ưu loss không kéo được metric.

| Tầng | Lúc TRAIN (loss tác động) | Lúc EVAL/INFER (đo) | Hệ quả |
|---|---|---|---|
| **Instance** | existence pool trên **teacher masks** (`student_instance_scores(prob, d["label"])`, `label`=teacher) | student tự sinh instance bằng **connected components của chính nó** (`student_predict`) | student được dạy "chấm điểm ô của teacher", nhưng lúc dùng lại tự cắt ô khác → tín hiệu học lệch hoàn toàn |
| **σ (độ bất định)** | σ_S = PB var **trên teacher masks**, khớp σ_T | σ dùng cho conformal = PB var **trên instance của student** | distill σ vào **sai cặp** → không ảnh hưởng σ mà conformal thực sự dùng |
| **Interval / coverage** | CCAD Winkler ở **k cố định = 1.64** (`soft_winkler_loss`) | conformal dùng **q recalibrate (~2–3)**, khác k | tối ưu khoảng train ≠ khoảng eval → không kéo được coverage eval |
| **Count** | không có số hạng ép **count suy luận** = GT | count = Σ (mean prob mỗi component) | không gradient nào đẩy count thật về đúng ở organ khó |

→ **Kết luận tầng này:** dù loss "đúng ý tưởng", nó **không chạm vào** đại lượng mà eval chấm. Đó là lý do
PBUD/CCAD ≈ KD, và tăng trọng số chỉ làm nhiễu (v3 tệ hơn).

---

## 2. Count estimator hiện tại vốn đã yếu (gây bias per-organ)

`student_predict`: `prob = sigmoid(unet(img))` → ngưỡng → `ndimage.label` (connected components) →
`s_i = mean(prob trong component)` → count = Σ s_i.

Ba vấn đề cụ thể:
1. **Count = Σ (mean prob mỗi cụm) ≈ (số cụm) × (độ tự tin trung bình)** — mỗi nhân đếm thành <1 →
   **under-count hệ thống**.
2. **Connected components KHÔNG tách được nhân dính nhau** — ở mô dày đặc (mouse spleen, human kidney)
   nhiều nhân dính → 1 cụm → under-count mạnh. Teacher (PathoSAM AIS) tách được; student thì không.
   **Đây chính là nguồn bias per-organ** làm conditional coverage sụp.
3. **Ngưỡng + label không khả vi** → không có gradient sửa bước tạo instance.

→ Student **kém teacher đúng ở chỗ khó nhất** (tách nhân mô dày), và distill **không sửa được** vì
teacher-signal là foreground map, không dạy cách tách.

---

## 3. Vì sao các bài KD + uncertainty/calibration KHÁC họ kết hợp ĐƯỢC

Điểm chung của các bài thành công: **độ bất định là output hạng nhất, được giám sát TRỰC TIẾP, và
đại lượng distill = đại lượng đánh giá (aligned).**

| Bài | Họ distill/eval cái gì | Vì sao work |
|---|---|---|
| Calibration Transfer via KD (ACCV 2024) — classification | distill **softmax teacher**, eval **calibration của softmax student** | distilled ≡ evaluated (cùng là softmax). Aligned tuyệt đối |
| Uncertainty-aware distillation (regression) | student có **head σ tường minh**, distill σ_T + train **NLL vs residual thật** | σ là output trực tiếp, học từ lỗi thật → calibrated |
| Deep Double Poisson Networks (count) | mạng xuất **tham số phân phối đếm**, train bằng **NLL** | uncertainty khớp độ phân tán quan sát, đo đúng cái train |

**Điểm chết của mình so với họ:** mình **chế** uncertainty *sau* (conformal trên PB-từ-instance), và
**không bao giờ giám sát trực tiếp** cái uncertainty mà eval đo. Họ để uncertainty là output hạng nhất
+ train/eval aligned. **Đó là khác biệt quyết định.**

---

## 4. Đặc thù bài này cần gì (rút ra từ mục 1–3)

1. **Align train ↔ eval:** cái được tối ưu phải ĐÚNG là cái được chấm. Nếu eval là (μ, σ) → phải train
   trực tiếp (μ, σ).
2. **σ là output HỌC ĐƯỢC, hạng nhất**, huấn luyện bằng **proper scoring rule (NLL/Winkler thật)** vs GT
   count → σ tự lớn ở organ khó (heteroscedastic), tự nhỏ ở organ dễ. **Đây là cách "nới khoảng đúng chỗ"
   mà CCAD tay-thiết-kế thất bại.** NLL phạt cả over- và under-confidence → không degenerate.
3. **Count estimator bền:** bỏ threshold+CC; dùng **density-sum** (chuẩn của Khan/crowd counting) hoặc
   instance head tách được nhân → giảm bias per-organ.
4. **Chấp nhận giới hạn lý thuyết:** conditional coverage HOÀN HẢO với ít mẫu/nhóm là **bất khả** (Vovk,
   Barber "limits of conditional inference"). Mục tiêu đúng = **"tốt hơn baseline theo Winkler"**, không
   phải "worst-org = 0.90". Winkler (phạt cả coverage lẫn width) mới là metric method thắng được.
5. **Conformal về đúng vai:** chỉ là lớp bọc mỏng recalibrate CUỐI trên (μ, σ) đã calibrated — không còn
   là nơi "sinh ra" uncertainty.

---

## 5. Code hiện tại có vấn đề gì (liệt kê cụ thể để sửa)

- `distill_student_pbud.student_instance_scores` dùng **teacher label** → train theo teacher proposals,
  nhưng `student_predict` dùng **CC của student** → **mismatch instance** (mục 1).
- `pbud_losses.soft_winkler_loss` dùng **k=1.64 cố định** ≠ q recalibrate của eval → **mismatch interval**.
- Không có số hạng **NLL / GT-count trực tiếp** trên count suy luận → σ và μ không được calibrate theo lỗi
  thật.
- `student_predict`: count = Σ mean-prob, threshold+CC → **bias + không tách nhân** (mục 2).
- σ_S/σ_T so trên teacher masks, không phải σ mà conformal dùng → **distill sai cặp**.

→ Nói thẳng: **không phải "loss chưa đủ tốt", mà là cả pipeline train một bài toán khác với bài eval.**
Sửa loss lẻ (PBUD/CCAD) không cứu được vì gốc là misalignment kiến trúc.

---

## 6. THIẾT KẾ LẠI (method, có cửa thắng — không chứng minh)

### Ý tưởng: Distributional Count Distillation — student xuất TRỰC TIẾP phân phối đếm (μ, σ)

Bỏ chuỗi "instance → PB → conformal-sinh-σ". Thay bằng: **student là bộ hồi quy phân phối đếm nhẹ**, xuất
`(μ, log σ)` cho mỗi ảnh, huấn luyện để (μ, σ) tự calibrated. Conformal chỉ recalibrate mỏng ở cuối.

**Kiến trúc student (nhẹ, edge):**
- Backbone nhẹ → **density head** (μ = Σ density map, bền, tách-nhân tốt hơn CC) **+ log-σ head** (một
  vô hướng/ảnh: độ bất định đếm, heteroscedastic).

**Loss (mọi thứ ALIGNED với eval):**
```
L = L_density   : MSE(density_S, density_teacher)            # KD: học đếm từ teacher (mean)
  + L_count     : |Σdensity_S − GT|                           # ép count suy luận đúng (trực tiếp!)
  + L_nll       : Gaussian/Poisson NLL(GT | μ_S, σ_S)         # ★ học σ CALIBRATED (lớn ở organ khó)
  + λ·L_distill_sigma : KD σ từ teacher nếu có                # tùy chọn
```
- **L_nll là chìa khóa:** ép σ khớp lỗi thật per-image → organ khó (kidney/spleen) tự có σ lớn → khoảng
  rộng **đúng chỗ đó thôi** → conditional coverage tăng mà width tổng không nổ. NLL phạt cả 2 phía → không
  degenerate như CCAD.

**Eval:** (μ, σ) → conformal recalibrate mỏng (q trên score `|GT−μ|/σ`) → coverage/Winkler. Giờ
**train và eval cùng đo (μ, σ)** → aligned.

### Vì sao cái này có cửa thắng nơi PBUD/CCAD thua
- **Aligned train↔eval** (sửa mục 1).
- **σ học trực tiếp từ lỗi thật** (giống các bài thành công ở mục 3), không phải chế post-hoc.
- **Density-sum bền hơn CC** (sửa mục 2).
- **Nới khoảng đúng chỗ qua NLL** thay vì CCAD tay-thiết-kế (sửa cái đã thất bại).
- **Metric mục tiêu = Winkler** (khả thi), không đòi conditional coverage hoàn hảo (bất khả).

### Novelty (thành thật, không tô hồng)
- "Heteroscedastic count + NLL" đã có (Deep Double Poisson). "Distill teacher variance" đã có
  (uncertainty-aware KD). **Cái mới = tổ hợp:** distill phân phối đếm từ **foundation model (SAM3/
  PathoSAM)** sang student nhẹ, **giữ reliability (Winkler/coverage) khi nén**, cho **đếm tế bào**. Vẫn là
  novelty tổ hợp — nhưng lần này **có cơ chế đúng để THẮNG baseline**, khác 3 vòng vừa rồi (không thắng nổi).
- Cần verify Scholar: "distributional distillation counting", "heteroscedastic knowledge distillation
  regression uncertainty".

### Biến thể R1 (giữ khung PB của paper 1, nếu muốn mạch luận văn liền)
Nếu muốn giữ per-instance PB (để nối paper 1): student phải **tách instance ĐÚNG như teacher** (distill
instance qua distance-map/boundary kiểu CellGenNet), để {s_i} lúc infer khớp lúc train → PB σ aligned.
Khó hơn (tách nhân dính) nhưng giữ khung. **Khuyến nghị: thử R2 (distributional) trước** vì aligned dễ đạt
và có cửa thắng cao hơn; R1 để sau nếu cần mạch PB.

---

## 7. Việc tiếp (nếu chọn thiết kế lại)
- [ ] Verify Scholar novelty cụm ở trên.
- [ ] Implement student (density + log-σ head) + loss (density KD + count + NLL). Test khả vi ở local.
- [ ] Cache teacher density map (đã có foreground map — dùng làm density target hoặc build density từ AIS).
- [ ] Train KD-density (baseline) vs Distributional (ours) → eval Winkler/coverage. **Cổng:** ours phải
      ≤ Winkler KD **và** worst-org ≥ KD. Nếu đạt → có method win thật.
- [ ] Nếu đạt: nén sweep + đa dataset (MoNuSAC/PanNuke K>1).

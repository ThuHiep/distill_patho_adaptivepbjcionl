# Paper 2 — Ý tưởng: Model mới + Loss mới cho distillation đếm tế bào giữ độ tin cậy

> Cập nhật: 2026-07-10. Đây là bài **METHOD** (tạo model mới + loss mới), KHÔNG phải bài chứng minh như
> paper 1. File này gồm: (1) ý tưởng, (2) model mới, (3) loss mới, (4) **đánh giá novelty KHÔNG tô hồng**
> — có khảo sát trực tiếp các bài va chạm, (5) đo bằng gì, (6) khác paper 1 chỗ nào, (7) phán đoán Q1 thẳng.
>
> Nguyên tắc: không bịa, **không tô hồng**. Chỗ nào ý tưởng KHÔNG mới thì ghi rõ là không mới.

---

## 1. Ý tưởng một câu

> **Nén một foundation model đếm (SAM3/PathoSAM) thành student nhẹ để chạy edge, nhưng giữ được độ tin
> cậy của khoảng dự đoán — bằng một loss distillation truyền cả CẤU TRÚC BẤT ĐỊNH per-instance
> (Poisson-Binomial), không chỉ số đếm.**

**Mạch:** foundation model đếm rất chính xác nhưng quá nặng cho bệnh viện/edge → phải distill. KD chuẩn
(Khan 2025, CellGenNet 2025) chỉ khớp **số đếm** (mean) → student mất **cấu trúc phương sai σ** → khoảng
tin cậy của student không còn đáng tin, đặc biệt **conditional coverage** (theo loại mô) sụp. Loss mới ép
student giữ σ → student vừa nhẹ, vừa chính xác, vừa giữ coverage.

---

## 2. Model mới (artifact chính — cái paper 1 không có)

**Student đếm tế bào nhẹ, có cấu trúc bất định per-instance khả vi.** Khác student của Khan/CellGenNet ở
chỗ: nó không chỉ xuất density/foreground map, mà xuất **per-instance existence prob sᵢ + class prob pᵢ**
để tái tạo được phân phối đếm Poisson-Binomial của teacher. Đây là model mới (student + thiết kế head),
paper 1 hoàn toàn không có student/không nén.

Thiết kế head khả vi (để loss ở mục 3 tính được σ khả vi lúc train) — hai đường:
- **(khuyến nghị, sạch):** giữ instance proposals nhẹ, student học **per-instance score head + type head**
  khớp trực tiếp (sᵢ, pᵢ) của teacher → σ khả vi ngay. Nén = nén backbone/heads.
- **(khó):** student density-based + xấp xỉ PB khả vi.

---

## 3. Loss mới

Count Poisson-Binomial: `N_k = Σᵢ sᵢ·p_{i,k}`, mean `μ_k = Σᵢ sᵢp_{i,k}`,
variance `σ²_k = Σᵢ (sᵢp_{i,k})(1 − sᵢp_{i,k})` (đúng `pb_variance()` trong `conformal.py`).

**Phiên bản tối thiểu — PBUD (Poisson-Binomial Uncertainty Distillation):**
```
L = α·L_task(N^S, GT)              ← nhãn thật
  + β·Σ_k (μ_k^S − μ_k^T)²         ← distill MEAN (Khan/CellGenNet đã có)
  + γ·Σ_k (σ_k^S − σ_k^T)²         ← distill VARIANCE (PB structure)
```
γ là số hạng thêm. Có nguyên lý: conformal dùng score chuẩn hoá `|N_k−μ_k|/σ_k`; student overconfident
(σ^S<σ^T) → score phồng → khoảng quá hẹp → under-coverage. Khớp σ là target suy từ toán.

**⚠️ Nhưng PBUD một mình KHÔNG đủ mới (xem mục 4).** Phiên bản mạnh hơn, đáng theo:

**CCAD (Conditional-Coverage-Aware Distillation) — bản khuyến nghị:**
Thay vì chỉ khớp **marginal** σ (thứ uncertainty-aware distillation đã làm), loss này ép student cân bằng
**phân phối nonconformity score theo NHÓM (organ/loại mô)** để giữ **conditional coverage** — đúng cái
conformal KHÔNG bảo đảm và KD chuẩn phá. Cụ thể: minimize độ phân tán của coverage/score chuẩn hoá giữa
các subgroup của student, dùng cấu trúc PB. Đây là target **chưa được giải** bởi dòng uncertainty-aware
distillation (họ khớp marginal moment, không đụng conditional coverage).

---

## 4. ĐÁNH GIÁ NOVELTY — KHÔNG TÔ HỒNG

Đã khảo sát trực tiếp (không đoán). Kết quả thẳng:

### 4a. Những gì KHÔNG mới — phải thừa nhận & trích, KHÔNG được nhận vơ
| Thành phần ý tưởng | Đã có ai làm | Hệ quả |
|---|---|---|
| **"Distill variance/uncertainty của teacher, không chỉ mean"** | ⚠️ **ĐÃ CÓ, có tên hẳn hoi: "uncertainty-aware distillation" / "explicit uncertainty transfer"** — literature nói thẳng: *"distills not just the teacher's mean prediction, but also its predictive variance, diversity, or higher-order moments"*. Có bài dense-prediction: **Uncertainty-Aware and Decoupled Distillation for Semantic Segmentation** (IJCV 2025) | ❌ **KHÔNG được** nói "distill variance là mới". Đây là đòn nặng nhất vào novelty của PBUD |
| Moment-matching giữa teacher–student | ĐÃ CÓ: Adversarial Moment-Matching Distillation (NeurIPS 2024); KD²M (MMD/Wasserstein match distribution) | Khớp moment teacher–student là kỹ thuật KD đã biết |
| Calibration-preserving KD | ĐÃ CÓ, đang nóng: Role of Teacher Calibration in KD (2025); Calibration Transfer via KD (ACCV 2024); Trust the Uncertain Teacher (2026) — **đều classification** | Ý "giữ calibration khi distill" không mới ở classification |
| Mô hình hoá phương sai count | ĐÃ CÓ: Deep Double Poisson Networks (2024); UNIC (counting uncertainty); DUMLO/Trihorn (Sci Rep 2025, OT chống nhiễu nhãn — **single-model, không distill**) | "Đo uncertainty của count" không mới |
| **Metric conditional coverage mới** | ⚠️ **ĐÔNG ĐÚC 2025–2026:** ERT/Conditional Coverage Diagnostics (2512.11779); CVP/Conformal Prediction Assessment (2603.27189); CReL/Conformal Reliability (2605.30807) | ❌ **KHÔNG nên** đề xuất metric conditional-coverage mới — sẽ trùng. **Dùng lại** metric của họ |
| Nén × conformal | ĐÃ CÓ: Pruning CNNs for inductive conformal (Neurocomputing 2024) — thấy conformal **khá bền** với pruning (classification) | Phải trích & phân biệt; và cảnh báo: nén có thể "lành", gap không rộng như kỳ vọng |

### 4b. Phần CÒN LẠI có thể mới (hẹp — phải verify Scholar trước khi chốt)
Chỉ còn **tổ hợp cụ thể** này chưa thấy ai làm (trong phạm vi search):
1. Cấu trúc **Poisson-Binomial per-instance** (σ từ existence × classification của **foundation
   segmentation model**) làm mục tiêu distillation — chưa thấy trong distillation đếm.
2. **CCAD**: loss distillation nhắm **conditional coverage** (không phải marginal moment) — dòng
   uncertainty-aware distillation chưa đụng conditional coverage; đây là chỗ sạch nhất.
3. Ghép: nén foundation-model đếm tế bào + giữ conditional coverage, đo trên pathology.

### 4c. Kết luận novelty (thẳng, không tô hồng)
- **PBUD (chỉ distill σ) là NOVELTY YẾU.** Về bản chất nó là "uncertainty-aware distillation chuyên biệt
  hoá cho count PB". Reviewer Q1-đỉnh (MIA/TMI) rất dễ gọi là *incremental* ("moment-matching KD applied
  to counting"). **Không nên** bán PBUD như loss đột phá.
- **CCAD (nhắm conditional coverage) MỚI HƠN**, vì nó tấn công đại lượng mà (i) conformal không bảo đảm,
  (ii) uncertainty-aware distillation không nhắm, (iii) tôi đã đo là đang vỡ thật. Nhưng nó vẫn là **tổ
  hợp** (conformal-aware training × distillation × counting) — không phải một nguyên thủy hoàn toàn mới.
- **Model mới (student nhẹ đáng tin) là artifact thật**, phòng thủ tốt hơn phần loss.
- **Metric: KHÔNG tự nhận metric mới.** Dùng ERT/CVP có sẵn + có thể đề xuất **giao thức đo** "coverage
  degradation dưới nén" (protocol, không phải metric mới).

---

## 5. Đo bằng gì (không tự chế metric)
- Accuracy: MAE, RMSE.
- Calibration: marginal coverage, **conditional coverage** (dùng metric có sẵn: ERT / CVP; per-organ),
  Winkler, width. (Instrument `eval_coverage_transfer.py` đã đo organ-wise coverage — dùng làm evaluation.)
- Efficiency: params, FLOPs, inference time.
- So sánh cốt lõi: student **KD chuẩn** vs **PBUD** vs **CCAD** vs student-from-scratch → xem loss mới có
  giữ conditional/transfer coverage nơi KD chuẩn vỡ không.

---

## 6. Khác paper 1 chỗ nào (để không trùng)
| | Paper 1 (đã nộp) | Paper 2 (bài này) |
|---|---|---|
| Trục | đổi **dữ liệu** (distribution shift) | đổi **model** (nén/distill) |
| Artifact | conformal online adaptive | **student nhẹ + loss distillation mới** |
| Conformal | là **đóng góp** | chỉ là **thước đo** để đánh giá student |
| Có student/nén? | Không | **Có** |
→ Conformal-counting của paper 1 ở đây chỉ dùng để **đo**, không lặp lại làm đóng góp → không trùng.

---

## 7. Phán đoán Q1 (thẳng, không tô hồng)
- **Q1 "vững" (Sci Rep — nơi Khan, CellGenNet, DUMLO đều đăng):** khả thi, **với điều kiện** loss mới thật
  sự giữ coverage tốt hơn KD chuẩn (chưa chứng minh — phải chạy).
- **Q1 "đỉnh" (MIA/TMI/TIP):** **rủi ro cao.** Vì (i) "distill uncertainty" không mới, (ii) metric
  conditional-coverage đã đông. Muốn có cửa: phải làm **CCAD** thành đóng góp rõ (không chỉ PBUD), có
  kết quả mạnh (đặc biệt tương tác nén×shift), và có thể cần một mảnh lý thuyết (bound coverage-vs-nén).
- **Thành thật:** đây là novelty **"tổ hợp + chuyên biệt hoá + artifact mới"**, KHÔNG phải nguyên thủy
  đột phá. Đủ cho Q1-vững nếu số đẹp; Q1-đỉnh thì phải gồng thêm và không chắc.

**Rủi ro lớn nhất:** (1) nén hoá ra "lành" (pruning-conformal 2024 đã thấy vậy ở classification) → loss
mới không tạo khác biệt đo được → bài yếu. (2) Reviewer gọi PBUD là uncertainty-aware-KD đội lốt. Phòng
thủ: dồn trọng số vào **CCAD + conditional coverage + nén×shift**, không dồn vào PBUD.

---

## 7b. Bỏ lý thuyết → bù bằng thực nghiệm + CỔNG effect-size (đã lọc từ nhận xét ngoài)

Quyết định: **không đuổi theo theorem/bound** (chấp nhận bỏ cửa "Q1-đỉnh bằng lý thuyết"). Thay vào đó
làm **strong empirical method paper**. Nhưng CCAD vẫn là **ngôi sao** — KHÔNG lùi về "systematic study"
làm đóng góp chính (đó là hướng proving đã bác). Study chỉ là bằng chứng hỗ trợ cho method.

**Kế hoạch thực nghiệm bù (áp dụng phần đúng của nhận xét):**
- **Compression sweep:** teacher → student {1×,2×,4×,8×,16×} nén. CCAD ổn định hơn KD chuẩn ở MỌI mức nén = thuyết phục.
- **Multi-domain:** train tissue này test tissue khác (PanNuke có tissue types; NuInsSeg có 31 organ). Thắng 1 dataset = reviewer nghi overfit.
- **Nén × shift (mạnh nhất):** lưới {KD, CCAD} × {ID, OOD}. Nếu KD giữ được ở ID nhưng vỡ ở OOD, CCAD giữ được cả hai → câu chuyện đẹp.

**⚠️ SỬA LỖI trong nhận xét (marginal vs conditional):** nhận xét vẽ "Teacher 94 / KD 80 / CCAD 92" cho
coverage — **SAI nếu đó là marginal coverage đã recalibrate**. Split conformal bảo đảm marginal ~target
theo cấu tạo cho MỌI student → recalibrate student KD sẽ cho ~90%, KHÔNG phải 80%. "Coverage collapse"
CHỈ đo được ở:
  - **Conditional coverage** (worst-organ): vd 90%→70%, recalibrate KHÔNG cứu → chỗ thật.
  - **Transfer coverage** (áp q teacher lên student, không recalibrate — edge không nhãn) → chỗ thật.
→ Cổng effect-size dưới đây phải đo **conditional + transfer**, KHÔNG phải marginal-sau-recalibrate,
nếu không sẽ thấy đường phẳng ~90% và kết luận nhầm "bài chết".

**CỔNG QUYẾT ĐỊNH (chạy TRƯỚC khi viết gì):** dùng `eval_coverage_transfer.py` (đã đo organ-wise) trên
student KD chuẩn:
  - Nếu conditional/transfer coverage **KHÔNG tụt đáng kể** so teacher → hiện tượng yếu → **CCAD không có
    gì để sửa** → dừng hoặc đổi hướng. (effect nhỏ thì thêm theorem cũng không cứu — điểm này nhận xét đúng.)
  - Nếu tụt rõ (vd worst-organ ≥10 điểm, hoặc transfer marginal tụt rõ) → có động lực → làm CCAD, đo lại,
    chứng minh CCAD khôi phục phần lớn.

## 8. Việc tiếp
- [ ] **Verify novelty trên Google Scholar / Semantic Scholar** cụm: "uncertainty aware distillation
      counting", "conditional coverage distillation", "calibration preserving distillation regression" —
      TRƯỚC khi viết đề cương. (Search engine của tôi có thể sót.)
- [ ] Implement PBUD + CCAD vào trainer (cần đổi student head sang xuất per-instance sᵢ,pᵢ khả vi).
- [ ] Chạy so sánh KD chuẩn vs PBUD vs CCAD → đo bằng `eval_coverage_transfer.py` (đã có, đo organ-wise).
- [ ] Nếu CCAD thắng rõ → viết method paper; nếu không → xem lại (đừng cố ép).

## Nguồn va chạm gần (đã đọc/tra — kiểm lại trước khi nộp)
- Uncertainty-aware / explicit uncertainty transfer distillation (khái niệm đã có) — How Is Uncertainty
  Propagated in KD (arXiv 2601.18909, Duke 2026, đã đọc: KHÔNG counting/conformal); Uncertainty-Aware &
  Decoupled Distillation for Semantic Segmentation (IJCV 2025, s11263-025-02585-2)
- Moment-matching distillation — Adversarial Moment-Matching Distillation (NeurIPS 2024, arXiv 2406.02959)
- Calibration-preserving KD — Role of Teacher Calibration in KD (arXiv 2508.20224); Calibration Transfer
  via KD (ACCV 2024); Trust the Uncertain Teacher (arXiv 2602.12687) — tất cả classification
- Count uncertainty — Deep Double Poisson Networks (arXiv 2406.09262); DUMLO/Trihorn (Sci Rep 2025,
  s41598-025-14056-2, đã đọc: single-model OT chống nhiễu nhãn, KHÔNG distill/nén/conformal)
- Conditional-coverage metrics (dùng lại, đừng tự chế) — ERT (arXiv 2512.11779); CVP (arXiv 2603.27189);
  CReL (arXiv 2605.30807)
- Nén × conformal — Pruning CNNs for inductive conformal prediction (Neurocomputing 2024)
- KD đếm nền — Khan et al. 2025 (s41598-025-90750-5); CellGenNet (arXiv 2511.15054)

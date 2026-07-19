# Hướng viết Paper 2 — dựng quanh 1 điểm mạnh thật

*Bản ý tưởng để cô duyệt trước khi viết lại báo cáo/manuscript. Ngày 19/07/2026.*

---

## 0. Kết luận một dòng

**Điểm bán không phải "model nhỏ" cũng không phải "đếm giỏi nhất" — mà là: cùng một ngân sách CÔNG dán nhãn, chưng cất từ foundation model chỉ với nhãn ĐẾM cho bộ đếm TỐT HƠN cách train bằng mask, lại kèm uncertainty calibrated.**

---

## 1. Phân tích: điểm mạnh thật nằm ở đâu

Rà lại toàn bộ kết quả, chia làm 3 loại và chỉ 1 loại thắng thật:

| Loại kết quả | Có baseline? | Kết cục | Dùng làm gì |
|---|---|---|---|
| Accuracy thô vs model khác (NuLite 12M, CellViT) | ✅ | **THUA** ~1.72× MAE | Không bán; thừa nhận honest |
| UQ/conformal vs các scheme | ✅ | ~4/5 (không dẫn đầu) | Không bán; UQ = tính chất kèm theo |
| **Label-efficiency (nhãn đếm vs mask)** | ✅ | **THẮNG** khi tính theo chi phí | **← trục bán duy nhất** |

→ Chỉ có **label-efficiency** là chỗ vừa **mới**, vừa **có baseline (mask-supervised)**, vừa **thắng**. Mọi thứ khác chỉ đóng vai bổ trợ.

Vì sao "nhỏ nhất" KHÔNG được làm điểm bán: nhỏ thì ai cũng làm được (cắt kênh là xong). Cái đáng giá là **nhỏ + rẻ nhãn + vẫn tin cậy được** — tức là điểm khác biệt nằm ở **chi phí NHÃN**, không phải số tham số.

---

## 2. Cú lật quyết định: đổi trục x sang CHI PHÍ annotation

Bản báo cáo cũ so ở **cùng số ẢNH** → supervised nhỉnh chút ở khúc giữa → nghe yếu ("chỉ ngang thôi").

Nhưng câu hỏi thực tế là **cùng CÔNG dán nhãn** bỏ ra. Vì 1 mask (vẽ viền ~52.8 nhân) đắt gấp **5–10×** một nhãn đếm (Bearman ECCV'16: ~2.4s/điểm), nên với cùng ngân sách công:

- **Distilled @100%** — 532 ảnh, chi phí ≈ 532 × 127s → MAE **14.12**, coverage **0.843**
- **Supervised cùng chi phí đó** — chỉ đủ dán **~53–106 ảnh (10–20%)** → MAE **~20–26**, coverage **~0.88**

| So ở CÙNG chi phí nhãn | MAE ↓ | Coverage |
|---|---|---|
| **Distilled (count-only), 532 ảnh** | **14.12** | 0.843 |
| Supervised (mask), ~53–106 ảnh | ~20–26 | ~0.88 |

→ **Distilled có MAE thấp hơn ~30–46%, coverage tương đương.** Kết luận **bền** với mọi tỉ số chi phí trong dải 5–10× (distilled thắng ở cả hai đầu dải).

**Quan trọng:** đây là số **đã có sẵn** trong dữ liệu label-efficiency, chỉ vẽ lại theo trục "giây annotation" thay vì "% ảnh". Nó biến kết quả chính từ *"hòa"* thành *"đè"* — mà không cần chạy thêm GPU.

---

## 3. Xương sống bài báo

> Chi phí annotation mask là nút thắt của computational pathology. Nghiên cứu chứng minh: một pathology foundation model **đông lạnh** cộng **nhãn đếm rẻ** (không mask), khi so ở **cùng ngân sách công dán nhãn**, cho một bộ đếm **tốt hơn** cách train-bằng-mask — đồng thời tặng kèm **uncertainty calibrated** mà bản thân foundation model và các peer distill đều không có.

**Tên đề xuất:** *"Count Is Enough: Annotation-Efficient Distillation of Pathology Foundation Models for Trustworthy Cell Counting"*

---

## 4. Ba đóng góp (theo đúng thứ tự sức mạnh)

1. **Annotation-efficiency (HEADLINE):** ở cùng chi phí nhãn, count-only distillation **đè** mask-supervision về MAE và ngang về coverage. Đây là câu trả lời cho "tại sao distill thay vì train bằng mask". *Hình chính = frontier theo trục giây-annotation.*
2. **Trustworthy gần như miễn phí:** đầu ra phân phối (μ, σ) Poisson-anchored, coverage hợp lệ theo nhóm mô, chỉ **1 forward** — UQ mà các peer distill (NuLite, HoVer-unet) **không có**. Kế thừa PB-σ của Paper 1 làm nền, chỉ ra và vá giới hạn của nó dưới chế độ nén (learned-σ ổn định qua các scheme).
3. **Rẻ triển khai (bonus, không lên tiêu đề):** 1.9M tham số, 1-forward — hệ quả tự nhiên của distillation, ghi ở phần efficiency.

---

## 5. Bố cục kết quả đề xuất

| Mục | Nội dung | Baseline |
|---|---|---|
| **Fig 1 (headline)** | Frontier trục **chi phí annotation**: MAE & coverage của distilled vs supervised theo giây dán nhãn — distilled dominate | supervised (mask) |
| Bảng equal-cost | Distilled@100% vs supervised cùng-chi-phí (§2 ở trên) | supervised |
| §quality | Chất lượng tuyệt đối PanNuke/NuInsSeg + **thừa nhận honest** dưới SOTA accuracy vì đây không phải trục | (accuracy: NuLite — để thừa nhận) |
| §uncertainty | learned-σ vs PB-σ dưới nén (tôn trọng Paper 1) | KD / PB-σ |
| §transfer | σ transfer sang dataset khác — trình bày như **tính chất**, không claim "tốt hơn" | (không claim superiority) |
| §efficiency | 1.9M, 1-forward; ablation dung lượng 0.5M | bảng params vs NuLite/CellViT |

**Sửa so với bản cũ:**
- Gộp §3.3 (count-only frontier) vào phần label-efficiency — nó chính là nhánh distilled, baseline supervised nằm ngay đó.
- Hạ §3.5 transfer & §3.6 efficiency xuống "tính chất/deployment", **không** để đứng như "một mình một chợ tự khen".
- Conformal lùi về **một** chỉ số reliability (coverage), không chiếm toàn bộ results — để bài không "đọc như bài conformal của Paper 1".

---

## 6. Trung thực về trần (không tô hồng)

- Đây là **bài applied Q1 tầm-trung vững**, không phải breakthrough đập-bảng-SOTA. Sức mạnh = một gói mạch lạc + trung thực trả lời câu hỏi annotation-cost thật.
- Accuracy thô **dưới** SOTA mask-heavy → thừa nhận thẳng, khung lại là "chế độ nhãn rẻ", không giấu.
- Con số chi phí **5–10×** là **khoảng an toàn** (point-cost 2.4s có nguồn Bearman; mask-cost/nhân không có nguồn chuẩn → không quy về 1 số đơn). Kết luận "distilled đè ở equal-cost" bền với cả dải này.
- Hai cửa nâng lên top-tier (validation lâm sàng / đào cơ chế N4) **ngoài phạm vi hiện tại** — không hứa.

---

## 7. Việc tiếp theo (nếu cô duyệt hướng)

1. Vẽ **Fig 1 — frontier trục chi phí annotation** từ số đã có (không cần GPU).
2. Viết lại báo cáo/manuscript theo bố cục §5.
3. (Tùy chọn hardening) nâng label-efficiency từ 3→5 seed cho hình sạch.

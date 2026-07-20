# Báo cáo tiến độ Paper 2

*Chưng cất pathology foundation model thành bộ đếm tế bào nhẹ, giám sát mức đếm (không cần mask)*

---

## 1. Vấn đề nghiên cứu

Các pathology foundation model (như PathoSAM ~640M) segment được nhân tế bào và về nguyên tắc đếm được (đếm số instance), nhưng vướng ba rào cản khi dùng thực tế:

1. **Kích thước lớn** (~640M tham số) → tốn tài nguyên, khó triển khai ở nơi thiếu GPU.
2. Muốn **thích nghi sang dataset/miền mới** theo cách thông thường vẫn cần **annotation mask mức instance** — rất tốn công vẽ.
3. Là mô hình segmentation, chúng **không xuất độ bất định** cho con số đếm.

Câu hỏi: **liệu có thể distill một foundation model xuống một bộ đếm rất nhẹ, chỉ dùng nhãn mức đếm (không mask), mà vẫn đạt độ chính xác đếm cạnh tranh — và kèm thêm được khoảng dự đoán có độ tin cậy?**

Đây cùng dòng với hai công trình mới nhất về distillation-counting trong bệnh học — **Shvetsov et al. 2025** (distill H-Optimus 1.1B → student 24M, dùng R² đánh giá đếm, headline là student ngang teacher ở 48× nhỏ hơn) và **CellGenNet 2025** (distill StarDist → U-Net, bán label-efficiency, thắng teacher + baseline cùng hạng). Cả hai **lead hiệu quả/nhãn, không lead uncertainty**, và **không model nào có đầu bất định**. PACT đi cùng hướng nhưng **nhỏ hơn nữa (1.9M), có định lượng label-efficiency, có transfer cross-dataset, và thêm đầu phân phối**.

## 2. Hướng giải quyết

Dùng PathoSAM (~640M) làm teacher, distill sang student **1.9M** đặt tên **PACT** (*Poisson-Anchored Calibrated counTer*). PACT học density map của teacher (không cần mask người vẽ) + một giá trị đếm mỗi ảnh, xuất đồng thời count (μ = tổng density) và độ bất định σ (Poisson-anchored, hiệu chuẩn bằng conformal).

**Ký hiệu:**
- **PACT (ours):** student 1.9M, đầu σ **học được** (Poisson-anchored β-NLL).
- **KD (mốc so):** cùng student 1.9M nhưng σ theo **công thức giải tích Poisson-Binomial (PB-σ)** — cách dựng bất định của **Paper 1** áp lên student nén (mốc so, không phải "đối thủ").
- **Nhãn count-only:** PACT cần **một con số đếm mỗi ảnh** (lấy được bằng chấm điểm), **KHÔNG cần mask** — nhưng vẫn là *giám sát nhẹ*, không phải "không nhãn".

## 3. Bằng chứng thực nghiệm

### 3.1 ★ Kết quả chính — Độ chính xác đếm so với các model có tên

So PACT với foundation/heavy net áp **off-the-shelf** (train PanNuke → dùng thẳng trên NuInsSeg) trên **cùng 665 ảnh, cùng một thước đếm** (`dump_counts.py`). Đây là bảng theo đúng cấu trúc CellGenNet (proposed so nhiều baseline có tên) nhưng bằng **metric đếm**:

| Method | Params | Thích nghi NuInsSeg | R² ↑ | MAE ↓ | RMSE ↓ | MAPE ↓ |
|---|---|---|---|---|---|---|
| CellViT-SAM-H | 699.7M | — (off-the-shelf) | 0.663 | 21.83 | 31.33 | 52.9% |
| LKCell-L | 163.8M | — (off-the-shelf) | 0.448 | 20.92 | 40.10 | 37.4% |
| NuLite-T | 12.0M | — (off-the-shelf) | 0.622 | 20.01 | 33.22 | 39.6% |
| PathoSAM teacher | ~640M | — (zero-shot) | 0.711 | 15.80 | 29.02 | **28.3%** |
| **PACT (ours, 5 seed)** | **1.9M** | **count (in-domain)** | **0.786** | **14.74** | **24.81** | 47.6% |

Cột "Thích nghi NuInsSeg" = nhãn **thực tế đã dùng** để thích nghi: chỉ **PACT** được thích nghi (bằng **count rẻ**); mọi net kia + teacher đều dùng **off-the-shelf** (không thích nghi).

**Đọc:** PACT đạt **R² + MAE + RMSE tốt nhất bảng**, ở **6–368× nhỏ hơn**, chỉ dùng **nhãn đếm**. Đáng chú ý: **LKCell (164M) còn thua NuLite (12M)** trên OOD, và không model dùng-sẵn nào (kể cả CellViT 699M) tiến gần PACT → foundation nặng áp off-the-shelf transfer không tốt hơn tương ứng kích thước. *(CellViT chạy ở native 1024; ở 256 R² chỉ 0.444 → đã dùng số 1024 cho fair.)*

**Hai điều nói thẳng (trung thực, đúng chuẩn CellGenNet):**
- **Claim đúng = thích-nghi-rẻ, KHÔNG phải "PACT là model đếm giỏi hơn".** Chỉ PACT được thích nghi trên NuInsSeg, bằng **nhãn count rẻ**; các net kia để off-the-shelf. Đây *chính xác* là thiết lập CellGenNet (họ train trên domain đích rồi so với Cellpose/StarDist/InstanSeg off-the-shelf). Lý do các net kia **kẹt** off-the-shelf: muốn thích nghi chúng phải có **mask pixel đắt**, còn PACT chỉ cần **count** → sự bất đối xứng nhãn đó *chính là* luận điểm label-efficiency.
- **PACT thua MAPE** (47.6% > teacher 28.3%): density-sum sai tương đối cao ở ảnh ít nhân — disclose thẳng (như CellGenNet disclose FPR cao của họ).

*(In-domain PanNuke: PACT MAE 3.36, worst-organ coverage 0.906, Winkler 19.28.)*

### 3.2 Vì sao distill? — Label-efficiency (phép so CÓ KIỂM SOÁT)

Để tách riêng ảnh hưởng của **loại giám sát**, so trên **cùng kiến trúc PACT, cùng data, cùng số ảnh** — chỉ khác nguồn nhãn: (a) **distilled** (target = density teacher + đếm mức ảnh, chỉ cần nhãn count); (b) **mask-supervised** (cùng mạng, target = density dựng từ mask GT từng nhân). NuInsSeg, 3 seed.

| Ngân sách ảnh | Distilled (count-only) — Worst-org / MAE | Mask-supervised — Worst-org / MAE |
|---|---|---|
| 10% (53 ảnh) | 0.865 / 24.26 | 0.879 / 26.09 |
| 25% (133 ảnh) | 0.858 / 21.85 | 0.824 / 18.49 |
| 50% (266 ảnh) | 0.840 / 20.34 | 0.830 / 17.73 |
| 100% (532 ảnh) | 0.843 / 14.12 | 0.897 / 14.61 |

Cùng số ảnh → reliability **không phân biệt được thống kê** và MAE tương đương (mask nhỉnh ~2–3 nhân ở giữa nhưng hòa ở hai đầu). **Distillation (chỉ nhãn đếm) đạt chất lượng ngang giám sát bằng mask** — nhưng nhãn đếm là **một con số/ảnh** thay vì vẽ ~52.8 mask/ảnh. **Đây là phép so công bằng NHẤT của bài** (cùng mọi thứ, chỉ khác nguồn nhãn) → xương sống luận điểm.

*Ghi chú trung thực: (i) single-split, chỉ đọc tương đối giữa các mức; (ii) GT count vẫn lấy từ mask → chứng minh **yêu cầu giám sát** của phương pháp là mức đếm, không phải "đã annotate rẻ hơn trong thực tế"; (iii) claim về **loại nhãn** (một số vs mask từng nhân) — factual; bội số thời gian cụ thể chưa đưa vì thiếu citation chuẩn.*

### 3.3 Hiệu quả mô hình

| Model | Params | GMACs @256² | Size (fp32) | Latency (ms/ảnh) | Peak VRAM |
|---|---|---|---|---|---|
| PathoSAM teacher (ViT-H) | 641M | n/a¹ | ~2.56 GB | n/a¹ | **5.87 GB** |
| **PACT (ch32, chính)** | **1.935M** | 10.49 | 7.74 MB | **4.91 ms** | **70.7 MB** |

*¹ GMACs/latency của teacher không so cross-model được (teacher 1024²/prompt-based vs PACT 256²/1-forward — khác thang & paradigm); so sánh dựa params·size·VRAM, như H-Optimus.*

PACT **1.935M** — nhỏ hơn teacher (641M) ~**330× tham số**, và về **bộ nhớ chạy: 70.7 MB so với 5.87 GB → nhỏ hơn ~85×** (đo thực Tesla T4: teacher = peak VRAM của image-encoder SAM ViT-H ở 1024² native; PACT ở 256² native). Đây đúng con số headline kiểu H-Optimus (16GB→3GB). PACT cũng nhỏ hơn student H-Optimus (24M) ~**12×**, NuLite (12M) ~**6×**; chạy **4.91 ms/ảnh (204 ảnh/giây)** trên T4 — deploy được trên GPU/CPU khiêm tốn. *(Không so latency teacher vì SAM prompt-based khác paradigm PACT 1-forward; teacher GMACs chưa đo.)*

*Ablation dung lượng (phụ): cấu hình ch16 (~0.5M) không mất chất lượng (NuInsSeg nhỉnh, PanNuke hòa) → phương pháp bền theo dung lượng. Đây là ablation, cấu hình chính vẫn là ch32.*

### 3.4 Transfer reliability cross-dataset

| Transfer | Coverage | Nhận xét |
|---|---|---|
| NuInsSeg → PanNuke | worst-org 0.897 | ≈ in-domain 0.906 |
| NuInsSeg → CryoNuSeg | marginal 0.967 | reliability duy trì (dataset thứ 3) |

Đầu ra phân phối (μ, σ) transfer sang dataset khác **không cần train lại** — điều cả hai paper mốc (Shvetsov, CellGenNet) để lại "future work". *Giới hạn trung thực: transfer sang MoNuSAC KHÔNG giữ được do chênh độ phân giải làm nhân co ~4× (scale gap) — ghi rõ là giới hạn về scale.*

### 3.5 (Phụ) Đầu bất định: learned-σ so với PB-σ giải tích

PB-σ là đóng góp **Paper 1** (kiểm chứng trên foundation model). Câu hỏi: mang PB-σ xuống student 1.9M nén, còn hiệu chuẩn không? So trên **cùng student**, khác cách lấy σ:

| Metric | PB-σ (KD, kiểu Paper 1) | Learned-σ (PACT) |
|---|---|---|
| Worst-organ (global) | 0.278 | **0.610** |
| Worst-organ (cluster) | 0.658 | **0.750** |
| MAE | 21.71 | **14.72** |

**Diễn giải (không hạ Paper 1):** PB-σ **không "hỏng"** — dưới scheme **cluster** vẫn tốt (+14%); chỉ dưới **global** (khi student nén không tái tạo trung thực điểm detection) σ giải tích mất hiệu chuẩn. PACT **kế thừa** Paper 1 và **vá đúng giới hạn của PB-σ trong chế độ nén** bằng σ học được.

**Trung thực:** trên thang UQ-floor (E-AURC) PACT xếp **~4/5** so các phương pháp UQ cùng backbone → **đầu bất định là tính năng bổ trợ (bonus), KHÔNG phải chỗ dẫn đầu**. Điểm riêng của nó là *phân phối tái dùng được + transfer* (§3.4), không phải coverage tốt nhất.

## 4. Đóng góp chính

1. **★ Distilled tiny counter đạt độ chính xác đếm dẫn đầu ở kích thước tí hon** (§3.1): PACT 1.9M đạt R²/MAE/RMSE tốt nhất so teacher + heavy net off-the-shelf, ở 6–368× nhỏ hơn — chỉ dùng nhãn count. *(cấu trúc so như CellGenNet/H-Optimus, nhưng nhỏ hơn nhiều.)*
2. **★ Chứng minh label-efficiency bằng thực nghiệm có kiểm soát** (§3.2): distillation (nhãn đếm) ngang giám sát mask ở cùng ngân sách ảnh — thứ hai paper mốc chỉ nói chứ không định lượng.
3. **Transfer reliability cross-dataset** (§3.4) — hai paper mốc để "future work".
4. **(Phụ) Đầu phân phối calibrated nhẹ** (§3.5): kế thừa PB-σ của Paper 1, vá giới hạn của nó dưới nén bằng learned-σ. **Không dòng distillation-counting nào có UQ** — nhưng đây là bonus, không phải trục bán.

## 5. Kết luận hiện tại

Khả thi để chuyển tri thức đếm từ một pathology foundation model 640M xuống một bộ đếm **1.9M**, chỉ cần **nhãn đếm mức ảnh (không mask)**, mà đạt độ chính xác đếm **dẫn đầu ở hạng tí hon** và reliability tốt, transfer được. Đóng góp là **một gói mạch lạc, nghiêng model lightweight**: model tí hon + nhãn count rẻ + (bonus) uncertainty calibrated. Hướng: *label-efficient distillation of a lightweight cell counter* — Q1 ứng dụng (CBM/CMIG/AIM), cùng hạng với Shvetsov 2025 & CellGenNet 2025 nhưng chặt hơn.

## 6. Việc còn lại (đang hoàn thiện)

- **Chốt CellViT-SAM-H ở native 1024** (số hiện tại @256 là tạm; smoke cho thấy lệch ~15%/ảnh — chạy lại cho fair).
- **Thêm 1 baseline recent/classic** (InstanSeg 2024–25 hoặc Cellpose) cho đa dạng paradigm — đủ 5 baseline.
- **Latency/VRAM thực đo** trên GPU (Bảng hiệu quả §3.3).
- Điền cột heavy-net cho **PanNuke** (leak-free).

*(Hướng multi-teacher committee đã thử và gác lại: probe cho thấy nó chỉ tạo thêm một tín hiệu σ (UQ) chứ không cải thiện accuracy — không phục vụ trục lightweight/accuracy đang là chính.)*

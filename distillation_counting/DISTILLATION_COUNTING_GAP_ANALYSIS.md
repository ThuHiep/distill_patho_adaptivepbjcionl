# Knowledge Distillation cho bài toán Đếm (Counting) — Rà soát 2025–2026, phân tích khe hở, đề xuất hướng

> Cập nhật: 2026-07-10. Người thực hiện rà soát: Claude (theo yêu cầu của bạn, dựa trên chỉ đạo của
> giáo viên hướng dẫn sau khi nộp bài "Adaptive PB-JCI Online").
>
> **Nguyên tắc bắt buộc của tài liệu này (theo đúng yêu cầu):**
> - **Không bịa số liệu, không bịa tên bài, không bịa arXiv ID.** Mọi bài nêu dưới đây đều có link
>   thật, được tra qua WebSearch và phần lớn được **mở trực tiếp** (WebFetch) để đọc abstract/nội dung
>   thật — không suy diễn từ tên bài.
> - **Có đánh dấu mức độ tin cậy** cho từng bài: 🟢 *đã mở full abstract/nội dung và đọc trực tiếp*,
>   🟡 *chỉ có snippet từ kết quả tìm kiếm (chưa mở được trang, có thể sai lệch)*, 🔴 *bị chặn
>   paywall/403, chỉ ghi lại thông tin gián tiếp*.
> - **Không khẳng định "chưa ai làm X"** như một chân lý tuyệt đối — chỉ khẳng định *"không tìm thấy
>   trong phạm vi các truy vấn đã chạy (liệt kê ở cuối file)"*. Đây là giới hạn cố hữu của rà soát bằng
>   search — trước khi chốt novelty trong bài nộp, **bắt buộc kiểm tra lại bằng Google Scholar /
>   Semantic Scholar / Connected Papers** như quy ước đã có trong `BASELINES_PAPER.md` và
>   `PHAN_TICH_LITERATURE_DOI_THU.md`.
> - Không có phần "kết quả thực nghiệm" nào trong file này là số thật của bạn — toàn bộ là **đề xuất
>   hướng nghiên cứu**, chưa chạy, chưa có số.

---

## 0. Tóm tắt điều hành (đọc trước, chi tiết ở dưới)

1. Bài cô đưa (Khan et al., *Sci. Rep.* 2025, "Crowd counting at the edge using weighted knowledge
   distillation") là **KD kinh điển cho đếm đám đông**: teacher/student đều là CNN nhỏ, KD kiểu
   response+feature với trọng số theo sai số của teacher trên từng mẫu. Đây là nền tốt để tham chiếu,
   nhưng **tự bài báo đã thừa nhận trong phần Related Work**: *"rất ít công trình dùng KD cho bài toán
   hồi quy (regression) như ước lượng mật độ đám đông"* — tức là **KD cho counting nói chung vẫn là
   một ngách nhỏ so với KD cho classification**. Đây là quan sát xuất phát điểm, không phải điều tôi
   suy diễn — nó nằm trong chính bài báo (mục "Related work", đoạn về "dark knowledge").
2. Sau khi rà rộng 2025–2026, bức tranh KD-cho-đếm chia làm các nhánh: (a) đếm đám đông/xe cộ CNN cổ
   điển — vẫn tiếp tục nhưng ít đổi mới về bản chất; (b) **distill foundation model (SAM3, CLIP, MLLM)
   sang student nhẹ** — đang nóng nhưng **hầu như chỉ nhắm segmentation/detection, không đo chỉ số
   ĐẾM**; (c) **distill cho đếm tế bào/nhân (pathology)** — có vài bài 2025 rất gần với bộ dữ liệu bạn
   đang dùng (NuInsSeg) nhưng **không báo cáo độ chính xác đếm**, chỉ báo Dice/IoU; (d) **distillation
   kết hợp uncertainty/conformal** — có tồn tại (AdaConG, 2025–2026) nhưng **chỉ làm trên classification,
   và tác giả tự nói rõ là chưa mở rộng sang regression và chưa có bảo đảm coverage cho student**.
3. **Khe hở rõ nhất, có bằng chứng cụ thể, và khớp nhất với năng lực + hạ tầng bạn đã có** (SAM3/
   PathoSAM, pipeline conformal PB-JCI-online, dữ liệu MoNuSAC/PanNuke/NuInsSeg/CoNSeP, có thể thuê
   vast.ai): **chưa có công trình nào kiểm tra việc distillation ảnh hưởng thế nào đến tính hợp lệ
   (validity/coverage) của một bộ định lượng bất định có bảo đảm (conformal) khi áp lên model đếm đã bị
   nén** — và do đó chưa có công trình nào **thiết kế loss distillation để bảo toàn coverage** thay vì
   chỉ bảo toàn MAE. Đây chính là phần mở rộng tự nhiên của "Adaptive PB-JCI Online" mà bạn vừa nộp.
4. Đề xuất chính (chi tiết ở mục 5): **"Calibration/Coverage-Preserving Distillation cho đếm tế bào đa
   lớp"** — dùng SAM3/PathoSAM (đã có bảo đảm conformal từ paper 1) làm teacher, distill sang student
   nhẹ để triển khai thời gian thực/edge, và nghiên cứu (i) coverage của khoảng dự đoán có bị vỡ khi
   dùng student không, (ii) nếu vỡ thì vỡ theo cơ chế nào, (iii) đề xuất mục tiêu distillation ngăn vỡ.
   Đây **chưa phải một phương pháp đã thiết kế xong** — mục 5 nêu câu hỏi nghiên cứu và không gian thiết
   kế, **không bịa công thức cụ thể** vì chưa có bằng chứng thực nghiệm để khẳng định công thức nào đúng.

---

## 1. Bối cảnh (để không lặp lại những gì bạn đã biết, nhưng ghi lại cho đầy đủ mạch)

- Bạn vừa nộp **"Adaptive PB-JCI Online"**: conformal prediction **online/adaptive dưới distribution
  shift** cho bài toán **đếm nhân tế bào đa lớp** (Poisson-Binomial structured score), chạy trên
  backbone **SAM3** và **PathoSAM**, đo bằng Winkler/coverage, trên dữ liệu MoNuSAC/PanNuke/NuInsSeg
  (theo `BASELINES_PAPER.md`, `PHAN_TICH_LITERATURE_DOI_THU.md` trong repo này).
- Cô hướng dẫn đưa bài Khan et al. 2025 (`s41598-025-90750-5.pdf`) và bảo tìm hiểu tiếp **distillation
  learning cho counting** — nếu ra kết quả tốt thì luận văn sẽ là "counting với nhiều cách tiếp cận"
  (tức là mở rộng số hướng con trong luận văn, không nhất thiết thay thế hướng conformal).
- Hạ tầng đã có: pipeline SAM3 LoRA (`kaggle/lib/lora_sam3.py`, `kaggle/lib/sam3_train.py`), type head
  phân loại tế bào, loader cho MoNuSAC/PanNuke/CoNIC, module conformal (`kaggle/lib/conformal.py`,
  `saocp.py`, `ogd.py`), và có thể thuê GPU trên vast.ai (đã có sẵn `kaggle/vast/` với script train/eval
  multi-seed). → Bất kỳ hướng distillation nào tái dùng được các phần này sẽ **rẻ hơn nhiều** để triển
  khai so với bắt đầu từ số 0.

---

## 2. Đọc kỹ bài cô đưa: Khan et al. 2025 — "Crowd counting at the edge using weighted KD"

**Nguồn:** 🟢 đọc trực tiếp toàn văn PDF (16 trang) — Khan, Menouar, Hamila, Abu-Dayya,
*Scientific Reports* 15:11932 (2025). DOI: 10.1038/s41598-025-90750-5.

### Phương pháp
- Teacher: **CSRNet** (16.26M tham số) hoặc **CSRNet_lite** (ablation). Student: **MCNN** (0.13M),
  **DroneNet** (0.15M, dùng Self-ONN thay Conv), **LCDnet** (0.05M) — cả ba đều **CNN đa cột kinh điển**,
  không transformer, không pretrain.
- KD dạng **response + feature hybrid** cho bài toán hồi quy mật độ (density map), công thức trọng số
  (Eq. 6–9 trong bài):
  ```
  L_reg = (1/n) Σ [ α‖p_S − p_GT‖² + (1−α)·φ_i·‖p_S − p_T‖² ]
  φ_i = 1 − ‖p_T − p_GT‖² / η          (η = max(L_T) − min(L_T) trên tập train)
  ```
  → **φ_i là trọng số theo mẫu**: nếu teacher sai nhiều trên ảnh đó (density map lệch ground-truth),
  trọng số soft-label giảm xuống — ý tưởng là "đừng bắt student tin teacher khi teacher cũng sai".
  Đây là đóng góp cụ thể nhất của bài (không phải KD chuẩn Hinton, không phải feature-matching chuẩn
  FitNets — mà là **feature/response distillation có trọng số theo độ tin cậy của teacher trên từng
  mẫu**, áp dụng cho hồi quy chứ không phải phân loại).
- Đánh giá: 6 dataset (Mall, ShanghaiTech A/B, CARPK, DroneRGBT, Aerial Sheep) — **cross-domain thật**
  (người, xe, cừu) — đây là điểm mạnh thực nghiệm của bài, không phải chỉ một domain.
- Ablation: đổi teacher (CSRNet ↔ CSRNet_lite), đổi tỉ lệ nhãn (40/60/80/100%), domain adaptation
  (xe, cừu).
- Kết quả: giảm MAE khoảng 8–26% tuỳ dataset/tỉ lệ nhãn; sinh viên nhẹ hơn teacher 8–18 lần về thời
  gian suy luận trên Jetson.

### Những gì bài **tự nhận** là hạn chế/định hướng tương lai (trích/diễn giải từ Conclusion)
- *"rất ít công trình dùng KD cho bài toán hồi quy như ước lượng mật độ đám đông"* — tác giả tự đặt
  mình vào một ngách hẹp.
- Hướng tương lai họ nêu: tích hợp KD với **federated learning** (student = model cục bộ trên edge,
  teacher = model toàn cục trên server), và kiến trúc **server-less FL**. **Không đề cập** đến
  uncertainty quantification, conformal prediction, foundation model teacher, hay class-agnostic/
  open-vocabulary counting.

### Hạn chế tôi quan sát được (không phải tác giả tự nói, nhưng suy ra trực tiếp từ nội dung bài — có
căn cứ, không phải suy diễn tuỳ tiện)
| # | Hạn chế | Bằng chứng trong bài |
|---|---|---|
| L1 | Teacher/student đều CNN nhỏ/cũ (2016–2023), không dùng transformer hay foundation model | Bảng 2: CSRNet 2018, MCNN 2016, DroneNet 2023 — không model nào sau 2023 |
| L2 | Mỗi model chỉ học **một lớp đối tượng cố định** (người HOẶC xe HOẶC cừu) — train riêng từng dataset, không phải một model đếm đa lớp/mở lớp | Mục Ablation "Cross-domain adaptation": chạy lại từ đầu trên từng dataset, không phải zero-shot |
| L3 | KD loss chỉ dùng **L2 pixel-wise trên density map** — không có relation-based KD (dù bài có nhắc lý thuyết relation-based ở phần "Knowledge distillation" nhưng **không dùng** trong phương pháp đề xuất) | So Eq. 6 (chỉ response+feature L2) với phần lý thuyết mục "Knowledge distillation" liệt kê 3 loại KD |
| L4 | Không có bất kỳ định lượng bất định/khoảng tin cậy nào cho số đếm ra — chỉ MAE/SSIM/PSNR điểm | Toàn bộ Bảng 1, 3, 4, 5 chỉ có giá trị điểm |
| L5 | τ (temperature) là hằng số cố định, không thích nghi theo mẫu/theo domain | Mục "Proposed scheme": nói rõ τ cần tinh chỉnh (finetune) nhưng không có cơ chế tự động |

---

## 3. Bản đồ literature 2025–2026: KD cho counting theo từng nhánh

> Ký hiệu độ tin cậy: 🟢 đã mở đọc trực tiếp (WebFetch), 🟡 chỉ có snippet tìm kiếm, 🔴 bị chặn/paywall.

### 3.1. Đếm đám đông / xe cộ — CNN/Transformer cổ điển (nối tiếp dòng của bài cô đưa)

| Bài | Năm/venue | Trạng thái | Cơ chế cốt lõi | Ghi chú novelty so với Khan 2025 |
|---|---|---|---|---|
| **D2PT** — Density to Point Transformer with KD for Crowd Counting and Localization | IEICE Trans. Inf. 2025 (2024EDL8067) | 🟡 snippet (không mở được toàn văn, chỉ trang tóm tắt IEICE/researchgate) | Transformer teacher-student, **feature-aligned KD giữa hai đầu ra khác dạng**: density-map branch và point-map branch huấn luyện phối hợp | Khác Khan 2025 ở chỗ dùng **transformer + đa đầu ra** (density + point) thay vì chỉ density map CNN |
| **DHMoE** — Towards trustworthy crowd counting by distillation hierarchical mixture of experts for edge-based cluster computing | *Cluster Computing* (Springer), 08/2025 | 🟡 snippet (trang Springer đòi đăng nhập, không mở được toàn văn) | KD (teacher lớn → student nhẹ) **+ Hierarchical Mixture-of-Experts**: 4 stage của student = 4 "expert" theo scale khác nhau, giải quyết scale variation | Đánh giá trên **4 dataset đám đông + 4 dataset xe cộ** (theo snippet, tên cụ thể chưa xác nhận được — TRANCOS được nhắc tới ở kết quả tìm kiếm liên quan nhưng chưa xác nhận có trong bài này) |
| Shen et al. — "A lightweight object counting network based on density map knowledge distillation" (EdgeCount) | IEEE TCSVT 2024 | 🟡 (đã được chính bài Khan 2025 trích dẫn và mô tả, xem mục Related Work của Khan) | Teacher/student cùng kiến trúc MobileViT + SCConv + module fusion nhẹ | Đã được Khan 2025 dùng làm related-work, không phải phát hiện mới của tôi — trích lại để đủ bức tranh |
| Remote Sensing Object Counting with Online Knowledge Learning | arXiv 2303.10318 (2023) | 🟡 snippet | **Online KD** (không cần teacher pretrain sẵn, học đồng thời) cho đếm ảnh viễn thám | Trước 2025 nhưng là tiền lệ quan trọng cho "online distillation" — khác self-distillation |
| Density Map Distillation for Incremental Object Counting | arXiv 2304.05255, IEEE 2023 | 🟡 snippet | KD dùng làm **regularizer chống quên** (catastrophic forgetting) khi học thêm lớp mới, không phải nén model | Hướng khác hẳn (continual learning), nêu để không nhầm lẫn khi tìm "distillation + counting" |

**Nhận xét nhánh 3.1:** Không có bài nào trong nhánh này (2025) dùng foundation-model teacher (CLIP/SAM/
MLLM), và không có bài nào gắn với uncertainty/conformal. Cơ chế "trọng số theo độ tin cậy teacher"
của Khan 2025 (φ_i) **chưa thấy bài nào khác dùng lại/so sánh trực tiếp** trong phạm vi tìm kiếm đã chạy.

### 3.2. Class-agnostic / open-vocabulary / few-shot counting — hiệu quả hoá bằng KIẾN TRÚC, không phải distillation

| Bài | Năm | Trạng thái | Ghi chú |
|---|---|---|---|
| A Survey on Class-Agnostic Counting: Advancements from Reference-Based to Open-World Text-Guided Approaches | arXiv 2501.19184, sửa đến v4 02/2026 | 🟢 đã mở, đọc abstract | Taxonomy 3 nhánh (reference-based / reference-less / open-world text-guided). **Không hề nhắc đến knowledge distillation, hiệu quả tính toán, hay triển khai edge** như một hướng mở — abstract chỉ nêu "annotation dependency và generalization" là vấn đề tồn đọng. → **Bản thân cộng đồng CAC còn chưa đặt vấn đề hiệu quả hoá/distillation** như một open problem chính thức. |
| MambaCount — Efficient Text-guided Open-vocabulary Object Counting with Spatial Sparse SSD | arXiv 2606.17650 (2026) | 🟡 snippet | Hiệu quả hoá bằng **kiến trúc Mamba/SSM**, không phải distillation từ teacher lớn |
| RT-Counter — Real-Time Text-Guided Open-Vocabulary Object Counting | arXiv 2606.17561 (2026) | 🟡 snippet | Tương tự — thiết kế kiến trúc nhẹ từ đầu, không có teacher-student |
| Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting | arXiv 2602.12774 (2026) | 🔴 không tải được nội dung (file quá lớn để fetch), chỉ có tiêu đề | Tên bài gợi ý dùng MLLM để tạo pseudo-label/bootstrap — **chưa xác nhận được có phải KD chính thống (loss teacher-student) hay chỉ pseudo-labeling**. Cần đọc trực tiếp trước khi dùng làm căn cứ. |
| T2ICount (CVPR 2025) | 2025 | 🔴 403 khi fetch trực tiếp CVF; chỉ có snippet | Dùng **prior của diffusion model đã pretrain** cho zero-shot counting — đây là "leverage pretrained model", **không rõ có phải distillation loss tường minh (KL/L2 giữa teacher-student) hay không** — cần đọc kỹ trước khi trích là "distillation". |

**Nhận xét nhánh 3.2 — đây là một khe hở quan trọng:** Cách cộng đồng "đếm mở lớp/text-guided" đang xử
lý bài toán hiệu quả hoá (MambaCount, RT-Counter, 2026) là **thiết kế kiến trúc nhẹ từ đầu**, **không
phải distill từ một teacher mạnh (SAM3/CLIP) sang student nhẹ**. Điều này có nghĩa là các model nhẹ này
phải **học lại khả năng tổng quát hoá open-vocabulary từ đầu** với ít tham số hơn — nhiều khả năng đánh
đổi generalization để lấy tốc độ. Không có bằng chứng nào (trong phạm vi tìm kiếm) cho thấy ai đã thử
"giữ khả năng open-vocabulary của SAM3 bằng cách distill thay vì train nhỏ từ đầu" cho riêng bài toán
đếm.

### 3.3. Distill foundation model (SAM/SAM3/CLIP) — chủ yếu nhắm segmentation, KHÔNG đo đếm

| Bài | Năm | Trạng thái | Teacher → Student | Có đo COUNT không? |
|---|---|---|---|---|
| **EfficientSAM3** — Progressive Hierarchical Distillation for Video Concept Segmentation from SAM1,2,3 | arXiv 2511.15833 (11/2025) | 🟢 đã mở, đọc abstract | SAM3 (shared vision backbone + DETR-style detector + dense memory tracker) → RepViT/TinyViT/EfficientViT, 3 giai đoạn (encoder distill → temporal memory distill → end-to-end finetune) | **Không** — abstract không có metric đếm, chỉ nói "VOS datasets" (video object segmentation), không có số liệu cụ thể trong phần đọc được |
| TinySAM — full-stage KD cho SAM (gốc SAM1) | AAAI (trước 2025, không tính vào phạm vi 2025-2026 nhưng là tiền lệ kỹ thuật quan trọng) | 🟡 snippet | SAM ViT-H → student nhẹ, hard-prompt sampling + hard-mask weighting | Không |
| On Efficient Variants of Segment Anything Model: A Survey | arXiv 2410.04960 (v4, 10/2024) | 🟡 snippet | Survey các cách nén SAM (distill toàn bộ vs chỉ distill encoder) | Survey, không tự đo đếm |
| SPPNet — Single-Point Prompt Network cho nuclei segmentation | arXiv 2308.12231 (2023, **trước phạm vi 2025-2026**, nêu để biết tiền lệ) | 🟡 snippet | SAM (image encoder nặng) → distilled lightweight ViT encoder, "one-prompt-all-nuclei" | **Không** — chỉ segmentation semantic, không phải instance/count |

**Nhận xét nhánh 3.3:** Đây là nhánh kỹ thuật distillation **tiên tiến nhất về mặt kiến trúc** (progressive/
hierarchical, multi-stage) nhưng **mục tiêu luôn là segmentation/tracking chất lượng cao (mask IoU),
chưa từng là "đếm chính xác + đếm có định lượng bất định"**. SAM3 tự nhận có cải thiện khả năng đếm
(theo trang blog Meta/Roboflow, xem mục 3.5), nhưng **chưa có ai kiểm chứng khả năng đếm đó còn giữ được
bao nhiêu sau khi distill sang student nhẹ** — đây là câu hỏi mở cụ thể, xuất hiện tự nhiên từ việc đọc
EfficientSAM3.

### 3.4. Distillation cho đếm tế bào/nhân — pathology (nhánh khớp trực tiếp với dữ liệu bạn đang dùng)

| Bài | Năm | Trạng thái | Teacher → Student | Dataset | Có đo COUNT không? |
|---|---|---|---|---|---|
| **CellGenNet** — A Knowledge-Distilled Framework for Robust Cell Segmentation in Cancer Tissues | arXiv 2511.15054 (11/2025), nộp IEEE SSIAI 2026 | 🟢 đã mở, đọc full text (arxiv.org/html) | **StarDist** (teacher, train trên nhãn thưa) → **U-Net** (student, 18 conv layer) qua pseudo-label response-based KD | **Osteosarcoma nội bộ + NuInsSeg** (⚠️ **trùng trực tiếp** với dataset bạn dùng trong PB-JCI-online) | **KHÔNG** — chỉ Dice, IoU, FPR/TPR, Hausdorff Distance, F1. Không có MAE/RMSE của số đếm nhân, không có coverage/interval. Bài kết thúc không có mục Limitation/Future Work rõ ràng (đã kiểm tra trực tiếp, không có). |
| SPPNet (xem 3.3) | 2023 | 🟡 | SAM → lightweight ViT | Nuclei nói chung | Không |
| "A generalizable pathology foundation model using a unified knowledge distillation pretraining framework" | PubMed, 09/2025 (PMID 40897898) | 🔴 bị chặn cookie-wall, chỉ có tiêu đề từ snippet gốc, **chưa xác nhận được nội dung** | Chưa rõ | Chưa rõ | Chưa rõ — **không dùng làm căn cứ cho đến khi đọc được bản đầy đủ** |
| DCSNet — lightweight KD cho chẩn đoán ung thư phổi từ ảnh mô bệnh học | arXiv 2505.09334 (05/2025) | 🟢 đã mở, đọc abstract | 8 CNN (ResNet50 chính) → student nhẹ | Ảnh mô bệnh học ung thư phổi | **Không phải bài toán đếm** — đây là **phân loại mô** (tissue classification), không phải đếm tế bào. Liệt kê ở đây chỉ để loại trừ, tránh nhầm là "KD cho đếm pathology". |
| Revisiting foundation models for cell instance segmentation | arXiv 2603.17845 (03/2026), MIDL 2026 | 🟢 đã mở, đọc abstract | Không phải bài distillation — là **benchmark/so sánh** SAM/SAM2/SAM3/CellPoseSAM/CellSAM/μSAM | Nhiều bộ dữ liệu kính hiển vi | Không đo đếm; **không nhắc đến hiệu quả tính toán/distillation** như một hướng ở phần đọc được — nghĩa là ngay cả bài "tổng kết lại foundation model cho tế bào" mới nhất (03/2026) **cũng chưa đặt vấn đề nén/distill** như một trục đánh giá |

**Nhận xét nhánh 3.4 — đây là khe hở rõ và cụ thể nhất:** CellGenNet (11/2025) là bài **gần bạn nhất về
mặt kỹ thuật lẫn dataset** (KD cho nhân tế bào, dùng đúng NuInsSeg) nhưng:
1. Dùng teacher **StarDist** (không phải foundation model như SAM3/PathoSAM),
2. **Không báo cáo độ chính xác đếm** — chỉ segmentation overlap metrics,
3. **Không có bất kỳ định lượng bất định nào**.

→ Nếu bạn làm distillation cho đếm nhân tế bào **có báo cáo MAE đếm + coverage conformal**, trên
**cùng dataset NuInsSeg** mà CellGenNet dùng, bạn sẽ có một so sánh trực diện, dễ định vị: *"CellGenNet
distill cho segmentation nhưng bỏ ngỏ câu hỏi liệu đếm suy ra từ mask có còn đáng tin sau distill hay
không — đây là câu hỏi bài này trả lời."*

### 3.5. SAM 3 (Meta, 11/2025) và khả năng đếm — bối cảnh cần biết trước khi chọn teacher

**Nguồn:** 🟢 xác nhận qua nhiều nguồn thứ cấp (Roboflow blog, Ultralytics docs, MarkTechPost) + có bản
arXiv 2511.16719 "SAM 3: Segment Anything with Concepts" (chưa mở full-text, chỉ xác nhận tồn tại).

- SAM 3 ra ngày 19/11/2025, thêm **Promptable Concept Segmentation**: cho một cụm danh từ ngắn (vd.
  "yellow school bus") hoặc ảnh mẫu, SAM3 trả về mask + ID cho **toàn bộ instance khớp cùng lúc** — khác
  SAM1/2 (mỗi lần 1 instance).
- Meta công bố có cải thiện trên bài toán **object counting**, đo trên benchmark **CountBench** và
  **PixMo-Count** (so với MLLM và detector chuyên dụng) — thông tin này lấy từ kết quả tìm kiếm, **chưa
  mở được bảng số liệu gốc**, chỉ nêu để biết SAM3 có claim về đếm.
- SAM3 đi kèm dataset **SA-Co** (120K ảnh, 1.7K video, hơn 200K khái niệm).
- ⚠️ Chưa tìm thấy bài nào (2025–2026, trong phạm vi search đã chạy) đánh giá **SAM3 trên FSC-147**
  (benchmark chuẩn của class-agnostic counting) — nghĩa là ngay cả việc so sánh SAM3 với dòng CAC
  truyền thống cũng còn thiếu, chưa nói đến distill SAM3 cho đếm.

### 3.6. Distillation + Uncertainty/Conformal — nhánh gần nhất về mặt PHƯƠNG PHÁP với "Adaptive PB-JCI Online"

| Bài | Năm | Trạng thái | Nội dung | Vì sao quan trọng cho bạn |
|---|---|---|---|---|
| **AdaConG** — Adaptive Conformal Guidance for Learning under Uncertainty | arXiv 2502.16736, v4 | 🟢 đã mở, đọc chi tiết (HTML) | Nhúng **split conformal prediction vào vòng lặp huấn luyện** để điều chỉnh trọng số loss theo độ bất định của teacher. Có thử nghiệm rõ ràng trên **knowledge distillation** (CIFAR-100, ResNet→ShuffleNet, có nhiễu domain-shift) | **Đây là bài "gần nhất" về ý tưởng "conformal + KD"**, nhưng: (1) **chỉ classification** (không regression/counting), (2) **conformal chỉ áp cho teacher** để tạo trọng số hướng dẫn — **không có bảo đảm coverage cho chính student sau khi train xong**, (3) tác giả tự liệt "mở rộng ngoài classification" là hướng tương lai chưa làm (đọc được trong phần Discussion/Future work, tuy bị cắt ngắn khi fetch). |
| Trust the Uncertain Teacher: Distilling Dark Knowledge via Calibrated Uncertainty (CUD) | arXiv 2602.12687, v1 02/2026 → v2 05/2026 | 🟢 đã mở, đọc abstract | Định hình lại phân phối xác suất của teacher (giảm overconfidence) trước khi distill, để student không thừa hưởng độ tự tin sai lệch của teacher | **Chỉ classification, nhiều lớp (long-tail)**. Không đụng đến counting/regression. Nhưng ý tưởng cốt lõi — *"KD chuẩn có thể truyền cả sự quá tự tin/miscalibration của teacher sang student"* — là một **cơ chế lý thuyết đáng tin để đặt giả thuyết** cho hướng đề xuất của bạn (mục 5): nếu điều này đúng cho classification, rất có khả năng cũng đúng cho counting → coverage của conformal-trên-student có thể bị lệch nếu không xử lý. |
| GNN's Uncertainty Quantification using Self-Distillation | arXiv 2506.20046 | 🟡 snippet | Self-distillation để ước lượng uncertainty cho graph neural network, không phải counting | Nêu để biết self-distillation-cho-UQ đã có tiền lệ ở domain khác (GNN), càng củng cố là "counting" là domain còn trống cho ý tưởng này |
| Uncertainty-Aware Dual-Student Knowledge Distillation for Efficient Image Classification | arXiv 2511.18826 (11/2025) | 🟡 snippet | Hai student học từ uncertainty của teacher | Classification, không counting |

**Kết luận nhánh 3.6 (quan trọng nhất của cả file):** Trong phạm vi tìm kiếm đã thực hiện (các truy vấn
liệt kê ở mục 7), **không tìm thấy bài nào kết hợp (i) knowledge distillation, (ii) bài toán đếm/hồi
quy, và (iii) bảo đảm định lượng bất định (conformal hoặc calibration) cho chính student model**.
AdaConG là bài gần nhất về mặt ý tưởng nhưng dừng ở classification và không bảo đảm coverage cho student.
Đây là khoảng trống ba chiều (distillation × counting × calibrated uncertainty) — và bạn **đã có sẵn hai
trong ba trục** (counting qua PathoSAM/SAM3, calibrated uncertainty qua PB-JCI-online) — chỉ thiếu trục
distillation, đúng như cô đang gợi ý.

---

## 4. Tổng hợp: 5 khe hở có bằng chứng cụ thể

| # | Khe hở | Bằng chứng | Mức độ chắc chắn |
|---|---|---|---|
| G1 | KD cho counting là ngách nhỏ so với classification; hầu hết KD-cho-đếm (2025) vẫn dùng CNN teacher/student cũ, một lớp đối tượng cố định | Tự nhận trong Khan et al. 2025 (mục Related work) + không tìm thấy bài KD-đếm 2025-2026 nào dùng foundation-model teacher cho một model đếm đa lớp/mở lớp | Cao — có trích dẫn trực tiếp từ nguồn |
| G2 | Nhánh "đếm mở lớp hiệu quả" (2026: MambaCount, RT-Counter) giải quyết hiệu quả bằng **kiến trúc mới**, không bằng **distillation từ teacher mạnh** → có thể đánh đổi khả năng tổng quát hoá open-vocab để lấy tốc độ, chưa ai kiểm chứng cách còn lại (distill) có tốt hơn không | Đối chiếu trực tiếp 2 bài 2026 (kiến trúc-first) với việc **không tìm thấy** bài nào distill CLIP/SAM3/MLLM → student nhẹ *cho riêng bài toán đếm* (phân biệt với EfficientSAM3 là distill cho segmentation) | Trung bình — dựa trên "không tìm thấy trong search", cần verify thêm |
| G3 | Distill foundation model (SAM3, StarDist) cho tế bào/nhân (CellGenNet, EfficientSAM3, SPPNet) đều dừng ở **segmentation metrics**, không báo cáo **độ chính xác đếm** | Đọc trực tiếp CellGenNet (không có MAE đếm) + EfficientSAM3 (không có metric đếm trong phần đọc được) + SPPNet (segmentation semantic, không phải instance/count) | Cao cho CellGenNet (đọc full text), trung bình cho 2 bài còn lại (chỉ đọc abstract) |
| G4 | Chưa có công trình kết hợp distillation + bảo đảm định lượng bất định (conformal/calibration) **cho bài toán hồi quy/đếm** | AdaConG (đọc chi tiết): tự nhận chỉ classification, không coverage cho student, liệt "mở rộng ngoài classification" là tương lai chưa làm. CUD: cùng hiện tượng (miscalibration truyền từ teacher→student) nhưng cũng chỉ classification | Cao — có 2 bài độc lập cùng dừng ở ranh giới classification, cùng chỉ ra hiện tượng liên quan |
| G5 | Ngay cả bài tổng kết foundation-model-cho-tế-bào mới nhất (03/2026, MIDL) cũng không đặt "hiệu quả hoá/distillation" như một trục đánh giá chính thức của lĩnh vực | Đọc abstract "Revisiting foundation models for cell instance segmentation" | Trung bình — chỉ đọc được abstract, có thể phần thân bài có nhắc mà abstract không nêu |

---

## 5. Đề xuất hướng nghiên cứu

> ⚠️ Đây là **đề xuất hướng và câu hỏi nghiên cứu**, không phải một phương pháp đã thiết kế xong và
> chắc chắn đúng. Tôi **không bịa công thức loss cụ thể** ở đây — vì chưa có thực nghiệm nào chứng minh
> công thức nào hiệu quả. Bước đầu tiên khi bạn triển khai phải là **thí nghiệm chẩn đoán** (mục 5.2)
> trước khi cam kết vào một thiết kế phương pháp.

### 5.1. Hướng chính (khuyến nghị): Calibration/Coverage-Preserving Distillation cho đếm tế bào đa lớp

**Câu hỏi nghiên cứu trung tâm:**
> Khi distill một model đếm tế bào đã được bọc conformal (teacher = SAM3/PathoSAM + PB-JCI-online, đã
> có bảo đảm coverage) sang một student nhẹ để triển khai thời gian thực/edge, thì:
> (RQ1) Độ chính xác đếm (MAE) của student giảm bao nhiêu so với teacher — đây là câu hỏi kiểu Khan 2025.
> (RQ2) **Coverage của khoảng dự đoán conformal khi áp trực tiếp lên student (không recalibrate) có còn
>   giữ đúng mức 1−α đã cam kết không, hay bị vỡ (under-coverage)?** — đây là câu hỏi CHƯA ai trả lời
>   (theo G4).
> (RQ3) Nếu vỡ, cơ chế nào gây ra (student mất "dark knowledge" về độ khó của từng ảnh → σ_k trong cấu
>   trúc Poisson-Binomial của bạn bị méo → nonconformity score lệch phân phối so với lúc calibrate)?
> (RQ4) Có thể thiết kế mục tiêu distillation (ví dụ: ép student học không chỉ density map/point map mà
>   còn học **cấu trúc bất định per-instance** mà teacher tạo ra, thay vì chỉ học điểm dự đoán) để giữ
>   coverage mà không cần recalibrate lại từ đầu trên device edge hay không?

**Vì sao đây là hướng mạnh cho Q1, xét trên 4 tiêu chí bạn quan tâm:**
1. **Novelty có bằng chứng (không phải cảm tính):** khớp trực tiếp với khe hở G3+G4 — hai khe hở được
   xác nhận bằng đọc trực tiếp nhiều bài, không chỉ suy đoán.
2. **Khớp năng lực + hạ tầng sẵn có:** tái dùng gần như toàn bộ pipeline PB-JCI-online (backbone SAM3/
   PathoSAM, module conformal, dataset MoNuSAC/PanNuke/NuInsSeg/CoNSeP, script vast.ai) — chỉ cần thêm
   bước huấn luyện student + đánh giá coverage-transfer. Rủi ro triển khai thấp hơn nhiều so với bắt đầu
   một hướng hoàn toàn mới.
3. **Nối mạch câu chuyện luận văn:** đúng như cô nói — "counting với nhiều cách tiếp cận" — đây là
   "paper 2" tự nhiên nối "paper 1" (PB-JCI-online), cùng framing "định lượng bất định có bảo đảm cho
   đếm", chỉ thêm trục "dưới nén mô hình" thay vì "dưới distribution shift". Câu chuyện tổng thể của
   luận văn sẽ mạch lạc: *"đếm có bảo đảm — dưới shift (paper 1), dưới nén để triển khai thực tế
   (paper 2)"*.
4. **Compute khả thi với vast.ai, không giới hạn deadline:** distill student nhẹ từ SAM3/PathoSAM trên
   patch bệnh học không đòi hỏi tài nguyên khủng như train foundation model từ đầu — nằm trong tầm một
   vài GPU thuê theo giờ, và bạn đã có sẵn script multi-seed để chạy nghiêm túc (nhiều seed, nhiều
   dataset) giống phong cách bài Khan 2025 (3 student × 2 teacher × 6 dataset) — bạn có thể làm tương tự
   nhưng thêm trục coverage.

**Thiết kế thực nghiệm sơ bộ (không phải cam kết cuối cùng, chỉ để hình dung quy mô):**
- Teacher: SAM3 và/hoặc PathoSAM (đã có, đã LoRA fine-tune) — tái dùng nguyên trạng.
- Student: 2–3 kiến trúc nhẹ (vd. một CNN nhỏ kiểu MCNN/LCDnet như Khan 2025 để so sánh trực tiếp được
  với bài cô đưa, cộng thêm 1 backbone nhẹ hiện đại hơn như MobileViT/TinyViT/RepViT — các tên này đã
  xuất hiện thật trong EfficientSAM3, không phải tôi tự chọn ngẫu nhiên).
- Dataset: giữ nguyên bộ đã dùng cho PB-JCI-online (MoNuSAC, PanNuke, NuInsSeg, CoNSeP) để so sánh dọc
  được với paper 1; cân nhắc thêm 1 dataset ngoài pathology (vd. một dataset đếm đám đông/xe) để chứng
  minh phương pháp không chỉ đúng cho một domain — giống cách Khan 2025 dùng 6 dataset đa domain để tăng
  sức thuyết phục.
- Quy trình đánh giá 3 tầng: (1) MAE đếm chuẩn (so trực tiếp với Khan 2025 làm baseline phương pháp);
  (2) coverage/Winkler của conformal-trên-teacher (đã có từ paper 1) **áp thẳng lên student mà không
  recalibrate** — đo độ vỡ coverage; (3) coverage sau khi recalibrate lại (split conformal chuẩn) trên
  student — để tách bạch "vỡ vì distillation" và "vỡ vì đổi model nói chung".
- Baseline bắt buộc phải có trong bảng so sánh: (a) Khan et al. 2025 (φ_i-weighted KD, adapt sang
  pathology counting để so công bằng), (b) KD chuẩn không trọng số (Hinton), (c) student train from
  scratch không distill, (d) AdaConG-style conformal-guided training (adapt từ classification sang
  regression — đây cũng là đóng góp phụ nếu làm được, vì AdaConG tự nhận chưa làm phần này).

### 5.2. Thí nghiệm chẩn đoán bắt buộc làm TRƯỚC khi cam kết vào hướng 5.1

Trước khi đầu tư thiết kế phương pháp phức tạp, cần một thí nghiệm nhỏ, nhanh, rẻ để biết **có hiện
tượng cần giải quyết hay không**:
1. Lấy một student baseline đơn giản nhất có thể (KD chuẩn kiểu Hinton, không cần trọng số φ_i).
2. Distill từ SAM3/PathoSAM sang student trên 1 dataset (vd. NuInsSeg, nhỏ nhất, rẻ nhất để thử nhanh).
3. Áp calibration set + conformal quantile **đã tính từ teacher** (paper 1) thẳng lên student, đo
   coverage thực tế trên test set.
4. Nếu coverage **vẫn giữ đúng mức 1−α** → hiện tượng "vỡ coverage do distillation" **không xảy ra**
   (hoặc yếu) → hướng 5.1 cần đổi khung: chuyển thành bài **thực nghiệm/chẩn đoán** ("chúng tôi chỉ ra
   rằng distillation, khi làm đúng cách X, bảo toàn coverage — đây là phát hiện thực nghiệm hữu ích cho
   triển khai lâm sàng") thay vì bài **phương pháp mới** (đề xuất kỹ thuật ép bảo toàn coverage). Cả hai
   khung đều có thể là bài Q1 tốt, nhưng đóng góp khác nhau — **phải chạy thí nghiệm này trước khi viết
   đề cương/abstract**, không suy đoán trước.
5. Nếu coverage vỡ rõ rệt (under-coverage đáng kể) → đây là bằng chứng thực nghiệm mạnh cho motivation,
   và bước tiếp theo là chẩn đoán *tại sao* (RQ3) trước khi thiết kế cách sửa (RQ4).

### 5.3. Hướng thay thế / bổ sung (nếu 5.1 sau khi chạy thử không đủ tín hiệu, hoặc muốn làm song song)

**Hướng thay thế A — Distill SAM3 cho đếm mở lớp (class-agnostic), giữ khả năng zero-shot:**
Nhắm thẳng vào G2: distill SAM3 (hoặc kết hợp SAM3 + text encoder) sang một student nhẹ **chuyên cho
đếm** (không phải segmentation chung chung như EfficientSAM3), đánh giá trên FSC-147/CARPK để so sánh
được với dòng CAC hiện có, và so sánh với hướng "kiến trúc nhẹ từ đầu" (MambaCount/RT-Counter) để trả
lời câu hỏi "distill từ teacher mạnh có giữ generalization tốt hơn train nhỏ từ đầu không". Ít gắn với
conformal/UQ hơn hướng 5.1, nhưng tái dùng trực tiếp pipeline SAM3 LoRA đã có, kỹ thuật triển khai đơn
giản hơn. Phù hợp nếu muốn một hướng "chắc ăn" hơn về mặt kỹ thuật (rủi ro thấp) để làm song song.

**Hướng thay thế B — Benchmark/chẩn đoán đa domain (không giới hạn pathology):**
Mở rộng thí nghiệm chẩn đoán ở 5.2 thành một nghiên cứu có hệ thống trên **nhiều domain đếm** (đám đông,
xe, tế bào) để trả lời câu hỏi tổng quát hơn: "KD có bảo toàn calibration cho bài toán đếm nói chung
không, hay chỉ đặc thù pathology?" — đóng góp dạng benchmark/empirical-study (giống tinh thần "Revisiting
foundation models for cell instance segmentation" nhưng cho trục distillation×calibration thay vì trục
foundation-model-comparison). Rủi ro: đóng góp dạng benchmark thường khó vào venue Q1 hàng đầu hơn đóng
góp phương pháp mới, trừ khi phát hiện thực sự bất ngờ/sâu.

### 5.4. Điều KHÔNG nên làm (để tránh lặp lại rủi ro đã ghi trong `PHAN_TICH_LITERATURE_DOI_THU.md`)

- Đừng khoe "đếm chính xác hơn SOTA" — không phải trọng tâm, và Khan 2025 + CellGenNet + EfficientSAM3
  đều đã có model mạnh hơn về độ chính xác thuần túy. Trọng tâm là **coverage/calibration được bảo toàn
  dưới nén model**, không phải MAE thấp nhất.
- Đừng nhận "trọng số φ_i theo độ tin cậy teacher" (Eq. 6-9 của Khan 2025) là ý tưởng của mình nếu tái
  dùng — đây là kỹ thuật đã có, trích dẫn rõ, chỉ dùng làm baseline hoặc điểm khởi đầu.
- Đừng nhận "conformal cho đếm" hay "joint coverage" là mới — đã tự thống nhất trong file cũ là các kỹ
  thuật nền đã có (Barber 2023, v.v.), paper 2 (hướng 5.1) thừa hưởng đúng các giới hạn/quy ước novelty
  đã thống nhất ở `PHAN_TICH_LITERATURE_DOI_THU.md`.

---

## 6. Việc cần làm tiếp theo (checklist)

- [ ] Chạy thí nghiệm chẩn đoán 5.2 trên NuInsSeg (rẻ, nhanh) — đây là bước quyết định khung bài.
- [ ] Đọc toàn văn 3 bài đang ở trạng thái 🔴/🟡 quan trọng nhất trước khi viết bất kỳ tuyên bố novelty
      nào dựa vào chúng: **DHMoE** (Cluster Computing, cần đăng nhập Springer hoặc xin qua thư viện
      trường), **"unified KD pretraining framework" pathology FM** (PubMed 40897898, cần bản PDF đầy đủ),
      **D2PT** (IEICE, kiểm tra có open access không).
- [ ] Xác nhận lại T2ICount và "Bootstrapping MLLM for CAC" (2602.12774) có thực sự dùng distillation
      loss tường minh hay chỉ pseudo-labeling/prompting — hiện chưa xác nhận được (mục 3.2).
- [ ] Kiểm tra SAM3 trên FSC-147 (chưa tìm thấy số liệu trong rà soát này) — nếu chưa ai công bố, đây có
      thể là một baseline cần tự chạy cho hướng 5.3.
- [ ] Trước khi nộp đề cương chính thức, chạy lại rà soát novelty qua Google Scholar/Semantic Scholar/
      Connected Papers cho cụm từ khoá: "conformal knowledge distillation counting", "calibration-aware
      distillation regression", "coverage preserving model compression" — để chắc chắn không có bài mới
      hơn lọt qua WebSearch.

---

## 7. Nhật ký truy vấn đã chạy (để tái lập/kiểm tra, và để biết giới hạn của rà soát này)

```
knowledge distillation crowd counting 2025
knowledge distillation class-agnostic counting 2025
distill SAM Segment Anything Model counting lightweight 2025
knowledge distillation cell nuclei counting pathology 2025
CLIP vision language model distillation object counting 2025
arxiv 2026 knowledge distillation counting
SAM3 Segment Anything Model 3 counting concept prompting Meta 2025
self-distillation online mutual learning object counting 2025
few-shot object counting knowledge distillation exemplar 2025 2026
"knowledge distillation" counting uncertainty quantification calibration
trustworthy crowd counting distillation hierarchical mixture of experts edge cluster computing 2025
density map distillation transformer point counting 2025 arxiv
knowledge distillation object counting remote sensing drone lightweight 2025
knowledge distillation fruit counting agriculture plant counting 2025
conformal prediction knowledge distillation student uncertainty guarantee 2025
blood cell counting knowledge distillation microscopy lightweight model
open vocabulary counting distillation foundation model edge deployment 2025 2026
"SAM3" distillation counting nuclei cell segmentation 2026
knowledge distillation domain generalization counting cross-dataset shift 2025
SAM3 evaluation counting benchmark FSC-147 zero-shot 2025 2026
"referring expression counting" OR "text-guided counting" knowledge distillation lightweight 2025 2026
"DHMoE" OR "distillation hierarchical mixture of experts" crowd counting vehicle datasets MAE
SPPNet one-prompt-all-nuclei distilled SAM lightweight nuclear segmentation
```
**Giới hạn đã biết:** WebSearch trả về snippet đôi khi do AI tóm tắt lại (đã bắt được ít nhất một trường
hợp cần đối chiếu lại bằng WebFetch — mục CellGenNet/NuInsSeg ban đầu snippet mô tả hơi khác so với đọc
full text, tuy sau khi đối chiếu thì khớp). Vì vậy **mọi khẳng định "quan trọng" trong file này đều dựa
trên bài đã 🟢 mở trực tiếp**; các bài 🟡/🔴 chỉ dùng để vẽ bức tranh tổng thể, không dùng làm căn cứ
novelty khi viết bài chính thức.

---

## 8. Nguồn tham khảo đầy đủ (kèm trạng thái xác minh)

**Bài cô giao**
- 🟢 Khan, Menouar, Hamila, Abu-Dayya (2025). *Crowd counting at the edge using weighted knowledge
  distillation.* Scientific Reports 15:11932. https://www.nature.com/articles/s41598-025-90750-5

**KD cho đếm đám đông/xe cộ**
- 🟡 D2PT: Density to Point Transformer with KD for Crowd Counting and Localization (IEICE Trans. Inf.,
  2025). https://www.jstage.jst.go.jp/article/transinf/E108.D/2/E108.D_2024EDL8067/_article
- 🟡 Towards trustworthy crowd counting by distillation hierarchical mixture of experts (DHMoE),
  Cluster Computing 08/2025. https://link.springer.com/article/10.1007/s10586-025-05226-y
- 🟡 Shen et al., A lightweight object counting network based on density map knowledge distillation,
  IEEE TCSVT 2024. https://doi.org/10.1109/TCSVT.2024.3469933
- 🟡 Remote Sensing Object Counting with Online Knowledge Learning, arXiv 2303.10318.
  https://arxiv.org/abs/2303.10318

**Class-agnostic / open-vocabulary counting hiệu quả**
- 🟢 A Survey on Class-Agnostic Counting, arXiv 2501.19184. https://arxiv.org/pdf/2501.19184
- 🟡 MambaCount, arXiv 2606.17650. https://arxiv.org/pdf/2606.17650
- 🟡 RT-Counter, arXiv 2606.17561. https://arxiv.org/pdf/2606.17561
- 🔴 Bootstrapping MLLM for Weakly-Supervised CAC, arXiv 2602.12774 (chưa đọc được nội dung)
- 🔴 T2ICount, CVPR 2025 (403 khi fetch trực tiếp). https://arxiv.org/pdf/2502.20625

**Distill foundation model (SAM/SAM3)**
- 🟢 EfficientSAM3, arXiv 2511.15833. https://arxiv.org/html/2511.15833v1
- 🟢 SAM 3: Segment Anything with Concepts, arXiv 2511.16719 (xác nhận tồn tại, chưa đọc full text).
  https://arxiv.org/pdf/2511.16719
- 🟡 TinySAM (AAAI, trước 2025). https://ojs.aaai.org/index.php/AAAI/article/view/34255
- 🟡 On Efficient Variants of SAM: A Survey, arXiv 2410.04960. https://arxiv.org/html/2410.04960v4

**Distillation cho tế bào/nhân (pathology)**
- 🟢 CellGenNet, arXiv 2511.15054. https://arxiv.org/html/2511.15054
- 🟡 SPPNet, arXiv 2308.12231 (trước 2025, tiền lệ kỹ thuật). https://arxiv.org/abs/2308.12231
- 🟢 DCSNet, arXiv 2505.09334 (không phải bài toán đếm — loại trừ). https://arxiv.org/pdf/2505.09334
- 🔴 Unified KD pretraining pathology foundation model, PubMed 40897898 (chưa đọc được nội dung).
  https://pubmed.ncbi.nlm.nih.gov/40897898/
- 🟢 Revisiting foundation models for cell instance segmentation, arXiv 2603.17845 (MIDL 2026).
  https://arxiv.org/abs/2603.17845

**Distillation + Uncertainty/Conformal**
- 🟢 AdaConG, arXiv 2502.16736. https://arxiv.org/html/2502.16736v4
- 🟢 Trust the Uncertain Teacher (CUD), arXiv 2602.12687. https://arxiv.org/abs/2602.12687
- 🟡 GNN's Uncertainty Quantification using Self-Distillation, arXiv 2506.20046.
  https://arxiv.org/pdf/2506.20046
- 🟡 Uncertainty-Aware Dual-Student KD, arXiv 2511.18826. https://arxiv.org/abs/2511.18826

**Nền tảng lý thuyết KD (đã biết từ bài cô đưa, không tìm lại)**
- Hinton, Vinyals, Dean (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
- Romero et al. (2014). FitNets. arXiv:1412.6550.
- Yim et al. (2017). A Gift from Knowledge Distillation (FSP/Gram matrix). CVPR 2017.
- A Comprehensive Survey on Knowledge Distillation, arXiv 2503.12067 (03/2025).
  https://arxiv.org/pdf/2503.12067 🟡 (chỉ đọc được metadata, không đọc được nội dung chi tiết phần
  "regression/open challenges")

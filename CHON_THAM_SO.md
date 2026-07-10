# CÁCH CHỌN CÁC THAM SỐ / CON SỐ TRONG ĐỀ TÀI

> Tài liệu giải trình: mỗi tham số được chọn bằng **(E) thực nghiệm** (có bảng/sweep), **(C) quy ước chuẩn** (theo literature, không cần thực nghiệm), hay **(Đ) định nghĩa** (không phải lựa chọn). Dùng để trả lời câu hỏi "vì sao số này?" khi bảo vệ.

---

## A. Conformal core

| Tham số | Giá trị | Loại | Lý do |
|---|---|:--:|---|
| **α (mức ý nghĩa)** | 0.1 → coverage 90% | C | Quy ước chuẩn của conformal (Vovk 2005, Angelopoulos-Bates 2023). 90% là mức danh nghĩa phổ biến nhất trong y tế; không tuned. Đổi α chỉ dịch điểm vận hành, không đổi kết luận so sánh. |
| **Mức phân vị** | $\lceil (n+1)(1-\alpha)\rceil/n$, method="higher" | Đ | Công thức conformal quantile hữu hạn-mẫu chuẩn (hiệu chỉnh $+1$ cho finite-sample validity). Không phải lựa chọn — là định nghĩa để bảo đảm $\ge 1-\alpha$. |
| **ε (ổn định số học)** | $10^{-6}$ trong $\sigma_k=\sqrt{\mathrm{Var}+\varepsilon}$ | Đ | Chỉ chống chia 0 khi $\mathrm{Var}\approx 0$ (ảnh rất ít nhân). KHÔNG phải hyperparameter tuned: $10^{-6}\ll$ mọi $\sigma_k$ thực (cỡ đơn vị) nên không ảnh hưởng kết quả; bất kỳ giá trị $\le 10^{-4}$ cho cùng số. |
| **Điểm bất tương hợp** | $R_k=|N_k^{gt}-\mathbb E[N_k]|/\sigma_k$ | Đ | Chuẩn hóa theo σ là dạng "studentized residual" chuẩn; chia σ để khoảng tự co giãn theo độ khó ảnh — hệ quả trực tiếp của mô hình Poisson-Binomial, không phải số tuned. |
| **Joint statistic** | $S=\max_k R_k$ | Đ/E | Chọn max (thay vì sum/Bonferroni) là **thiết kế** để có joint coverage; ưu thế so với Bonferroni được **đo** (Bảng 9b: max-statistic hẹp hơn Bonferroni 31%; ~1.2× vs 1.8× marginal). |

---

## B. Online windowed recalibration

| Tham số | Giá trị | Loại | Lý do |
|---|---|:--:|---|
| **W (cửa sổ PB-JCI)** | **300** | **E** | **Sweep trực tiếp — Bảng 9c** (PathoSAM→NuInsSeg): W=100→87.5%/60.5; 200→84.6%/56.3; **300→81.8%/53.1**; 500→77.1%/47.4. Window nhỏ thích nghi nhanh (coverage cao) nhưng rộng; lớn thì ngược lại. W=300 = điểm cân bằng coverage/width. Chính tradeoff này là **động lực cho cửa sổ thích nghi** (không phải cố định một W). |
| **detector window** | 100 | C/E | Cửa sổ trượt tính median nonconformity cho `RollingShiftDetector`. 100 đủ dài để median ổn định (giảm nhiễu), đủ ngắn để bắt shift trong vài chục mẫu. Bền với lựa chọn này: Bảng 8h cho thấy coverage bất biến quanh ngưỡng; cơ chế không nhạy với window detector trong khoảng hợp lý. |
| **local window** | 100 | C | Cửa sổ tính `local_coverage_stats` (coverage cục bộ để chẩn đoán). Chỉ dùng để báo cáo/giám sát, không vào vòng quyết định chính → không nhạy. |

---

## C. Adaptive window (Adaptive PB-JCI Online)

| Tham số | Giá trị | Loại | Lý do |
|---|---|:--:|---|
| **cov_win (cửa sổ đo coverage gần đây)** | 50 | C | Trung bình coverage trên 50 mẫu gần nhất để quyết định co/giãn. 50 đủ để ước coverage ổn định quanh mục tiêu 0.9 (kỳ vọng ~45/50 phủ) mà vẫn phản ứng nhanh. Nhỏ hơn → ước nhiễu; lớn hơn → phản ứng chậm. |
| **target (ngưỡng coverage)** | 0.90 | Đ | Bằng đúng $1-\alpha$ — không phải số riêng, kế thừa từ α. |
| **hệ số co** | ×0.9 mỗi bước khi under | C | Co hình học chậm (10%/bước) để tránh giật; từ 300 về tối thiểu 40 mất ~19 bước → mượt, không sốc. |
| **hệ số giãn** | ×1.05 khi over (coverage > target+0.03) | C | Giãn chậm hơn co (5% vs 10%) → ưu tiên giữ coverage (an toàn) hơn là siết width. Biên +0.03 tạo vùng chết (dead-band) tránh dao động quanh mục tiêu. |
| **w_min / w_max** | 40 / 300 | C | Trần = W gốc (300); sàn 40 đủ điểm để quantile 90% còn nghĩa (cần ≥~10 điểm cho phân vị 0.9 ổn định; 40 dư an toàn). |

> **Ghi chú:** các hệ số adaptive (0.9 / 1.05 / cov_win 50 / dead-band 0.03) là **lựa chọn thiết kế hợp lý theo nguyên tắc** (co nhanh-giãn chậm để thiên về an toàn), không sweep từng giá trị. Kết quả tổng (Bảng 8f, Winkler tốt nhất) là bằng chứng cấu hình này hoạt động; chưa tuyên bố là tối ưu — đó là lý do đề tài không "khoe" width mà định vị đóng góp ở conditional validity.

---

## D. Detector-flush (biến thể)

| Tham số | Giá trị | Loại | Lý do |
|---|---|:--:|---|
| **flush threshold** | **0.5** | **E** | **Ablation trực tiếp — Bảng 8h(i)**: coverage 88.5–88.7% **bất biến** với ngưỡng 0.2→1.0; nhưng false-alarm trên stream KHÔNG shift giảm mạnh theo ngưỡng: 0.2→15.4% (quá nhạy), 0.35→1.5%, **0.5→0.1%**, 0.7→0%. Chọn 0.5 = điểm cân bằng: phát hiện tức thì shift thật (flush ngay vị trí 0) mà gần như không báo nhầm. |
| **baseline detector** | median(cal scores) + 1e-6 | Đ | Mốc so sánh = median nonconformity trên calibration; +1e-6 chống chia 0. Tín hiệu = $(\text{median gần đây} - \text{baseline})/\text{baseline}$, clip [0,1]. |

---

## E. Baselines (để so công bằng)

| Tham số | Giá trị | Loại | Lý do |
|---|---|:--:|---|
| **ACI γ (learning rate)** | 0.05 | C | Giá trị mặc định trong Gibbs-Candès 2021. Dùng đúng default để so công bằng, không tuning chống baseline. |
| **SA-ACI γ₀/γ_max/λ** | 0.05 / 0.15 / 3.0 | C | Mở rộng ACI (gamma tăng theo độ shift δ). SA-ACI là **kết quả âm** (≈ACI) nên các số này không quan trọng cho kết luận; giữ để minh bạch ablation. |
| **NexCP ρ (decay)** | 0.99 | C | Trọng số suy giảm hình học theo Barber et al. 2023; 0.99 là decay chậm điển hình cho stream cỡ vài trăm. |
| **COP η (learning rate)** | sweep 0.05→5.0, báo η=5 | E | **Cho COP cơ hội tốt nhất**: sweep learning-rate và lấy run có coverage gần 90% nhất (chuẩn "strong baseline"). Ngay cả η cực đại COP vẫn 87.9%<90% → kết luận trung thực, không dìm baseline. λ=1, w=100 đúng theo paper COP. |
| **COP λ, w** | 1.0, 100 | C | Đúng cấu hình trong paper COP (ICLR 2026); không đổi. |

---

## F. Predictor nền

| Tham số | Giá trị | Loại | Lý do |
|---|---|:--:|---|
| **LoRA rank** | 16 | C | Rank phổ biến cho LoRA (8–32); 16 cân bằng dung lượng/chi phí. Áp lên 2 lớp FFN (linear1, linear2) của decoder → 442,368 tham số (0.05%). |
| **TypeHead** | 256→128→5, 33,664 tham số | C | MLP nhỏ phân loại 5 loại từ embedding; kiến trúc tối giản đủ đạt macro-acc 80.5% (Bảng 2b). |
| **Resolution** | 1008 (RoPE 72×72) | C | Độ phân giải native của SAM3. |
| **3 prompt** | Medical / LLM-gen / Generic | E | Tái hiện đúng protocol Kong 2025 để đối chứng (Bảng 1); không phải tuning. |

---

## G. Giao thức thực nghiệm

| Tham số | Giá trị | Loại | Lý do |
|---|---|:--:|---|
| **Số seed (Phase C)** | model [42,100,200] × cal ×5 = **15 run** | C | Hai chiều bất định (model-seed + cal-seed); 15 run đủ cho CI (mean±std) ổn định mà chi phí GPU chấp nhận được. |
| **Số stream seed (cross-dataset)** | 5 | C | 5 hoán vị thứ tự stream cho CI; đủ vì cross-dataset là single fold. |
| **Reference shift detection** | Fold 1, N=200 | C | Mẫu tham chiếu cố định để đo δ; N=200 đủ ước MMD²/W1/Energy ổn định (Bảng 3 std nhỏ). |
| **Holdout** | Fold 3 | C | Tách hẳn fold test khỏi train/calibration → tránh rò rỉ; nghiêm ngặt theo chuẩn PanNuke. |
| **Loại 494 ảnh colon (PathoSAM)** | — | E | PathoSAM train có Lizard (chứa colon trùng PanNuke) → loại 494 ảnh colon để Fold-3 **sạch tuyệt đối** (2228 ảnh), tránh leakage. Bắt buộc, không tùy chọn. |
| **N_test** | SAM3 1361 / PathoSAM 1114, 2228 | Đ | Hệ quả của split đôi Fold-3 (SAM3) và lọc sạch (PathoSAM); không phải lựa chọn. |

---

## H. Thước đo

| Đại lượng | Công thức | Loại | Lý do |
|---|---|:--:|---|
| **Coverage** | tỉ lệ khoảng phủ giá trị thật | Đ | Mục tiêu trực tiếp (≥90%). |
| **Width** | $U-L$ trung bình | Đ | Độ rộng khoảng; KHÔNG đọc một mình (xem Winkler). |
| **Winkler / Interval score** | $(U-L)+\frac{2}{\alpha}\big[(L-y)^+ + (y-U)^+\big]$ | Đ | Proper scoring rule chuẩn cho prediction interval. Hệ số $2/\alpha=20$ (với α=0.1) là **định nghĩa**, không phải tuned: nó là đạo hàm của pinball loss tại hai biên → phạt miss đúng mức để thước đo "proper". Thấp = tốt. |
| **W@90% (matched-coverage width)** | width sau khi scale khoảng để coverage = đúng 90% | E | Phép kiểm để tách "hiệu quả thật" khỏi "đạt coverage nhờ rộng". Kết quả: online/flush/adaptive ~bằng nhau → chứng minh **width không phải đóng góp**, đóng góp là conditional validity. |

---

## I. Các số bị LOẠI sau thực nghiệm (minh bạch)

| Thử nghiệm | Kết quả | Quyết định |
|---|---|---|
| **Router theo loại shift** (fast/slow detector w20/w100, τ_div=0.15, τ_fast=0.25, τ_slow=0.15) | Không hơn adaptive đơn lẻ trên mọi metric (Winkler, W@90, conditional) | **Loại** → adaptive một cơ chế là đủ. Báo cáo như negative result. |
| **SA-ACI** (γ tăng theo δ) | ≈ ACI, không cải thiện | **Loại khỏi main**, giữ làm ablation. |
| **Fallback-multiplier / Hybrid** (cơ chế C/D ở Bảng 8f) | Đạt coverage nhưng Winkler kém hơn adaptive | **Không chọn** — adaptive Winkler tốt nhất. |

---

*Mọi con số "E" tái lập bằng code trong `kaggle/` (đặc biệt `kaggle/sam3_pathosam_winkler.ipynb` và `kaggle/vast/pathosam_router_shift.py`). Số "C/Đ" theo literature/định nghĩa, không cần thực nghiệm.*

# Thiết kế Paper 2 (bản mới) — Count-only counter chống CROWDING bằng module + loss mới

> Bản phác trên giấy để soi (2026-07-22). Lõi = **module mới + loss mới cho ĐẾM (accuracy)**,
> hướng mới KHÔNG phải UQ. Paper 1 (adaptive-window) + UQ = phụ. Chưa code.

## 1. Vấn đề (đã kiểm chứng bằng data)
Đếm nhân bằng density-map (μ = Σ density). Ở **mô xếp dày** (lympho/tạo máu — NuInsSeg thymus/lách/tủy,
GT 250–370), density **bão hoà/nhoè**: nhân chồng lấp gộp thành mảng → tổng density **sai** (đếm thiếu ở
cực dày, đếm dư ở dày vừa). Chỉ có **nhãn count mức-ảnh** (không mask/dot) → không có tín hiệu không gian
để học tách. Đây là chỗ MAE dồn (per_organ_error: thymus MAE 58, spleen 36, femur 33).

## 2. Nguyên lý mới (linh hồn) — ĐẾM là đại lượng BẢO TOÀN
Một bộ đếm đúng phải thoả 2 bất biến vật lý — mà crowding **phá vỡ**:
- **Cộng tính theo phân vùng:** chia ảnh thành các ô → **Σ đếm(ô) = đếm(ảnh)**.
- **Bất biến theo tỉ lệ:** phóng to/thu nhỏ (trong giới hạn phân giải) → đếm **không đổi**.

Vùng dày là nơi model **vi phạm** 2 bất biến này (bão hoà). ⟹ **Ép model tuân thủ bảo toàn = trực tiếp
chữa crowding**, mà **chỉ cần nhãn count** (self-supervised về mặt không gian). Đây là ẩn dụ "co giãn cửa
sổ" của cô áp vào **accuracy** (không phải UQ).

## 3. MODULE MỚI — Stain-Guided Multi-Scale Density (SG-MSD)
Đầu density hiện tại thay bằng:
1. **Stain prior (không nhãn):** stain-deconvolution (Macenko/Vahadane, per-ảnh → tự thích nghi nhuộm) tách
   **kênh hematoxylin** = "khối lượng nhân". Tín hiệu này **bất biến với độ dày** (nhân bắt tím dù dày hay thưa)
   và **phổ quát mọi H&E** → dùng làm **cổng chú ý** điều biến feature density: dồn density vào vùng nhân thật,
   né stroma/hồng-cầu (eosin).
2. **Nhánh đa tỉ lệ có cổng:** vài nhánh dilated (thị trường nhỏ↔lớn). **Cổng = mức crowding cục bộ ước từ
   hematoxylin**: vùng dày → nhánh phân giải mịn (tách nhân sát nhau); vùng thưa → nhánh ngữ cảnh rộng.
   (Khác crowd-counting multi-column ở chỗ **cổng bằng stain**, và ở bối cảnh count-only histopath.)

→ Output: density map + (nhánh σ Poisson giữ nguyên, phụ).

## 4. LOSS MỚI — Scale-Tiling Count Consistency (STC²) ★ ngôi sao
Ngoài L_count = |μ − GT| thường lệ, thêm 2 hạng **self-supervised (chỉ dùng GT count mức-ảnh)**:

- **(a) Cộng-tính tiling:** cắt ảnh thành K ô, **resize mỗi ô về full-res** rồi cho qua model → đếm(ô_i).
  - `L_tile = | Σ_i đếm(ô_i) − GT |`  (mỗi ô đếm ở phân giải cao hơn → tách nhân dày tốt hơn → tổng kéo
    ước lượng "ảnh-dày" lên, phá bão hoà).
  - `L_consist = | đếm(ảnh) − Σ_i đếm(ô_i) |`  (ép ảnh-nguyên khớp tổng-các-ô → bất biến phân vùng).
- **(b) Bất biến tỉ lệ:** cho ảnh qua ở 2 mức scale s∈{1, 0.5…}; `L_scale = | đếm(s·ảnh) − đếm(ảnh) |`.

**Vì sao chữa crowding:** ở vùng dày, đếm(ảnh) bão hoà < Σđếm(ô); các hạng này **kéo đếm(ảnh) lên đúng**,
đồng thời dạy model **đếm nhất quán qua độ-dày** → robust cả crowding lẫn scale-gap (MoNuSAC).
**Vì sao count-only:** không cần mask — chỉ cần GT count + phép cắt/resize (miễn phí).
**Bonus giám sát:** L_tile biến 1 nhãn-ảnh thành **nhiều ràng buộc cục bộ** → như "disaggregation" đa-vùng,
giúp học đếm cục bộ từ nhãn tổng.

## 5. Vì sao TỔNG QUÁT (chống overfit thiết kế)
- Bảo toàn đếm + bất biến tỉ lệ + hematoxylin = **quy luật đúng ở MỌI dataset H&E**, không phải con số NuInsSeg.
- Stain-deconv **per-ảnh** tự thích nghi chênh nhuộm giữa lab.
- **Protocol dev-freeze-test:** thiết kế+chỉnh trên NuInsSeg → KHOÁ → test PanNuke/CryoNuSeg/**Lizard (dày thật)**
  không sửa. + **scale stress-test** (resize tạo độ-dày) trên MỌI dataset. Scope = **H&E** (nói thẳng, IHC=future).

## 6. Mới so với ai (đã lit-check 2023–26)
- Ye 2025 (PR, count-only cell): họ ra **mask** IHC bằng superpixel; mình ra **đếm+density H&E chống crowding**. Khác.
- WaveSeg-UNet / HA2PNet (crowding nhân): **cần MASK** (segmentation+watershed); mình **count-only**. Khác.
- Crowd counting multi-scale (DADNet/MRCNet): scale-aware có, nhưng **có dot annotation** + không stain + không
  bảo-toàn-tiling làm loss count-only histopath. ⟹ tổ hợp **[STC² count-only] + [stain-gated multi-scale] + [histopath crowding]** = khe hở.
- ⚠️ PHẢI lit-check kỹ: "self-supervised scale/tiling consistency" trong crowd counting đã có phần nào →
  định vị rõ điểm khác (count-only + stain + histopath crowding), không overclaim.

## 7. Kế hoạch validate
1. Dev trên NuInsSeg: SG-MSD+STC² vs baseline count-only (efflite0 0.925) — cải thiện ở **mô dày** (per_organ MAE thymus/spleen/femur).
2. Khoá → test PanNuke/CryoNuSeg/**Lizard** không sửa (generality).
3. **Scale stress-test**: đường cong đếm theo scale — chứng minh bất biến tỉ lệ (so baseline gãy như MoNuSAC).
4. Ablation: bỏ stain-gate / bỏ STC² / bỏ multi-scale → tách đóng góp từng phần.
5. Phụ: UQ + adaptive-window Paper 1 (coverage bám regime mật độ) — 1 mục nhỏ.

## 8. Rủi ro (honest)
- **Nhân ở biên ô** bị cắt đôi → nhiễu L_tile. Xử: ô **chồng lấp** + trung bình, hoặc trọng số mềm ở biên.
- STC² có thể chỉ **regularize** chứ không nâng accuracy → phải thử sớm trên NuInsSeg (đo ở mô dày).
- Lizard = colon (đơn mô) → generality vẫn hạn chế về mô; nói thẳng.
- Stain-deconv lỗi ở ảnh nhuộm lệch nặng → cần kiểm robustness.

## 9. Câu contribution (nháp)
> *"A count-only histopathology cell counter robust to nuclear crowding: a **stain-gated multi-scale density
> module** + a **scale–tiling count-consistency loss** that enforce count conservation using only image-level
> labels, improving counting in densely-packed lymphoid/hematopoietic tissue where density counters saturate —
> validated cross-dataset and under controlled density stress. (Calibrated uncertainty via adaptive-online
> windows = secondary.)"*

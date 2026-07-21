# Runbook — đóng 3 critique reviewer (C1 / A-1 / B)

Chạy trên **Kaggle** (GPU + dataset PACT 5-seed `hipinhththu/sam3-r2-nuinsseg-seeds` + teacher pkl).
Bash trên Mac KHÔNG có PACT preds/CUDA → dán lệnh + output về đây.

Chuẩn bị (đầu session):
```bash
cd /kaggle/working && rm -rf repo && git clone <repo> repo && cd repo/distillation_counting
python prep_nuinsseg_as_pannuke.py --root /kaggle/input/datasets/ipateam/nuinsseg   # nếu train baseline
PKL="/kaggle/input/sam3-r2-nuinsseg-seeds/*.pkl"
TCH="/kaggle/input/<teacher-ds>/pathosam_nuinsseg_preds.pkl"
```

---

## C1 — lỗi theo tầng mật độ (local đã validate teacher: bin Thấp MAE 2.95 / MAPE 37.2%)
```bash
python stratified_error.py --pkl_glob "$PKL" --name PACT
python stratified_error.py --teacher_pkl "$TCH" --name "PathoSAM teacher"   # đối chiếu
```
→ Điền **Bảng phân tích lỗi** (§ mới): Thấp(1-20)/TB(21-50)/Cao(>50). Kỳ vọng: MAPE PACT dồn ở bin
Thấp (mẫu số nhỏ), MAE/R² ổn giữa/cao → biến MAPE-47.6% từ "điểm yếu" thành "artifact của metric".

## B — significance Bảng 1 (PACT vs teacher, paired per-ảnh)
```bash
python significance_counting.py --pkl_glob "$PKL" --teacher_pkl "$TCH" --name_vs teacher
```
→ Điền vào caption Bảng 1: "ΔMAE = −X.X (95%CI […]), paired-Wilcoxon p=… ***". Đơn vị lặp = ẢNH
(không per-seed). ±sd 5-seed ĐÃ CÓ trong Bảng 1 (0.786±0.052…). Cần bổ sung heavy-net thì pkl phải
có key `names` (hiện chưa) → nếu thiếu, chỉ report vs teacher (là claim chính).

## A-1 — baseline KHÁC kiến trúc, count-only (không teacher) — control diệt phản biện "chỉ nhờ in-domain"
Codebase đã đủ: `--backbone` timm + `--w_density 0` (bỏ distill) + `--use_gt_density` (chỉ để nạp
field density, KHÔNG dùng khi w_density=0) + count-loss mang tải.
```bash
# EfficientNet-Lite0 (~họ MobileNet), count-only, 5-fold leak-free (giống bảng chính):
python distill_student_r2.py --dataset nuinsseg --kfold 5 \
  --backbone efficientnet_lite0 --use_gt_density \
  --w_density 0 --w_count 1.0 --w_nll 0.01 --epochs 80 \
  --out /kaggle/working/repo/work/baseline_countonly_efflite0.pkl
# rồi eval cùng cách bảng chính:
python compute_r2_counting.py --pkl_glob "/kaggle/working/repo/work/baseline_countonly_efflite0.pkl"
# (tùy chọn) fastvit_t8 để có 2 kiến trúc:
#   --backbone fastvit_t8 ... --out .../baseline_countonly_fastvit.pkl
```
Nếu cần 5-seed cho ±sd: lặp `--seed {0..4}` (kfold đã leak-free; seed đổi để đo dao động).

**Diễn giải TRUNG THỰC (khớp §4.8 supervised≈distilled):**
- Baseline count-only ≈ PACT  → lợi thế Bảng 1 đến từ **in-domain count-label**, KHÔNG phải magic
  kiến trúc PACT. **KHÔNG làm yếu bài** — claim bài là *label-efficiency* (distill = bỏ mask hẳn +
  cho UQ calibrated), không phải "PACT arch thắng". Đây là control biến phản biện A thành bằng chứng.
- Baseline count-only < PACT  → teacher-distill giúp cả accuracy full-budget (mạnh hơn nữa).
Cách nào cũng honest. **Tuyệt đối không viết "PACT architecture superior".**

---
### Backup (BẮT BUỘC khi có kết quả)
- `baseline_countonly_*.pkl` → thêm vào dataset `hipinhththu/sam3-r2-nuinsseg-seeds` (Save Version).
- Ghi số vào PAPER2_MASTER §4.8 (three-way: supervised / distilled / count-only-otherarch).

# RUN_LIST — thí nghiệm strengthening cho Q1 (chạy trên vast)

> Mọi code ĐÃ VIẾT XONG. File này là danh sách lệnh chạy trên vast (RTX 5090) + nơi ghi số vào bảng.
> Chạy theo thứ tự ROI/compute. `REPO=/workspace/sam3_research`, `cd $REPO/distillation_counting`.
> Cần: teacher caches (`work/teacher_density_*.pkl`) — có sẵn từ bảng chính; kaggle dataset `hipinhththu/sam3-paper2-work`.

---

## Ưu tiên 1 — Cross-dataset transfer (#2) — RẺ NHẤT, tín hiệu novelty mạnh
Chứng minh (μ,σ) distilled CHUYỂN sang dataset khác (không dán chết vào 1 dataset).

```bash
# PanNuke -> NuInsSeg
python eval_cross_dataset.py --train_dataset pannuke --test_dataset nuinsseg \
    --exclude_tissue colon --detach_mu --out work/xfer_pannuke2nuinsseg.pkl
python eval_r2_grouped.py --preds work/xfer_pannuke2nuinsseg.pkl --seeds 20 --n_clusters 5

# NuInsSeg -> PanNuke
python eval_cross_dataset.py --train_dataset nuinsseg --test_dataset pannuke \
    --exclude_tissue colon --detach_mu --out work/xfer_nuinsseg2pannuke.pkl
python eval_r2_grouped.py --preds work/xfer_nuinsseg2pannuke.pkl --seeds 20 --n_clusters 3
```
**Ghi:** MAE + Winkler + worst-org của 2 chiều transfer → bảng "Cross-dataset transfer".
**Đọc kỳ vọng:** MAE transfer sẽ CAO hơn in-domain (đương nhiên) nhưng σ vẫn informative → worst-org
vẫn giữ được ở mức khá = bằng chứng (μ,σ) generalize. Trung thực ghi cả khi degrade.

---

## Ưu tiên 2 — UQ floor thành HÀNG THẬT trong bảng chính (#1)
Novelty = learned (μ,σ). Cần 4 baseline UQ chuẩn để bảo vệ. Code: `baselines_uq.py` + `eval_cqr_grouped.py`.
Bắt đầu bằng PanNuke (story chính), rồi NuInsSeg. Ensemble tốn nhất → để cuối, hạ M=3 nếu cần.

### PanNuke (3-fold no-colon: chạy mỗi method 3 lần test_fold=1,2,3 rồi trung bình)
```bash
for F in 1 2 3; do
  # MC-Dropout (σ epistemic)
  python baselines_uq.py --method mcdropout --dataset pannuke --test_fold $F --exclude_tissue colon \
      --T 30 --p_drop 0.2 --out work/uq_mcdropout_pannuke_f$F.pkl
  python eval_r2_grouped.py --preds work/uq_mcdropout_pannuke_f$F.pkl --seeds 20 --n_clusters 3

  # CQR (Romano 2019) -> eval_cqr_grouped
  python baselines_uq.py --method cqr --dataset pannuke --test_fold $F --exclude_tissue colon \
      --out work/uq_cqr_pannuke_f$F.pkl
  python eval_cqr_grouped.py --preds work/uq_cqr_pannuke_f$F.pkl --seeds 20 --n_clusters 3

  # CHDQR (2024) -> eval_cqr_grouped
  python baselines_uq.py --method chdqr --dataset pannuke --test_fold $F --exclude_tissue colon \
      --n_taus 21 --out work/uq_chdqr_pannuke_f$F.pkl
  python eval_cqr_grouped.py --preds work/uq_chdqr_pannuke_f$F.pkl --seeds 20 --n_clusters 3

  # Deep Ensemble (Lakshminarayanan 2017) — M=3 để tiết kiệm (nêu rõ M trong bài)
  python baselines_uq.py --method ensemble --dataset pannuke --test_fold $F --exclude_tissue colon \
      --M 3 --detach_mu --out work/uq_ensemble_pannuke_f$F.pkl
  python eval_r2_grouped.py --preds work/uq_ensemble_pannuke_f$F.pkl --seeds 20 --n_clusters 3
done
```

### NuInsSeg (cross-fit 5-fold: 1 lần/method)
```bash
python baselines_uq.py --method mcdropout --dataset nuinsseg --kfold 5 --T 30 --p_drop 0.2 \
    --out work/uq_mcdropout_nuinsseg.pkl
python eval_r2_grouped.py --preds work/uq_mcdropout_nuinsseg.pkl --seeds 20 --n_clusters 5

python baselines_uq.py --method cqr --dataset nuinsseg --kfold 5 --out work/uq_cqr_nuinsseg.pkl
python eval_cqr_grouped.py --preds work/uq_cqr_nuinsseg.pkl --seeds 20 --n_clusters 5

python baselines_uq.py --method chdqr --dataset nuinsseg --kfold 5 --n_taus 21 --out work/uq_chdqr_nuinsseg.pkl
python eval_cqr_grouped.py --preds work/uq_chdqr_nuinsseg.pkl --seeds 20 --n_clusters 5

python baselines_uq.py --method ensemble --dataset nuinsseg --kfold 5 --M 3 --detach_mu \
    --out work/uq_ensemble_nuinsseg.pkl
python eval_r2_grouped.py --preds work/uq_ensemble_nuinsseg.pkl --seeds 20 --n_clusters 5
```
**Ghi:** 4 dòng (MC-Dropout / Deep Ensemble / CQR / CHDQR) × 2 dataset → thêm vào bảng C2 (mục 8c).
**Đọc kỳ vọng:** R2 (1 model 1.9M) vẫn thắng Winkler/worst-org kể cả Deep Ensemble (3× compute) ⇒ novelty vững.

---

## Ưu tiên 3 — A2 + A6 trên PanNuke (#3) — đối xứng với NuInsSeg
Hiện A2 (ablation σ-mode) + A6 (seed) chỉ đầy đủ ở NuInsSeg. Reviewer sẽ hỏi "giữ trên PanNuke không?".

### A2 — ablation σ-mode (poisson vs raw vs nb), 3-fold no-colon
```bash
for MODE in poisson raw nb; do
  for F in 1 2 3; do
    python distill_student_r2.py --dataset pannuke --test_fold $F --exclude_tissue colon \
        --sigma_mode $MODE --detach_mu --epochs 80 \
        --out work/r2_${MODE}_pannuke_f$F.pkl
  done
  # gộp 3 fold rồi phân tích σ (NLL, z-std, corr) — analysis_sigma nhận 'a,b,c' (phẩy, KHÔNG cách)
  python analysis_sigma.py --name PanNuke-${MODE} \
      --preds work/r2_${MODE}_pannuke_f1.pkl,work/r2_${MODE}_pannuke_f2.pkl,work/r2_${MODE}_pannuke_f3.pkl
done
```
**Ghi:** NLL/z-std 3 mode trên PanNuke → xác nhận poisson-anchor thắng (như NuInsSeg NLL 4.21<4.58).

### A6 — 3 training-seed poisson, worst-org stability, 3-fold
```bash
for S in 0 1 2; do
  for F in 1 2 3; do
    python distill_student_r2.py --dataset pannuke --test_fold $F --exclude_tissue colon \
        --sigma_mode poisson --detach_mu --seed $S --epochs 80 \
        --out work/r2_poisson_s${S}_pannuke_f$F.pkl
  done
  python eval_r2_grouped.py --preds work/r2_poisson_s${S}_pannuke_f1.pkl --seeds 20 --n_clusters 3  # lặp f2,f3 hoặc gộp
done
```
**Ghi:** worst-org PanNuke ± std qua 3 seed → xác nhận 0.90x ổn định (như md 0.906).

---

## Ghi chú compute
- 5090 ~$0.015/hr. Nặng nhất = ensemble (M models × folds) + A2 PanNuke (3 mode × 3 fold = 9 train). Tổng vài giờ.
- Nếu gấp: chạy Ưu tiên 1 + PanNuke của Ưu tiên 2 trước (đủ mạnh cho vòng nộp đầu), phần còn lại bổ sung khi review.
- Sau mỗi mẻ: backup pkl lên kaggle `hipinhththu/sam3-paper2-work` (nhớ -m message; ĐỪNG ghi đè teacher cache).

## Không cần compute
- **#5** mệnh đề lý thuyết detach_mu (mean-variance conflict) — viết trong manuscript.
- **#6** PathoSAM GFLOPs — đo 1 lần (thop) hoặc trích micro_sam nếu có sẵn số.

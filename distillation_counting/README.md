# distillation_counting — Paper 2 (method: model mới + loss mới)

Hướng: **nén foundation model đếm (SAM3/PathoSAM) thành student nhẹ giữ được độ tin cậy** — bài METHOD
(model + loss mới), KHÔNG phải bài chứng minh như paper 1. Chi tiết ý tưởng: `Y_TUONG_PAPER_2.md`.

## File
| File | Vai trò | Trạng thái |
|---|---|---|
| `Y_TUONG_PAPER_2.md` | Ý tưởng, loss PBUD/CCAD, **novelty không tô hồng**, phán đoán Q1 | — |
| `DISTILLATION_COUNTING_GAP_ANALYSIS.md` | Rà soát literature KD-đếm 2025–2026 | — |
| `pbud_losses.py` | **Loss mới** PBUD + CCAD (torch, khả vi) | ✅ test local 16/16 |
| `test_pbud_losses.py` | De-risk toán loss (local, không cần GPU) | ✅ 16/16 pass |
| `distill_student_pbud.py` | **Trainer** distill PathoSAM→student với `--loss kd/pbud/ccad/pbud_ccad` | ✅ logic test synthetic; cần PathoSAM để chạy thật |
| `distill_student_nuinsseg.py` | Base (KD foreground) + student/infer dùng lại | ✅ compile; cần vast |
| `preflight_checks.py` | **TEST RỦI RO trước train** (chạy vast, ~vài phút) | cần vast |
| `eval_coverage_transfer.py` | Thước đo coverage transfer + organ-wise | ✅ self-test dữ liệu thật |
| `s41598-025-90750-5.pdf` | Bài nền (Khan 2025) | — |

## Đã de-risk được ở LOCAL (không cần vast — thật, không bịa)
- `pbud_losses`: **16/16 test pass**. `pb_moments` khớp CHÍNH XÁC numpy `conformal.py` (không lệch định
  nghĩa PB). PBUD + CCAD khả vi, gradient khác 0, số hạng variance đóng góp, edge cases (N=0, K=1, K=4) OK.
- Trainer logic (`distill_student_pbud.train` + ROI-pool per-instance + `student_predict`): chạy end-to-end
  trên **data synthetic** cho cả 4 loss, params hữu hạn, inference ra đúng schema `{scores,probs,K}`.
- `eval_coverage_transfer` self-test trên dữ liệu NuInsSeg thật: coverage 0.898 ≈ 0.90, organ-wise chạy đúng.

## CHƯA test được ở local (cần vast — preflight sẽ bắt lỗi)
PathoSAM load, teacher foreground/instance thật, build cache thật, train ở quy mô thật. → Đó chính là
việc của `preflight_checks.py`.

## Quy trình khi thuê vast (theo thứ tự)
```bash
cd distillation_counting
# 1) TEST RỦI RO trước (rẻ, ~vài phút) — chỉ chạy full khi tất cả PASS
REPO=/workspace/sam3_research python preflight_checks.py

# 2) train 3 student để so sánh loss (KD chuẩn vs PBUD vs PBUD+CCAD)
python distill_student_pbud.py --loss kd        --student_ch 32 --epochs 60 --out work/student_kd.pkl
python distill_student_pbud.py --loss pbud      --student_ch 32 --epochs 60 --out work/student_pbud.pkl
python distill_student_pbud.py --loss pbud_ccad --student_ch 32 --epochs 60 --out work/student_pbudccad.pkl

# 3) đo — CỔNG EFFECT-SIZE: so conditional/transfer coverage KD vs PBUD/CCAD
python eval_coverage_transfer.py --teacher ../data/pathosam_nuinsseg_preds.pkl --student work/student_kd.pkl        --seeds 20 --out cov_kd.json
python eval_coverage_transfer.py --teacher ../data/pathosam_nuinsseg_preds.pkl --student work/student_pbud.pkl      --seeds 20 --out cov_pbud.json
python eval_coverage_transfer.py --teacher ../data/pathosam_nuinsseg_preds.pkl --student work/student_pbudccad.pkl  --seeds 20 --out cov_pbudccad.json
```
Đọc kết quả theo mục 7b của `Y_TUONG_PAPER_2.md`: nhìn **worst-organ coverage** + **transfer (T→S)**, KHÔNG
nhìn marginal-sau-recalibrate. Nếu PBUD/CCAD giữ conditional/transfer coverage nơi KD chuẩn vỡ → có bài.

## Việc còn lại
- [ ] Verify novelty Google Scholar (mục 8 `Y_TUONG_PAPER_2.md`).
- [ ] Chạy preflight + full trên vast.
- [ ] (nếu tín hiệu tốt) mở rộng đa lớp MoNuSAC/PanNuke (cần distill type-head K>1) + nén sweep + nén×shift.
```

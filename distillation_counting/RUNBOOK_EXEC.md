# RUNBOOK — Thứ tự thực hiện phần còn lại (Paper 2)

> Soạn 2026-07-17. Mọi lệnh chạy trên **vast** (paste tay). Bash tool = Mac, KHÔNG chạy vast.
> `$REPO=/workspace/sam3_research`, code ở `$REPO/distillation_counting`, cache ở `$REPO/work`.
> Nguồn chân lý số = [PAPER2_MASTER.md](PAPER2_MASTER.md). Chạy xong bước nào → điền số vào MASTER đúng mục.

---

## 0. KIỂM KÊ PKL — cái gì KHỎI chạy (tránh trùng)

| Artifact | Ở đâu | Trạng thái |
|---|---|---|
| R2 NuInsSeg 5-seed (`_s42..s46`) | kaggle `sam3-r2-nuinsseg-seeds` | ✅ XONG — **đừng chạy lại** |
| R2 PanNuke 3-fold (`_f1/f2/f3_feat`) | kaggle `sam3-paper2-work` | ✅ XONG — **đừng chạy lại** |
| teacher_density_nuinsseg.pkl (305MB, chứa `img,density,gt,organ`) | kaggle `sam3-paper2-work` | ✅ tái dùng |
| teacher_density_pannuke_f123.pkl (3.6GB) | kaggle `sam3-paper2-work` | ✅ tái dùng |
| `_feat` NuInsSeg (thiếu detach_mu) | kaggle | ⚠️ CHỈ dùng feat R2CCP, **KHÔNG lấy số R2** |
| MoNuSAC raw (`monusac_converted.pkl` 180MB) | **local Mac** `data/` | ⬆️ cần upload cho bước 4 |
| **UQ-floor 5-seed** | không đâu có | ❌ phải chạy (bước 1) |
| **KD 5-seed** | không đâu có (pkl mất, `fg` PathoSAM không cache) | ❌ phải chạy (bước 3, cần PathoSAM) |
| **Ablations leak-free** | không đâu có (số cũ single-split) | ❌ phải chạy (bước 2) |

---

## BOOTSTRAP (instance mới — bỏ qua nếu đã dựng)

```bash
# clone repo vào /workspace/sam3_research (giữ work/ nếu đã có cache)
cd /workspace/sam3_research 2>/dev/null || { mkdir -p /workspace/sam3_research && cd /workspace/sam3_research; }
git init -q
git remote add origin https://github.com/ThuHiep/distill_patho_adaptivepbjcionl.git 2>/dev/null
git fetch --depth 1 origin main && git checkout -t origin/main 2>/dev/null || git checkout main
export REPO=/workspace/sam3_research
cd $REPO/distillation_counting
pip install -q conditionalconformal statsmodels tqdm thop kaggle

# kaggle credential: scp kaggle.json từ Mac (chạy Ở MAC):
#   scp -P <PORT> /Users/thuhiep/Documents/kaggle/kaggle.json root@<IP>:~/.kaggle/kaggle.json
#   ssh -p <PORT> root@<IP> 'chmod 600 ~/.kaggle/kaggle.json'

# kéo teacher cache NuInsSeg (KHÔNG kéo PanNuke 3.6GB nếu chỉ làm NuInsSeg)
mkdir -p $REPO/work
kaggle datasets download -d hipinhththu/sam3-paper2-work -p $REPO/work --unzip
ls -la $REPO/work/teacher_density_nuinsseg.pkl   # ~305MB → OK

# (tuỳ chọn) kéo luôn 5 seed R2 để eval/so sánh
kaggle datasets download -d hipinhththu/sam3-r2-nuinsseg-seeds -p $REPO/work --unzip
```

---

## BƯỚC 1 — UQ-floor 5-seed (nhẹ, chỉ cần density cache) → bảo vệ C2

```bash
export REPO=/workspace/sam3_research
cd $REPO/distillation_counting
for M in ensemble cqr chdqr mcdropout; do
  for S in 42 43 44 45 46; do
    echo "==== UQ $M seed $S ===="
    python baselines_uq.py --method $M --dataset nuinsseg --kfold 5 --seed $S \
      --out $REPO/work/uq_${M}_nuinsseg_s${S}.pkl \
      2>&1 | tee -a /tmp/uqfloor_5seed.log
  done
done
echo "DONE UQ-FLOOR"
# gom số:
grep -Ei "worst|winkler|marg|cov" /tmp/uqfloor_5seed.log
```
→ Điền trung bình±sd (5 seed) mỗi method vào **PAPER2_MASTER.md §4.3** (thay số pkl-cũ ⚠️).

---

## BƯỚC 2 — Ablations leak-free (nhẹ, cùng density cache) → bảo vệ N3 + "distillation đáng giá"

> Thay số single-split (leaky) trong §4.8 bằng số cross-fit 5-fold. Seed 42 (thêm seed nếu dư thời gian).

```bash
export REPO=/workspace/sam3_research
cd $REPO/distillation_counting

# 2a. detach_mu OFF (chứng minh coupled NLL hỏng MAE) — canonical ON đã có sẵn (seed pkl)
python distill_student_r2.py --dataset nuinsseg --kfold 5 --seed 42 \
  --out $REPO/work/student_r2_nuinsseg_cv5_nodetach.pkl          # (KHÔNG có --detach_mu)

# 2b. compression sweep ch16 / ch64 (ch32 = canonical đã có)
python distill_student_r2.py --dataset nuinsseg --kfold 5 --detach_mu --seed 42 --student_ch 16 \
  --out $REPO/work/student_r2_nuinsseg_cv5_ch16.pkl
python distill_student_r2.py --dataset nuinsseg --kfold 5 --detach_mu --seed 42 --student_ch 64 \
  --out $REPO/work/student_r2_nuinsseg_cv5_ch64.pkl

# 2c. distilled vs GT-supervised (cùng student, teacher OFF)
python distill_student_r2.py --dataset nuinsseg --kfold 5 --detach_mu --seed 42 --use_gt_density \
  --out $REPO/work/student_r2_nuinsseg_cv5_gtsup.pkl

# EVAL từng cái (NuInsSeg = cluster, n_clusters=5):
for P in nodetach ch16 ch64 gtsup; do
  echo "==== ablation $P ===="
  python eval_r2_grouped.py --preds $REPO/work/student_r2_nuinsseg_cv5_${P}.pkl \
    --pit_scheme cluster --n_clusters 5 2>&1 | tee -a /tmp/ablation.log
done
grep -Ei "worst|winkler|mae" /tmp/ablation.log
```
→ Điền vào **§4.8** (thay 3 dòng số single-split).

---

## BƯỚC 3 — KD 5-seed + per-image significance (NẶNG, cần PathoSAM) → bảo vệ C1/N4 (LÕI)

> **Blocker thật:** KD cần `teacher_targets_nuinsseg.pkl = {img, fg, gtbin, gt, organ}` với `fg` = foreground
> soft-map DÀY của PathoSAM. `fg` KHÔNG cache ở đâu → **buộc chạy PathoSAM**. `pathosam_nuinsseg_preds.pkl`
> (chỉ scores tóm tắt) KHÔNG thay được.

### 3.0 Setup PathoSAM env + raw NuInsSeg (một lần — TỰ TÌM từ kaggle/vast/README_pathosam.md)
> Env: micro_sam cần conda (vigra/nifty/elf — KHÔNG pip được) → micromamba env `/workspace/penv`.
> Raw NuInsSeg: có sẵn kaggle public `ipateam/nuinsseg` → **KHỎI upload**. KD trainer tìm ở `$REPO/data/nuinsseg`
> (code `NUINSSEG_CANDS` dòng 50). `distill_student_nuinsseg.py` tự thêm `$REPO/kaggle/vast` vào sys.path (dòng 45-48)
> nên `import pathosam_lib` chạy được khi có biến `REPO`. Model = `vit_l_histopathology` (KHÔNG train trên PanNuke → sạch).
> Disk: env ≈16GB + torch image ≈15GB + NuInsSeg nhỏ → **~50-60GB đủ** (100GB chỉ khi làm PanNuke).

```bash
export REPO=/workspace/sam3_research
cd $REPO/kaggle/vast
bash setup_pathosam_vast.sh                       # tạo /workspace/penv (~5-10 phút)
MM="/workspace/bin/micromamba run -p /workspace/penv"
$MM python -c "import micro_sam, torch; print('micro_sam OK, cuda', torch.cuda.is_available())"

# raw NuInsSeg (public) → đúng path KD trainer mong đợi
kaggle datasets download -d ipateam/nuinsseg --unzip -p $REPO/data/nuinsseg/
ls $REPO/data/nuinsseg | head
```

### 3.1 Chạy KD 5-seed
```bash
export REPO=/workspace/sam3_research
cd $REPO/distillation_counting
MM="/workspace/bin/micromamba run -p /workspace/penv"
# seed 42 (chạy đầu) sẽ build teacher_targets_nuinsseg.pkl = chạy PathoSAM 1 LẦN (~5-10 phút);
# seed 43-46 TÁI DÙNG cache đó (code dòng 155-157) → nhanh, KHỎI PathoSAM lại.
for S in 42 43 44 45 46; do
  echo "==== KD seed $S ===="
  REPO=$REPO $MM python distill_student_nuinsseg.py --dataset nuinsseg --kfold 5 \
    --lambda_kd 1.0 --seed $S \
    --out $REPO/work/student_kd_nuinsseg_cv5_s${S}.pkl \
    2>&1 | tee -a /tmp/kd_5seed.log
done
echo "DONE KD"
# backup teacher_targets cache NGAY (đắt, đừng mất lần nữa):
kaggle datasets version -p $REPO/work -m "add teacher_targets + KD 5-seed" 2>/dev/null || echo "tạo dataset mới nếu cần"
```

### 3.2 Per-image paired significance (thay p=1.9e−6 pseudoreplication)
```bash
# so R2 vs KD trên CÙNG seed, replication = ẢNH test (không phải seed)
for S in 42 43 44 45 46; do
  echo "==== per-image test seed $S ===="
  python eval_r2_grouped.py \
    --preds $REPO/work/student_r2_nuinsseg_cv5_poisson_s${S}.pkl \
    --kd    $REPO/work/student_kd_nuinsseg_cv5_s${S}.pkl \
    --per_image_test --pit_scheme cluster --n_clusters 5 \
    2>&1 | tee -a /tmp/per_image_test.log
done
grep -Ei "wilcoxon|p=|p-value|boot|CI" /tmp/per_image_test.log
```
→ Điền cột KD (verified) + p-value đúng vào **§4.1** và **N4 (§3)**.

---

## BƯỚC 4 — MoNuSAC UQ-transfer (dataset 3) → bảo vệ N5 + bề rộng

```bash
# 4a. upload data từ MAC (chạy Ở MAC):
#   scp -P <PORT> /Users/thuhiep/Documents/1LUANVAN/counting/sam3_research/data/monusac_converted.pkl \
#       root@<IP>:/workspace/sam3_research/data/monusac_converted.pkl

# 4b. transfer PanNuke → MoNuSAC (kiểm tra tên script + flag trước khi chạy)
export REPO=/workspace/sam3_research
cd $REPO/distillation_counting
ls eval_cross_dataset.py eval_coverage_transfer.py 2>/dev/null   # xác nhận script tồn tại
# → soạn lệnh transfer đúng theo argparse script tìm được (chưa cố định ở đây)
```
> ⚠️ Lệnh 4b để **rỗng có chủ đích** — cần xác nhận script transfer (`eval_cross_dataset.py`?) + cách nạp
> `monusac_converted.pkl` trước khi cố định, tránh bịa flag.

---

## BƯỚC 5 — Gom số → cập nhật PAPER2_MASTER.md

| Bước | Số ra | Mục MASTER cập nhật |
|---|---|---|
| 1 UQ-floor | worst-org/Winkler ×4 method (mean±sd 5 seed) | §4.3 (thay số ⚠️ pkl-cũ) |
| 2 Ablations | detach/ch16/ch64/gtsup: MAE+worst-org | §4.8 (thay 3 dòng single-split) |
| 3 KD + sig | cột KD verified + p-value per-image | §4.1, §3-N4, §0 (đổi ❌→✅) |
| 4 MoNuSAC | transfer worst-org/coverage | §4.4 (thêm dataset 3) |

---

## BƯỚC 6 — Backup (sau MỖI bước quan trọng, ĐỪNG mất lần nữa)

```bash
# code + md → GitHub (chạy Ở MAC hoặc vast)
cd $REPO && git add -A && git commit -m "Paper 2: <mô tả>" && git push

# pkl mới → kaggle (version dataset)
cd $REPO/work
kaggle datasets version -p . -m "add <artifact>" -d   # -d nếu dataset đã tồn tại, hoặc tạo mới
```

---

## THỨ TỰ THỰC DỤNG (đề nghị)

1. **Bước 1 (UQ-floor)** — đang/đã chạy, nhẹ.
2. **Bước 2 (Ablations)** — nhẹ, cùng cache, làm ngay sau.
3. **Bước 3 (KD)** — nặng (PathoSAM); lỗ LÕI phải bịt. Dựng env rồi chạy.
4. **Bước 4 (MoNuSAC)** — bề rộng, sau cùng của phần chạy.
5. → **Viết manuscript** khi §4.1/§4.3/§4.8 đã "✅ verified" hết.

**Quyết định treo:** C3 (giữ CondConf/PCP làm trục conformal-comparison hay bỏ cho gọn tiêu điểm) — chốt trước khi viết.

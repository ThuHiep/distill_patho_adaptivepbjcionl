# Kế hoạch bảng LIGHTWEIGHT COMPARISON (fair, dựa số thật) — 2026-07-16

> Thay bảng heavy-net OOD cũ (unfair) bằng **bảng công bằng**. Thiết kế theo phản hồi chi tiết của user.
> Nguyên tắc: **không đoán** (đo params/GMACs bằng cùng script), **fair** (cùng input/GT-rule), **official code** (không retrain ai).

## Bằng chứng nền (đã verify, KHÔNG đoán)
- **NuLite** checkpoint official = **trained on WHOLE PanNuke**, không per-fold (README repo). Tên official = **NuLite-H/M/T** (KHÔNG phải S/M/H như abstract). Nhẹ nhất = **NuLite-T** (Zenodo 13272655).
- **CellViT** checkpoint = "trained on 90% of all folds", cũng KHÔNG per-fold.
- ⇒ **So fair IN-DOMAIN PanNuke với model published = BẤT KHẢ THI** (không ai có per-fold → leak). Đường fair = **all-OOD** (mọi model train PanNuke → test dataset KHÁC) + in-domain chỉ so với **baseline mình tự kiểm soát**.

## Protocol CÔNG BẰNG (bắt buộc, áp cho MỌI model)
1. **Cùng input set**: cùng danh sách ảnh, cùng exclusion rule, cùng xử lý ambiguous mask.
2. **Cùng GT-count rule**: NuInsSeg `len(unique(mask))−bg`; MoNuSAC `counts.sum()` (4 lớp). Quyết định ambiguous-area 1 lần, áp nhất quán.
3. **Native input mỗi model, ghi rõ**: student 256, CellViT-SAM-H/NuLite 1024, CellViT-256 256. KHÔNG để model này tile, model kia resize mà không kiểm tra ảnh hưởng → ghi cột resolution.
4. **Count = official post-proc**: NuLite/CellViT/LKCell = #instance sau post-proc official (`len(instance_types)`); R2 = μ=Σdensity. **KHÔNG tune threshold theo target** (tune = adaptation, không còn zero-shot).
5. **Cùng regime**: MỌI model = PanNuke checkpoint, **KHÔNG target fine-tuning** → thêm cột "Source / Target-FT".
6. **Params/GMACs đo bằng CÙNG script** (`dump_cellvit_counts.py --measure_cost` cho heavy/lightweight; `count_student_cost.py` cho student) — không lấy số từ paper.

## BẢNG A — Zero-shot count transfer (trained on PanNuke, test OOD)
| Model | Source | Target-FT | Params↓ | GMACs↓ | NuInsSeg MAE↓ | MoNuSAC MAE↓ | RMSE↓ | MAPE↓ | Bias | Latency | Count UQ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **R2 student (ours)** | PanNuke | Không | 1.935M | 10.49 | 44.68 | TBD | … | … | … | 1.87ms | **✓** |
| **NuLite-T** | PanNuke | Không | đo | đo | TBD | TBD | … | … | … | đo | ✗ |
| **CellViT-256** | PanNuke | Không | đo | đo | TBD | TBD | … | … | … | đo | ✗ |
| LKCell-L | PanNuke | Không | 163.84M* | 47.86* | 16.54† | TBD | … | … | … | đo | ✗ |
| CellViT-SAM-H | PanNuke | Không | 699.74M* | 214.33* | 24.24† | TBD | … | … | … | đo | ✗ |

`*` params/GMACs cũ từ paper → **đo lại bằng --measure_cost** cho nhất quán.
`†` số NuInsSeg cũ (Bước 2) — **PHẢI re-verify** cùng GT-rule/preprocessing trước khi ghép (footnote user). Nếu protocol khác → chạy lại.

## BẢNG B — R2 OOD uncertainty transfer (riêng R2, vì baseline không có UQ)
| Source→Target | Calibration | MAE↓ | Coverage↑ | Winkler↓ | Worst-organ↑ | #Under↓ |
|---|---|---|---|---|---|---|
| PanNuke→NuInsSeg | source-only | 44.68 | … | … | … | … |
| PanNuke→NuInsSeg | small target-cal | 44.68 | ~0.90 | 214.83 | 0.685 | 4/27 |
| PanNuke→MoNuSAC | source-only | TBD | … | … | … | … |
| PanNuke→MoNuSAC | small target-cal | TBD | … | … | … | … |
(số small-target-cal NuInsSeg lấy từ 8c-bis; source-only = KHÔNG recalibrate → coverage vỡ, ghi trung thực.)

## Điều bảng NÀY trả lời / KHÔNG trả lời
- ✅ Trả lời: **model nào transfer zero-shot tốt nhất từ PanNuke** + **student nhỏ nhất + duy nhất có UQ**.
- ❌ KHÔNG trả lời: nếu train in-domain thì lightweight nào đếm tốt nhất → cần **in-domain NuLite** (retrain per-fold = compute + rủi ro; để làm nếu nhắm Q1 mạnh). In-domain hiện chỉ so **baseline mình kiểm soát** (R2 vs KD vs supervised same-size).
- **Đọc trung thực dự kiến**: count MoNuSAC 2→772 (mean 150) rất cao → student under-predict mạnh (Bias âm lớn) → OOD MAE kém. Bảng fair này **document tradeoff** (nhỏ nhất + UQ, đổi lấy OOD accuracy), KHÔNG phải "student thắng".

## RUNBOOK vast (harness đã viết + compile OK)
```bash
cd $REPO/distillation_counting && git pull
# ---- MoNuSAC prep (nếu chưa có trên vast) ----
python prep_monusac_counts.py --pkl ../data/monusac_converted.pkl --out ../work/monusac_png

# ---- NuLite: clone + tải NuLite-T (Zenodo 13272655) ----
cd $REPO && git clone https://github.com/CosmoIknosLab/NuLite.git
# tải NuLite-T-Weights.pth vào /workspace/ckpt/nulite_t/  (từ zenodo.org/records/13272655)
cd $REPO/distillation_counting

# ---- đo params/GMACs (cùng script) ----
python dump_cellvit_counts.py --nulite --cellvit_dir $REPO/NuLite --ckpt /workspace/ckpt/nulite_t/NuLite-T-Weights.pth \
    --images_dir ../work/nuinsseg_png/images --out_csv /tmp/x.csv --measure_cost --infer_size 256

# ---- dump count NuLite: NuInsSeg + MoNuSAC ----
python dump_cellvit_counts.py --nulite --cellvit_dir $REPO/NuLite --ckpt <NuLite-T.pth> \
    --images_dir ../work/nuinsseg_png/images --out_csv nulite_nuinsseg.csv --mag 40 --infer_size 0
python dump_cellvit_counts.py --nulite --cellvit_dir $REPO/NuLite --ckpt <NuLite-T.pth> \
    --images_dir ../work/monusac_png/images --out_csv nulite_monusac.csv --mag 40 --infer_size 0
python eval_heavy_count.py --gt ../work/nuinsseg_png/gt_counts.csv --preds nulite_nuinsseg.csv --label NuLite-T
python eval_heavy_count.py --gt ../work/monusac_png/gt_counts.csv --preds nulite_monusac.csv --label NuLite-T

# ---- student PanNuke -> MoNuSAC (OOD, ô student bảng A) ----
python eval_cross_dataset.py --train_dataset pannuke --exclude_tissue colon --detach_mu \
    --test_images_dir ../work/monusac_png/images --test_gt_csv ../work/monusac_png/gt_counts.csv \
    --out ../work/xfer_pannuke2monusac.pkl
# MAE in ra; UQ: python eval_r2_grouped.py --preds ../work/xfer_pannuke2monusac.pkl --seeds 20 (organ=_all_ -> chỉ global)
```
CellViT-256: `--ckpt CellViT-256-x40.pth` (KHÔNG --nulite, KHÔNG --lkcell), `--infer_size 256`.

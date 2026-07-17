# PathoSAM 2nd-backbone pipeline (PB-JCI predictor-agnostic)

Goal: reproduce the PB-JCI main table with **PathoSAM** instead of weak SAM3 — proving PB-JCI Online is predictor-agnostic.

⚠️ **CORRECTED 2026-07-17 (verified against arXiv 2502.00408 Table 1, p.17 — italic=train, bold=eval):**
PathoSAM generalist (`vit_l_histopathology`, micro_sam) **TRAINING datasets** = CPM15, CPM17, Lizard, MoNuSeg,
**PanNuke**, PUMA. **EVAL/out-of-domain** = CoNSeP, CryoNuSeg, LyNSec, **NuInsSeg**, IHC-TMA, TNBC.
→ **PanNuke IS in training → PanNuke = LEAKY teacher (in-domain mechanism-check, NOT clean OOD).**
→ **NuInsSeg = clean OOD (not trained) = the leak-free anchor.** Colon still excluded (Lizard-train overlaps PanNuke-colon).
(Previous line here said "not PanNuke / CoNSeP-trained" — WRONG, contradicted the official paper; see memory `pathosam-training-data`.)

Trial verdict (2026-06-03): Fold-3 count MAE **2.88** (vs SAM3 ~15.7), instance-only,
494/2722 colon to exclude → clean Fold-3 = 2228 imgs.

## Why conda + Vast
micro_sam hard-imports `vigra`/`nifty`/`elf` (conda-forge C++ only, NOT PyPI). Kaggle/pip
can't. Use Vast + micromamba (`setup_pathosam_vast.sh`).

## Disk
PanNuke unzipped is ~35GB (≈12GB/fold). Env ≈16GB + PyTorch image ≈15GB.
→ rent **disk ≥ 100GB**. (Or stage: train on Fold1+2, delete, download Fold3, build.)

## Run order (Vast SSH, env at /workspace/penv)

```bash
# 0. one-time: repo + kaggle + conda env  (see README_vast.md for SSH/kaggle.json)
cd /workspace && git clone https://github.com/duonguwu/sam3_research.git
cd sam3_research && pip install -q kaggle && mkdir -p ~/.kaggle
#   scp kaggle.json -> /workspace/, then:
cp /workspace/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
mkdir -p data/pannuke && kaggle datasets download -d hipinhththu/pannuke --unzip -p data/pannuke/
cd kaggle/vast && bash setup_pathosam_vast.sh        # micro_sam conda env

MM="/workspace/bin/micromamba run -p /workspace/penv"

# 1. PROBE (confirm micro_sam AIS shapes before the long runs)
$MM python pathosam_lib.py
#    expect: predictor.features (1,256,64,64), pooled (M,256), s_i in [0,1]

# 2. TRAIN TypeHead on Fold 1+2 (needs fold1+fold2; ~45-60 min GPU)
$MM python run_pathosam_typehead_train.py
#    -> checkpoints/type_head_pathosam.pt  (features cached to work/*.npz)
#    quick retrain only:  $MM python run_pathosam_typehead_train.py --reuse-cache

# 3. BUILD preds on clean Fold 3 x 3 settings (~30-40 min GPU)
$MM python run_pathosam_build_preds.py
#    -> work/pathosam_predictions.pkl   + prints PathoSAM total-count MAE

# 4. CONFORMAL benchmark (CPU, minutes) — the PathoSAM PB-JCI table
$MM python run_pathosam_conformal.py
#    -> work/pathosam_conformal_results.json + printed table

# 5. backup before destroy
kaggle datasets version -p work/ -m "pathosam preds+conformal"   # or scp work/ down
```

## Extra runs (mIoU + cross-dataset) — optional, same env

```bash
# A. PathoSAM segmentation quality on Fold-3 — GPU ~15min. Outputs BOTH:
#    - PER-CLASS semantic mIoU (via TypeHead) = FAIR comparison vs SAM3 ~15%
#    - binary foreground IoU/Dice = raw nucleus separation (different/easier metric)
$MM python run_pathosam_miou.py            # clean 2228 (or --all for 2722)
#    -> work/pathosam_miou.json (macro_per_class_miou + binary_fg_iou)
#    NOTE: only the per-class macro is comparable to SAM3; binary is NOT.

# B. Cross-dataset PathoSAM -> NuInsSeg (total-count K=1). NuInsSeg only (CoNSeP leaky).
kaggle datasets download -d ipateam/nuinsseg --unzip -p data/nuinsseg/
$MM python run_pathosam_nuinsseg.py        # GPU: build NuInsSeg preds pkl
$MM python run_pathosam_crossdataset.py    # CPU: cal PanNuke-clean -> test NuInsSeg
#    -> work/pathosam_nuinsseg_preds.pkl, work/pathosam_crossdataset_results.json
```
Backup these jsons too before destroy.

## Files
- `setup_pathosam_vast.sh` — micromamba env `pathosam` at `-p /workspace/penv`
- `pathosam_lib.py` — load PathoSAM, AIS → per-instance (mask, s_i=fg-prob, pooled feat). `__main__` = probe
- `run_pathosam_typehead_train.py` — TypeHead(256,128,5) on Fold 1+2 (majority-vote GT type)
- `run_pathosam_build_preds.py` — clean Fold-3 × 3 settings → pkl (Phase-C schema)
- `run_pathosam_conformal.py` — 6-method × 4-setting PB-JCI benchmark (reuses conformal.py)

## Cross-dataset note
For PathoSAM, cross-dataset target = **NuInsSeg ONLY** (CoNSeP is in PathoSAM's training
→ leaky). SAM3 keeps both NuInsSeg + CoNSeP.

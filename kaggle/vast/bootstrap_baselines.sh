#!/bin/bash
#
# bootstrap_baselines.sh — Dựng lại TOÀN BỘ pipeline Paper 2 trên MỘT vast instance MỚI (ổ trắng),
# rồi chạy hết 8 baseline UQ. Dùng khi máy cũ mất (không cứu được teacher cache).
#
# LÀM GÌ (tuần tự, tự bỏ qua bước đã xong):
#   1. clone/pull repo
#   2. tải PanNuke (hipinhththu/pannuke) + NuInsSeg (ipateam/nuinsseg) qua kaggle
#   3. precompute counts.npy + XOÁ masks.npy (giải phóng ~23G)
#   4. setup PathoSAM env (micromamba /workspace/penv)  [teacher targets]
#   5. train R2 --dump_feat (PanNuke 3 fold + NuInsSeg cv5)  -> TỰ build teacher cache + pkl có feature
#   6. train KD (mốc chính)
#   7. cài dep baseline + clone repo baseline (pcp / R2CCP / CPCP)
#   8. chạy 8 baseline -> lưu log work/baseline_logs/
#
# TIÊN QUYẾT (làm THỦ CÔNG trước khi chạy):
#   - Instance disk >= 100G (52G TỪNG CHẬT — xem memory vast-pannuke-disk-ops).
#   - Upload kaggle.json -> /workspace/  (script tự copy sang ~/.kaggle).
#   - Repo đã clone ở /workspace/sam3_research (hoặc script tự clone nếu đặt GIT_URL).
#
# Chạy:  bash /workspace/sam3_research/kaggle/vast/bootstrap_baselines.sh
set -e

WORK=/workspace
REPO=$WORK/sam3_research
PENV=$WORK/penv
MM="$WORK/bin/micromamba"
export MAMBA_ROOT_PREFIX=$WORK/micromamba
DC=$REPO/distillation_counting
WK=$REPO/work
GIT_URL="${GIT_URL:-}"   # đặt GIT_URL=... nếu muốn script tự clone
RUN="$MM run -p $PENV"   # chạy trong env PathoSAM (train cần teacher targets)
mkdir -p $WK $WK/baseline_logs

echo "############ [0] kaggle.json ############"
if [ ! -f ~/.kaggle/kaggle.json ]; then
    mkdir -p ~/.kaggle
    cp $WORK/kaggle.json ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
fi
pip install -q kaggle 2>/dev/null || true

echo "############ [1] repo ############"
if [ ! -d "$REPO/.git" ] && [ -n "$GIT_URL" ]; then git clone "$GIT_URL" "$REPO"; fi
cd $REPO && git pull --ff-only 2>/dev/null || true

echo "############ [2] tải data (PanNuke + NuInsSeg) ############"
mkdir -p $REPO/data/pannuke $REPO/data/nuinsseg
if [ ! -e "$REPO/data/pannuke"/*/images.npy ] && [ -z "$(find $REPO/data/pannuke -name images.npy 2>/dev/null | head -1)" ]; then
    cd $REPO/data && kaggle datasets download -d hipinhththu/pannuke --unzip -p pannuke/
fi
if [ -z "$(find $REPO/data/nuinsseg -iname 'tissue images' -type d 2>/dev/null | head -1)" ]; then
    cd $REPO/data && kaggle datasets download -d ipateam/nuinsseg --unzip -p nuinsseg/
fi

echo "############ [3] precompute counts + XOÁ masks.npy (giải phóng đĩa) ############"
python $DC/precompute_pannuke_counts.py --pannuke_root $REPO/data/pannuke --folds 1,2,3 || true
find $REPO/data/pannuke -name masks.npy -delete 2>/dev/null || true
df -h $WORK | tail -1

echo "############ [4] setup PathoSAM env ############"
if [ ! -d "$PENV/conda-meta" ]; then
    bash $REPO/kaggle/vast/setup_pathosam_vast.sh
    $MM install -y -p $PENV -c conda-forge "micro_sam>=1.1" vigra nifty || true
fi

echo "############ [5] train R2 --dump_feat (tự build teacher cache) ############"
for F in 1 2 3; do
  OUT=$WK/student_r2_pannuke_f${F}_nocolon_poisson_feat.pkl
  [ -f "$OUT" ] || $RUN python $DC/distill_student_r2.py --dataset pannuke --pannuke_folds 1,2,3 \
      --test_fold $F --exclude_tissue colon --dump_feat --out "$OUT"
done
OUT=$WK/student_r2_nuinsseg_cv5_poisson_feat.pkl
[ -f "$OUT" ] || $RUN python $DC/distill_student_r2.py --dataset nuinsseg --kfold 5 --dump_feat --out "$OUT"

echo "############ [6] train KD (mốc) ############"
for F in 1 2 3; do
  OUT=$WK/student_kd_pannuke_f${F}_nocolon.pkl
  [ -f "$OUT" ] || $RUN python $DC/distill_student_nuinsseg.py --dataset pannuke --pannuke_folds 1,2,3 \
      --test_fold $F --exclude_tissue colon --out "$OUT"
done
OUT=$WK/student_kd_nuinsseg_cv5.pkl
[ -f "$OUT" ] || $RUN python $DC/distill_student_nuinsseg.py --dataset nuinsseg --kfold 5 --out "$OUT"

echo "############ [7] cài dep baseline + clone repo (chạy baseline ở python BASE image) ############"
pip install -q conditionalconformal statsmodels tqdm pytorch_lightning configargparse 2>&1 | tail -2 || true
cd $DC
[ -d pcp ]   || git clone -q https://github.com/yaozhang24/pcp.git pcp
[ -d R2CCP ] || git clone -q https://github.com/EtashGuha/R2CCP.git R2CCP
[ -d CPCP ]  || git clone -q https://github.com/Cqyiiii/Colorful-Pinball-Conformal-Prediction-CPCP.git CPCP

echo "############ [8] chạy 8 baseline -> log ############"
cd $DC
run_pk () {  # $1=eval script  $2=preds  $3+=extra
  local s=$1 p=$2; shift 2
  echo ">>> $s  $(basename $p)"; python $s --preds "$p" "$@" 2>&1 | tee "$WK/baseline_logs/$(basename $s .py)_$(basename $p .pkl).log" | tail -6
}
for F in 1 2 3; do
  P=$WK/student_r2_pannuke_f${F}_nocolon_poisson_feat.pkl
  KD=$WK/student_kd_pannuke_f${F}_nocolon.pkl
  # (μ,σ) recent: CondConf 2025, PCP 2024
  run_pk eval_condconf_grouped.py "$P" --seeds 10 --min_organ_imgs 10
  run_pk eval_pcp_grouped.py      "$P" --pcp_dir ./pcp --seeds 10 --min_organ_imgs 10
  # feature recent: R2CCP 2024, CPCP 2026
  run_pk eval_r2ccp.py "$P" --r2ccp_dir ./R2CCP --seeds 5 --max_epochs 100 --min_organ_imgs 10
  run_pk eval_cpcp.py  "$P" --cpcp_dir ./CPCP  --seeds 5 --min_organ_imgs 10
  # sàn UQ (train, xem md mục 10 -> baselines_uq.py) — chạy riêng nếu cần
done
# NuInsSeg
P=$WK/student_r2_nuinsseg_cv5_poisson_feat.pkl
run_pk eval_condconf_grouped.py "$P" --seeds 10 --min_organ_imgs 10
run_pk eval_pcp_grouped.py      "$P" --pcp_dir ./pcp --seeds 10 --min_organ_imgs 10
run_pk eval_r2ccp.py "$P" --r2ccp_dir ./R2CCP --seeds 5 --max_epochs 100 --min_organ_imgs 10
run_pk eval_cpcp.py  "$P" --cpcp_dir ./CPCP  --seeds 5 --min_organ_imgs 10

echo "############ DONE — log ở $WK/baseline_logs/ ############"
echo "Sàn UQ (mcdropout/ensemble/cqr/chdqr) + eval R2 gốc: xem md mục 10 (baselines_uq.py)."

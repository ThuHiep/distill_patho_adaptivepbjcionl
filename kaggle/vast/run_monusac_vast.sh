#!/bin/bash
#
# MoNuSAC (K=4) multi-class conformal pipeline tren Vast — dataset da lop SACH cho PathoSAM
# (eval-only, KHONG nam trong 7 dataset train PathoSAM; xac minh o get_generalist_datasets.py).
# Tai su dung CUNG env micromamba voi PathoSAM (setup_pathosam_vast.sh) — micro_sam da co.
#
# CHUAN BI (1 trong 2 cach co data):
#   (A) KHUYEN: upload san file da convert+validate o local:
#         scp data/monusac_converted.pkl  -> /workspace/sam3_research/data/monusac_converted.pkl
#       (173 MB, da PNG-nen, KHONG can svs/openslide tren Vast).
#   (B) Neu pkl thieu, script tu gdown zip goc + convert (can .tif doc duoc; .svs can openslide).
#
# Yeu cau truoc: repo clone o /workspace/sam3_research + da chay setup_pathosam_vast.sh.
# Chay:  bash run_monusac_vast.sh
set -e

WORK=/workspace
REPO=$WORK/sam3_research
MM="$WORK/bin/micromamba"
ENV=$WORK/penv
export MAMBA_ROOT_PREFIX=$WORK/micromamba
PKL=$REPO/data/monusac_converted.pkl
RUN="$MM run -p $ENV python"

echo "=========================================="
echo "MoNuSAC K=4 conformal pipeline (PathoSAM)"
echo "=========================================="

# ---- 0. dam bao co data (uu tien pkl da upload) -----------------------------
if [ -f "$PKL" ]; then
    echo "[0/3] found converted pkl: $PKL ($(du -h "$PKL" | cut -f1))"
else
    echo "[0/3] pkl MISSING -> tu tai + convert MoNuSAC tren Vast (cach B)"
    $MM run -p $ENV pip install -q gdown tifffile
    cd $WORK
    [ -f monusac_train.zip ] || $MM run -p $ENV gdown 1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq -O monusac_train.zip
    rm -rf monusac_raw && $MM run -p $ENV python -c "import zipfile;zipfile.ZipFile('monusac_train.zip').extractall('monusac_raw')"
    mkdir -p $REPO/data
    # converter glob **/*.xml de quy -> truyen thang thu muc goc, du long nong sao cung tim ra
    $RUN $REPO/kaggle/lib/monusac_converter.py "$WORK/monusac_raw" "$PKL"
fi

cd $REPO/kaggle/vast

# ---- 1. TypeHead K=4 tren CAL patients (tach theo patient) -------------------
echo ""
echo "=== [1/3] TypeHead(256,128,4) tren cal patients ==="
$RUN run_monusac_typehead_train.py --epochs 60 --seed 0

# ---- 2. build preds tren TEST patients (TypeHead chua thay -> sach) ----------
echo ""
echo "=== [2/3] build PathoSAM preds tren test patients ==="
$RUN run_monusac_build_preds.py --seed 0
echo ">>> CHU Y: doc dong 'per-class count MAE' o tren. Neu thap bat thuong (~3 nhu"
echo ">>>        PanNuke in-domain) -> nghi TCGA-overlap; neu muc OOD binh thuong -> sach."

# ---- 3. joint conformal K=4 (CPU, vai giay) ---------------------------------
echo ""
echo "=== [3/3] joint conformal K=4 (PB-JCI vs Bonferroni vs Marginal) ==="
$RUN run_monusac_conformal.py

echo ""
echo "=========================================="
echo "DONE. Ket qua:"
echo "  - work/monusac_predictions.pkl   (preds, de backup ve local)"
echo "  - checkpoints/type_head_monusac.pt"
echo "Backup ve local de chay/ve lai bang offline:"
echo "  scp ...:$REPO/work/monusac_predictions.pkl  data/"
echo "=========================================="

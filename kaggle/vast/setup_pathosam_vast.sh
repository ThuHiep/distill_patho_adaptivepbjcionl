#!/bin/bash
#
# Setup PathoSAM (micro_sam + patho_sam) on a Vast.ai instance via micromamba.
#
# WHY conda: micro_sam hard-imports vigra/nifty/elf (C++, conda-forge ONLY — not on
# PyPI). Kaggle/pip cannot install them. micromamba gives us a real conda-forge env
# fast and non-interactively, no full Anaconda.
#
# Env is created at an EXPLICIT prefix ($ENV_PREFIX) via `-p`, NOT `-n NAME`. The Vast
# PyTorch image ships its own conda config (base env at /venv) which hijacks `-n` envs
# to /venv. `-p` pins the location deterministically so the run command always finds it.
#
# Installs: python 3.11, micro_sam (pulls vigra/nifty/elf/torch-em + GPU torch via
# pytorch-gpu), then pip-installs patho_sam from source (--no-deps so it doesn't
# clobber the conda torch). Cleans the pkg cache at the end to reclaim disk.
#
# DISK: the conda env is ~16-18 GB. PanNuke uncompressed is ~35 GB — for the discovery
# trial you only need Fold 3, so delete fold1/fold2 first if disk is tight.
#
# Run AFTER repo clone + PanNuke download.
# Usage:  bash setup_pathosam_vast.sh
set -e

echo "=========================================="
echo "PathoSAM (micro_sam) conda setup — start"
echo "=========================================="

WORK=/workspace
REPO=$WORK/sam3_research
MAMBA_ROOT=$WORK/micromamba
ENV_PREFIX=$WORK/penv          # explicit env location (avoids image's /venv hijack)
MM="$WORK/bin/micromamba"

# ---- 1. install micromamba (static binary, no root deps) -------------------
if [ ! -x "$MM" ]; then
    echo "[1/5] Installing micromamba..."
    mkdir -p $WORK/bin
    cd $WORK
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C $WORK bin/micromamba
else
    echo "[1/5] micromamba already present, skip."
fi
export MAMBA_ROOT_PREFIX=$MAMBA_ROOT

# ---- 2. create env at explicit prefix with micro_sam + GPU torch -----------
if [ ! -d "$ENV_PREFIX/conda-meta" ]; then
    echo "[2/5] Creating conda env at $ENV_PREFIX (micro_sam + pytorch-gpu)... (~5-10 min)"
    $MM create -y -p $ENV_PREFIX -c conda-forge \
        python=3.11 micro_sam pytorch-gpu torchvision
else
    echo "[2/5] Env at $ENV_PREFIX exists, skip create."
fi

# ---- 3. patho_sam from source (no-deps so conda torch stays) ---------------
echo "[3/5] Installing patho_sam from source (--no-deps)..."
$MM run -p $ENV_PREFIX pip install -q --no-deps \
    git+https://github.com/computational-cell-analytics/patho-sam.git

# ---- 4. reclaim disk: drop the downloaded package tarballs -----------------
echo "[4/5] Cleaning conda pkg cache to reclaim disk..."
$MM clean -a -y >/dev/null 2>&1 || true

# ---- 5. verify the conda-only deps actually import -------------------------
echo "[5/5] Verifying imports inside env $ENV_PREFIX..."
$MM run -p $ENV_PREFIX python - <<'PY'
import importlib
mods = ["vigra", "nifty", "elf", "torch_em", "segment_anything",
        "micro_sam", "patho_sam", "torch"]
import torch
for m in mods:
    try:
        importlib.import_module(m); print(f"  import {m:16s} OK")
    except Exception as e:
        print(f"  import {m:16s} FAIL -> {repr(e)[:120]}")
print(f"\n  torch {torch.__version__} | CUDA available = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  WARN: conda landed CPU torch. Trial still works (use a small SUBSET).")
PY

echo ""
echo "=========================================="
echo "PathoSAM env ready. Run the discovery trial:"
echo "  cd $REPO/kaggle/vast"
echo "  $MM run -p $ENV_PREFIX python run_pathosam_trial.py --n 60"
echo "=========================================="

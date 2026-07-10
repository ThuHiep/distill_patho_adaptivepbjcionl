#!/bin/bash

set -e

REPO=/workspace/sam3_research

echo "=========================================="
echo "Vast.ai data download — start"
echo "=========================================="

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found"
    echo "Upload kaggle.json to /workspace/ then run setup_vast.sh first"
    exit 1
fi

echo ""
echo "[1/4] Downloading PanNuke..."
mkdir -p $REPO/data/pannuke
cd $REPO/data
if [ ! -f "pannuke/Fold_1/images.npy" ]; then
    kaggle datasets download -d hipinhththu/pannuke --unzip -p pannuke/
    echo "  PanNuke OK"
else
    echo "  PanNuke already exists, skip"
fi

echo ""
echo "[2/4] Downloading SAM3 native weights..."
mkdir -p $REPO/checkpoints
cd $REPO/checkpoints
if [ ! -f "sam3.pt" ]; then
    kaggle datasets download -d hipinhththu/sam3-native-pt --unzip
    echo "  SAM3 weights OK"
else
    echo "  SAM3 weights already exist, skip"
fi

echo ""
echo "[3/4] Downloading A2 LoRA weights..."
if [ ! -f "sam3_lora_rank16_final.pt" ]; then
    kaggle datasets download -d hipinhththu/phase-a2-lora-weights --unzip
    echo "  A2 LoRA OK"
else
    echo "  A2 LoRA already exists, skip"
fi

echo ""
echo "[4/4] Downloading A3 TypeHead weights..."
if [ ! -f "type_head_final.pt" ]; then
    kaggle datasets download -d hipinhththu/phase-a3-typehead-weights --unzip
    echo "  A3 TypeHead OK"
else
    echo "  A3 TypeHead already exists, skip"
fi

echo ""
echo "=========================================="
echo "Data verification:"
echo "=========================================="
ls -lh $REPO/data/pannuke/Fold_*/images.npy 2>/dev/null | head -5
ls -lh $REPO/checkpoints/*.pt
du -sh $REPO/data $REPO/checkpoints

echo ""
echo "Download DONE. Ready to run training."

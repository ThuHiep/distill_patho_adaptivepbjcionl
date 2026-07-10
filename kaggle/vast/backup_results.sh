#!/bin/bash

set -e

USERNAME="${1:-hipinhththu}"
REPO=/workspace/sam3_research

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle API not setup"
    exit 1
fi

echo "=========================================="
echo "Backup to Kaggle datasets (user: $USERNAME)"
echo "=========================================="

CKPT_DIR=$REPO/checkpoints_multiseed
META_CKPT=$CKPT_DIR/dataset-metadata.json
if [ ! -f "$META_CKPT" ]; then
    cat > $META_CKPT <<EOF
{
  "title": "SAM3 Q1 Multi-seed Checkpoints",
  "id": "$USERNAME/sam3-q1-multiseed-ckpts",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
    echo "[1] Create new dataset: $USERNAME/sam3-q1-multiseed-ckpts"
    kaggle datasets create -p $CKPT_DIR -r zip
else
    echo "[1] Version existing dataset"
    kaggle datasets version -p $CKPT_DIR -m "Q1 multi-seed update $(date +%Y%m%d-%H%M)" -r zip
fi

WORK_DIR=$REPO/work
META_WORK=$WORK_DIR/dataset-metadata.json
if [ ! -f "$META_WORK" ]; then
    cat > $META_WORK <<EOF
{
  "title": "SAM3 Q1 Results",
  "id": "$USERNAME/sam3-q1-results",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
    echo "[2] Create new dataset: $USERNAME/sam3-q1-results"
    kaggle datasets create -p $WORK_DIR -r zip
else
    echo "[2] Version existing dataset"
    kaggle datasets version -p $WORK_DIR -m "Q1 results update $(date +%Y%m%d-%H%M)" -r zip
fi

echo ""
echo "=========================================="
echo "Backup DONE. Verify in Kaggle:"
echo "  https://www.kaggle.com/datasets/$USERNAME/sam3-q1-multiseed-ckpts"
echo "  https://www.kaggle.com/datasets/$USERNAME/sam3-q1-results"
echo "=========================================="
echo ""
echo "Safe to destroy Vast instance now."

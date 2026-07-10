#!/bin/bash

set -e

echo "=========================================="
echo "Vast.ai SAM3 setup — start"
echo "=========================================="

echo ""
echo "[1/6] apt update + build tools..."
apt-get update -qq
apt-get install -qq -y build-essential git wget curl unzip nano htop tmux

WORK="/workspace"
mkdir -p $WORK
cd $WORK
echo "[2/6] Workspace: $WORK"

if [ ! -d "$WORK/sam3_research" ]; then
    echo ""
    echo "[3/6] Cloning repo..."
    git clone https://github.com/duonguwu/sam3_research.git
else
    echo "[3/6] Repo exists, pulling latest..."
    cd $WORK/sam3_research && git pull
fi
cd $WORK/sam3_research

echo ""
echo "[4/6] Installing SAM3 + Python deps..."
pip install --upgrade pip -q
pip install -e sam3 -q
pip install -q scikit-image scikit-learn matplotlib opencv-python \
                pycocotools einops tqdm seaborn pandas

echo ""
echo "[5/6] Kaggle API setup..."
pip install -q kaggle
mkdir -p ~/.kaggle
if [ -f "/workspace/kaggle.json" ]; then
    cp /workspace/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    echo "  Kaggle API configured."
else
    echo "  WARN: /workspace/kaggle.json not found."
    echo "  Upload it via Jupyter Lab (drag-drop to /workspace/)"
    echo "  Then re-run: bash setup_vast.sh"
fi

echo ""
echo "[6/6] Verify environment..."
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "  WARN: no GPU"
echo ""
echo "Python:"
python -c "import sys; print(f'  Python {sys.version.split()[0]}')"
echo ""
echo "PyTorch:"
python -c "import torch; print(f'  Torch {torch.__version__} | CUDA {torch.cuda.is_available()} | Device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo ""
echo "SAM3:"
python -c "from sam3.model_builder import build_sam3_image_model; print('  SAM3 import OK')" || echo "  FAIL: SAM3 import"

echo ""
echo "=========================================="
echo "Setup DONE. Next steps:"
echo "  1. Upload kaggle.json if not yet (drag-drop to /workspace/)"
echo "  2. bash download_data.sh"
echo "  3. python run_a2_multiseed.py  (or other multi-seed script)"
echo "=========================================="

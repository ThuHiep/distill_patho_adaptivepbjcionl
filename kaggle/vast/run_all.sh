#!/bin/bash

set -e
cd /workspace/sam3_research/kaggle/vast

mkdir -p logs

echo "========================================="
echo "SAM3 Q1 Multi-seed Pipeline — start"
echo "========================================="

echo ""
echo "[1/6] A2 LoRA train × 3 seeds..."
python run_a2_multiseed.py 2>&1 | tee logs/a2_train.log

echo ""
echo "[2/6] A2 eval × 3 seeds..."
python run_a2_eval_multiseed.py 2>&1 | tee logs/a2_eval.log

echo ""
echo "[3/6] A3 TypeHead train × 3 seeds..."
python run_a3_multiseed.py 2>&1 | tee logs/a3_train.log

echo ""
echo "[4/6] A3 eval × 3 seeds..."
python run_a3_eval_multiseed.py 2>&1 | tee logs/a3_eval.log

echo ""
echo "[5/6] Phase B shift × 5 seeds..."
python run_phaseB_multiseed.py 2>&1 | tee logs/phaseB.log

echo ""
echo "[6/6] Phase C conformal multi-seed..."
python run_phaseC_multiseed.py 2>&1 | tee logs/phaseC.log

echo ""
echo "========================================="
echo "PIPELINE COMPLETE"
echo "========================================="
ls -lh /workspace/sam3_research/work/*.json
ls -lh /workspace/sam3_research/checkpoints_multiseed/*.pt

echo ""
echo "Next steps:"
echo "  1. Backup results to Kaggle Dataset:"
echo "     cd /workspace/sam3_research"
echo "     kaggle datasets version -p checkpoints_multiseed -m 'Q1 multiseed'"
echo "     kaggle datasets version -p work -m 'Q1 results JSON'"
echo "  2. Download via Jupyter Lab"
echo "  3. DESTROY Vast instance"

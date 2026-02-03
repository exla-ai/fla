#!/bin/bash
# Corrected Pi0.6 training with proper weight loading
#
# ISSUE FIXED: Previous training used gemma3_4b model but loaded pi05_base
# (gemma_2b) weights, causing architecture mismatch. Model trained from
# near-random initialization, resulting in 24% success vs 60% baseline.
#
# This script uses the corrected config (gemma_2b) that matches pi05_base weights.

set -e

CONFIG="pi06_multi"
EXP_NAME="pi06_multi_v2"  # v2 = corrected version
NUM_STEPS=50000

echo "============================================================"
echo "Pi0.6 Training (Corrected)"
echo "============================================================"
echo "Config: ${CONFIG}"
echo "Exp name: ${EXP_NAME}"
echo "Training steps: ${NUM_STEPS}"
echo ""
echo "Key fix: Using gemma_2b architecture to match pi05_base weights"
echo "============================================================"
echo ""

cd /lambda/nfs/arizona/pi-openpi

# Activate environment if needed
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True"

# Start training
echo "Starting training at $(date)"
python scripts/train.py "${CONFIG}" \
    --exp-name="${EXP_NAME}" \
    --overwrite=false 2>&1 | tee "training_${EXP_NAME}.log"

echo ""
echo "Training completed at $(date)"
echo "Checkpoint: ./checkpoints/${CONFIG}/${EXP_NAME}"

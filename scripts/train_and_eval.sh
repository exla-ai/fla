#!/bin/bash
# Complete training and evaluation pipeline for Pi0.6
#
# This script:
# 1. Trains the corrected pi06_multi model
# 2. Waits for completion
# 3. Runs evaluation on both ALOHA tasks
# 4. Saves results summary

set -e

CONFIG="pi06_aloha_sim"
EXP_NAME="pi06_aloha_sim_v1"
CHECKPOINT_BASE="./checkpoints/${CONFIG}/${EXP_NAME}"
LOG_DIR="./logs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

cd /lambda/nfs/arizona/pi-openpi

# Activate environment if needed
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
# Memory optimization for JAX
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl

echo "============================================================" | tee -a "${LOG_DIR}/pipeline.log"
echo "Pi0.6 Training & Evaluation Pipeline" | tee -a "${LOG_DIR}/pipeline.log"
echo "Started at: $(date)" | tee -a "${LOG_DIR}/pipeline.log"
echo "============================================================" | tee -a "${LOG_DIR}/pipeline.log"

# Phase 1: Training
echo "" | tee -a "${LOG_DIR}/pipeline.log"
echo "[Phase 1] Starting training..." | tee -a "${LOG_DIR}/pipeline.log"
echo "Config: ${CONFIG}" | tee -a "${LOG_DIR}/pipeline.log"
echo "Exp name: ${EXP_NAME}" | tee -a "${LOG_DIR}/pipeline.log"
echo "" | tee -a "${LOG_DIR}/pipeline.log"

WANDB_MODE=disabled python scripts/train.py "${CONFIG}" \
    --exp-name="${EXP_NAME}" 2>&1 | tee "${LOG_DIR}/training.log"

# Find the latest checkpoint
LATEST_CKPT=$(ls -d ${CHECKPOINT_BASE}/*/ 2>/dev/null | grep -E '/[0-9]+/$' | sort -t/ -k6 -n | tail -1 | sed 's/\/$//')

if [ -z "${LATEST_CKPT}" ]; then
    echo "ERROR: No checkpoint found!" | tee -a "${LOG_DIR}/pipeline.log"
    exit 1
fi

echo "" | tee -a "${LOG_DIR}/pipeline.log"
echo "Training complete. Latest checkpoint: ${LATEST_CKPT}" | tee -a "${LOG_DIR}/pipeline.log"

# Phase 2: Evaluation
echo "" | tee -a "${LOG_DIR}/pipeline.log"
echo "[Phase 2] Starting evaluation..." | tee -a "${LOG_DIR}/pipeline.log"

# Transfer Cube task
echo "" | tee -a "${LOG_DIR}/pipeline.log"
echo "Evaluating Transfer Cube..." | tee -a "${LOG_DIR}/pipeline.log"
python scripts/evaluate_aloha_sim.py \
    --checkpoint_dir "${LATEST_CKPT}" \
    --config "${CONFIG}" \
    --task gym_aloha/AlohaTransferCube-v0 \
    --num_episodes 50 2>&1 | tee "${LOG_DIR}/eval_transfer_cube.log"

# Insertion task
echo "" | tee -a "${LOG_DIR}/pipeline.log"
echo "Evaluating Insertion..." | tee -a "${LOG_DIR}/pipeline.log"
python scripts/evaluate_aloha_sim.py \
    --checkpoint_dir "${LATEST_CKPT}" \
    --config "${CONFIG}" \
    --task gym_aloha/AlohaInsertion-v0 \
    --num_episodes 50 2>&1 | tee "${LOG_DIR}/eval_insertion.log"

# Phase 3: Generate Summary
echo "" | tee -a "${LOG_DIR}/pipeline.log"
echo "[Phase 3] Generating summary..." | tee -a "${LOG_DIR}/pipeline.log"

cat > "${LOG_DIR}/results_summary.md" << 'SUMMARY_EOF'
# Pi0.6 Multi-Dataset Evaluation Results

## Training Details
- Config: pi06_multi (corrected)
- Architecture: gemma_2b VLM + gemma_300m action expert
- Pretrained weights: pi05_base
- Training: Frozen VLM backbone, trainable action expert

## Evaluation Results

### Transfer Cube (AlohaTransferCube-v0)
SUMMARY_EOF

grep -E "Success Rate|Average Reward|Episodes" "${LOG_DIR}/eval_transfer_cube.log" >> "${LOG_DIR}/results_summary.md" || true

cat >> "${LOG_DIR}/results_summary.md" << 'SUMMARY_EOF'

### Insertion (AlohaInsertion-v0)
SUMMARY_EOF

grep -E "Success Rate|Average Reward|Episodes" "${LOG_DIR}/eval_insertion.log" >> "${LOG_DIR}/results_summary.md" || true

cat >> "${LOG_DIR}/results_summary.md" << 'SUMMARY_EOF'

## Comparison to Baselines
| Method | Transfer Cube | Insertion |
|--------|--------------|-----------|
| BC Baseline | 60% | ? |
| Pi0.6 RECAP (paper) | 85% | ? |
| This model | See above | See above |

## Notes
- Previous training (v1) had architecture mismatch bug
- This run (v2) uses corrected gemma_2b weights
SUMMARY_EOF

echo "" | tee -a "${LOG_DIR}/pipeline.log"
echo "============================================================" | tee -a "${LOG_DIR}/pipeline.log"
echo "Pipeline Complete!" | tee -a "${LOG_DIR}/pipeline.log"
echo "Finished at: $(date)" | tee -a "${LOG_DIR}/pipeline.log"
echo "Results saved to: ${LOG_DIR}/results_summary.md" | tee -a "${LOG_DIR}/pipeline.log"
echo "============================================================" | tee -a "${LOG_DIR}/pipeline.log"

cat "${LOG_DIR}/results_summary.md"

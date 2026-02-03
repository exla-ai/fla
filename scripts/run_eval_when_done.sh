#!/bin/bash
# Wait for training to finish and run evaluation

CHECKPOINT_BASE="./checkpoints/pi06_multi/pi06_multi_v1"
TARGET_STEP="200000"
CONFIG="pi06_multi"

echo "Waiting for training to complete (200k steps)..."

while true; do
    if [ -d "${CHECKPOINT_BASE}/${TARGET_STEP}" ]; then
        echo "Checkpoint ${TARGET_STEP} found!"
        break
    fi
    
    LATEST=$(ls -1 "${CHECKPOINT_BASE}" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
    echo -ne "\rProgress: ${LATEST:-0}/${TARGET_STEP}    "
    sleep 60
done

# Wait for training process to exit
echo "Waiting for training process to finish..."
while pgrep -f "train.py pi06_multi" > /dev/null; do
    sleep 30
done

echo "Training complete! Starting evaluation..."

# Create symlinks for norm_stats in all checkpoints
for ckpt in ${CHECKPOINT_BASE}/*/; do
    if [ ! -f "${ckpt}assets/multi-dataset/norm_stats.json" ]; then
        mkdir -p "${ckpt}assets/multi-dataset"
        ln -sf ../lerobot/aloha_sim_transfer_cube_human/norm_stats.json "${ckpt}assets/multi-dataset/norm_stats.json" 2>/dev/null
    fi
done

# Run evaluation with GPU (should be free now)
export MUJOCO_GL=egl
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

echo ""
echo "========================================"
echo "Running evaluation on ${TARGET_STEP} checkpoint"
echo "========================================"

python scripts/evaluate_aloha_sim.py \
    --checkpoint_dir "${CHECKPOINT_BASE}/${TARGET_STEP}" \
    --config "${CONFIG}" \
    --task gym_aloha/AlohaTransferCube-v0 \
    --num_episodes 50 2>&1 | tee eval_transfer_cube.log

python scripts/evaluate_aloha_sim.py \
    --checkpoint_dir "${CHECKPOINT_BASE}/${TARGET_STEP}" \
    --config "${CONFIG}" \
    --task gym_aloha/AlohaInsertion-v0 \
    --num_episodes 50 2>&1 | tee eval_insertion.log

echo ""
echo "Evaluation complete! Results in eval_*.log"

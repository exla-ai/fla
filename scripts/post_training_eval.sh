#!/bin/bash
# Post-training evaluation script
# Monitors training completion and runs evaluations automatically

set -e

LOG_FILE="/lambda/nfs/arizona/pi-openpi/logs/pi06_multi_training.log"
CHECKPOINT_DIR="/lambda/nfs/arizona/pi-openpi/checkpoints/pi06_multi/pi06_multi_v3"
EVAL_LOG="/lambda/nfs/arizona/pi-openpi/logs/post_training_eval.log"
CONFIG_NAME="pi06_multi"
EXP_NAME="pi06_multi_v3"

echo "$(date): Starting post-training monitor..." | tee -a "$EVAL_LOG"
echo "Monitoring: $LOG_FILE" | tee -a "$EVAL_LOG"
echo "Checkpoint dir: $CHECKPOINT_DIR" | tee -a "$EVAL_LOG"

# Function to check if training is complete
check_training_complete() {
    # Check if the final checkpoint (50000 or 49999) exists
    if [ -d "$CHECKPOINT_DIR/50000" ] || [ -d "$CHECKPOINT_DIR/49999" ]; then
        return 0
    fi

    # Also check if log shows completion
    if grep -q "Waiting for checkpoint manager to finish" "$LOG_FILE" 2>/dev/null; then
        return 0
    fi

    return 1
}

# Wait for training to complete
echo "$(date): Waiting for training to complete..." | tee -a "$EVAL_LOG"
while ! check_training_complete; do
    sleep 60  # Check every minute
done

echo "$(date): Training complete! Starting evaluations..." | tee -a "$EVAL_LOG"

# Give it a moment for checkpoint to fully save
sleep 30

# Find the latest checkpoint
LATEST_CKPT=$(ls -d "$CHECKPOINT_DIR"/[0-9]* 2>/dev/null | sort -t/ -k9 -n | tail -1)
echo "$(date): Using checkpoint: $LATEST_CKPT" | tee -a "$EVAL_LOG"

cd /lambda/nfs/arizona/pi-openpi

# 1. Compute EWC state for continual learning
echo "$(date): Computing EWC state for continual learning..." | tee -a "$EVAL_LOG"
PYTHONPATH=./src python scripts/compute_ewc_state.py "$CONFIG_NAME" --exp-name="$EXP_NAME" >> "$EVAL_LOG" 2>&1 || echo "EWC computation failed, continuing..."

# 2. Run ALOHA sim evaluation
echo "$(date): Starting ALOHA sim evaluation..." | tee -a "$EVAL_LOG"

# Start the policy server in background
echo "$(date): Starting policy server..." | tee -a "$EVAL_LOG"
PYTHONPATH=./src python scripts/serve_policy.py policy:checkpoint \
    --policy.config="$CONFIG_NAME" \
    --policy.dir="$LATEST_CKPT" \
    --port=8000 >> "$EVAL_LOG" 2>&1 &
SERVER_PID=$!
echo "$(date): Policy server started with PID $SERVER_PID" | tee -a "$EVAL_LOG"

# Wait for server to be ready
sleep 60

# Run evaluation script
echo "$(date): Running evaluation..." | tee -a "$EVAL_LOG"
PYTHONPATH=./src python scripts/evaluate_aloha_sim.py \
    --host=localhost \
    --port=8000 \
    --num_episodes=50 >> "$EVAL_LOG" 2>&1 || echo "Evaluation failed"

# Kill the server
echo "$(date): Stopping policy server..." | tee -a "$EVAL_LOG"
kill $SERVER_PID 2>/dev/null || true

echo "$(date): All post-training tasks complete!" | tee -a "$EVAL_LOG"
echo "Results logged to: $EVAL_LOG" | tee -a "$EVAL_LOG"

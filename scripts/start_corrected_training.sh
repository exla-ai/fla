#!/bin/bash
# Wait for current eval to finish, then start corrected training
# Run with: nohup ./scripts/start_corrected_training.sh > training_pipeline.log 2>&1 &

set -e

cd /lambda/nfs/arizona/pi-openpi

echo "============================================================"
echo "Waiting for current evaluation to complete..."
echo "Started at: $(date)"
echo "============================================================"

# Wait for any running python evaluation to complete
while pgrep -f "evaluate_aloha_sim.py" > /dev/null 2>&1; do
    echo "$(date): Eval still running, waiting 30s..."
    sleep 30
done

echo ""
echo "Evaluation complete at $(date)"
echo "Starting corrected training..."
echo ""

# Run the training and eval pipeline
./scripts/train_and_eval.sh

echo ""
echo "Pipeline complete at $(date)"

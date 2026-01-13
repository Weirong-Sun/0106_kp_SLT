#!/bin/bash
# Extract keypoints from all PHOENIX-2014-T videos using MediaPipe (Background)
# This script runs the extraction in the background with nohup

# Configuration
DATASET_PATH="/data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"
OUTPUT_PATH="phoenix_keypoints.pkl"
NUM_WORKERS=4
NUM_GPUS=4

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Log file
LOG_FILE="extract_keypoints_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="extract_keypoints.pid"

echo "=========================================="
echo "PHOENIX-2014-T Keypoint Extraction"
echo "Complete Dataset (Background Mode)"
echo "=========================================="
echo "Dataset Path: $DATASET_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Number of Workers: $NUM_WORKERS"
echo "Number of GPUs: $NUM_GPUS"
echo "Log File: $LOG_FILE"
echo "PID File: $PID_FILE"
echo "=========================================="
echo ""

# Run extraction in background
nohup torchrun --nproc_per_node=$NUM_GPUS \
    data/extract_phoenix_keypoints_distributed.py \
    --dataset_path "$DATASET_PATH" \
    --output_path "$OUTPUT_PATH" \
    --num_workers $NUM_WORKERS \
    > "$LOG_FILE" 2>&1 &

# Save PID
echo $! > "$PID_FILE"
PID=$(cat "$PID_FILE")

echo "Extraction started in background"
echo "Process ID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "To check progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if running:"
echo "  ps -p $PID"
echo ""
echo "To stop:"
echo "  kill $PID"
echo "=========================================="



#!/bin/bash
# Extract keypoints from all PHOENIX-2014-T videos using MediaPipe
# This script processes the complete dataset (no sample limit)

# Configuration
DATASET_PATH="/data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"
OUTPUT_PATH="phoenix_keypoints.pkl"
NUM_WORKERS=4
NUM_GPUS=4

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Log file
LOG_FILE="extract_keypoints_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "PHOENIX-2014-T Keypoint Extraction"
echo "Complete Dataset (No Sample Limit)"
echo "=========================================="
echo "Dataset Path: $DATASET_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Number of Workers: $NUM_WORKERS"
echo "Number of GPUs: $NUM_GPUS"
echo "Log File: $LOG_FILE"
echo "=========================================="
echo ""

# Run extraction
# Note: Removing --max_samples_per_split to process all videos
torchrun --nproc_per_node=$NUM_GPUS \
    data/extract_phoenix_keypoints_distributed.py \
    --dataset_path "$DATASET_PATH" \
    --output_path "$OUTPUT_PATH" \
    --num_workers $NUM_WORKERS \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Extraction completed successfully!"
    echo "Output files:"
    echo "  - ${OUTPUT_PATH%.pkl}.train"
    echo "  - ${OUTPUT_PATH%.pkl}.dev"
    echo "  - ${OUTPUT_PATH%.pkl}.test"
else
    echo "Extraction failed with exit code: $EXIT_CODE"
fi
echo "Log file: $LOG_FILE"
echo "=========================================="

exit $EXIT_CODE






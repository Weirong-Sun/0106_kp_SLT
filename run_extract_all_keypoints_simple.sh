#!/bin/bash
# Simple script to extract keypoints from all PHOENIX-2014-T videos

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 \
    data/extract_phoenix_keypoints_distributed.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints.pkl \
    --num_workers 4






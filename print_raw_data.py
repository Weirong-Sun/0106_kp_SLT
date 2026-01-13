#!/usr/bin/env python
"""
直接打印 phoenix_keypoints.train 中第一条数据的原始内容
"""
import pickle
import sys

try:
    import torch
except ImportError:
    print("Error: torch module required", file=sys.stderr)
    sys.exit(1)

# Load data
with open('phoenix_keypoints.train', 'rb') as f:
    data = pickle.load(f)

# Get first item
first_key = sorted(list(data.keys()))[0]
first_value = data[first_key]

# Print directly
print(first_value)



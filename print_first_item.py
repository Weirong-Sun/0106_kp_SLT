#!/usr/bin/env python
import pickle

with open('phoenix_keypoints.train', 'rb') as f:
    data = pickle.load(f)

first_key = sorted(list(data.keys()))[0]
first_value = data[first_key]

for key, value in first_value.items():
    print(f'{key}:')
    print(repr(value))
    print()




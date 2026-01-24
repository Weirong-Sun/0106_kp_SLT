"""
关键点工具函数
"""
import numpy as np
import torch


def flatten_keypoint_dict(kp_dict):
    """
    将关键点字典展平为 [143, 3] 的数组

    Args:
        kp_dict: 关键点字典，包含 'face', 'left_hand', 'right_hand', 'pose'

    Returns:
        flattened: numpy array of shape [143, 3]
    """
    kp_list = []

    # Face (68 points)
    if kp_dict.get('face') is not None:
        kp_list.append(kp_dict['face'])
    else:
        kp_list.append(np.zeros((68, 3), dtype=np.float32))

    # Left hand (21 points)
    if kp_dict.get('left_hand') is not None:
        kp_list.append(kp_dict['left_hand'])
    else:
        kp_list.append(np.zeros((21, 3), dtype=np.float32))

    # Right hand (21 points)
    if kp_dict.get('right_hand') is not None:
        kp_list.append(kp_dict['right_hand'])
    else:
        kp_list.append(np.zeros((21, 3), dtype=np.float32))

    # Pose (33 points)
    if kp_dict.get('pose') is not None:
        kp_list.append(kp_dict['pose'])
    else:
        kp_list.append(np.zeros((33, 3), dtype=np.float32))

    # Concatenate all keypoints
    flattened = np.concatenate(kp_list, axis=0)  # [143, 3]

    return flattened






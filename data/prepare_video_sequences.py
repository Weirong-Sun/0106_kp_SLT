"""
Prepare video sequences from extracted keypoints for temporal training
Organizes keypoints into video sequences
"""
import pickle
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

def organize_keypoints_by_video(keypoints_data, video_info=None):
    """
    Organize keypoints into video sequences
    
    Args:
        keypoints_data: List of keypoint dictionaries or dict with 'keypoints' key
        video_info: Optional dict mapping frame indices to video IDs
    
    Returns:
        video_sequences: List of video sequences, each is a list of keypoint dicts
    """
    # Handle different input formats
    if isinstance(keypoints_data, dict):
        if 'keypoints' in keypoints_data:
            kp_list = keypoints_data['keypoints']
        else:
            raise ValueError("Dict format should have 'keypoints' key")
    elif isinstance(keypoints_data, list):
        kp_list = keypoints_data
    else:
        raise ValueError("Unknown data format")
    
    # If video_info is provided, group by video ID
    if video_info is not None:
        video_dict = defaultdict(list)
        for idx, kp_dict in enumerate(kp_list):
            if idx in video_info:
                video_id = video_info[idx]
                video_dict[video_id].append(kp_dict)
        
        video_sequences = list(video_dict.values())
    else:
        # If no video info, assume all frames belong to one video
        # Or try to detect video boundaries based on frame gaps
        # For now, treat as single video
        video_sequences = [kp_list]
    
    return video_sequences

def extract_video_id_from_path(image_path):
    """
    Extract video ID from image path
    Assumes structure like: extracted_frames/video_name/frame_001.jpg
    or: extracted_frames/path/to/video_name/frame_001.jpg
    """
    path = Path(image_path)
    # Get parent directory name (video name)
    # If path is: extracted_frames/video_name/frame_001.jpg
    # Then parent is: video_name
    video_id = path.parent.name
    return video_id

def organize_by_video_from_paths(keypoints_data, image_paths):
    """
    Organize keypoints by video using image paths
    
    Args:
        keypoints_data: List of keypoint dictionaries
        image_paths: List of image paths corresponding to keypoints
    
    Returns:
        video_sequences: List of video sequences (list of keypoint dicts)
    """
    video_dict = defaultdict(list)
    
    for kp_dict, img_path in zip(keypoints_data, image_paths):
        video_id = extract_video_id_from_path(img_path)
        video_dict[video_id].append(kp_dict)
    
    return list(video_dict.values())

def detect_video_boundaries(keypoints_data, gap_threshold=10):
    """
    Detect video boundaries based on large gaps in keypoint detection
    
    Args:
        keypoints_data: List of keypoint dictionaries
        gap_threshold: Number of consecutive missing detections to consider as boundary
    
    Returns:
        video_sequences: List of video sequences
    """
    video_sequences = []
    current_sequence = []
    missing_count = 0
    
    for kp_dict in keypoints_data:
        # Check if frame has valid keypoints
        has_valid_kp = False
        for key in ['face', 'left_hand', 'right_hand', 'pose']:
            if kp_dict.get(key) is not None:
                has_valid_kp = True
                break
        
        if has_valid_kp:
            current_sequence.append(kp_dict)
            missing_count = 0
        else:
            missing_count += 1
            if missing_count >= gap_threshold and len(current_sequence) > 0:
                # End current sequence
                video_sequences.append(current_sequence)
                current_sequence = []
                missing_count = 0
    
    # Add last sequence
    if len(current_sequence) > 0:
        video_sequences.append(current_sequence)
    
    return video_sequences

def prepare_sequences_from_directory(keypoints_dir, output_path, min_seq_len=10):
    """
    Prepare video sequences from directory of keypoint files
    
    Args:
        keypoints_dir: Directory containing keypoint pickle files (one per video)
        output_path: Path to save sequences pickle file
        min_seq_len: Minimum sequence length to include
    """
    keypoints_dir = Path(keypoints_dir)
    video_sequences = []
    
    # Find all pickle files
    pickle_files = list(keypoints_dir.glob("*.pkl"))
    
    print(f"Found {len(pickle_files)} keypoint files")
    
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different formats
            if isinstance(data, dict) and 'keypoints' in data:
                kp_list = data['keypoints']
            elif isinstance(data, list):
                kp_list = data
            else:
                print(f"Warning: Unknown format in {pkl_file}, skipping")
                continue
            
            # Filter sequences by minimum length
            if len(kp_list) >= min_seq_len:
                video_sequences.append(kp_list)
                print(f"Added sequence from {pkl_file.name}: {len(kp_list)} frames")
            else:
                print(f"Skipped {pkl_file.name}: too short ({len(kp_list)} frames)")
        
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
            continue
    
    # Save sequences
    with open(output_path, 'wb') as f:
        pickle.dump({'sequences': video_sequences}, f)
    
    print(f"\nSaved {len(video_sequences)} video sequences to {output_path}")
    print(f"Total frames: {sum(len(seq) for seq in video_sequences)}")
    
    return video_sequences

def extract_video_id_from_path(image_path):
    """
    Extract video ID from image path
    Assumes structure like: extracted_frames/video_name/frame_001.jpg
    or: extracted_frames/path/to/video_name/frame_001.jpg
    """
    path = Path(image_path)
    # Get parent directory name (video name)
    # If path is: extracted_frames/video_name/frame_001.jpg
    # Then parent is: video_name
    video_id = path.parent.name
    return video_id

def organize_by_video_from_paths(keypoints_data, image_paths):
    """
    Organize keypoints by video using image paths
    
    Args:
        keypoints_data: List of keypoint dictionaries
        image_paths: List of image paths corresponding to keypoints
    
    Returns:
        video_sequences: Dict mapping video_id to list of keypoint dicts
    """
    video_dict = defaultdict(list)
    
    for kp_dict, img_path in zip(keypoints_data, image_paths):
        video_id = extract_video_id_from_path(img_path)
        video_dict[video_id].append(kp_dict)
    
    return dict(video_dict)

def prepare_sequences_from_single_file(keypoints_path, output_path, min_seq_len=10, gap_threshold=10, use_path_info=True):
    """
    Prepare video sequences from a single keypoint file
    
    Args:
        keypoints_path: Path to keypoints pickle file
        output_path: Path to save sequences pickle file
        min_seq_len: Minimum sequence length to include
        gap_threshold: Gap threshold for detecting video boundaries (if not using path info)
        use_path_info: If True, use image_paths to organize by video; otherwise detect boundaries
    """
    print(f"Loading keypoints from {keypoints_path}...")
    with open(keypoints_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract keypoints and paths
    if isinstance(data, dict) and 'keypoints' in data:
        kp_list = data['keypoints']
        image_paths = data.get('image_paths', None)
    elif isinstance(data, list):
        kp_list = data
        image_paths = None
    else:
        raise ValueError("Unknown data format")
    
    # Organize by video
    if use_path_info and image_paths is not None and len(image_paths) == len(kp_list):
        print(f"Organizing {len(kp_list)} frames by video using image paths...")
        video_dict = organize_by_video_from_paths(kp_list, image_paths)
        video_sequences = list(video_dict.values())
        print(f"Found {len(video_sequences)} videos from path information")
        
        # Print video info
        for video_id, seq in video_dict.items():
            print(f"  Video '{video_id}': {len(seq)} frames")
    else:
        if use_path_info:
            print("Warning: image_paths not available or length mismatch, using boundary detection instead")
        print(f"Detecting video boundaries (gap_threshold={gap_threshold})...")
        video_sequences = detect_video_boundaries(kp_list, gap_threshold=gap_threshold)
        print(f"Found {len(video_sequences)} sequences from boundary detection")
    
    # Filter by minimum length
    filtered_sequences = [seq for seq in video_sequences if len(seq) >= min_seq_len]
    
    print(f"\nFiltered to {len(filtered_sequences)} sequences (min_len={min_seq_len})")
    
    # Save sequences
    with open(output_path, 'wb') as f:
        pickle.dump({'sequences': filtered_sequences}, f)
    
    print(f"\nSaved {len(filtered_sequences)} video sequences to {output_path}")
    print(f"Total frames: {sum(len(seq) for seq in filtered_sequences)}")
    
    # Print statistics
    if filtered_sequences:
        seq_lens = [len(seq) for seq in filtered_sequences]
        print(f"\nSequence length statistics:")
        print(f"  Min: {min(seq_lens)}")
        print(f"  Max: {max(seq_lens)}")
        print(f"  Mean: {np.mean(seq_lens):.1f}")
        print(f"  Median: {np.median(seq_lens):.1f}")
    
    return filtered_sequences

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare video sequences for temporal training")
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input keypoints file or directory')
    parser.add_argument('--output_path', type=str, default='video_sequences.pkl',
                        help='Path to save sequences pickle file')
    parser.add_argument('--min_seq_len', type=int, default=10,
                        help='Minimum sequence length to include')
    parser.add_argument('--gap_threshold', type=int, default=10,
                        help='Gap threshold for detecting video boundaries')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'single', 'directory'],
                        help='Mode: auto (detect), single (single file), directory (multiple files)')
    parser.add_argument('--use_path_info', action='store_true', default=True,
                        help='Use image_paths to organize by video (default: True)')
    parser.add_argument('--no_path_info', dest='use_path_info', action='store_false',
                        help='Disable using image_paths, use boundary detection instead')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if args.mode == 'auto':
        if input_path.is_file():
            mode = 'single'
        elif input_path.is_dir():
            mode = 'directory'
        else:
            raise ValueError(f"Path does not exist: {input_path}")
    else:
        mode = args.mode
    
    if mode == 'single':
        prepare_sequences_from_single_file(
            args.input_path,
            args.output_path,
            min_seq_len=args.min_seq_len,
            gap_threshold=args.gap_threshold,
            use_path_info=args.use_path_info
        )
    elif mode == 'directory':
        prepare_sequences_from_directory(
            args.input_path,
            args.output_path,
            min_seq_len=args.min_seq_len
        )


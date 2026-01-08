"""
Create text data from video sequences
Extracts video names/IDs from video_sequences.pkl and creates corresponding text descriptions
"""
import pickle
import json
from collections import defaultdict

def extract_texts_from_video_sequences(video_sequences_path, output_path='text_data.json'):
    """
    Extract text descriptions from video sequences
    
    Args:
        video_sequences_path: Path to video_sequences.pkl file
        output_path: Path to save text data JSON file
    """
    print(f"Loading video sequences from {video_sequences_path}...")
    with open(video_sequences_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'sequences' in data:
        video_sequences = data['sequences']
    elif isinstance(data, list):
        video_sequences = data
    else:
        raise ValueError("Unknown data format")
    
    print(f"Found {len(video_sequences)} video sequences")
    
    # Extract texts from video sequences
    texts = []
    video_names = []
    
    # Try to extract video names from keypoint data
    for idx, seq in enumerate(video_sequences):
        # Check if sequence has metadata with video name
        video_name = None
        
        # Method 1: Check if first keypoint dict has video_id or name
        if len(seq) > 0 and isinstance(seq[0], dict):
            first_kp = seq[0]
            if 'video_id' in first_kp:
                video_name = first_kp['video_id']
            elif 'video_name' in first_kp:
                video_name = first_kp['video_name']
            elif 'name' in first_kp:
                video_name = first_kp['name']
        
        # Method 2: Try to extract from image_paths if available
        if video_name is None and len(seq) > 0:
            # Check if we can get path info from the original data
            # This would require loading the original keypoints file
            pass
        
        # If we can't extract name, use a default
        if video_name is None:
            video_name = f"video_sequence_{idx+1}"
        
        video_names.append(video_name)
        
        # Use video name as text description
        # Clean up the name (remove numbers, extra spaces, etc.)
        text = video_name.strip()
        # Remove leading numbers and spaces (e.g., "1 hello you how" -> "hello you how")
        parts = text.split()
        if parts and parts[0].isdigit():
            text = ' '.join(parts[1:])
        if not text:
            text = f"sign language sequence {idx+1}"
        
        texts.append(text)
    
    # Save as JSON
    output_data = {
        'texts': texts,
        'video_names': video_names  # Keep for reference
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nCreated text data file: {output_path}")
    print(f"Total texts: {len(texts)}")
    print(f"\nFirst 10 texts:")
    for i, text in enumerate(texts[:10]):
        print(f"  {i+1}: {text}")
    
    return texts, video_names

def extract_texts_from_keypoints(keypoints_path, output_path='text_data.json'):
    """
    Extract video names from original keypoints file (which has image_paths)
    and create text descriptions
    
    Args:
        keypoints_path: Path to original keypoints pickle file (e.g., sign_language_keypoints.pkl)
        output_path: Path to save text data JSON file
    """
    print(f"Loading keypoints from {keypoints_path}...")
    with open(keypoints_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'keypoints' in data:
        keypoints_list = data['keypoints']
        image_paths = data.get('image_paths', None)
    elif isinstance(data, list):
        keypoints_list = data
        image_paths = None
    else:
        raise ValueError("Unknown data format")
    
    print(f"Found {len(keypoints_list)} keypoint frames")
    
    if image_paths is None or len(image_paths) != len(keypoints_list):
        print("Warning: image_paths not available or length mismatch")
        print("Cannot extract video names from paths")
        return None, None
    
    # Extract video IDs from image paths
    from collections import defaultdict
    video_dict = defaultdict(list)
    
    for idx, path in enumerate(image_paths):
        # Extract video ID from path
        # Assuming format like: extracted_frames/video_name/frame_xxx.jpg
        if isinstance(path, str):
            parts = path.replace('\\', '/').split('/')
            # Find the video name (usually second to last part)
            video_id = None
            for part in reversed(parts):
                if part and part != 'extracted_frames' and not part.startswith('frame_'):
                    video_id = part
                    break
            
            if video_id:
                video_dict[video_id].append(idx)
            else:
                # Fallback: use filename
                video_id = parts[-1] if parts else f"video_{idx}"
                video_dict[video_id].append(idx)
    
    print(f"Found {len(video_dict)} unique videos")
    
    # Create texts from video IDs
    texts = []
    video_names = []
    
    for video_id, frame_indices in sorted(video_dict.items()):
        video_names.append(video_id)
        
        # Clean up video ID to create text
        text = video_id.strip()
        # Remove leading numbers and spaces
        parts = text.split()
        if parts and parts[0].isdigit():
            text = ' '.join(parts[1:])
        if not text:
            text = video_id
        
        texts.append(text)
    
    # Save as JSON
    output_data = {
        'texts': texts,
        'video_names': video_names
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nCreated text data file: {output_path}")
    print(f"Total texts: {len(texts)}")
    print(f"\nFirst 10 texts:")
    for i, text in enumerate(texts[:10]):
        print(f"  {i+1}: {text}")
    
    return texts, video_names

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create text data from video sequences")
    parser.add_argument('--video_sequences_path', type=str, default='video_sequences.pkl',
                        help='Path to video_sequences.pkl file')
    parser.add_argument('--keypoints_path', type=str, default=None,
                        help='Path to original keypoints file (for extracting video names from paths)')
    parser.add_argument('--output_path', type=str, default='text_data.json',
                        help='Path to save text data JSON file')
    
    args = parser.parse_args()
    
    # Try to extract from keypoints file first (has more info)
    if args.keypoints_path:
        try:
            texts, video_names = extract_texts_from_keypoints(
                args.keypoints_path,
                args.output_path
            )
        except Exception as e:
            print(f"Error extracting from keypoints file: {e}")
            print("Trying video_sequences.pkl instead...")
            texts, video_names = extract_texts_from_video_sequences(
                args.video_sequences_path,
                args.output_path
            )
    else:
        # Extract from video sequences
        texts, video_names = extract_texts_from_video_sequences(
            args.video_sequences_path,
            args.output_path
        )
    
    print("\nDone!")


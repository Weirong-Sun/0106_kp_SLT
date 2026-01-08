"""
Test script to visualize extracted body keypoints
Load keypoints and plot skeleton images
"""
import pickle
import cv2
import numpy as np
import os
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_skeleton import draw_full_skeleton, draw_face_skeleton, draw_hand_skeleton, draw_pose_skeleton

def test_keypoints(data_path, num_samples=5, output_dir="test_keypoints_visualization"):
    """
    Test and visualize extracted keypoints
    
    Args:
        data_path: Path to keypoints pickle file
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualization images
    """
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Keypoints file not found: {data_path}")
        print("\nPlease run the keypoint extraction script first:")
        print("  python extract_body_keypoints.py --dataset_path extracted_frames --output_path body_keypoints_data.pkl")
        return
    
    # Load keypoints data
    print("Loading keypoints data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    keypoints_list = data['keypoints']
    print(f"Loaded {len(keypoints_list)} samples")
    
    # Print keypoint info
    if 'keypoint_info' in data:
        print("\nKeypoint Information:")
        for kp_type, info in data['keypoint_info'].items():
            if kp_type != 'total_points':
                print(f"  {kp_type}: {info['num_points']} points - {info['description']}")
        print(f"  Total points: {data['keypoint_info']['total_points']}")
    
    # Print statistics
    if 'stats' in data:
        print("\nDetection Statistics:")
        stats = data['stats']
        total = len(keypoints_list)
        if total > 0:
            print(f"  Face detected: {stats['face_detected']} ({stats['face_detected']/total*100:.2f}%)")
            print(f"  Left hand detected: {stats['left_hand_detected']} ({stats['left_hand_detected']/total*100:.2f}%)")
            print(f"  Right hand detected: {stats['right_hand_detected']} ({stats['right_hand_detected']/total*100:.2f}%)")
            print(f"  Pose detected: {stats['pose_detected']} ({stats['pose_detected']/total*100:.2f}%)")
            print(f"  Full body detected: {stats['full_body_detected']} ({stats['full_body_detected']/total*100:.2f}%)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize samples
    num_samples_to_show = min(num_samples, len(keypoints_list))
    print(f"\nVisualizing {num_samples_to_show} samples...")
    
    image_size = 512
    
    for idx in range(num_samples_to_show):
        print(f"\nProcessing sample {idx+1}/{num_samples_to_show}")
        kp_dict = keypoints_list[idx]
        
        # Print keypoint availability
        print(f"  Face: {'✓' if kp_dict.get('face') is not None else '✗'}")
        print(f"  Left hand: {'✓' if kp_dict.get('left_hand') is not None else '✗'}")
        print(f"  Right hand: {'✓' if kp_dict.get('right_hand') is not None else '✗'}")
        print(f"  Pose: {'✓' if kp_dict.get('pose') is not None else '✗'}")
        
        # Draw full skeleton
        full_skeleton = draw_full_skeleton(kp_dict, image_size=image_size, point_radius=3, line_thickness=2)
        
        # Draw individual components for comparison
        components = {}
        
        # Face only
        face_canvas = np.ones((image_size, image_size), dtype=np.uint8) * 255
        if kp_dict.get('face') is not None:
            draw_face_skeleton(face_canvas, kp_dict['face'], image_size, point_radius=3, line_thickness=2)
        components['face'] = face_canvas
        
        # Hands only
        hands_canvas = np.ones((image_size, image_size), dtype=np.uint8) * 255
        if kp_dict.get('left_hand') is not None:
            draw_hand_skeleton(hands_canvas, kp_dict['left_hand'], image_size, point_radius=3, line_thickness=2)
        if kp_dict.get('right_hand') is not None:
            draw_hand_skeleton(hands_canvas, kp_dict['right_hand'], image_size, point_radius=3, line_thickness=2)
        components['hands'] = hands_canvas
        
        # Pose only
        pose_canvas = np.ones((image_size, image_size), dtype=np.uint8) * 255
        if kp_dict.get('pose') is not None:
            draw_pose_skeleton(pose_canvas, kp_dict['pose'], image_size, point_radius=4, line_thickness=3)
        components['pose'] = pose_canvas
        
        # Save individual images
        full_path = os.path.join(output_dir, f"sample_{idx}_full_skeleton.png")
        face_path = os.path.join(output_dir, f"sample_{idx}_face_only.png")
        hands_path = os.path.join(output_dir, f"sample_{idx}_hands_only.png")
        pose_path = os.path.join(output_dir, f"sample_{idx}_pose_only.png")
        
        cv2.imwrite(full_path, full_skeleton)
        cv2.imwrite(face_path, components['face'])
        cv2.imwrite(hands_path, components['hands'])
        cv2.imwrite(pose_path, components['pose'])
        
        # Create combined visualization
        # Top row: Full skeleton (need to resize to match bottom row width)
        # Bottom row: Face | Hands | Pose
        bottom_row = np.hstack([components['face'], components['hands'], components['pose']])
        bottom_width = bottom_row.shape[1]
        
        # Resize top row to match bottom row width
        top_row_resized = cv2.resize(full_skeleton, (bottom_width, image_size))
        
        combined = np.vstack([top_row_resized, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f'Sample {idx+1} - Full Skeleton', (10, 30), font, 1, 0, 2)
        cv2.putText(combined, 'Face', (10, image_size + 30), font, 1, 0, 2)
        cv2.putText(combined, 'Hands', (image_size + 10, image_size + 30), font, 1, 0, 2)
        cv2.putText(combined, 'Pose', (image_size * 2 + 10, image_size + 30), font, 1, 0, 2)
        
        combined_path = os.path.join(output_dir, f"sample_{idx}_combined.png")
        cv2.imwrite(combined_path, combined)
        
        print(f"  Saved visualization: {combined_path}")
        
        # Print keypoint counts
        if kp_dict.get('face') is not None:
            print(f"  Face keypoints: {kp_dict['face'].shape}")
        if kp_dict.get('left_hand') is not None:
            print(f"  Left hand keypoints: {kp_dict['left_hand'].shape}")
        if kp_dict.get('right_hand') is not None:
            print(f"  Right hand keypoints: {kp_dict['right_hand'].shape}")
        if kp_dict.get('pose') is not None:
            print(f"  Pose keypoints: {kp_dict['pose'].shape}")
    
    # Create grid of all full skeletons
    print(f"\nCreating grid visualization...")
    font = cv2.FONT_HERSHEY_SIMPLEX
    cols = min(3, num_samples_to_show)
    rows = (num_samples_to_show + cols - 1) // cols
    
    grid_height = rows * image_size
    grid_width = cols * image_size
    
    grid_image = np.ones((grid_height, grid_width), dtype=np.uint8) * 255
    
    for idx in range(num_samples_to_show):
        kp_dict = keypoints_list[idx]
        skeleton_img = draw_full_skeleton(kp_dict, image_size=image_size, point_radius=3, line_thickness=2)
        
        row = idx // cols
        col = idx % cols
        y_start = row * image_size
        x_start = col * image_size
        grid_image[y_start:y_start+image_size, x_start:x_start+image_size] = skeleton_img
        
        # Add sample label
        cv2.putText(grid_image, f'Sample {idx+1}', 
                   (x_start + 10, y_start + 30), font, 1, 0, 2)
    
    grid_path = os.path.join(output_dir, f"grid_all_samples.png")
    cv2.imwrite(grid_path, grid_image)
    print(f"Saved grid visualization: {grid_path}")
    
    print(f"\n{'='*60}")
    print("Visualization Complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test and visualize extracted body keypoints')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to keypoints pickle file (auto-detect if not specified)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='test_keypoints_visualization',
                        help='Directory to save visualization images')
    
    args = parser.parse_args()
    
    # Auto-detect keypoints file if not specified
    if args.data_path is None:
        possible_files = [
            'body_keypoints_data.pkl',
            'sign_language_keypoints.pkl',
            'keypoints_data.pkl'
        ]
        
        data_path = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                data_path = file_path
                print(f"Auto-detected keypoints file: {file_path}")
                break
        
        if data_path is None:
            print("Error: No keypoints file found!")
            print("\nPlease run the keypoint extraction script first:")
            print("  python extract_body_keypoints.py --dataset_path extracted_frames --output_path body_keypoints_data.pkl")
            print("\nOr specify the path manually:")
            print("  python test_keypoints.py --data_path your_keypoints_file.pkl")
            exit(1)
        
        args.data_path = data_path
    
    print("="*60)
    print("Keypoint Visualization Test")
    print("="*60)
    
    test_keypoints(
        data_path=args.data_path,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )


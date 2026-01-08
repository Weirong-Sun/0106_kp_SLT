"""
Extract full body keypoints from images using MediaPipe
Includes: Face (68 points), Hands (21 points each), Pose (33 points)
"""
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse

class FullBodyKeypointExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe models for full body keypoint extraction
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        # Face Mesh (468 points, we'll extract 68)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        )
        
        # Hands (21 points per hand)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Pose (33 points)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Mapping from MediaPipe 468 face points to 68-point standard landmarks
        self.face_68_indices = self._get_68_face_indices()
    
    def _get_68_face_indices(self):
        """Map MediaPipe 468 points to 68-point standard landmarks"""
        return [
            # Face outline (17 points: 0-16)
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            # Right eyebrow (5 points: 17-21)
            107, 55, 65, 52, 53,
            # Left eyebrow (5 points: 22-26)
            336, 296, 334, 293, 300,
            # Nose (9 points: 27-35)
            19, 1, 2, 5, 4, 6, 98, 97, 94,
            # Right eye (6 points: 36-41)
            33, 7, 163, 144, 145, 153,
            # Left eye (6 points: 42-47)
            362, 382, 381, 380, 374, 373,
            # Mouth outer (12 points: 48-59)
            61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321,
            # Mouth inner (8 points: 60-67)
            78, 95, 88, 178, 87, 14, 317, 402
        ]
    
    def extract_face_keypoints(self, image_rgb, face_landmarks):
        """
        Extract 68 facial keypoints
        
        Args:
            image_rgb: RGB image
            face_landmarks: MediaPipe face landmarks
        
        Returns:
            keypoints: numpy array of shape (68, 3) or None
        """
        if not face_landmarks:
            return None
        
        keypoints = []
        for idx in self.face_68_indices:
            landmark = face_landmarks.landmark[idx]
            keypoints.append([landmark.x, landmark.y, landmark.z])
        
        return np.array(keypoints, dtype=np.float32)
    
    def extract_hand_keypoints(self, image_rgb, hand_landmarks_list):
        """
        Extract hand keypoints (21 points per hand)
        
        Args:
            image_rgb: RGB image
            hand_landmarks_list: List of MediaPipe hand landmarks
        
        Returns:
            left_hand: numpy array of shape (21, 3) or None
            right_hand: numpy array of shape (21, 3) or None
        """
        left_hand = None
        right_hand = None
        
        if not hand_landmarks_list:
            return left_hand, right_hand
        
        for hand_landmarks in hand_landmarks_list:
            # Determine if left or right hand
            hand_label = hand_landmarks.classification[0].label if hasattr(hand_landmarks, 'classification') else None
            
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
            
            keypoints = np.array(keypoints, dtype=np.float32)
            
            # MediaPipe returns 'Left' for right hand and 'Right' for left hand (from camera perspective)
            # We'll use the order: first hand = right, second hand = left
            if right_hand is None:
                right_hand = keypoints
            elif left_hand is None:
                left_hand = keypoints
        
        return left_hand, right_hand
    
    def extract_pose_keypoints(self, image_rgb, pose_landmarks):
        """
        Extract pose keypoints (33 points)
        
        Args:
            image_rgb: RGB image
            pose_landmarks: MediaPipe pose landmarks
        
        Returns:
            keypoints: numpy array of shape (33, 3) or None
        """
        if not pose_landmarks:
            return None
        
        keypoints = []
        for landmark in pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])
        
        return np.array(keypoints, dtype=np.float32)
    
    def extract_keypoints(self, image_path):
        """
        Extract all keypoints from an image
        
        Args:
            image_path: Path to the image file
        
        Returns:
            keypoints_dict: Dictionary containing all keypoints or None if no detection
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Extract face keypoints
        face_results = self.face_mesh.process(image_rgb)
        face_kp = None
        if face_results.multi_face_landmarks:
            face_kp = self.extract_face_keypoints(image_rgb, face_results.multi_face_landmarks[0])
        
        # Extract hand keypoints
        hand_results = self.hands.process(image_rgb)
        left_hand_kp, right_hand_kp = self.extract_hand_keypoints(
            image_rgb, 
            hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None
        )
        
        # Extract pose keypoints
        pose_results = self.pose.process(image_rgb)
        pose_kp = None
        if pose_results.pose_landmarks:
            pose_kp = self.extract_pose_keypoints(image_rgb, pose_results.pose_landmarks)
        
        # Check if we have at least some keypoints
        if face_kp is None and left_hand_kp is None and right_hand_kp is None and pose_kp is None:
            return None
        
        return {
            'face': face_kp,  # (68, 3) or None
            'left_hand': left_hand_kp,  # (21, 3) or None
            'right_hand': right_hand_kp,  # (21, 3) or None
            'pose': pose_kp,  # (33, 3) or None
            'image_shape': (h, w)
        }
    
    def process_dataset(self, dataset_path, output_path, max_samples=None):
        """
        Process all images in the dataset and extract keypoints
        
        Args:
            dataset_path: Path to the dataset directory (containing extracted frames)
            output_path: Path to save the extracted keypoints
            max_samples: Maximum number of samples to process (None for all)
        
        Returns:
            keypoints_data: Dictionary with all extracted keypoints
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
        
        print(f"Scanning for images in: {dataset_path}")
        image_files = []
        for ext in image_extensions:
            pattern = f'**/*{ext}'
            found_files = list(dataset_path.glob(pattern))
            image_files.extend(found_files)
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        print(f"\nFound {len(image_files)} image files")
        
        if max_samples:
            image_files = image_files[:max_samples]
            print(f"Processing first {len(image_files)} samples (limited by max_samples)")
        else:
            print(f"Processing all {len(image_files)} samples")
        
        # Extract keypoints
        all_keypoints = []
        valid_paths = []
        stats = {
            'face_detected': 0,
            'left_hand_detected': 0,
            'right_hand_detected': 0,
            'pose_detected': 0,
            'full_body_detected': 0  # All components detected
        }
        
        for image_path in tqdm(image_files, desc="Extracting keypoints"):
            kp_dict = self.extract_keypoints(str(image_path))
            if kp_dict is not None:
                all_keypoints.append(kp_dict)
                valid_paths.append(str(image_path))
                
                # Update statistics
                if kp_dict['face'] is not None:
                    stats['face_detected'] += 1
                if kp_dict['left_hand'] is not None:
                    stats['left_hand_detected'] += 1
                if kp_dict['right_hand'] is not None:
                    stats['right_hand_detected'] += 1
                if kp_dict['pose'] is not None:
                    stats['pose_detected'] += 1
                if (kp_dict['face'] is not None and 
                    kp_dict['pose'] is not None and
                    (kp_dict['left_hand'] is not None or kp_dict['right_hand'] is not None)):
                    stats['full_body_detected'] += 1
        
        print(f"\nExtraction Summary:")
        print(f"  Total images found: {len(image_files)}")
        print(f"  Successfully extracted: {len(all_keypoints)}")
        print(f"  Failed (no detection): {len(image_files) - len(all_keypoints)}")
        print(f"  Success rate: {len(all_keypoints)/len(image_files)*100:.2f}%")
        print(f"\nDetection Statistics:")
        print(f"  Face detected: {stats['face_detected']} ({stats['face_detected']/len(all_keypoints)*100:.2f}%)" if len(all_keypoints) > 0 else "  Face detected: 0")
        print(f"  Left hand detected: {stats['left_hand_detected']} ({stats['left_hand_detected']/len(all_keypoints)*100:.2f}%)" if len(all_keypoints) > 0 else "  Left hand detected: 0")
        print(f"  Right hand detected: {stats['right_hand_detected']} ({stats['right_hand_detected']/len(all_keypoints)*100:.2f}%)" if len(all_keypoints) > 0 else "  Right hand detected: 0")
        print(f"  Pose detected: {stats['pose_detected']} ({stats['pose_detected']/len(all_keypoints)*100:.2f}%)" if len(all_keypoints) > 0 else "  Pose detected: 0")
        print(f"  Full body detected: {stats['full_body_detected']} ({stats['full_body_detected']/len(all_keypoints)*100:.2f}%)" if len(all_keypoints) > 0 else "  Full body detected: 0")
        
        # Save keypoints
        output_data = {
            'keypoints': all_keypoints,
            'image_paths': valid_paths,
            'stats': stats,
            'keypoint_info': {
                'face': {'num_points': 68, 'description': 'Facial landmarks'},
                'left_hand': {'num_points': 21, 'description': 'Left hand landmarks'},
                'right_hand': {'num_points': 21, 'description': 'Right hand landmarks'},
                'pose': {'num_points': 33, 'description': 'Body pose landmarks'},
                'total_points': 68 + 21 + 21 + 33  # 143 points total
            }
        }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"\nSaved keypoints to {output_path}")
        print(f"Total samples: {len(all_keypoints)}")
        print(f"Total keypoints per sample: up to 143 points (68 face + 21 left hand + 21 right hand + 33 pose)")
        
        return output_data

def main():
    parser = argparse.ArgumentParser(description='Extract full body keypoints from images')
    parser.add_argument('--dataset_path', type=str, default='extracted_frames',
                        help='Path to the dataset directory (containing extracted frames)')
    parser.add_argument('--output_path', type=str, default='body_keypoints_data.pkl',
                        help='Path to save the extracted keypoints')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None for all)')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5,
                        help='Minimum confidence for detection (0.0-1.0)')
    parser.add_argument('--min_tracking_confidence', type=float, default=0.5,
                        help='Minimum confidence for tracking (0.0-1.0)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Full Body Keypoint Extraction")
    print("="*60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output path: {args.output_path}")
    print(f"Detection confidence: {args.min_detection_confidence}")
    print("="*60)
    
    extractor = FullBodyKeypointExtractor(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    
    extractor.process_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        max_samples=args.max_samples
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main()


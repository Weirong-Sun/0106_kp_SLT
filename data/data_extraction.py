"""
Extract facial keypoints using MediaPipe Face Landmark Detection
Extracts 68 facial keypoints from images in the dataset
"""
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

class FaceKeypointExtractor:
    def __init__(self):
        # Use Face Mesh with specific landmark model that provides 468 points
        # We'll select 68 representative points from the 468 points
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        # Mapping from MediaPipe 468 points to 68-point standard landmarks
        self.landmark_indices_68 = self._get_68_landmark_indices()
    
    def _get_68_landmark_indices(self):
        """
        Map MediaPipe 468 points to 68-point format
        Using standard facial landmark positions
        """
        # Standard 68-point facial landmark mapping from MediaPipe 468 points
        # Face outline (17 points)
        face_outline = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 
            361, 288, 397, 365, 379, 378, 400
        ]
        
        # Right eyebrow (5 points)
        right_eyebrow = [107, 55, 65, 52, 53]
        
        # Left eyebrow (5 points)
        left_eyebrow = [336, 296, 334, 293, 300]
        
        # Nose (9 points)
        nose = [19, 1, 2, 5, 4, 6, 98, 97, 94]
        
        # Right eye (6 points)
        right_eye = [33, 7, 163, 144, 145, 153]
        
        # Left eye (6 points)
        left_eye = [362, 382, 381, 380, 374, 373]
        
        # Mouth outer (12 points)
        mouth_outer = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321]
        
        # Mouth inner (8 points)
        mouth_inner = [78, 95, 88, 178, 87, 14, 317, 402]
        
        # Combine all indices
        indices = (face_outline[:17] + right_eyebrow[:5] + left_eyebrow[:5] + 
                  nose[:9] + right_eye[:6] + left_eye[:6] + 
                  mouth_outer[:12] + mouth_inner[:8])
        
        # Ensure exactly 68 points
        return indices[:68]
    
    def extract_keypoints(self, image_path):
        """
        Extract 68 facial keypoints from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            keypoints: numpy array of shape (68, 3) with (x, y, z) coordinates
                      Returns None if no face detected
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract 68 key points using the mapping
        keypoints = []
        for idx in self.landmark_indices_68:
            landmark = face_landmarks.landmark[idx]
            # Normalize coordinates to [0, 1] (MediaPipe already provides normalized coordinates)
            x = landmark.x
            y = landmark.y
            z = landmark.z
            keypoints.append([x, y, z])
        
        return np.array(keypoints, dtype=np.float32)
    
    def process_dataset(self, dataset_path, output_path, max_samples=None):
        """
        Process all images in the dataset and extract keypoints
        
        Args:
            dataset_path: Path to the dataset directory
            output_path: Path to save the extracted keypoints
            max_samples: Maximum number of samples to process (None for all)
        """
        dataset_path = Path(dataset_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_path.rglob(f'*{ext}'))
            image_files.extend(dataset_path.rglob(f'*{ext.upper()}'))
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        print(f"Found {len(image_files)} images")
        
        keypoints_list = []
        valid_paths = []
        
        for image_path in tqdm(image_files, desc="Extracting keypoints"):
            kp = self.extract_keypoints(str(image_path))
            if kp is not None:
                keypoints_list.append(kp)
                valid_paths.append(str(image_path))
        
        print(f"Successfully extracted keypoints from {len(keypoints_list)} images")
        
        # Save keypoints
        keypoints_array = np.array(keypoints_list)
        output_data = {
            'keypoints': keypoints_array,
            'image_paths': valid_paths
        }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"Saved keypoints to {output_path}")
        print(f"Keypoints shape: {keypoints_array.shape}")
        
        return keypoints_array

if __name__ == "__main__":
    dataset_path = r"D:\dataset\Celebrity Faces Dataset"
    output_path = "keypoints_data.pkl"
    
    extractor = FaceKeypointExtractor()
    
    # Process first 100 samples for testing
    print("Processing first 100 samples...")
    extractor.process_dataset(dataset_path, output_path, max_samples=100)


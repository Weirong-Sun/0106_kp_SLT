"""
Utility functions for generating ground truth images from keypoints
Projects keypoints onto a white canvas
"""
import cv2
import numpy as np
import torch

def draw_keypoints_on_canvas(keypoints, image_size=256, point_radius=2, line_thickness=1, draw_lines=True):
    """
    Draw keypoints on a white canvas
    
    Args:
        keypoints: numpy array of shape [68, 3] or [68, 2] with (x, y) or (x, y, z) coordinates
                   Coordinates should be normalized to [0, 1]
        image_size: Size of output image
        point_radius: Radius of keypoint circles
        line_thickness: Thickness of connecting lines
        draw_lines: Whether to draw connecting lines between keypoints
    
    Returns:
        image: Grayscale image [image_size, image_size] with values in [0, 255]
    """
    # Create white canvas
    canvas = np.ones((image_size, image_size), dtype=np.uint8) * 255
    
    # Convert normalized coordinates to pixel coordinates
    if keypoints.shape[1] >= 2:
        kp_2d = keypoints[:, :2]  # Use only x, y
    else:
        kp_2d = keypoints
    
    pixel_coords = (kp_2d * image_size).astype(np.int32)
    
    # Define connections for facial landmarks (68-point format)
    # Face outline (0-16)
    face_outline_connections = list(range(17)) + [0]
    
    # Right eyebrow (17-21)
    right_eyebrow_connections = list(range(17, 22))
    
    # Left eyebrow (22-26)
    left_eyebrow_connections = list(range(22, 27))
    
    # Nose (27-35)
    nose_connections = [
        (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
        (27, 31), (31, 35)
    ]
    
    # Right eye (36-41)
    right_eye_connections = list(range(36, 42)) + [36]
    
    # Left eye (42-47)
    left_eye_connections = list(range(42, 48)) + [42]
    
    # Mouth outer (48-59)
    mouth_outer_connections = list(range(48, 60)) + [48]
    
    # Mouth inner (60-67)
    mouth_inner_connections = list(range(60, 68)) + [60]
    
    # Draw connections
    if draw_lines:
        all_connections = [
            (face_outline_connections, 0),
            (right_eyebrow_connections, 0),
            (left_eyebrow_connections, 0),
            (nose_connections, 0),
            (right_eye_connections, 0),
            (left_eye_connections, 0),
            (mouth_outer_connections, 0),
            (mouth_inner_connections, 0)
        ]
        
        for connections, _ in all_connections:
            if isinstance(connections[0], tuple):
                # List of tuples
                for start_idx, end_idx in connections:
                    if 0 <= start_idx < len(pixel_coords) and 0 <= end_idx < len(pixel_coords):
                        pt1 = tuple(pixel_coords[start_idx])
                        pt2 = tuple(pixel_coords[end_idx])
                        cv2.line(canvas, pt1, pt2, 0, line_thickness)
            else:
                # List of indices
                for i in range(len(connections) - 1):
                    idx1 = connections[i]
                    idx2 = connections[i + 1]
                    if 0 <= idx1 < len(pixel_coords) and 0 <= idx2 < len(pixel_coords):
                        pt1 = tuple(pixel_coords[idx1])
                        pt2 = tuple(pixel_coords[idx2])
                        cv2.line(canvas, pt1, pt2, 0, line_thickness)
    
    # Draw keypoints
    for i, (x, y) in enumerate(pixel_coords):
        if 0 <= x < image_size and 0 <= y < image_size:
            cv2.circle(canvas, (x, y), point_radius, 0, -1)
    
    return canvas

def generate_image_dataset(keypoints_data, image_size=256, output_path=None):
    """
    Generate ground truth images from keypoints dataset
    
    Args:
        keypoints_data: numpy array of shape [num_samples, 68, 3]
        image_size: Size of output images
        output_path: Optional path to save images as numpy array
    
    Returns:
        images: numpy array of shape [num_samples, image_size, image_size]
    """
    num_samples = len(keypoints_data)
    images = []
    
    for i in range(num_samples):
        kp = keypoints_data[i]
        img = draw_keypoints_on_canvas(kp, image_size=image_size)
        images.append(img)
    
    images = np.array(images, dtype=np.uint8)
    
    if output_path:
        np.save(output_path, images)
        print(f"Saved {num_samples} images to {output_path}")
    
    return images



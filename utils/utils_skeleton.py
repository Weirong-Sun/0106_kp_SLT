"""
Utility functions for generating skeleton images from body keypoints
Draws skeleton connections for face, hands, and pose
"""
import cv2
import numpy as np

def draw_face_skeleton(canvas, face_keypoints, image_size, point_radius=2, line_thickness=1):
    """
    Draw face skeleton on canvas
    
    Args:
        canvas: Image canvas
        face_keypoints: numpy array of shape [68, 3] or [68, 2]
        image_size: Size of image
        point_radius: Radius of keypoint circles
        line_thickness: Thickness of connecting lines
    """
    if face_keypoints is None:
        return
    
    # Convert normalized coordinates to pixel coordinates
    if face_keypoints.shape[1] >= 2:
        kp_2d = face_keypoints[:, :2]
    else:
        kp_2d = face_keypoints
    
    pixel_coords = (kp_2d * image_size).astype(np.int32)
    
    # Face outline (0-16)
    face_outline = list(range(17)) + [0]
    for i in range(len(face_outline) - 1):
        idx1, idx2 = face_outline[i], face_outline[i + 1]
        if 0 <= idx1 < len(pixel_coords) and 0 <= idx2 < len(pixel_coords):
            pt1 = tuple(pixel_coords[idx1])
            pt2 = tuple(pixel_coords[idx2])
            cv2.line(canvas, pt1, pt2, 0, line_thickness)
    
    # Right eyebrow (17-21)
    for i in range(17, 21):
        if i + 1 < len(pixel_coords):
            pt1 = tuple(pixel_coords[i])
            pt2 = tuple(pixel_coords[i + 1])
            cv2.line(canvas, pt1, pt2, 0, line_thickness)
    
    # Left eyebrow (22-26)
    for i in range(22, 26):
        if i + 1 < len(pixel_coords):
            pt1 = tuple(pixel_coords[i])
            pt2 = tuple(pixel_coords[i + 1])
            cv2.line(canvas, pt1, pt2, 0, line_thickness)
    
    # Nose (27-35)
    nose_connections = [(27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (27, 31), (31, 35)]
    for start_idx, end_idx in nose_connections:
        if 0 <= start_idx < len(pixel_coords) and 0 <= end_idx < len(pixel_coords):
            pt1 = tuple(pixel_coords[start_idx])
            pt2 = tuple(pixel_coords[end_idx])
            cv2.line(canvas, pt1, pt2, 0, line_thickness)
    
    # Right eye (36-41)
    right_eye = list(range(36, 42)) + [36]
    for i in range(len(right_eye) - 1):
        idx1, idx2 = right_eye[i], right_eye[i + 1]
        if 0 <= idx1 < len(pixel_coords) and 0 <= idx2 < len(pixel_coords):
            pt1 = tuple(pixel_coords[idx1])
            pt2 = tuple(pixel_coords[idx2])
            cv2.line(canvas, pt1, pt2, 0, line_thickness)
    
    # Left eye (42-47)
    left_eye = list(range(42, 48)) + [42]
    for i in range(len(left_eye) - 1):
        idx1, idx2 = left_eye[i], left_eye[i + 1]
        if 0 <= idx1 < len(pixel_coords) and 0 <= idx2 < len(pixel_coords):
            pt1 = tuple(pixel_coords[idx1])
            pt2 = tuple(pixel_coords[idx2])
            cv2.line(canvas, pt1, pt2, 0, line_thickness)
    
    # Mouth outer (48-59)
    mouth_outer = list(range(48, 60)) + [48]
    for i in range(len(mouth_outer) - 1):
        idx1, idx2 = mouth_outer[i], mouth_outer[i + 1]
        if 0 <= idx1 < len(pixel_coords) and 0 <= idx2 < len(pixel_coords):
            pt1 = tuple(pixel_coords[idx1])
            pt2 = tuple(pixel_coords[idx2])
            cv2.line(canvas, pt1, pt2, 0, line_thickness)
    
    # Mouth inner (60-67)
    mouth_inner = list(range(60, 68)) + [60]
    for i in range(len(mouth_inner) - 1):
        idx1, idx2 = mouth_inner[i], mouth_inner[i + 1]
        if 0 <= idx1 < len(pixel_coords) and 0 <= idx2 < len(pixel_coords):
            pt1 = tuple(pixel_coords[idx1])
            pt2 = tuple(pixel_coords[idx2])
            cv2.line(canvas, pt1, pt2, 0, line_thickness)
    
    # Draw keypoints
    for x, y in pixel_coords:
        if 0 <= x < image_size and 0 <= y < image_size:
            cv2.circle(canvas, (x, y), point_radius, 0, -1)

def draw_hand_skeleton(canvas, hand_keypoints, image_size, point_radius=3, line_thickness=2):
    """
    Draw hand skeleton on canvas with enhanced visibility
    
    Args:
        canvas: Image canvas
        hand_keypoints: numpy array of shape [21, 3] or [21, 2] or None
        image_size: Size of image
        point_radius: Radius of keypoint circles (increased for better visibility)
        line_thickness: Thickness of connecting lines (increased for better visibility)
    """
    if hand_keypoints is None:
        return
    
    # Convert normalized coordinates to pixel coordinates
    if hand_keypoints.shape[1] >= 2:
        kp_2d = hand_keypoints[:, :2]
    else:
        kp_2d = hand_keypoints
    
    pixel_coords = (kp_2d * image_size).astype(np.int32)
    
    # Hand connections (MediaPipe hand landmarks)
    # Thumb (thicker lines for main structure)
    thumb_connections = [(0, 1), (1, 2), (2, 3), (3, 4)]
    # Fingers (thicker lines)
    index_connections = [(0, 5), (5, 6), (6, 7), (7, 8)]
    middle_connections = [(0, 9), (9, 10), (10, 11), (11, 12)]
    ring_connections = [(0, 13), (13, 14), (14, 15), (15, 16)]
    pinky_connections = [(0, 17), (17, 18), (18, 19), (19, 20)]
    # Base connections
    base_connections = [(5, 9), (9, 13), (13, 17)]
    
    all_connections = (thumb_connections + index_connections + middle_connections + 
                       ring_connections + pinky_connections + base_connections)
    
    # Draw connections with varying thickness
    for start_idx, end_idx in all_connections:
        if 0 <= start_idx < len(pixel_coords) and 0 <= end_idx < len(pixel_coords):
            pt1 = tuple(pixel_coords[start_idx])
            pt2 = tuple(pixel_coords[end_idx])
            # Check if points are valid
            if (0 <= pt1[0] < image_size and 0 <= pt1[1] < image_size and
                0 <= pt2[0] < image_size and 0 <= pt2[1] < image_size):
                # Use thicker lines for finger connections
                thickness = line_thickness + 1 if start_idx > 0 else line_thickness
                cv2.line(canvas, pt1, pt2, 0, thickness)
    
    # Draw keypoints with larger radius for fingertips
    for i, (x, y) in enumerate(pixel_coords):
        if 0 <= x < image_size and 0 <= y < image_size:
            # Fingertips (indices 4, 8, 12, 16, 20) get larger radius
            radius = point_radius + 1 if i in [4, 8, 12, 16, 20] else point_radius
            cv2.circle(canvas, (x, y), radius, 0, -1)

def draw_pose_skeleton(canvas, pose_keypoints, image_size, point_radius=3, line_thickness=2):
    """
    Draw pose skeleton on canvas
    
    Args:
        canvas: Image canvas
        pose_keypoints: numpy array of shape [33, 3] or [33, 2]
        image_size: Size of image
        point_radius: Radius of keypoint circles
        line_thickness: Thickness of connecting lines
    """
    if pose_keypoints is None:
        return
    
    # Convert normalized coordinates to pixel coordinates
    if pose_keypoints.shape[1] >= 2:
        kp_2d = pose_keypoints[:, :2]
    else:
        kp_2d = pose_keypoints
    
    pixel_coords = (kp_2d * image_size).astype(np.int32)
    
    # MediaPipe Pose connections
    # Torso
    torso_connections = [
        (11, 12),  # Shoulders
        (11, 23), (12, 24),  # Shoulder to hip
        (23, 24),  # Hips
    ]
    
    # Left arm
    left_arm_connections = [
        (11, 13),  # Left shoulder to left elbow
        (13, 15),  # Left elbow to left wrist
    ]
    
    # Right arm
    right_arm_connections = [
        (12, 14),  # Right shoulder to right elbow
        (14, 16),  # Right elbow to right wrist
    ]
    
    # Left leg
    left_leg_connections = [
        (23, 25),  # Left hip to left knee
        (25, 27),  # Left knee to left ankle
    ]
    
    # Right leg
    right_leg_connections = [
        (24, 26),  # Right hip to right knee
        (26, 28),  # Right knee to right ankle
    ]
    
    # Face (if available)
    face_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Face outline
        (0, 4), (4, 5), (5, 6), (6, 8),  # Face outline
    ]
    
    all_connections = (torso_connections + left_arm_connections + right_arm_connections + 
                       left_leg_connections + right_leg_connections + face_connections)
    
    for start_idx, end_idx in all_connections:
        if 0 <= start_idx < len(pixel_coords) and 0 <= end_idx < len(pixel_coords):
            pt1 = tuple(pixel_coords[start_idx])
            pt2 = tuple(pixel_coords[end_idx])
            # Check if points are valid (not zero or out of bounds)
            if (0 <= pt1[0] < image_size and 0 <= pt1[1] < image_size and
                0 <= pt2[0] < image_size and 0 <= pt2[1] < image_size):
                cv2.line(canvas, pt1, pt2, 0, line_thickness)
    
    # Draw keypoints
    for x, y in pixel_coords:
        if 0 <= x < image_size and 0 <= y < image_size:
            cv2.circle(canvas, (x, y), point_radius, 0, -1)

def draw_full_skeleton(keypoints_dict, image_size=256, point_radius=2, line_thickness=1):
    """
    Draw full body skeleton from keypoints dictionary
    
    Args:
        keypoints_dict: Dictionary with 'face', 'left_hand', 'right_hand', 'pose' keypoints
        image_size: Size of output image
        point_radius: Radius of keypoint circles
        line_thickness: Thickness of connecting lines
    
    Returns:
        image: Grayscale image [image_size, image_size] with values in [0, 255]
    """
    # Create white canvas
    canvas = np.ones((image_size, image_size), dtype=np.uint8) * 255
    
    # Draw pose first (largest structure)
    if 'pose' in keypoints_dict:
        draw_pose_skeleton(canvas, keypoints_dict['pose'], image_size, 
                          point_radius=point_radius+1, line_thickness=line_thickness+1)
    
    # Draw hands with enhanced visibility
    if 'left_hand' in keypoints_dict:
        draw_hand_skeleton(canvas, keypoints_dict['left_hand'], image_size, 
                          point_radius=max(point_radius, 3), line_thickness=max(line_thickness, 2))
    
    if 'right_hand' in keypoints_dict:
        draw_hand_skeleton(canvas, keypoints_dict['right_hand'], image_size, 
                          point_radius=max(point_radius, 3), line_thickness=max(line_thickness, 2))
    
    # Draw face last (most detailed)
    if 'face' in keypoints_dict:
        draw_face_skeleton(canvas, keypoints_dict['face'], image_size, 
                          point_radius=point_radius, line_thickness=line_thickness)
    
    return canvas

def generate_skeleton_dataset(keypoints_data, image_size=256, output_path=None):
    """
    Generate ground truth skeleton images from keypoints dataset
    
    Args:
        keypoints_data: List of keypoint dictionaries
        image_size: Size of output images
        output_path: Optional path to save images as numpy array
    
    Returns:
        images: numpy array of shape [num_samples, image_size, image_size]
    """
    num_samples = len(keypoints_data)
    images = []
    
    for i in range(num_samples):
        kp_dict = keypoints_data[i]
        img = draw_full_skeleton(kp_dict, image_size=image_size)
        images.append(img)
    
    images = np.array(images, dtype=np.uint8)
    
    if output_path:
        np.save(output_path, images)
        print(f"Saved {num_samples} skeleton images to {output_path}")
    
    return images


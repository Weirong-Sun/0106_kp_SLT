"""
Configuration file for video-language alignment pipeline
All training and inference parameters are centralized here
"""
import os

# ==================== Paths ====================
# Data paths
DATASET_PATH = r"D:\dataset\Celebrity Faces Dataset"
KEYPOINTS_DATA_PATH = "keypoints_data.pkl"
BODY_KEYPOINTS_DATA_PATH = "sign_language_keypoints.pkl"
VIDEO_SEQUENCES_PATH = "video_sequences.pkl"
TEXT_DATA_PATH = "text_data.json"

# Model checkpoint paths
CHECKPOINTS = {
    'skeleton': 'checkpoints_skeleton_hierarchical',
    'temporal': 'checkpoints_temporal',
    'alignment': 'checkpoints_alignment',
    # Experimental models (not used in sign language pipeline)
    'hierarchical_keypoint': 'checkpoints_hierarchical',
    'hierarchical_image': 'checkpoints_hierarchical_image'
}

# Output paths
OUTPUT_DIRS = {
    'temporal_reprs': 'temporal_representations_all',
    'visualizations': 'visualizations'
}

# mBART model path
MBART_MODEL_PATH = r"../model/mbart-large-cc25"  # Local path to mBART model

# ==================== Skeleton Model ====================
# Note: HIERARCHICAL_KEYPOINT and HIERARCHICAL_IMAGE are experimental models
# for facial keypoint reconstruction (68 points), not used in sign language pipeline
SKELETON = {
    'model': {
        'd_global': 256,
        'd_region': 128,
        'num_regions': 4,  # Face, Left Hand, Right Hand, Pose
        'nhead': 8,
        'num_region_layers': 2,
        'num_interaction_layers': 2,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'image_size': 256,
        'num_keypoints': 143  # 68 face + 21 left hand + 21 right hand + 33 pose
    },
    'training': {
        'data_path': BODY_KEYPOINTS_DATA_PATH,
        'batch_size': 16,
        'epochs': 100,
        'lr': 1e-4,
        'use_weighted_loss': True,
        'hand_weight': 2.0,
        'face_weight': 1.5,
        'save_dir': CHECKPOINTS['skeleton']
    },
    'inference': {
        'checkpoint': os.path.join(CHECKPOINTS['skeleton'], 'best_model.pth'),
        'data_path': BODY_KEYPOINTS_DATA_PATH,
        'num_samples': 10
    }
}

# ==================== Temporal Model ====================
TEMPORAL = {
    'model': {
        'd_global': 256,
        'd_region': 128,
        'num_regions': 4,
        'd_temporal': 512,
        'd_final': 512,
        'nhead': 8,
        'num_temporal_layers': 4,
        'num_local_vars': 2,  # Number of local temporal variables
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_seq_len': 300,
        'fusion_method': 'concat',
        'freeze_frame_encoder': True
    },
    'training': {
        'video_data_path': VIDEO_SEQUENCES_PATH,
        'frame_encoder_checkpoint': os.path.join(CHECKPOINTS['skeleton'], 'best_model.pth'),
        'batch_size': 8,
        'seq_len': 6,
        'min_seq_len': 4,
        'epochs': 50,
        'lr': 1e-4,
        'save_dir': CHECKPOINTS['temporal']
    },
    'inference': {
        'checkpoint': os.path.join(CHECKPOINTS['temporal'], 'best_model_stage1.pth'),
        'video_sequences': VIDEO_SEQUENCES_PATH,
        'num_samples': 152,
        'output_dir': OUTPUT_DIRS['temporal_reprs']
    }
}

# ==================== Alignment Model ====================
ALIGNMENT = {
    'model': {
        'video_repr_dim': 1536,  # global (512) + local (2*512)
        'mbart_model_name': 'facebook/mbart-large-50',
        'mbart_model_path': MBART_MODEL_PATH,
        'd_model': 1024,
        'dropout': 0.1,
        'freeze_mbart': False
    },
    'training': {
        'video_reprs_path': os.path.join(OUTPUT_DIRS['temporal_reprs'], 'all_representations.npz'),
        'text_data_path': TEXT_DATA_PATH,
        'batch_size': 4,
        'epochs': 20,
        'lr': 1e-4,
        'max_text_length': 128,
        'save_dir': CHECKPOINTS['alignment']
    },
    'inference': {
        'checkpoint': os.path.join(CHECKPOINTS['alignment'], 'best_model.pth'),
        'video_reprs_path': os.path.join(OUTPUT_DIRS['temporal_reprs'], 'all_representations.npz'),
        'text_data_path': TEXT_DATA_PATH,
        'num_samples': 10,
        'max_length': 128,
        'num_beams': 4,
        'output_path': 'generation_results.json'
    }
}

# ==================== Device Configuration ====================
DEVICE = 'cuda'  # or 'cpu'

# ==================== Pipeline Configuration ====================
PIPELINE = {
    'enabled': True,
    'stages': [
        'skeleton',      # Train skeleton model first
        'temporal',      # Then train temporal model
        'alignment'      # Finally train alignment model
    ],
    'skip_trained': True,  # Skip if checkpoint already exists
    'run_inference': True  # Run inference after training
}

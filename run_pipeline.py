"""
Pipeline script to run the complete video-language alignment pipeline
Reads configuration from config/config.py and executes training/inference stages
"""
import os
import sys
import subprocess
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config.config import *
except ImportError as e:
    print(f"❌ Error: Failed to import config: {e}")
    print("Please ensure config/config.py exists and is valid.")
    sys.exit(1)

def run_command(cmd, description):
    """Run a command and print output"""
    print("\n" + "="*60)
    print(f"{description}")
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    print("-"*60)
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"\n✅ Success: {description} completed")
        return True

def train_skeleton(config):
    """Train skeleton model"""
    train_script = "training/skeleton/train.py"
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.py")
    cmd = [
        "python", train_script,
        "--config", config_path
    ]
    
    return run_command(cmd, "Training Skeleton Model")

def train_temporal(config):
    """Train temporal model"""
    train_script = "training/temporal/train.py"
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.py")
    cmd = [
        "python", train_script,
        "--config", config_path
    ]
    
    return run_command(cmd, "Training Temporal Model")

def train_alignment(config):
    """Train alignment model"""
    train_script = "training/alignment/train.py"
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.py")
    cmd = [
        "python", train_script,
        "--config", config_path
    ]
    
    return run_command(cmd, "Training Alignment Model")

def inference_temporal(config):
    """Run temporal model inference"""
    inference_script = "inference/temporal/inference.py"
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.py")
    cmd = [
        "python", inference_script,
        "--config", config_path
    ]
    
    return run_command(cmd, "Temporal Model Inference")

def inference_alignment(config):
    """Run alignment model inference"""
    inference_script = "inference/alignment/inference.py"
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.py")
    cmd = [
        "python", inference_script,
        "--config", config_path
    ]
    
    return run_command(cmd, "Alignment Model Inference")

def check_checkpoint_exists(checkpoint_path):
    """Check if checkpoint file exists"""
    return os.path.exists(checkpoint_path) if checkpoint_path else False

def main():
    parser = argparse.ArgumentParser(description="Run video-language alignment pipeline")
    parser.add_argument('--stage', type=str, default=None,
                        choices=['skeleton', 'temporal', 'alignment', 'all'],
                        help='Which stage to run (default: all from config)')
    parser.add_argument('--skip-trained', action='store_true',
                        help='Skip stages if checkpoint already exists')
    parser.add_argument('--no-inference', action='store_true',
                        help='Skip inference after training')
    parser.add_argument('--config-override', type=str, default=None,
                        help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Override config if provided
    if args.config_override:
        global SKELETON, TEMPORAL, ALIGNMENT, PIPELINE
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config_override)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        SKELETON = config_module.SKELETON
        TEMPORAL = config_module.TEMPORAL
        ALIGNMENT = config_module.ALIGNMENT
        PIPELINE = config_module.PIPELINE
    
    # Determine which stages to run
    if args.stage:
        if args.stage == 'all':
            stages = ['skeleton', 'temporal', 'alignment']
        else:
            stages = [args.stage]
    else:
        stages = PIPELINE.get('stages', ['skeleton', 'temporal', 'alignment'])
    
    skip_trained = args.skip_trained or PIPELINE.get('skip_trained', True)
    run_inference = not args.no_inference and PIPELINE.get('run_inference', True)
    
    print("\n" + "="*60)
    print("VIDEO-LANGUAGE ALIGNMENT PIPELINE")
    print("="*60)
    print(f"Stages to run: {stages}")
    print(f"Skip trained: {skip_trained}")
    print(f"Run inference: {run_inference}")
    print("="*60)
    
    # Run training stages
    for stage in stages:
        if stage == 'skeleton':
            checkpoint = os.path.join(SKELETON['training']['save_dir'], 'best_model.pth')
            if skip_trained and check_checkpoint_exists(checkpoint):
                print(f"\n⏭️  Skipping skeleton training (checkpoint exists: {checkpoint})")
            else:
                if not train_skeleton(SKELETON):
                    print("\n❌ Pipeline failed at skeleton training stage")
                    return 1
        
        elif stage == 'temporal':
            checkpoint = os.path.join(TEMPORAL['training']['save_dir'], 'best_model_stage1.pth')
            if skip_trained and check_checkpoint_exists(checkpoint):
                print(f"\n⏭️  Skipping temporal training (checkpoint exists: {checkpoint})")
            else:
                if not train_temporal(TEMPORAL):
                    print("\n❌ Pipeline failed at temporal training stage")
                    return 1
            
            # Run temporal inference to generate representations
            if run_inference:
                inference_temporal(TEMPORAL)
        
        elif stage == 'alignment':
            checkpoint = os.path.join(ALIGNMENT['training']['save_dir'], 'best_model.pth')
            if skip_trained and check_checkpoint_exists(checkpoint):
                print(f"\n⏭️  Skipping alignment training (checkpoint exists: {checkpoint})")
            else:
                if not train_alignment(ALIGNMENT):
                    print("\n❌ Pipeline failed at alignment training stage")
                    return 1
            
            # Run alignment inference
            if run_inference:
                inference_alignment(ALIGNMENT)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    return 0

if __name__ == "__main__":
    exit(main())


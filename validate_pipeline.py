"""
Pipeline Validation Script
检查所有配置参数和依赖关系，确保 pipeline 可以正常运行
"""
import os
import sys

def check_file_exists(filepath, name):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"[OK] {name}: {filepath}")
        return True
    else:
        print(f"[FAIL] {name}: {filepath} (NOT FOUND)")
        return False

def check_dir_exists(dirpath, name, create=False):
    """检查目录是否存在"""
    if os.path.exists(dirpath):
        print(f"[OK] {name}: {dirpath}")
        return True
    else:
        if create:
            os.makedirs(dirpath, exist_ok=True)
            print(f"[WARN] {name}: {dirpath} (CREATED)")
            return True
        else:
            print(f"[FAIL] {name}: {dirpath} (NOT FOUND)")
            return False

def validate_config():
    """验证配置文件"""
    print("\n" + "="*60)
    print("VALIDATING CONFIGURATION")
    print("="*60)
    
    # 导入配置
    try:
        from config.config import (
            SKELETON, TEMPORAL, ALIGNMENT,
            BODY_KEYPOINTS_DATA_PATH, VIDEO_SEQUENCES_PATH, TEXT_DATA_PATH,
            CHECKPOINTS, OUTPUT_DIRS, MBART_MODEL_PATH
        )
        print("[OK] Config file loaded successfully")
    except Exception as e:
        print(f"[FAIL] Failed to load config: {e}")
        return False, False
    
    all_ok = True
    mbart_exists = False
    
    # 检查数据文件
    print("\n--- Data Files ---")
    all_ok &= check_file_exists(BODY_KEYPOINTS_DATA_PATH, "Body keypoints data")
    all_ok &= check_file_exists(VIDEO_SEQUENCES_PATH, "Video sequences data")
    all_ok &= check_file_exists(TEXT_DATA_PATH, "Text data")
    
    # 检查 mBART 模型路径（可选，如果不存在会从 HuggingFace 下载）
    print("\n--- Model Paths ---")
    mbart_exists = check_dir_exists(MBART_MODEL_PATH, "mBART model path")
    if not mbart_exists:
        print("  [INFO] mBART model not found locally. Will use HuggingFace if path not set.")
        print("  [INFO] This is OK if using HuggingFace model name instead of local path.")
        # mBART 路径不存在不影响整体配置验证
        # all_ok 保持不变
    
    # 检查 checkpoint 目录（不存在则创建）
    print("\n--- Checkpoint Directories ---")
    for name, path in CHECKPOINTS.items():
        all_ok &= check_dir_exists(path, f"{name} checkpoint dir", create=True)
    
    # 检查输出目录（不存在则创建）
    print("\n--- Output Directories ---")
    for name, path in OUTPUT_DIRS.items():
        all_ok &= check_dir_exists(path, f"{name} output dir", create=True)
    
    # 验证配置参数
    print("\n--- Configuration Parameters ---")
    
    # Skeleton 配置
    print("\nSkeleton Model:")
    required_keys = ['model', 'training', 'inference']
    for key in required_keys:
        if key in SKELETON:
            print(f"  [OK] {key} config exists")
        else:
            print(f"  [FAIL] {key} config missing")
            all_ok = False
    
    # Temporal 配置
    print("\nTemporal Model:")
    for key in required_keys:
        if key in TEMPORAL:
            print(f"  [OK] {key} config exists")
        else:
            print(f"  [FAIL] {key} config missing")
            all_ok = False
    
    # Alignment 配置
    print("\nAlignment Model:")
    for key in required_keys:
        if key in ALIGNMENT:
            print(f"  [OK] {key} config exists")
        else:
            print(f"  [FAIL] {key} config missing")
            all_ok = False
    
    # 检查依赖关系
    print("\n--- Dependency Check ---")
    
    # Temporal 依赖 Skeleton
    skeleton_checkpoint = os.path.join(CHECKPOINTS['skeleton'], 'best_model.pth')
    if os.path.exists(skeleton_checkpoint):
        print(f"  [OK] Skeleton checkpoint exists (required by Temporal): {skeleton_checkpoint}")
    else:
        print(f"  [WARN] Skeleton checkpoint not found (Temporal will need it): {skeleton_checkpoint}")
    
    # Alignment 依赖 Temporal 推理输出
    temporal_reprs = os.path.join(OUTPUT_DIRS['temporal_reprs'], 'all_representations.npz')
    if os.path.exists(temporal_reprs):
        print(f"  [OK] Temporal representations exist (required by Alignment): {temporal_reprs}")
    else:
        print(f"  [WARN] Temporal representations not found (Alignment will need it): {temporal_reprs}")
    
    # 验证路径一致性
    print("\n--- Path Consistency Check ---")
    
    # Temporal frame encoder checkpoint 路径
    expected_frame_checkpoint = os.path.join(CHECKPOINTS['skeleton'], 'best_model.pth')
    actual_frame_checkpoint = TEMPORAL['training'].get('frame_encoder_checkpoint', '')
    if expected_frame_checkpoint == actual_frame_checkpoint or os.path.exists(actual_frame_checkpoint):
        print(f"  [OK] Temporal frame encoder checkpoint path: {actual_frame_checkpoint}")
    else:
        print(f"  [WARN] Temporal frame encoder checkpoint path may be incorrect: {actual_frame_checkpoint}")
    
    # Alignment video reprs 路径
    expected_video_reprs = os.path.join(OUTPUT_DIRS['temporal_reprs'], 'all_representations.npz')
    actual_video_reprs = ALIGNMENT['training'].get('video_reprs_path', '')
    if expected_video_reprs == actual_video_reprs:
        print(f"  [OK] Alignment video reprs path: {actual_video_reprs}")
    else:
        print(f"  [WARN] Alignment video reprs path may be incorrect: {actual_video_reprs}")
    
    # 返回配置验证结果和 mbart 状态
    return all_ok, mbart_exists

def validate_training_scripts():
    """验证训练脚本"""
    print("\n" + "="*60)
    print("VALIDATING TRAINING SCRIPTS")
    print("="*60)
    
    scripts = [
        'training/skeleton/train.py',
        'training/temporal/train.py',
        'training/alignment/train.py'
    ]
    
    all_ok = True
    for script in scripts:
        if os.path.exists(script):
            print(f"[OK] {script}")
        else:
            print(f"[FAIL] {script} (NOT FOUND)")
            all_ok = False
    
    return all_ok

def validate_inference_scripts():
    """验证推理脚本"""
    print("\n" + "="*60)
    print("VALIDATING INFERENCE SCRIPTS")
    print("="*60)
    
    scripts = [
        'inference/temporal/inference.py',
        'inference/alignment/inference.py'
    ]
    
    all_ok = True
    for script in scripts:
        if os.path.exists(script):
            print(f"[OK] {script}")
        else:
            print(f"[FAIL] {script} (NOT FOUND)")
            all_ok = False
    
    return all_ok

def validate_model_files():
    """验证模型文件"""
    print("\n" + "="*60)
    print("VALIDATING MODEL FILES")
    print("="*60)
    
    models = [
        'models/skeleton/model.py',
        'models/temporal/model.py',
        'models/alignment/model.py'
    ]
    
    all_ok = True
    for model in models:
        if os.path.exists(model):
            print(f"[OK] {model}")
        else:
            print(f"[FAIL] {model} (NOT FOUND)")
            all_ok = False
    
    return all_ok

def main():
    """主验证函数"""
    print("="*60)
    print("PIPELINE VALIDATION")
    print("="*60)
    
    results = []
    mbart_exists = False
    
    # 验证配置
    config_ok, mbart_exists = validate_config()
    results.append(("Configuration", config_ok))
    
    # 验证训练脚本
    results.append(("Training Scripts", validate_training_scripts()))
    
    # 验证推理脚本
    results.append(("Inference Scripts", validate_inference_scripts()))
    
    # 验证模型文件
    results.append(("Model Files", validate_model_files()))
    
    # 总结
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    critical_failures = []
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
            # Configuration 失败可能只是因为 mBART 路径，不算严重问题
            if name != "Configuration" or not mbart_exists:
                critical_failures.append(name)
    
    print("\n" + "="*60)
    if all_passed or (len(critical_failures) == 0 and not mbart_exists):
        print("[SUCCESS] ALL CRITICAL VALIDATIONS PASSED")
        if not mbart_exists:
            print("[INFO] mBART model path not found, but this is OK.")
            print("        The model will be loaded from HuggingFace if needed.")
        print("Pipeline should be ready to run!")
    else:
        print("[WARNING] SOME VALIDATIONS FAILED")
        if critical_failures:
            print(f"Critical failures: {', '.join(critical_failures)}")
        else:
            print("Non-critical issues detected. Pipeline may still work.")
        print("Please review the issues above before running the pipeline.")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())


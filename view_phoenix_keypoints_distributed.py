#!/usr/bin/env python
"""
查看 extract_phoenix_keypoints_distributed.py 生成的数据结构
支持查看 .train, .dev, .test 文件
"""
import pickle
import argparse
import sys
from pathlib import Path
import numpy as np

# 可选导入 torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def view_keypoint_file(file_path, detailed=False):
    """
    查看关键点文件的结构

    Args:
        file_path: pickle 文件路径
        detailed: 是否显示详细信息
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return False

    print("="*70)
    print(f"查看文件: {file_path}")
    print("="*70)
    print(f"文件大小: {file_path.stat().st_size / (1024*1024):.2f} MB")
    print()

    # 加载数据
    print("加载数据...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"错误: 加载文件失败: {e}")
        return False

    # 基本信息
    print("\n" + "="*70)
    print("数据结构")
    print("="*70)
    print(f"类型: {type(data)}")

    if isinstance(data, dict):
        print(f"键数量: {len(data)}")
        print(f"\n顶层键示例 (前10个):")
        keys = list(data.keys())[:10]
        for key in keys:
            print(f"  - {key}")
        if len(data) > 10:
            print(f"  ... (共 {len(data)} 个键)")

        # 查看第一个样本的结构
        if len(data) > 0:
            first_key = list(data.keys())[0]
            first_value = data[first_key]

            print(f"\n第一个样本键: {first_key}")
            print(f"第一个样本值类型: {type(first_value)}")

            if isinstance(first_value, dict):
                print(f"\n第一个样本的字段:")
                for field, value in first_value.items():
                    print(f"  - {field}: {type(value).__name__}")

                    if field == 'keypoint':
                        if HAS_TORCH and isinstance(value, torch.Tensor):
                            print(f"     形状: {value.shape}")
                            print(f"     数据类型: {value.dtype}")
                        elif isinstance(value, np.ndarray):
                            print(f"     形状: {value.shape}")
                            print(f"     数据类型: {value.dtype}")
                        elif hasattr(value, 'shape'):
                            print(f"     形状: {value.shape}")
                            if hasattr(value, 'dtype'):
                                print(f"     数据类型: {value.dtype}")
                        else:
                            print(f"     类型: {type(value).__name__}")
                    elif field == 'name':
                        print(f"     值: {value}")
                    elif field == 'gloss':
                        print(f"     值: {value[:100] if len(str(value)) > 100 else value}")
                    elif field == 'text':
                        print(f"     值: {value[:100] if len(str(value)) > 100 else value}")
                    elif field == 'num_frames':
                        print(f"     值: {value}")

            # 详细统计
            if detailed:
                print(f"\n" + "="*70)
                print("详细统计")
                print("="*70)

                # 按划分统计
                splits = {}
                for key in data.keys():
                    split = key.split('/')[0] if '/' in key else 'unknown'
                    if split not in splits:
                        splits[split] = []
                    splits[split].append(key)

                print(f"\n按划分统计:")
                for split, keys in splits.items():
                    print(f"  {split}: {len(keys)} 个视频")

                # 关键点统计
                keypoint_shapes = []
                num_frames_list = []
                for key, value in data.items():
                    if isinstance(value, dict) and 'keypoint' in value:
                        kp = value['keypoint']
                        if hasattr(kp, 'shape'):
                            keypoint_shapes.append(kp.shape)
                            num_frames_list.append(kp.shape[0] if len(kp.shape) >= 2 else 1)
                    elif isinstance(value, dict) and 'num_frames' in value:
                        num_frames_list.append(value['num_frames'])

                if keypoint_shapes:
                    print(f"\n关键点形状统计:")
                    unique_shapes = {}
                    for shape in keypoint_shapes:
                        shape_str = str(shape)
                        unique_shapes[shape_str] = unique_shapes.get(shape_str, 0) + 1

                    for shape_str, count in sorted(unique_shapes.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"  {shape_str}: {count} 个视频")

                    print(f"\n帧数统计:")
                    if num_frames_list:
                        print(f"  最小帧数: {min(num_frames_list)}")
                        print(f"  最大帧数: {max(num_frames_list)}")
                        print(f"  平均帧数: {sum(num_frames_list) / len(num_frames_list):.1f}")

                # 样本示例
                print(f"\n" + "="*70)
                print("样本示例 (前3个)")
                print("="*70)
                for i, (key, value) in enumerate(list(data.items())[:3]):
                    print(f"\n样本 {i+1}: {key}")
                    if isinstance(value, dict):
                        for field, field_value in value.items():
                            if field == 'keypoint':
                                if HAS_TORCH and isinstance(field_value, torch.Tensor):
                                    print(f"  {field}: tensor, shape={field_value.shape}, dtype={field_value.dtype}")
                                elif isinstance(field_value, np.ndarray):
                                    print(f"  {field}: array, shape={field_value.shape}, dtype={field_value.dtype}")
                                elif hasattr(field_value, 'shape'):
                                    print(f"  {field}: {type(field_value).__name__}, shape={field_value.shape}")
                                    if hasattr(field_value, 'dtype'):
                                        print(f"    dtype={field_value.dtype}")
                                else:
                                    print(f"  {field}: {type(field_value).__name__}")
                            elif field == 'name':
                                print(f"  {field}: {field_value}")
                            elif field == 'gloss':
                                print(f"  {field}: {field_value}")
                            elif field == 'text':
                                print(f"  {field}: {field_value[:80]}..." if len(str(field_value)) > 80 else f"  {field}: {field_value}")
                            elif field == 'num_frames':
                                print(f"  {field}: {field_value}")

    print("\n" + "="*70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='查看 extract_phoenix_keypoints_distributed.py 生成的数据结构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 查看 train 文件
  python view_phoenix_keypoints_distributed.py phoenix_keypoints.train

  # 查看 dev 文件（详细信息）
  python view_phoenix_keypoints_distributed.py phoenix_keypoints.dev --detailed

  # 查看所有划分的文件
  python view_phoenix_keypoints_distributed.py phoenix_keypoints.train
  python view_phoenix_keypoints_distributed.py phoenix_keypoints.dev
  python view_phoenix_keypoints_distributed.py phoenix_keypoints.test
        """
    )

    parser.add_argument(
        'file_path',
        type=str,
        help='要查看的 pickle 文件路径（例如: phoenix_keypoints.train）'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='显示详细信息（统计、示例等）'
    )

    args = parser.parse_args()

    success = view_keypoint_file(args.file_path, detailed=args.detailed)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()


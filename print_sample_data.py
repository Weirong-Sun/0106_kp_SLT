#!/usr/bin/env python
"""
打印 phoenix_keypoints.train 文件中一条数据的详细内容
"""
import pickle
import sys
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    print("错误: 需要 torch 模块来加载数据")
    print("请安装: pip install torch")
    sys.exit(1)


def print_sample_data(file_path, sample_idx=0):
    """
    打印一条数据的详细内容

    Args:
        file_path: pickle 文件路径
        sample_idx: 要打印的样本索引（默认: 0，即第一个）
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return False

    print("="*70)
    print(f"加载文件: {file_path}")
    print("="*70)

    # 加载数据
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"错误: 加载文件失败: {e}")
        return False

    print(f"总视频数: {len(data)}")
    print()

    # 获取指定索引的样本
    if len(data) == 0:
        print("错误: 文件中没有数据")
        return False

    if sample_idx >= len(data):
        print(f"错误: 索引 {sample_idx} 超出范围（总共 {len(data)} 个视频）")
        return False

    video_keys = sorted(list(data.keys()))
    selected_key = video_keys[sample_idx]
    selected_data = data[selected_key]

    print("="*70)
    print(f"样本 {sample_idx + 1}/{len(data)}")
    print("="*70)
    print(f"视频键: {selected_key}")
    print()

    if isinstance(selected_data, dict):
        print("数据内容:")
        print("-"*70)

        for field, value in selected_data.items():
            print(f"\n【{field}】")

            if field == 'keypoint':
                if isinstance(value, torch.Tensor):
                    print(f"  类型: torch.Tensor")
                    print(f"  形状: {value.shape}")
                    print(f"  数据类型: {value.dtype}")
                    print(f"  设备: {value.device}")

                    # 显示部分关键点数据
                    print(f"\n  关键点数据预览（前5帧，前10个关键点）:")
                    if len(value.shape) >= 2:
                        num_frames = value.shape[0]
                        num_keypoints = value.shape[1] if len(value.shape) >= 2 else 1

                        print(f"  形状说明: [帧数={num_frames}, 关键点数={num_keypoints}, 坐标维度=3]")
                        print(f"  前5帧前10个关键点的坐标:")

                        # 显示前5帧的前10个关键点
                        for frame_idx in range(min(5, num_frames)):
                            print(f"\n    帧 {frame_idx + 1}:")
                            for kp_idx in range(min(10, num_keypoints)):
                                kp_coords = value[frame_idx, kp_idx].cpu().numpy()
                                print(f"      关键点 {kp_idx:2d}: x={kp_coords[0]:.6f}, y={kp_coords[1]:.6f}, z={kp_coords[2]:.6f}")

                        if num_frames > 5 or num_keypoints > 10:
                            print(f"    ... (共 {num_frames} 帧, 每帧 {num_keypoints} 个关键点)")

                elif hasattr(value, 'shape'):
                    print(f"  类型: {type(value).__name__}")
                    print(f"  形状: {value.shape}")
                    if hasattr(value, 'dtype'):
                        print(f"  数据类型: {value.dtype}")

            elif field == 'name':
                print(f"  类型: {type(value).__name__}")
                print(f"  值: {value}")

            elif field == 'gloss':
                print(f"  类型: {type(value).__name__}")
                print(f"  值: {value}")
                print(f"  长度: {len(str(value))} 字符")

            elif field == 'text':
                print(f"  类型: {type(value).__name__}")
                print(f"  值: {value}")
                print(f"  长度: {len(str(value))} 字符")

            elif field == 'num_frames':
                print(f"  类型: {type(value).__name__}")
                print(f"  值: {value}")

            else:
                print(f"  类型: {type(value).__name__}")
                print(f"  值: {value}")

    print("\n" + "="*70)

    # 显示关键点信息说明
    print("\n关键点说明:")
    print("-"*70)
    print("关键点总数: 143")
    print("  - 面部 (Face): 68 个关键点 (索引 0-67)")
    print("  - 左手 (Left hand): 21 个关键点 (索引 68-88)")
    print("  - 右手 (Right hand): 21 个关键点 (索引 89-109)")
    print("  - 姿态 (Pose): 33 个关键点 (索引 110-142)")
    print()
    print("坐标维度: 3 (x, y, z)")
    print("  - x: 归一化的 x 坐标 [0, 1]")
    print("  - y: 归一化的 y 坐标 [0, 1]")
    print("  - z: 深度信息")
    print("="*70)

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='打印 phoenix_keypoints.train 文件中一条数据的详细内容'
    )

    parser.add_argument(
        'file_path',
        type=str,
        nargs='?',
        default='phoenix_keypoints.train',
        help='要查看的文件路径（默认: phoenix_keypoints.train）'
    )
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='要打印的样本索引（默认: 0，即第一个）'
    )

    args = parser.parse_args()

    success = print_sample_data(args.file_path, sample_idx=args.index)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()



"""
准备模型输入数据
从关键点提取结果中准备按 train/dev/test 划分的数据，方便模型训练使用
"""
import pickle
import argparse
from pathlib import Path
import numpy as np


def prepare_model_input(keypoints_file, output_dir, splits=None, normalize=False):
    """
    准备模型输入数据

    Args:
        keypoints_file: 关键点 pickle 文件路径
        output_dir: 输出目录
        splits: 要准备的划分列表（默认: 所有划分）
    """
    if splits is None:
        splits = ['train', 'dev', 'test']

    keypoints_file = Path(keypoints_file)
    if not keypoints_file.exists():
        raise ValueError(f"关键点文件不存在: {keypoints_file}")

    print("="*60)
    print("准备模型输入数据")
    print("="*60)
    print(f"输入文件: {keypoints_file}")
    print(f"输出目录: {output_dir}")
    print(f"处理的划分: {splits}")
    print("="*60)

    # 加载关键点数据
    print("\n加载关键点数据...")
    with open(keypoints_file, 'rb') as f:
        data = pickle.load(f)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 准备每个划分的数据
    for split in splits:
        if split not in data:
            print(f"警告: 划分 '{split}' 不存在，跳过")
            continue

        print(f"\n处理 {split.upper()} 划分...")
        split_data = data[split]
        keypoints_list = split_data.get('keypoints', [])
        image_paths = split_data.get('image_paths', [])
        video_ids = split_data.get('video_ids', [])

        if len(keypoints_list) == 0:
            print(f"警告: {split} 划分没有数据，跳过")
            continue

        # 准备关键点数组（方便模型输入）
        # 格式: [N, 143, 3] - N 个样本，每个样本 143 个关键点，每个关键点 3 个坐标
        prepared_keypoints = []
        valid_indices = []

        for idx, kp_dict in enumerate(keypoints_list):
            if kp_dict is None:
                continue

            # 组合所有关键点
            kp_array = []

            # 面部关键点 (68)
            if kp_dict.get('face') is not None:
                kp_array.append(kp_dict['face'])
            else:
                kp_array.append(np.zeros((68, 3), dtype=np.float32))

            # 左手关键点 (21)
            if kp_dict.get('left_hand') is not None:
                kp_array.append(kp_dict['left_hand'])
            else:
                kp_array.append(np.zeros((21, 3), dtype=np.float32))

            # 右手关键点 (21)
            if kp_dict.get('right_hand') is not None:
                kp_array.append(kp_dict['right_hand'])
            else:
                kp_array.append(np.zeros((21, 3), dtype=np.float32))

            # 姿态关键点 (33)
            if kp_dict.get('pose') is not None:
                kp_array.append(kp_dict['pose'])
            else:
                kp_array.append(np.zeros((33, 3), dtype=np.float32))

            # 组合成 [143, 3]
            combined_kp = np.concatenate(kp_array, axis=0)
            prepared_keypoints.append(combined_kp)
            valid_indices.append(idx)

        if len(prepared_keypoints) == 0:
            print(f"警告: {split} 划分没有有效关键点，跳过")
            continue

        # 转换为 numpy 数组 [N, 143, 3]
        keypoints_array = np.array(prepared_keypoints, dtype=np.float32)

        print(f"  有效样本数: {len(keypoints_array)}")
        print(f"  关键点形状: {keypoints_array.shape}")

        # 归一化（可选）
        if normalize:
            print("  归一化关键点...")
            # 归一化到 [0, 1]（假设原始坐标在 [0, 1] 范围内）
            # 如果坐标不在 [0, 1] 范围内，需要根据实际情况调整
            keypoints_array = np.clip(keypoints_array, 0, 1)

        # 准备对应的元数据
        valid_image_paths = [image_paths[i] for i in valid_indices]
        valid_video_ids = [video_ids[i] for i in valid_indices]

        # 保存为不同格式
        # 1. NumPy 格式（方便模型直接加载）
        npz_path = output_dir / f"{split}_keypoints.npz"
        np.savez_compressed(
            str(npz_path),
            keypoints=keypoints_array,
            image_paths=valid_image_paths,
            video_ids=valid_video_ids
        )
        print(f"  ✓ 保存 NumPy 格式: {npz_path.name}")

        # 2. Pickle 格式（保持完整信息）
        pkl_path = output_dir / f"{split}_keypoints.pkl"
        split_output = {
            'keypoints': keypoints_array,
            'keypoints_dict': [keypoints_list[i] for i in valid_indices],  # 原始字典格式
            'image_paths': valid_image_paths,
            'video_ids': valid_video_ids,
            'split': split,
            'num_samples': len(keypoints_array),
            'keypoint_shape': keypoints_array.shape
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(split_output, f)
        print(f"  ✓ 保存 Pickle 格式: {pkl_path.name}")

        # 3. 统计信息文件
        stats_path = output_dir / f"{split}_stats.json"
        import json
        stats = {
            'split': split,
            'num_samples': len(keypoints_array),
            'num_videos': len(set(valid_video_ids)),
            'keypoint_shape': list(keypoints_array.shape),
            'face_detected': sum(1 for i in valid_indices if keypoints_list[i].get('face') is not None),
            'left_hand_detected': sum(1 for i in valid_indices if keypoints_list[i].get('left_hand') is not None),
            'right_hand_detected': sum(1 for i in valid_indices if keypoints_list[i].get('right_hand') is not None),
            'pose_detected': sum(1 for i in valid_indices if keypoints_list[i].get('pose') is not None)
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 保存统计信息: {stats_path.name}")

    print(f"\n{'='*60}")
    print("数据准备完成!")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"\n生成的文件:")
    for split in splits:
        print(f"  {split}:")
        print(f"    - {split}_keypoints.npz (NumPy 格式，模型输入)")
        print(f"    - {split}_keypoints.pkl (Pickle 格式，完整信息)")
        print(f"    - {split}_stats.json (统计信息)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='准备模型输入数据，从关键点提取结果中准备按 train/dev/test 划分的数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 准备所有划分的数据
  python data/prepare_model_input.py \\
      --keypoints_file phoenix_keypoints_full.pkl \\
      --output_dir model_input_data

  # 只准备训练集
  python data/prepare_model_input.py \\
      --keypoints_file phoenix_keypoints_full.pkl \\
      --output_dir model_input_data \\
      --splits train

  # 归一化关键点
  python data/prepare_model_input.py \\
      --keypoints_file phoenix_keypoints_full.pkl \\
      --output_dir model_input_data \\
      --normalize
        """
    )

    parser.add_argument(
        '--keypoints_file',
        type=str,
        required=True,
        help='关键点 pickle 文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='model_input_data',
        help='输出目录（默认: model_input_data）'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        choices=['train', 'dev', 'test'],
        default=None,
        help='要准备的划分列表 (默认: 所有划分)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='是否归一化关键点坐标'
    )

    args = parser.parse_args()

    prepare_model_input(
        keypoints_file=args.keypoints_file,
        output_dir=args.output_dir,
        splits=args.splits,
        normalize=args.normalize
    )

    print("\n完成!")


if __name__ == "__main__":
    main()



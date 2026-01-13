#!/usr/bin/env python
"""
将 phoenix_keypoints.pkl 按照 train、dev、test 划分成三个独立的文件
输出文件:
- mediaPipe_Phoenix.train
- mediaPipe_Phoenix.dev
- mediaPipe_Phoenix.test
"""
import pickle
import argparse
from pathlib import Path
import sys


def split_keypoints_file(input_file, output_prefix="mediaPipe_Phoenix"):
    """
    将关键点文件按照划分分割成三个独立文件

    Args:
        input_file: 输入 pickle 文件路径
        output_prefix: 输出文件前缀（默认: mediaPipe_Phoenix）
    """
    input_file = Path(input_file)
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return False

    print("="*60)
    print("分割 PHOENIX 关键点文件")
    print("="*60)
    print(f"输入文件: {input_file}")
    print(f"文件大小: {input_file.stat().st_size / (1024*1024):.2f} MB")
    print("="*60)

    # 加载数据
    print("\n加载数据...")
    try:
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"错误: 加载文件失败: {e}")
        return False

    # 检查数据格式
    required_splits = ['train', 'dev', 'test']
    for split in required_splits:
        if split not in data:
            print(f"错误: 数据中缺少 '{split}' 划分")
            return False

    # 创建输出文件
    output_files = {}
    for split in required_splits:
        output_file = Path(f"{output_prefix}.{split}")
        output_files[split] = output_file

    print(f"\n输出文件:")
    for split, output_file in output_files.items():
        print(f"  {split.upper()}: {output_file}")

    # 分割数据
    print(f"\n分割数据...")

    for split in required_splits:
        print(f"\n处理 {split.upper()} 划分...")
        split_data = data[split]

        # 准备输出数据
        output_data = {
            split: {
                'keypoints': split_data.get('keypoints', []),
                'image_paths': split_data.get('image_paths', []),
                'video_ids': split_data.get('video_ids', [])
            },
            'keypoint_info': data.get('keypoint_info', {}),
            'dataset_info': {
                **data.get('dataset_info', {}),
                'source_file': str(input_file),
                'split': split,
                'original_splits': data.get('dataset_info', {}).get('splits', [])
            }
        }

        # 如果有统计信息，只包含当前划分的统计
        if 'stats' in data and split in data['stats']:
            output_data['stats'] = {
                split: data['stats'][split]
            }

        # 保存文件
        output_file = output_files[split]
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(output_data, f)

            # 显示信息
            file_size = output_file.stat().st_size / (1024*1024)
            num_samples = len(output_data[split]['keypoints'])
            num_videos = len(set(output_data[split]['video_ids'])) if output_data[split].get('video_ids') else 0

            print(f"  ✓ 保存完成: {output_file}")
            print(f"    文件大小: {file_size:.2f} MB")
            print(f"    样本数: {num_samples:,}")
            print(f"    视频数: {num_videos}")

        except Exception as e:
            print(f"  ✗ 保存失败: {e}")
            return False

    print(f"\n{'='*60}")
    print("分割完成!")
    print("="*60)
    print(f"\n生成的文件:")
    total_size = 0
    for split, output_file in output_files.items():
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024*1024)
            total_size += file_size
            num_samples = len(data[split]['keypoints'])
            print(f"  {output_file.name}: {file_size:.2f} MB ({num_samples:,} 个样本)")

    print(f"\n总文件大小: {total_size:.2f} MB")
    print("="*60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='将 PHOENIX 关键点文件按照 train、dev、test 划分成三个独立文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认输出文件名前缀
  python split_phoenix_keypoints.py phoenix_keypoints.pkl

  # 指定自定义输出文件名前缀
  python split_phoenix_keypoints.py phoenix_keypoints.pkl --output_prefix my_prefix

  输出文件:
    - mediaPipe_Phoenix.train
    - mediaPipe_Phoenix.dev
    - mediaPipe_Phoenix.test
        """
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='输入的关键点 pickle 文件路径'
    )
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='mediaPipe_Phoenix',
        help='输出文件前缀（默认: mediaPipe_Phoenix）'
    )

    args = parser.parse_args()

    success = split_keypoints_file(args.input_file, args.output_prefix)

    if success:
        print("\n✓ 成功完成!")
        sys.exit(0)
    else:
        print("\n✗ 处理失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()



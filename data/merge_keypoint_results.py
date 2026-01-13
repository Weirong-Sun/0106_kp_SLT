"""
合并多个关键点提取结果文件
用于整合分块处理的结果，方便按 train/dev/test 划分使用
"""
import pickle
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict


def merge_keypoint_files(file_paths, output_path, splits=None):
    """
    合并多个关键点 pickle 文件

    Args:
        file_paths: 要合并的文件路径列表
        output_path: 输出文件路径
        splits: 要合并的划分列表（默认: 所有划分）
    """
    if splits is None:
        splits = ['train', 'dev', 'test']

    print("="*60)
    print("合并关键点提取结果")
    print("="*60)
    print(f"输入文件数: {len(file_paths)}")
    print(f"输出文件: {output_path}")
    print(f"处理的划分: {splits}")
    print("="*60)

    # 存储合并后的数据
    merged_data = {
        'train': {'keypoints': [], 'image_paths': [], 'video_ids': []},
        'dev': {'keypoints': [], 'image_paths': [], 'video_ids': []},
        'test': {'keypoints': [], 'image_paths': [], 'video_ids': []}
    }

    # 统计信息
    merged_stats = {
        'train': {'total': 0, 'success': 0, 'failed': 0},
        'dev': {'total': 0, 'success': 0, 'failed': 0},
        'test': {'total': 0, 'success': 0, 'failed': 0}
    }

    # 处理信息
    processing_info = {
        'merged_files': [],
        'total_files': len(file_paths)
    }

    # 加载并合并所有文件
    for file_path in file_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"警告: 文件不存在，跳过: {file_path}")
            continue

        print(f"\n加载文件: {file_path.name}")
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"加载文件失败: {e}")
            continue

        # 合并每个划分
        for split in splits:
            if split not in data:
                continue

            split_data = data[split]
            keypoints = split_data.get('keypoints', [])
            image_paths = split_data.get('image_paths', [])
            video_ids = split_data.get('video_ids', [])

            # 添加到合并数据
            merged_data[split]['keypoints'].extend(keypoints)
            merged_data[split]['image_paths'].extend(image_paths)
            merged_data[split]['video_ids'].extend(video_ids)

            # 累计统计
            if 'stats' in data:
                split_stats = data['stats'].get(split, {})
                merged_stats[split]['total'] += split_stats.get('total', len(keypoints))
                merged_stats[split]['success'] += split_stats.get('success', len([k for k in keypoints if k is not None]))
                merged_stats[split]['failed'] += split_stats.get('failed', 0)

        processing_info['merged_files'].append(str(file_path))

        # 保存处理信息
        if 'dataset_info' in data and 'processing_info' in data['dataset_info']:
            proc_info = data['dataset_info']['processing_info']
            if 'processing_info_list' not in processing_info:
                processing_info['processing_info_list'] = []
            processing_info['processing_info_list'].append(proc_info)

    # 去重（根据图像路径）
    print("\n去重处理...")
    for split in splits:
        # 使用字典去重，保留第一个出现的
        seen = set()
        unique_keypoints = []
        unique_paths = []
        unique_video_ids = []

        for kp, path, vid in zip(
            merged_data[split]['keypoints'],
            merged_data[split]['image_paths'],
            merged_data[split]['video_ids']
        ):
            if path not in seen:
                seen.add(path)
                unique_keypoints.append(kp)
                unique_paths.append(path)
                unique_video_ids.append(vid)

        original_count = len(merged_data[split]['keypoints'])
        merged_data[split]['keypoints'] = unique_keypoints
        merged_data[split]['image_paths'] = unique_paths
        merged_data[split]['video_ids'] = unique_video_ids

        if original_count != len(unique_keypoints):
            print(f"  {split}: 去重 {original_count - len(unique_keypoints)} 个重复样本")

    # 准备输出数据
    output_data = {
        'train': {
            'keypoints': merged_data['train']['keypoints'],
            'image_paths': merged_data['train']['image_paths'],
            'video_ids': merged_data['train']['video_ids']
        },
        'dev': {
            'keypoints': merged_data['dev']['keypoints'],
            'image_paths': merged_data['dev']['image_paths'],
            'video_ids': merged_data['dev']['video_ids']
        },
        'test': {
            'keypoints': merged_data['test']['keypoints'],
            'image_paths': merged_data['test']['image_paths'],
            'video_ids': merged_data['test']['video_ids']
        },
        'stats': merged_stats,
        'keypoint_info': {
            'face': {'num_points': 68, 'description': 'Facial landmarks'},
            'left_hand': {'num_points': 21, 'description': 'Left hand landmarks'},
            'right_hand': {'num_points': 21, 'description': 'Right hand landmarks'},
            'pose': {'num_points': 33, 'description': 'Body pose landmarks'},
            'total_points': 68 + 21 + 21 + 33  # 143 points total
        },
        'dataset_info': {
            'name': 'PHOENIX-2014-T',
            'source_path': 'merged',
            'splits': splits,
            'processing_info': processing_info
        }
    }

    # 保存合并后的数据
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n{'='*60}")
    print("合并完成!")
    print(f"{'='*60}")
    print(f"输出文件: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"\n合并后统计:")
    for split in splits:
        num_samples = len(merged_data[split]['keypoints'])
        num_videos = len(set(merged_data[split]['video_ids'])) if merged_data[split]['video_ids'] else 0
        print(f"  {split}:")
        print(f"    样本数: {num_samples:,}")
        print(f"    视频数: {num_videos}")
        if merged_stats[split]['total'] > 0:
            success_rate = merged_stats[split]['success'] / merged_stats[split]['total'] * 100
            print(f"    成功率: {success_rate:.2f}%")

    total_samples = sum(len(merged_data[split]['keypoints']) for split in splits)
    print(f"\n  总样本数: {total_samples:,}")
    print("="*60)

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='合并多个关键点提取结果文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 合并多个结果文件
  python data/merge_keypoint_results.py \\
      --input_files result1.pkl result2.pkl result3.pkl \\
      --output_path merged_keypoints.pkl

  # 只合并特定划分
  python data/merge_keypoint_results.py \\
      --input_files result1.pkl result2.pkl \\
      --output_path merged_keypoints.pkl \\
      --splits train dev
        """
    )

    parser.add_argument(
        '--input_files',
        type=str,
        nargs='+',
        required=True,
        help='要合并的输入文件路径列表'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='输出合并后的文件路径'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        choices=['train', 'dev', 'test'],
        default=None,
        help='要合并的划分列表 (默认: 所有划分)'
    )

    args = parser.parse_args()

    merge_keypoint_files(
        file_paths=args.input_files,
        output_path=args.output_path,
        splits=args.splits
    )

    print("\n完成!")


if __name__ == "__main__":
    main()



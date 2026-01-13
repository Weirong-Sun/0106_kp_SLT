"""
从 PHOENIX-2014-T 数据集中提取全身关键点
使用 MediaPipe 提取面部、双手和姿态关键点

数据集结构：
PHOENIX-2014-T-release-v3/
  └── PHOENIX-2014-T/
      └── features/
          └── fullFrame-210x260px/
              ├── train/
              ├── dev/
              └── test/
# 处理所有划分
python data/extract_phoenix_keypoints.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints.pkl
# 只处理训练集
python data/extract_phoenix_keypoints.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_train.pkl \
    --splits train


"""
import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse
import json

# 导入现有的关键点提取器
# 支持相对导入和绝对导入
try:
    from .extract_body_keypoints import FullBodyKeypointExtractor
except ImportError:
    from extract_body_keypoints import FullBodyKeypointExtractor

class PhoenixKeypointExtractor:
    """
    专门用于处理 PHOENIX-2014-T 数据集的关键点提取器
    保持数据集的 train/dev/test 划分
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化关键点提取器

        Args:
            min_detection_confidence: 检测的最小置信度
            min_tracking_confidence: 跟踪的最小置信度
        """
        self.extractor = FullBodyKeypointExtractor(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_phoenix_dataset(self, dataset_path, output_path, splits=None, max_samples_per_split=None):
        """
        处理 PHOENIX-2014-T 数据集

        Args:
            dataset_path: PHOENIX 数据集根目录路径
                        例如: /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T
            output_path: 输出 pickle 文件路径
            splits: 要处理的划分列表，例如 ['train', 'dev', 'test']，None 表示处理所有
            max_samples_per_split: 每个划分最多处理的样本数（None 表示处理全部）

        Returns:
            output_data: 包含所有关键点数据的字典
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"数据集路径不存在: {dataset_path}")

        # PHOENIX 数据集的图像帧路径
        frames_dir = dataset_path / "features" / "fullFrame-210x260px"
        if not frames_dir.exists():
            raise ValueError(f"未找到图像帧目录: {frames_dir}")

        # 确定要处理的划分
        if splits is None:
            splits = ['train', 'dev', 'test']

        # 存储所有关键点数据
        all_data = {
            'train': {'keypoints': [], 'image_paths': [], 'video_ids': []},
            'dev': {'keypoints': [], 'image_paths': [], 'video_ids': []},
            'test': {'keypoints': [], 'image_paths': [], 'video_ids': []}
        }

        # 统计信息
        stats = {
            'train': {'total': 0, 'success': 0, 'failed': 0},
            'dev': {'total': 0, 'success': 0, 'failed': 0},
            'test': {'total': 0, 'success': 0, 'failed': 0}
        }

        # 处理每个划分
        for split in splits:
            split_dir = frames_dir / split
            if not split_dir.exists():
                print(f"警告: 划分目录不存在，跳过: {split_dir}")
                continue

            print(f"\n{'='*60}")
            print(f"处理 {split.upper()} 划分")
            print(f"{'='*60}")

            # 获取所有视频文件夹
            video_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            print(f"找到 {len(video_dirs)} 个视频文件夹")

            # 限制处理的视频数量
            if max_samples_per_split and len(video_dirs) > max_samples_per_split:
                video_dirs = video_dirs[:max_samples_per_split]
                print(f"限制处理前 {max_samples_per_split} 个视频")

            # 处理每个视频
            for video_dir in tqdm(video_dirs, desc=f"处理 {split} 视频"):
                video_id = video_dir.name

                # 获取该视频的所有图像帧
                image_files = sorted([
                    f for f in video_dir.iterdir()
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
                ])

                if len(image_files) == 0:
                    print(f"警告: 视频 {video_id} 中没有找到图像文件")
                    continue

                stats[split]['total'] += len(image_files)

                # 处理该视频的所有帧
                video_keypoints = []
                video_paths = []

                for image_path in image_files:
                    kp_dict = self.extractor.extract_keypoints(str(image_path))
                    if kp_dict is not None:
                        video_keypoints.append(kp_dict)
                        video_paths.append(str(image_path))
                        stats[split]['success'] += 1
                    else:
                        stats[split]['failed'] += 1

                # 如果该视频有成功提取的关键点，添加到数据中
                if len(video_keypoints) > 0:
                    all_data[split]['keypoints'].extend(video_keypoints)
                    all_data[split]['image_paths'].extend(video_paths)
                    all_data[split]['video_ids'].extend([video_id] * len(video_keypoints))

            # 打印该划分的统计信息
            print(f"\n{split.upper()} 划分统计:")
            print(f"  总图像数: {stats[split]['total']}")
            print(f"  成功提取: {stats[split]['success']} ({stats[split]['success']/stats[split]['total']*100:.2f}%)" if stats[split]['total'] > 0 else "  成功提取: 0")
            print(f"  失败: {stats[split]['failed']} ({stats[split]['failed']/stats[split]['total']*100:.2f}%)" if stats[split]['total'] > 0 else "  失败: 0")
            print(f"  关键点样本数: {len(all_data[split]['keypoints'])}")

        # 准备输出数据
        output_data = {
            'train': {
                'keypoints': all_data['train']['keypoints'],
                'image_paths': all_data['train']['image_paths'],
                'video_ids': all_data['train']['video_ids']
            },
            'dev': {
                'keypoints': all_data['dev']['keypoints'],
                'image_paths': all_data['dev']['image_paths'],
                'video_ids': all_data['dev']['video_ids']
            },
            'test': {
                'keypoints': all_data['test']['keypoints'],
                'image_paths': all_data['test']['image_paths'],
                'video_ids': all_data['test']['video_ids']
            },
            'stats': stats,
            'keypoint_info': {
                'face': {'num_points': 68, 'description': 'Facial landmarks'},
                'left_hand': {'num_points': 21, 'description': 'Left hand landmarks'},
                'right_hand': {'num_points': 21, 'description': 'Right hand landmarks'},
                'pose': {'num_points': 33, 'description': 'Body pose landmarks'},
                'total_points': 68 + 21 + 21 + 33  # 143 points total
            },
            'dataset_info': {
                'name': 'PHOENIX-2014-T',
                'source_path': str(dataset_path),
                'splits': splits
            }
        }

        # 保存数据
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)

        print(f"\n{'='*60}")
        print("提取完成!")
        print(f"{'='*60}")
        print(f"输出文件: {output_path}")
        print(f"\n总体统计:")
        total_samples = sum(len(all_data[split]['keypoints']) for split in splits)
        print(f"  总关键点样本数: {total_samples}")
        for split in splits:
            if len(all_data[split]['keypoints']) > 0:
                print(f"  {split}: {len(all_data[split]['keypoints'])} 个样本")

        return output_data

def main():
    parser = argparse.ArgumentParser(
        description='从 PHOENIX-2014-T 数据集中提取全身关键点',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理所有划分 (train, dev, test)
  python data/extract_phoenix_keypoints.py \\
      --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
      --output_path phoenix_keypoints.pkl

  # 只处理训练集
  python data/extract_phoenix_keypoints.py \\
      --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
      --output_path phoenix_keypoints_train.pkl \\
      --splits train

  # 限制每个划分处理的样本数（用于测试）
  python data/extract_phoenix_keypoints.py \\
      --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
      --output_path phoenix_keypoints_test.pkl \\
      --max_samples_per_split 10
        """
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T',
        help='PHOENIX-2014-T 数据集根目录路径'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='phoenix_keypoints.pkl',
        help='输出 pickle 文件路径'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        choices=['train', 'dev', 'test'],
        default=None,
        help='要处理的划分列表 (默认: 处理所有划分)'
    )
    parser.add_argument(
        '--max_samples_per_split',
        type=int,
        default=None,
        help='每个划分最多处理的视频数量 (None 表示处理全部，用于测试)'
    )
    parser.add_argument(
        '--min_detection_confidence',
        type=float,
        default=0.5,
        help='检测的最小置信度 (0.0-1.0)'
    )
    parser.add_argument(
        '--min_tracking_confidence',
        type=float,
        default=0.5,
        help='跟踪的最小置信度 (0.0-1.0)'
    )

    args = parser.parse_args()

    print("="*60)
    print("PHOENIX-2014-T 数据集关键点提取")
    print("="*60)
    print(f"数据集路径: {args.dataset_path}")
    print(f"输出路径: {args.output_path}")
    print(f"处理的划分: {args.splits if args.splits else '全部 (train, dev, test)'}")
    print(f"每个划分最大样本数: {args.max_samples_per_split if args.max_samples_per_split else '无限制'}")
    print(f"检测置信度: {args.min_detection_confidence}")
    print("="*60)

    extractor = PhoenixKeypointExtractor(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    extractor.process_phoenix_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        splits=args.splits,
        max_samples_per_split=args.max_samples_per_split
    )

    print("\n完成!")

if __name__ == "__main__":
    main()


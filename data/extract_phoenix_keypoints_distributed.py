"""
从 PHOENIX-2014-T 数据集中提取全身关键点（分布式版本）
使用 MediaPipe 提取面部、双手和姿态关键点
支持多进程并行处理，可在多块GPU服务器上运行

数据集结构：
PHOENIX-2014-T-release-v3/
  └── PHOENIX-2014-T/
      └── features/
          └── fullFrame-210x260px/
              ├── train/
              ├── dev/
              └── test/

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 data/extract_phoenix_keypoints_distributed.py --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T --output_path phoenix_keypoints.pkl --num_workers 4
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
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
from collections import defaultdict
import time

# 导入现有的关键点提取器
try:
    from .extract_body_keypoints import FullBodyKeypointExtractor
    from .load_phoenix_annotations import load_phoenix_annotations
    from .utils_keypoints import flatten_keypoint_dict
except ImportError:
    from extract_body_keypoints import FullBodyKeypointExtractor
    from load_phoenix_annotations import load_phoenix_annotations
    from utils_keypoints import flatten_keypoint_dict

import torch


def extract_keypoints_worker(args):
    """
    工作进程函数：处理一批图像的关键点提取

    Args:
        args: 元组 (image_paths, video_ids, process_id, min_detection_confidence, min_tracking_confidence)

    Returns:
        results: 列表，每个元素是 (image_path, video_id, kp_dict) 或 None
    """
    image_paths, video_ids, process_id, min_detection_confidence, min_tracking_confidence = args

    # 每个进程创建自己的 MediaPipe 实例（MediaPipe 不是线程安全的）
    extractor = FullBodyKeypointExtractor(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

    results = []
    for image_path, video_id in zip(image_paths, video_ids):
        try:
            kp_dict = extractor.extract_keypoints(str(image_path))
            if kp_dict is not None:
                results.append((str(image_path), video_id, kp_dict))
            else:
                results.append((str(image_path), video_id, None))
        except Exception as e:
            print(f"进程 {process_id} 处理 {image_path} 时出错: {e}")
            results.append((str(image_path), video_id, None))

    return results


def split_list(lst, n):
    """将列表分割成 n 个大致相等的部分"""
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


class PhoenixKeypointExtractorDistributed:
    """
    支持分布式处理的 PHOENIX-2014-T 数据集关键点提取器
    使用多进程并行处理，提高处理速度
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, num_workers=4):
        """
        初始化关键点提取器

        Args:
            min_detection_confidence: 检测的最小置信度
            min_tracking_confidence: 跟踪的最小置信度
            num_workers: 并行工作进程数（默认: 4，对应4块GPU）
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_workers = num_workers

        # 检查可用CPU核心数
        available_cores = cpu_count()
        if num_workers > available_cores:
            print(f"警告: 请求 {num_workers} 个工作进程，但只有 {available_cores} 个CPU核心")
            print(f"将使用 {available_cores} 个工作进程")
            self.num_workers = available_cores

    def process_split_distributed(self, image_files, video_ids, split_name):
        """
        使用多进程处理一个划分的图像

        Args:
            image_files: 图像文件路径列表
            video_ids: 对应的视频ID列表
            split_name: 划分名称（用于显示）

        Returns:
            keypoints: 关键点列表
            image_paths: 图像路径列表
            video_ids_result: 视频ID列表
        """
        if len(image_files) == 0:
            return [], [], []

        print(f"\n使用 {self.num_workers} 个进程并行处理 {len(image_files)} 个图像...")

        # 将图像列表分割到不同的进程
        image_chunks = split_list(image_files, self.num_workers)
        video_id_chunks = split_list(video_ids, self.num_workers)

        # 准备进程参数
        process_args = []
        for i, (img_chunk, vid_chunk) in enumerate(zip(image_chunks, video_id_chunks)):
            if len(img_chunk) > 0:  # 只添加非空的块
                process_args.append((
                    img_chunk,
                    vid_chunk,
                    i,
                    self.min_detection_confidence,
                    self.min_tracking_confidence
                ))

        # 使用进程池并行处理
        start_time = time.time()
        with Pool(processes=self.num_workers) as pool:
            # 使用 tqdm 显示进度
            results_list = []
            with tqdm(total=len(image_files), desc=f"处理 {split_name}") as pbar:
                # 提交所有任务
                async_results = [pool.apply_async(extract_keypoints_worker, (args,)) for args in process_args]

                # 收集结果
                for async_result in async_results:
                    results = async_result.get()
                    results_list.extend(results)
                    pbar.update(len(results))

        elapsed_time = time.time() - start_time
        print(f"处理完成，耗时: {elapsed_time:.2f} 秒")
        print(f"平均速度: {len(image_files)/elapsed_time:.2f} 图像/秒")

        # 整理结果
        keypoints = []
        image_paths = []
        video_ids_result = []

        for image_path, video_id, kp_dict in results_list:
            if kp_dict is not None:
                keypoints.append(kp_dict)
                image_paths.append(image_path)
                video_ids_result.append(video_id)

        return keypoints, image_paths, video_ids_result

    def process_phoenix_dataset(self, dataset_path, output_path, splits=None, max_samples_per_split=None, annotations_dir=None):
        """
        处理 PHOENIX-2014-T 数据集（分布式版本）
        按视频组织数据，包含 gloss、text、num_frames 信息

        Args:
            dataset_path: PHOENIX 数据集根目录路径
            output_path: 输出 pickle 文件路径
            splits: 要处理的划分列表，例如 ['train', 'dev', 'test']，None 表示处理所有
            max_samples_per_split: 每个划分最多处理的视频数量（None 表示处理全部）
            annotations_dir: 注释文件目录路径（默认: dataset_path/annotations/manual）

        Returns:
            output_data: 按视频组织的字典，格式为 {split/video_name: {keypoint, name, gloss, num_frames, text}}
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"数据集路径不存在: {dataset_path}")

        # PHOENIX 数据集的图像帧路径
        frames_dir = dataset_path / "features" / "fullFrame-210x260px"
        if not frames_dir.exists():
            raise ValueError(f"未找到图像帧目录: {frames_dir}")

        # 加载注释文件
        if annotations_dir is None:
            annotations_dir = dataset_path / "annotations" / "manual"
        else:
            annotations_dir = Path(annotations_dir)

        print(f"\n加载注释文件: {annotations_dir}")
        annotations = load_phoenix_annotations(str(annotations_dir))

        # 确定要处理的划分
        if splits is None:
            splits = ['train', 'dev', 'test']

        # 存储所有关键点数据（按视频组织）
        all_data = {}

        # 统计信息
        stats = {
            'train': {'total_videos': 0, 'success_videos': 0, 'failed_videos': 0, 'total_frames': 0, 'success_frames': 0},
            'dev': {'total_videos': 0, 'success_videos': 0, 'failed_videos': 0, 'total_frames': 0, 'success_frames': 0},
            'test': {'total_videos': 0, 'success_videos': 0, 'failed_videos': 0, 'total_frames': 0, 'success_frames': 0}
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

            # 收集所有图像文件和对应的视频ID
            all_image_files = []
            all_video_ids = []

            for video_dir in video_dirs:
                video_id = video_dir.name

                # 获取该视频的所有图像帧
                image_files = sorted([
                    f for f in video_dir.iterdir()
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
                ])

                if len(image_files) == 0:
                    print(f"警告: 视频 {video_id} 中没有找到图像文件")
                    continue

                all_image_files.extend(image_files)
                all_video_ids.extend([video_id] * len(image_files))

            stats[split]['total_frames'] = len(all_image_files)
            stats[split]['total_videos'] = len(video_dirs)

            if len(all_image_files) == 0:
                print(f"警告: {split} 划分没有图像文件")
                continue

            # 使用分布式处理提取关键点
            keypoints_list, image_paths_list, video_ids_result = self.process_split_distributed(
                all_image_files, all_video_ids, split
            )

            # 按视频分组关键点
            video_keypoints_dict = defaultdict(list)

            for kp_dict, img_path, vid_id in zip(keypoints_list, image_paths_list, video_ids_result):
                video_keypoints_dict[vid_id].append(kp_dict)

            # 处理每个视频
            for video_name in sorted(video_keypoints_dict.keys()):
                video_keypoints = video_keypoints_dict[video_name]

                if len(video_keypoints) == 0:
                    stats[split]['failed_videos'] += 1
                    continue

                stats[split]['success_frames'] += len(video_keypoints)
                stats[split]['success_videos'] += 1

                # 将关键点字典转换为数组 [num_frames, 143, 3]
                keypoint_arrays = []
                for kp_dict in video_keypoints:
                    flattened = flatten_keypoint_dict(kp_dict)  # [143, 3]
                    keypoint_arrays.append(flattened)

                # 堆叠为 [num_frames, 143, 3]
                keypoints_array = np.array(keypoint_arrays)  # [num_frames, 143, 3]

                # 转换为 torch tensor
                keypoints_tensor = torch.from_numpy(keypoints_array).double()  # [num_frames, 143, 3]

                # 获取注释信息
                video_annotation = annotations.get(split, {}).get(video_name, {})
                gloss = video_annotation.get('orth', '')
                text = video_annotation.get('translation', '')

                # 存储视频数据
                video_key = f"{split}/{video_name}"
                all_data[video_key] = {
                    'keypoint': keypoints_tensor,  # [num_frames, 143, 3]
                    'name': video_key,  # split/video_name
                    'gloss': gloss,
                    'num_frames': len(video_keypoints),
                    'text': text
                }

            # 打印该划分的统计信息
            print(f"\n{split.upper()} 划分统计:")
            print(f"  总视频数: {stats[split]['total_videos']}")
            print(f"  成功处理: {stats[split]['success_videos']}")
            print(f"  失败: {stats[split]['failed_videos']}")
            print(f"  总帧数: {stats[split]['total_frames']}")
            print(f"  成功提取: {stats[split]['success_frames']}")

        # 保存数据（按划分分别保存）
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 如果输出路径有扩展名，移除它，然后添加 .split
        if output_path.suffix:
            base_path = output_path.with_suffix('')
        else:
            base_path = output_path

        saved_files = []

        # 按划分分别保存
        for split in splits:
            # 提取该划分的数据
            split_data = {}
            for key, value in all_data.items():
                if key.startswith(f"{split}/"):
                    split_data[key] = value

            if len(split_data) == 0:
                print(f"警告: {split} 划分没有数据，跳过保存")
                continue

            # 生成文件名：base_path.split（例如：phoenix_keypoints.train）
            split_file = base_path.parent / f"{base_path.name}.{split}"

            # 保存该划分的数据
            with open(split_file, 'wb') as f:
                pickle.dump(split_data, f)

            saved_files.append(str(split_file))
            print(f"保存 {split} 划分: {split_file} ({len(split_data)} 个视频)")

        print(f"\n{'='*60}")
        print("提取完成!")
        print(f"{'='*60}")
        print(f"输出文件:")
        for file_path in saved_files:
            print(f"  {file_path}")
        print(f"\n总体统计:")
        total_videos = sum(stats[split]['success_videos'] for split in splits)
        total_frames = sum(stats[split]['success_frames'] for split in splits)
        print(f"  总视频数: {total_videos}")
        print(f"  总帧数: {total_frames}")
        for split in splits:
            if stats[split]['success_videos'] > 0:
                print(f"  {split}: {stats[split]['success_videos']} 个视频, {stats[split]['success_frames']} 帧")
        print(f"\n处理配置:")
        print(f"  工作进程数: {self.num_workers}")
        print(f"  分布式处理: 是")

        return all_data


def main():
    parser = argparse.ArgumentParser(
        description='从 PHOENIX-2014-T 数据集中提取全身关键点（分布式版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用4个进程处理所有划分
  python data/extract_phoenix_keypoints_distributed.py \\
      --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
      --output_path phoenix_keypoints.pkl \\
      --num_workers 4

  # 使用8个进程只处理训练集
  python data/extract_phoenix_keypoints_distributed.py \\
      --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
      --output_path phoenix_keypoints_train.pkl \\
      --splits train \\
      --num_workers 8

  # 限制每个划分处理的样本数（用于测试）
  python data/extract_phoenix_keypoints_distributed.py \\
      --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
      --output_path phoenix_keypoints_test.pkl \\
      --max_samples_per_split 10 \\
      --num_workers 4
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
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='并行工作进程数（默认: 4，对应4块GPU）'
    )
    parser.add_argument(
        '--annotations_dir',
        type=str,
        default=None,
        help='注释文件目录路径（默认: dataset_path/annotations/manual）'
    )

    args = parser.parse_args()

    print("="*60)
    print("PHOENIX-2014-T 数据集关键点提取（分布式版本）")
    print("="*60)
    print(f"数据集路径: {args.dataset_path}")
    print(f"输出路径: {args.output_path}")
    print(f"处理的划分: {args.splits if args.splits else '全部 (train, dev, test)'}")
    print(f"每个划分最大样本数: {args.max_samples_per_split if args.max_samples_per_split else '无限制'}")
    print(f"检测置信度: {args.min_detection_confidence}")
    print(f"工作进程数: {args.num_workers}")
    print(f"可用CPU核心数: {cpu_count()}")
    print("="*60)

    extractor = PhoenixKeypointExtractorDistributed(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        num_workers=args.num_workers
    )

    extractor.process_phoenix_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        splits=args.splits,
        max_samples_per_split=args.max_samples_per_split,
        annotations_dir=args.annotations_dir
    )

    print("\n完成!")


if __name__ == "__main__":
    # 确保在 Windows 上使用 spawn 方法（如果需要）
    # 在 Linux 上通常使用 fork
    if sys.platform == 'win32':
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

    main()


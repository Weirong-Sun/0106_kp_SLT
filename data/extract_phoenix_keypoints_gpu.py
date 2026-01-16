"""
从 PHOENIX-2014-T 数据集中提取全身关键点（GPU 优化版本）
使用 MediaPipe 提取面部、双手和姿态关键点
支持多 GPU 分布式处理和数据整合

特点：
1. 每个进程绑定到不同的 GPU（通过 CUDA_VISIBLE_DEVICES）
2. 优化批处理，提高处理效率
3. 支持结果合并，方便按 train/dev/test 划分使用
4. 自动整合分块处理的结果
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
import time
import subprocess

# 导入现有的关键点提取器
try:
    from .extract_body_keypoints import FullBodyKeypointExtractor
except ImportError:
    from extract_body_keypoints import FullBodyKeypointExtractor


def extract_keypoints_worker_gpu(args):
    """
    工作进程函数：处理一批图像的关键点提取（GPU 绑定版本）

    Args:
        args: 元组 (image_paths, video_ids, process_id, gpu_id, min_detection_confidence, min_tracking_confidence, output_dir)

    Returns:
        output_file: 临时输出文件路径，包含该进程的处理结果
    """
    image_paths, video_ids, process_id, gpu_id, min_detection_confidence, min_tracking_confidence, output_dir = args

    # 设置 GPU 可见性（每个进程绑定到不同的 GPU）
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"进程 {process_id}: 绑定到 GPU {gpu_id}")

    # 减少 MediaPipe 日志输出
    os.environ['GLOG_minloglevel'] = '2'

    # 每个进程创建自己的 MediaPipe 实例（MediaPipe 不是线程安全的）
    extractor = FullBodyKeypointExtractor(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

    # 处理图像
    results = []
    start_time = time.time()

    for idx, (image_path, video_id) in enumerate(zip(image_paths, video_ids)):
        try:
            kp_dict = extractor.extract_keypoints(str(image_path))
            if kp_dict is not None:
                results.append({
                    'image_path': str(image_path),
                    'video_id': video_id,
                    'keypoints': kp_dict
                })
        except Exception as e:
            print(f"进程 {process_id} 处理 {image_path} 时出错: {e}", flush=True)
            results.append({
                'image_path': str(image_path),
                'video_id': video_id,
                'keypoints': None
            })

        # 每处理 100 个图像报告一次进度
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"进程 {process_id} (GPU {gpu_id}): 已处理 {idx+1}/{len(image_paths)} 个图像, 速度: {speed:.2f} 图像/秒", flush=True)

    elapsed_time = time.time() - start_time

    # 保存该进程的结果到临时文件
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_output_file = output_dir / f"temp_split_{process_id}_gpu{gpu_id}.pkl"

    process_results = {
        'process_id': process_id,
        'gpu_id': gpu_id,
        'results': results,
        'total_images': len(image_paths),
        'success_count': sum(1 for r in results if r['keypoints'] is not None),
        'elapsed_time': elapsed_time,
        'speed': len(image_paths) / elapsed_time if elapsed_time > 0 else 0
    }

    with open(temp_output_file, 'wb') as f:
        pickle.dump(process_results, f)

    print(f"进程 {process_id} (GPU {gpu_id}) 完成: 处理了 {len(image_paths)} 个图像, "
          f"成功 {process_results['success_count']} 个, "
          f"耗时 {elapsed_time:.2f} 秒, "
          f"速度 {process_results['speed']:.2f} 图像/秒", flush=True)

    return str(temp_output_file)


def split_list(lst, n):
    """将列表分割成 n 个大致相等的部分"""
    if n == 0:
        return []
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def merge_results(temp_files, split_name, output_file):
    """
    合并多个临时结果文件

    Args:
        temp_files: 临时文件路径列表
        split_name: 划分名称
        output_file: 输出文件路径

    Returns:
        merged_data: 合并后的数据字典
    """
    print(f"\n合并 {len(temp_files)} 个临时结果文件...")

    all_results = []
    total_stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'total_time': 0,
        'total_speed': 0
    }

    # 加载所有临时文件
    for temp_file in tqdm(temp_files, desc=f"加载 {split_name} 临时文件"):
        if not Path(temp_file).exists():
            print(f"警告: 临时文件不存在，跳过: {temp_file}")
            continue

        try:
            with open(temp_file, 'rb') as f:
                process_data = pickle.load(f)

            # 合并结果
            all_results.extend(process_data['results'])

            # 累计统计
            total_stats['total'] += process_data['total_images']
            total_stats['success'] += process_data['success_count']
            total_stats['failed'] += process_data['total_images'] - process_data['success_count']
            total_stats['total_time'] += process_data['elapsed_time']
            total_stats['total_speed'] += process_data['speed']

        except Exception as e:
            print(f"加载临时文件 {temp_file} 时出错: {e}")
            continue

    # 整理数据
    keypoints = []
    image_paths = []
    video_ids = []

    for result in all_results:
        if result['keypoints'] is not None:
            keypoints.append(result['keypoints'])
            image_paths.append(result['image_path'])
            video_ids.append(result['video_id'])

    merged_data = {
        'keypoints': keypoints,
        'image_paths': image_paths,
        'video_ids': video_ids,
        'stats': {
            'total': total_stats['total'],
            'success': total_stats['success'],
            'failed': total_stats['failed'],
            'total_time': total_stats['total_time'],
            'avg_speed': total_stats['total_speed'] / len(temp_files) if temp_files else 0
        }
    }

    # 保存合并后的数据
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(merged_data, f)

    print(f"合并完成: {len(keypoints)} 个有效样本")
    print(f"总处理时间: {total_stats['total_time']:.2f} 秒")
    print(f"平均速度: {merged_data['stats']['avg_speed']:.2f} 图像/秒")

    return merged_data


class PhoenixKeypointExtractorGPU:
    """
    支持 GPU 绑定的 PHOENIX-2014-T 数据集关键点提取器
    使用多进程并行处理，每个进程绑定到不同的 GPU
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, num_gpus=4, num_workers_per_gpu=1):
        """
        初始化关键点提取器

        Args:
            min_detection_confidence: 检测的最小置信度
            min_tracking_confidence: 跟踪的最小置信度
            num_gpus: GPU 数量（默认: 4）
            num_workers_per_gpu: 每个 GPU 的工作进程数（默认: 1，即总共 4 个进程）
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_gpus = num_gpus
        self.num_workers_per_gpu = num_workers_per_gpu
        self.num_workers = num_gpus * num_workers_per_gpu

        # 检查 GPU 可用性
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                available_gpus = len(result.stdout.strip().split('\n'))
                print(f"检测到 {available_gpus} 个 GPU")
                if num_gpus > available_gpus:
                    print(f"警告: 请求 {num_gpus} 个 GPU，但只有 {available_gpus} 个可用")
                    self.num_gpus = available_gpus
                    self.num_workers = available_gpus * num_workers_per_gpu
            else:
                print("警告: 无法检测 GPU，将使用 CPU 模式")
                self.num_gpus = 0
        except Exception as e:
            print(f"警告: GPU 检测失败 ({e})，将使用 CPU 模式")
            self.num_gpus = 0

    def process_split_gpu(self, image_files, video_ids, split_name, temp_output_dir, output_file):
        """
        使用多 GPU 进程处理一个划分的图像

        Args:
            image_files: 图像文件路径列表
            video_ids: 对应的视频ID列表
            split_name: 划分名称
            temp_output_dir: 临时输出目录
            output_file: 最终输出文件路径

        Returns:
            merged_data: 合并后的数据字典
        """
        if len(image_files) == 0:
            return {'keypoints': [], 'image_paths': [], 'video_ids': []}

        print(f"\n使用 {self.num_workers} 个进程 ({self.num_gpus} 个 GPU) 并行处理 {len(image_files)} 个图像...")

        # 将图像列表分割到不同的进程
        image_chunks = split_list(image_files, self.num_workers)
        video_id_chunks = split_list(video_ids, self.num_workers)

        # 准备进程参数
        process_args = []
        for i, (img_chunk, vid_chunk) in enumerate(zip(image_chunks, video_id_chunks)):
            if len(img_chunk) > 0:
                # 计算该进程应该使用的 GPU ID
                gpu_id = (i // self.num_workers_per_gpu) % self.num_gpus if self.num_gpus > 0 else None

                process_args.append((
                    img_chunk,
                    vid_chunk,
                    i,
                    gpu_id,
                    self.min_detection_confidence,
                    self.min_tracking_confidence,
                    temp_output_dir
                ))

        # 使用进程池并行处理
        start_time = time.time()
        temp_files = []

        with Pool(processes=self.num_workers) as pool:
            # 提交所有任务
            async_results = []
            for args in process_args:
                async_result = pool.apply_async(extract_keypoints_worker_gpu, (args,))
                async_results.append(async_result)

            # 收集结果（临时文件路径）
            print(f"等待 {len(async_results)} 个进程完成...")
            for async_result in async_results:
                try:
                    temp_file = async_result.get()
                    temp_files.append(temp_file)
                except Exception as e:
                    print(f"进程执行失败: {e}")

        total_elapsed = time.time() - start_time
        print(f"所有进程完成，总耗时: {total_elapsed:.2f} 秒")

        # 合并结果
        merged_data = merge_results(temp_files, split_name, output_file)

        # 清理临时文件
        print("清理临时文件...")
        for temp_file in temp_files:
            try:
                Path(temp_file).unlink()
            except Exception as e:
                print(f"删除临时文件 {temp_file} 时出错: {e}")

        return merged_data

    def process_phoenix_dataset(self, dataset_path, output_path, splits=None, max_samples_per_split=None, temp_dir="temp_keypoints"):
        """
        处理 PHOENIX-2014-T 数据集（GPU 优化版本）

        Args:
            dataset_path: PHOENIX 数据集根目录路径
            output_path: 输出 pickle 文件路径
            splits: 要处理的划分列表，例如 ['train', 'dev', 'test']，None 表示处理所有
            max_samples_per_split: 每个划分最多处理的样本数（None 表示处理全部）
            temp_dir: 临时文件目录（用于存储分块处理的结果）

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

        # 创建临时目录
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 存储所有关键点数据
        all_data = {
            'train': {'keypoints': [], 'image_paths': [], 'video_ids': []},
            'dev': {'keypoints': [], 'image_paths': [], 'video_ids': []},
            'test': {'keypoints': [], 'image_paths': [], 'video_ids': []}
        }

        # 统计信息
        stats = {
            'train': {'total': 0, 'success': 0, 'failed': 0, 'total_time': 0},
            'dev': {'total': 0, 'success': 0, 'failed': 0, 'total_time': 0},
            'test': {'total': 0, 'success': 0, 'failed': 0, 'total_time': 0}
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

            stats[split]['total'] = len(all_image_files)

            if len(all_image_files) == 0:
                print(f"警告: {split} 划分没有图像文件")
                continue

            # 使用 GPU 分布式处理
            split_temp_dir = temp_dir / split
            split_temp_dir.mkdir(parents=True, exist_ok=True)

            split_output_file = split_temp_dir / f"{split}_merged.pkl"

            merged_data = self.process_split_gpu(
                all_image_files, all_video_ids, split, split_temp_dir, split_output_file
            )

            # 更新数据
            all_data[split]['keypoints'] = merged_data['keypoints']
            all_data[split]['image_paths'] = merged_data['image_paths']
            all_data[split]['video_ids'] = merged_data['video_ids']

            # 更新统计信息
            stats[split]['success'] = merged_data['stats']['success']
            stats[split]['failed'] = merged_data['stats']['failed']
            stats[split]['total_time'] = merged_data['stats']['total_time']

            # 打印该划分的统计信息
            print(f"\n{split.upper()} 划分统计:")
            print(f"  总图像数: {stats[split]['total']}")
            print(f"  成功提取: {stats[split]['success']} ({stats[split]['success']/stats[split]['total']*100:.2f}%)" if stats[split]['total'] > 0 else "  成功提取: 0")
            print(f"  失败: {stats[split]['failed']} ({stats[split]['failed']/stats[split]['total']*100:.2f}%)" if stats[split]['total'] > 0 else "  失败: 0")
            print(f"  处理时间: {stats[split]['total_time']:.2f} 秒")
            print(f"  关键点样本数: {len(all_data[split]['keypoints'])}")

        # 准备最终输出数据
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
                'splits': splits,
                'processing_info': {
                    'num_gpus': self.num_gpus,
                    'num_workers': self.num_workers,
                    'num_workers_per_gpu': self.num_workers_per_gpu,
                    'gpu_optimized': True,
                    'distributed': True
                }
            }
        }

        # 保存最终数据
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
        total_time = sum(stats[split]['total_time'] for split in splits)
        print(f"  总关键点样本数: {total_samples:,}")
        print(f"  总处理时间: {total_time:.2f} 秒 ({total_time/3600:.2f} 小时)")
        for split in splits:
            if len(all_data[split]['keypoints']) > 0:
                print(f"  {split}: {len(all_data[split]['keypoints']):,} 个样本")
        print(f"\n处理配置:")
        print(f"  GPU 数量: {self.num_gpus}")
        print(f"  工作进程数: {self.num_workers}")
        print(f"  每个 GPU 进程数: {self.num_workers_per_gpu}")
        print(f"  分布式处理: 是")
        print(f"  GPU 优化: 是")

        return output_data


def main():
    parser = argparse.ArgumentParser(
        description='从 PHOENIX-2014-T 数据集中提取全身关键点（GPU 优化版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用 4 个 GPU，每个 GPU 1 个进程（总共 4 个进程）
  python data/extract_phoenix_keypoints_gpu.py \\
      --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
      --output_path phoenix_keypoints_full.pkl \\
      --num_gpus 4 \\
      --num_workers_per_gpu 1

  # 使用 4 个 GPU，每个 GPU 2 个进程（总共 8 个进程，更激进）
  python data/extract_phoenix_keypoints_gpu.py \\
      --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
      --output_path phoenix_keypoints_full.pkl \\
      --num_gpus 4 \\
      --num_workers_per_gpu 2

  # 只处理训练集
  python data/extract_phoenix_keypoints_gpu.py \\
      --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
      --output_path phoenix_keypoints_train.pkl \\
      --splits train \\
      --num_gpus 4
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
        default='phoenix_keypoints_full.pkl',
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
        help='每个划分最多处理的视频数量 (None 表示处理全部)'
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
        '--num_gpus',
        type=int,
        default=4,
        help='使用的 GPU 数量（默认: 4）'
    )
    parser.add_argument(
        '--num_workers_per_gpu',
        type=int,
        default=1,
        help='每个 GPU 的工作进程数（默认: 1，即总共 4 个进程）'
    )
    parser.add_argument(
        '--temp_dir',
        type=str,
        default='temp_keypoints',
        help='临时文件目录（默认: temp_keypoints）'
    )

    args = parser.parse_args()

    print("="*60)
    print("PHOENIX-2014-T 数据集关键点提取（GPU 优化版本）")
    print("="*60)
    print(f"数据集路径: {args.dataset_path}")
    print(f"输出路径: {args.output_path}")
    print(f"处理的划分: {args.splits if args.splits else '全部 (train, dev, test)'}")
    print(f"每个划分最大样本数: {args.max_samples_per_split if args.max_samples_per_split else '无限制'}")
    print(f"检测置信度: {args.min_detection_confidence}")
    print(f"GPU 数量: {args.num_gpus}")
    print(f"每个 GPU 进程数: {args.num_workers_per_gpu}")
    print(f"总工作进程数: {args.num_gpus * args.num_workers_per_gpu}")
    print("="*60)

    extractor = PhoenixKeypointExtractorGPU(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        num_gpus=args.num_gpus,
        num_workers_per_gpu=args.num_workers_per_gpu
    )

    extractor.process_phoenix_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        splits=args.splits,
        max_samples_per_split=args.max_samples_per_split,
        temp_dir=args.temp_dir
    )

    print("\n完成!")


if __name__ == "__main__":
    # 确保在 Windows 上使用 spawn 方法
    if sys.platform == 'win32':
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

    main()




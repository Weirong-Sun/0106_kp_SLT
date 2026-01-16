#!/usr/bin/env python
"""
分析 PHOENIX 数据集实际大小和提取结果对比
"""
from pathlib import Path
import pickle
import argparse


def analyze_dataset(dataset_path, result_pkl=None):
    """分析数据集大小"""
    dataset_path = Path(dataset_path)

    print("="*60)
    print("PHOENIX 数据集大小分析")
    print("="*60)

    # 检查数据集结构
    frames_dir = dataset_path / "features" / "fullFrame-210x260px"
    if not frames_dir.exists():
        print(f"错误: 未找到数据集目录: {frames_dir}")
        return

    dataset_stats = {}

    for split in ['train', 'dev', 'test']:
        split_dir = frames_dir / split
        if not split_dir.exists():
            print(f"\n警告: {split} 目录不存在")
            continue

        # 统计视频文件夹
        video_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        num_videos = len(video_dirs)

        # 统计图像文件
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        all_images = []
        for video_dir in video_dirs:
            for ext in image_extensions:
                all_images.extend(list(video_dir.glob(f'*{ext}')))
                all_images.extend(list(video_dir.glob(f'*{ext.upper()}')))

        num_images = len(all_images)
        avg_images_per_video = num_images / num_videos if num_videos > 0 else 0

        dataset_stats[split] = {
            'num_videos': num_videos,
            'num_images': num_images,
            'avg_images': avg_images_per_video
        }

        print(f"\n{split.upper()} 划分:")
        print(f"  视频文件夹数: {num_videos}")
        print(f"  总图像文件数: {num_images:,}")
        print(f"  平均每个视频: {avg_images_per_video:.1f} 个图像")

    # 如果提供了结果文件，进行对比
    if result_pkl and Path(result_pkl).exists():
        print("\n" + "="*60)
        print("提取结果对比")
        print("="*60)

        with open(result_pkl, 'rb') as f:
            result_data = pickle.load(f)

        for split in ['train', 'dev', 'test']:
            if split not in dataset_stats:
                continue

            ds_stats = dataset_stats[split]
            result_keypoints = result_data.get(split, {}).get('keypoints', [])
            result_video_ids = result_data.get(split, {}).get('video_ids', [])
            result_stats = result_data.get('stats', {}).get(split, {})

            num_result_videos = len(set(result_video_ids)) if result_video_ids else 0
            num_result_images = len(result_keypoints)

            print(f"\n{split.upper()} 划分对比:")
            print(f"  数据集:")
            print(f"    视频数: {ds_stats['num_videos']}")
            print(f"    图像数: {ds_stats['num_images']:,}")
            print(f"  提取结果:")
            print(f"    视频数: {num_result_videos}")
            print(f"    图像数: {num_result_images}")

            # 计算覆盖率
            video_coverage = (num_result_videos / ds_stats['num_videos'] * 100) if ds_stats['num_videos'] > 0 else 0
            image_coverage = (num_result_images / ds_stats['num_images'] * 100) if ds_stats['num_images'] > 0 else 0

            print(f"  覆盖率:")
            print(f"    视频覆盖率: {video_coverage:.2f}%")
            print(f"    图像覆盖率: {image_coverage:.2f}%")

            # 分析原因
            if num_result_videos < ds_stats['num_videos']:
                print(f"  ⚠️  视频数量不匹配!")
                print(f"     可能原因: 使用了 --max_samples_per_split 参数限制了处理数量")
                print(f"     建议: 不使用 --max_samples_per_split 参数，处理全部数据")

            if num_result_images < ds_stats['num_images']:
                print(f"  ⚠️  图像数量不匹配!")
                print(f"     差异: {ds_stats['num_images'] - num_result_images:,} 个图像未处理")

    print("\n" + "="*60)
    print("总结")
    print("="*60)
    total_videos = sum(s['num_videos'] for s in dataset_stats.values())
    total_images = sum(s['num_images'] for s in dataset_stats.values())
    print(f"数据集总计:")
    print(f"  总视频数: {total_videos:,}")
    print(f"  总图像数: {total_images:,}")

    if result_pkl and Path(result_pkl).exists():
        total_result = sum(len(result_data.get(s, {}).get('keypoints', [])) for s in ['train', 'dev', 'test'])
        print(f"提取结果总计:")
        print(f"  总图像数: {total_result:,}")
        coverage = (total_result / total_images * 100) if total_images > 0 else 0
        print(f"  总体覆盖率: {coverage:.2f}%")

        if coverage < 100:
            print(f"\n⚠️  注意: 提取结果不完整!")
            print(f"   缺少: {total_images - total_result:,} 个图像")
            print(f"   可能原因: 使用了 --max_samples_per_split 参数限制了处理数量")
            print(f"   建议运行完整提取:")
            print(f"   python data/extract_phoenix_keypoints_distributed.py \\")
            print(f"       --dataset_path {dataset_path} \\")
            print(f"       --output_path phoenix_keypoints_full.pkl \\")
            print(f"       --num_workers 4")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='分析 PHOENIX 数据集大小和提取结果对比'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T',
        help='PHOENIX 数据集根目录路径'
    )
    parser.add_argument(
        '--result_pkl',
        type=str,
        default=None,
        help='提取结果的 pickle 文件路径（用于对比）'
    )

    args = parser.parse_args()

    analyze_dataset(args.dataset_path, args.result_pkl)


if __name__ == "__main__":
    main()




#!/usr/bin/env python
"""
验证分布式处理结果与单进程结果的一致性
用于确保分布式处理不影响关键点输出
"""
import pickle
import numpy as np
import argparse
from pathlib import Path


def sort_by_path(data, split):
    """按图像路径排序数据"""
    items = list(zip(
        data[split]['image_paths'],
        data[split]['keypoints'],
        data[split]['video_ids']
    ))
    items.sort(key=lambda x: x[0])
    return items


def compare_keypoints(kp1, kp2, path, tolerance=1e-6):
    """比较两个关键点字典"""
    differences = []

    for key in ['face', 'left_hand', 'right_hand', 'pose']:
        if kp1[key] is None and kp2[key] is None:
            continue

        if kp1[key] is None or kp2[key] is None:
            differences.append(f"  {key}: 一个为None，另一个不为None")
            continue

        if kp1[key].shape != kp2[key].shape:
            differences.append(f"  {key}: 形状不匹配 {kp1[key].shape} vs {kp2[key].shape}")
            continue

        if not np.allclose(kp1[key], kp2[key], atol=tolerance):
            max_diff = np.max(np.abs(kp1[key] - kp2[key]))
            differences.append(f"  {key}: 数值差异最大 {max_diff:.2e}")

    return differences


def verify_results(single_path, distributed_path, splits=None, tolerance=1e-6):
    """
    验证单进程和分布式处理结果的一致性

    Args:
        single_path: 单进程版本的结果文件路径
        distributed_path: 分布式版本的结果文件路径
        splits: 要验证的划分列表
        tolerance: 数值比较的容差
    """
    print("="*60)
    print("验证分布式处理结果")
    print("="*60)

    # 加载文件
    print(f"\n加载单进程结果: {single_path}")
    with open(single_path, 'rb') as f:
        single = pickle.load(f)

    print(f"加载分布式结果: {distributed_path}")
    with open(distributed_path, 'rb') as f:
        distributed = pickle.load(f)

    # 确定要验证的划分
    if splits is None:
        splits = ['train', 'dev', 'test']

    all_passed = True

    # 验证每个划分
    for split in splits:
        if split not in single or split not in distributed:
            print(f"\n警告: 划分 '{split}' 在某个文件中不存在，跳过")
            continue

        print(f"\n{'='*60}")
        print(f"验证 {split.upper()} 划分")
        print(f"{'='*60}")

        # 排序数据
        single_items = sort_by_path(single, split)
        distributed_items = sort_by_path(distributed, split)

        # 验证数量
        single_count = len(single_items)
        distributed_count = len(distributed_items)

        print(f"单进程样本数: {single_count}")
        print(f"分布式样本数: {distributed_count}")

        if single_count != distributed_count:
            print(f"✗ 样本数量不匹配!")
            all_passed = False
            continue

        # 验证每个样本
        differences_count = 0
        for idx, ((path1, kp1, vid1), (path2, kp2, vid2)) in enumerate(zip(single_items, distributed_items)):
            # 验证路径
            if path1 != path2:
                print(f"✗ 样本 {idx}: 路径不匹配")
                print(f"  单进程: {path1}")
                print(f"  分布式: {path2}")
                all_passed = False
                differences_count += 1
                continue

            # 验证视频ID
            if vid1 != vid2:
                print(f"✗ 样本 {idx} ({path1}): 视频ID不匹配")
                print(f"  单进程: {vid1}")
                print(f"  分布式: {vid2}")
                all_passed = False
                differences_count += 1
                continue

            # 验证关键点
            diffs = compare_keypoints(kp1, kp2, path1, tolerance)
            if diffs:
                print(f"✗ 样本 {idx} ({Path(path1).name}): 关键点不匹配")
                for diff in diffs:
                    print(diff)
                all_passed = False
                differences_count += 1

        if differences_count == 0:
            print(f"✓ {split.upper()} 划分: 所有 {single_count} 个样本完全匹配!")
        else:
            print(f"✗ {split.upper()} 划分: {differences_count}/{single_count} 个样本有差异")

    # 验证统计信息
    print(f"\n{'='*60}")
    print("验证统计信息")
    print(f"{'='*60}")

    single_stats = single.get('stats', {})
    distributed_stats = distributed.get('stats', {})

    for split in splits:
        if split in single_stats and split in distributed_stats:
            s_stats = single_stats[split]
            d_stats = distributed_stats[split]

            if s_stats == d_stats:
                print(f"✓ {split.upper()} 统计信息匹配: {s_stats}")
            else:
                print(f"✗ {split.upper()} 统计信息不匹配:")
                print(f"  单进程: {s_stats}")
                print(f"  分布式: {d_stats}")
                all_passed = False

    # 总结
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ 验证通过: 分布式处理结果与单进程结果完全一致!")
        print("✓ 分布式处理不会影响关键点输出!")
    else:
        print("✗ 验证失败: 发现差异，请检查!")
    print(f"{'='*60}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='验证分布式处理结果与单进程结果的一致性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 验证两个结果文件
  python verify_distributed_results.py \\
      --single result_single.pkl \\
      --distributed result_distributed.pkl

  # 只验证训练集
  python verify_distributed_results.py \\
      --single result_single.pkl \\
      --distributed result_distributed.pkl \\
      --splits train
        """
    )

    parser.add_argument(
        '--single',
        type=str,
        required=True,
        help='单进程版本的结果文件路径'
    )
    parser.add_argument(
        '--distributed',
        type=str,
        required=True,
        help='分布式版本的结果文件路径'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        choices=['train', 'dev', 'test'],
        default=None,
        help='要验证的划分列表 (默认: 所有划分)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-6,
        help='数值比较的容差 (默认: 1e-6)'
    )

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.single).exists():
        print(f"错误: 文件不存在: {args.single}")
        return 1

    if not Path(args.distributed).exists():
        print(f"错误: 文件不存在: {args.distributed}")
        return 1

    # 验证
    passed = verify_results(
        args.single,
        args.distributed,
        splits=args.splits,
        tolerance=args.tolerance
    )

    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())



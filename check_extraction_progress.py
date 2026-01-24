#!/usr/bin/env python
"""
检查 PHOENIX 关键点提取进度
"""
import pickle
import os
from pathlib import Path


def check_progress(pkl_path=None):
    """检查关键点提取进度"""
    print("="*60)
    print("PHOENIX 关键点提取进度检查")
    print("="*60)

    # 查找结果文件
    if pkl_path is None:
        possible_files = [
            'result_distributed.pkl',
            'phoenix_keypoints.pkl',
            'phoenix_keypoints_test.pkl'
        ]
        pkl_path = None
        for f in possible_files:
            if os.path.exists(f):
                pkl_path = f
                break

        if pkl_path is None:
            print("未找到关键点文件！")
            print("\n可能的位置:")
            for f in possible_files:
                print(f"  - {f}")
            return

    print(f"\n检查文件: {pkl_path}")

    if not os.path.exists(pkl_path):
        print(f"文件不存在: {pkl_path}")
        return

    # 文件大小
    file_size = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"文件大小: {file_size:.2f} MB")

    # 加载数据
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return

    # 显示进度
    print("\n" + "="*60)
    print("提取进度")
    print("="*60)

    for split in ['train', 'dev', 'test']:
        if split in data:
            split_data = data[split]
            keypoints = split_data.get('keypoints', [])
            video_ids = split_data.get('video_ids', [])
            stats = data.get('stats', {}).get(split, {})

            num_samples = len(keypoints)
            num_videos = len(set(video_ids)) if video_ids else 0
            total = stats.get('total', 0)
            success = stats.get('success', 0)
            failed = stats.get('failed', 0)

            print(f"\n{split.upper()} 划分:")
            print(f"  样本数: {num_samples}")
            print(f"  视频数: {num_videos}")
            if total > 0:
                print(f"  总图像数: {total}")
                print(f"  成功: {success} ({success/total*100:.2f}%)")
                print(f"  失败: {failed} ({failed/total*100:.2f}%)")
                progress = (success / total * 100) if total > 0 else 0
                print(f"  进度: {progress:.1f}%")

    # 显示处理信息
    if 'dataset_info' in data:
        proc_info = data['dataset_info'].get('processing_info', {})
        if proc_info.get('distributed'):
            print(f"\n处理方式: 分布式")
            print(f"工作进程数: {proc_info.get('num_workers', 'N/A')}")
        else:
            print(f"\n处理方式: 单进程")

    # 估算总进度（如果知道总数）
    print("\n" + "="*60)
    total_samples = sum(len(data.get(s, {}).get('keypoints', [])) for s in ['train', 'dev', 'test'])
    print(f"总样本数: {total_samples}")

    # 如果知道完整数据集大小，可以估算进度
    # 这里只是示例，实际需要根据数据集大小调整
    print("="*60)


if __name__ == "__main__":
    import sys
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else None
    check_progress(pkl_path)






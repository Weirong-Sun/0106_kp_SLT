#!/usr/bin/env python
"""
查看 PHOENIX 关键点数据集的结构和格式
"""
import pickle
import numpy as np
import argparse
from pathlib import Path
import json


def view_keypoints_structure(pkl_file, detailed=False, save_json=None):
    """
    查看关键点文件的结构

    Args:
        pkl_file: pickle 文件路径
        detailed: 是否显示详细信息
        save_json: 是否保存为 JSON 格式（仅结构，不含数据）
    """
    pkl_file = Path(pkl_file)
    if not pkl_file.exists():
        print(f"错误: 文件不存在: {pkl_file}")
        return

    print("="*60)
    print("PHOENIX 关键点数据集结构")
    print("="*60)
    print(f"文件: {pkl_file}")
    print(f"大小: {pkl_file.stat().st_size / (1024*1024):.2f} MB")
    print("="*60)

    # 加载数据
    print("\n加载数据...")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # 显示顶级结构
    print(f"\n顶级键: {list(data.keys())}")

    # 数据结构说明
    structure_info = {
        'file_path': str(pkl_file),
        'file_size_mb': round(pkl_file.stat().st_size / (1024*1024), 2),
        'top_level_keys': list(data.keys()),
        'splits': {}
    }

    # 查看每个划分的结构
    for split in ['train', 'dev', 'test']:
        if split in data:
            print(f"\n{split.upper()} 划分:")
            print("-" * 60)
            split_data = data[split]

            # 基本信息
            keypoints = split_data.get('keypoints', [])
            image_paths = split_data.get('image_paths', [])
            video_ids = split_data.get('video_ids', [])

            print(f"  键: {list(split_data.keys())}")
            print(f"  样本数: {len(keypoints):,}")
            print(f"  图像路径数: {len(image_paths):,}")
            print(f"  视频ID数: {len(video_ids):,}")

            if len(video_ids) > 0:
                unique_videos = len(set(video_ids))
                print(f"  唯一视频数: {unique_videos}")
                print(f"  前5个视频ID: {list(set(video_ids))[:5]}")

            # 关键点结构
            if len(keypoints) > 0:
                first_kp = keypoints[0]
                print(f"\n  关键点结构 (第一个样本):")

                if isinstance(first_kp, dict):
                    print(f"    类型: dict")
                    print(f"    键: {list(first_kp.keys())}")

                    kp_structure = {}
                    for k, v in first_kp.items():
                        if v is not None:
                            arr = np.array(v)
                            print(f"    {k}:")
                            print(f"      shape: {arr.shape}")
                            print(f"      dtype: {arr.dtype}")
                            if detailed:
                                print(f"      范围: [{arr.min():.4f}, {arr.max():.4f}]")
                                print(f"      均值: {arr.mean():.4f}")
                                if arr.ndim == 2 and arr.shape[1] == 3:
                                    # 3D 坐标点
                                    print(f"      前3个点:")
                                    for i in range(min(3, len(arr))):
                                        print(f"        点 {i}: ({arr[i][0]:.4f}, {arr[i][1]:.4f}, {arr[i][2]:.4f})")
                                elif arr.ndim == 1:
                                    # 一维数组（如 image_shape）
                                    print(f"      值: {arr.tolist()}")

                            kp_structure[k] = {
                                'shape': list(arr.shape),
                                'dtype': str(arr.dtype),
                                'has_data': True
                            }
                        else:
                            print(f"    {k}: None")
                            kp_structure[k] = {'has_data': False}

                    structure_info['splits'][split] = {
                        'num_samples': len(keypoints),
                        'num_images': len(image_paths),
                        'num_videos': len(set(video_ids)) if video_ids else 0,
                        'keypoint_structure': kp_structure
                    }
                else:
                    print(f"    类型: {type(first_kp)}")
                    if isinstance(first_kp, np.ndarray):
                        print(f"    shape: {first_kp.shape}")
                        print(f"    dtype: {first_kp.dtype}")
                        structure_info['splits'][split] = {
                            'num_samples': len(keypoints),
                            'keypoint_type': 'numpy_array',
                            'shape': list(first_kp.shape),
                            'dtype': str(first_kp.dtype)
                        }
            else:
                print(f"  警告: 没有关键点数据")
                structure_info['splits'][split] = {'num_samples': 0}

    # 统计信息
    if 'stats' in data:
        print(f"\n统计信息:")
        print("-" * 60)
        stats = data['stats']
        for split in ['train', 'dev', 'test']:
            if split in stats:
                print(f"  {split.upper()}:")
                for k, v in stats[split].items():
                    print(f"    {k}: {v}")

    # 关键点信息
    if 'keypoint_info' in data:
        print(f"\n关键点信息:")
        print("-" * 60)
        kp_info = data['keypoint_info']
        for k, v in kp_info.items():
            print(f"  {k}: {v}")

    # 数据集信息
    if 'dataset_info' in data:
        print(f"\n数据集信息:")
        print("-" * 60)
        ds_info = data['dataset_info']
        for k, v in ds_info.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2}")
            else:
                print(f"  {k}: {v}")

    # 数据汇总
    print(f"\n{'='*60}")
    print("数据汇总")
    print("="*60)

    total_samples = 0
    total_videos = 0
    for split in ['train', 'dev', 'test']:
        if split in data:
            num_samples = len(data[split]['keypoints'])
            num_videos = len(set(data[split]['video_ids'])) if data[split].get('video_ids') else 0
            total_samples += num_samples
            total_videos += num_videos
            print(f"{split.upper()}: {num_samples:,} 个样本, {num_videos} 个视频")

    print(f"\n总计: {total_samples:,} 个样本, {total_videos} 个视频")

    # 关键点完整性统计（每个划分前100个样本）
    if detailed:
        print(f"\n关键点完整性统计 (每个划分前100个样本):")
        print("-" * 60)
        for split in ['train', 'dev', 'test']:
            if split in data and len(data[split]['keypoints']) > 0:
                samples = data[split]['keypoints'][:100]
                face_count = sum(1 for s in samples if isinstance(s, dict) and s.get('face') is not None)
                left_hand_count = sum(1 for s in samples if isinstance(s, dict) and s.get('left_hand') is not None)
                right_hand_count = sum(1 for s in samples if isinstance(s, dict) and s.get('right_hand') is not None)
                pose_count = sum(1 for s in samples if isinstance(s, dict) and s.get('pose') is not None)
                total_checked = len(samples)

                print(f"  {split.upper()}:")
                print(f"    面部: {face_count}/{total_checked} ({face_count/total_checked*100:.1f}%)")
                print(f"    左手: {left_hand_count}/{total_checked} ({left_hand_count/total_checked*100:.1f}%)")
                print(f"    右手: {right_hand_count}/{total_checked} ({right_hand_count/total_checked*100:.1f}%)")
                print(f"    姿态: {pose_count}/{total_checked} ({pose_count/total_checked*100:.1f}%)")

    # 保存结构信息为 JSON
    if save_json:
        structure_info['total_samples'] = total_samples
        structure_info['total_videos'] = total_videos

        output_file = Path(save_json)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structure_info, f, indent=2, ensure_ascii=False)
        print(f"\n结构信息已保存到: {output_file}")

    print("="*60)

    return structure_info


def main():
    parser = argparse.ArgumentParser(
        description='查看 PHOENIX 关键点数据集的结构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本查看
  python view_phoenix_keypoints_structure.py phoenix_keypoints.pkl

  # 详细信息
  python view_phoenix_keypoints_structure.py phoenix_keypoints.pkl --detailed

  # 保存结构信息为 JSON
  python view_phoenix_keypoints_structure.py phoenix_keypoints.pkl --save_json structure.json
        """
    )

    parser.add_argument(
        'pkl_file',
        type=str,
        help='关键点 pickle 文件路径'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='显示详细信息（包括关键点坐标示例）'
    )
    parser.add_argument(
        '--save_json',
        type=str,
        default=None,
        help='将结构信息保存为 JSON 文件'
    )

    args = parser.parse_args()

    view_keypoints_structure(
        args.pkl_file,
        detailed=args.detailed,
        save_json=args.save_json
    )


if __name__ == "__main__":
    main()


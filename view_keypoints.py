#!/usr/bin/env python
"""
查看和分析关键点 pickle 文件
支持查看 PHOENIX 数据集关键点文件
"""
import pickle
import numpy as np
import argparse
from pathlib import Path
import json

def view_keypoints_file(file_path, detailed=False, save_json=None):
    """
    查看关键点文件内容

    Args:
        file_path: pickle 文件路径
        detailed: 是否显示详细信息
        save_json: 是否保存为 JSON（仅统计信息）
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return

    print("="*60)
    print(f"查看关键点文件: {file_path.name}")
    print("="*60)

    # 加载数据
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"错误: 无法加载文件: {e}")
        return

    # 检查数据格式
    print("\n1. 文件基本信息:")
    print(f"   文件大小: {file_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   数据类型: {type(data).__name__}")

    # PHOENIX 数据集格式（包含 train/dev/test）
    if isinstance(data, dict) and 'train' in data:
        print("\n2. 数据集结构 (PHOENIX 格式):")
        print(f"   包含划分: {list(data.keys())}")

        for split in ['train', 'dev', 'test']:
            if split in data:
                split_data = data[split]
                num_samples = len(split_data.get('keypoints', []))
                print(f"\n   {split.upper()} 划分:")
                print(f"     样本数: {num_samples}")
                if 'video_ids' in split_data:
                    unique_videos = len(set(split_data['video_ids']))
                    print(f"     唯一视频数: {unique_videos}")
                if 'image_paths' in split_data:
                    print(f"     图像路径数: {len(split_data['image_paths'])}")

        # 显示统计信息
        if 'stats' in data:
            print("\n3. 提取统计信息:")
            for split in ['train', 'dev', 'test']:
                if split in data.get('stats', {}):
                    stats = data['stats'][split]
                    total = stats.get('total', 0)
                    success = stats.get('success', 0)
                    failed = stats.get('failed', 0)
                    if total > 0:
                        print(f"\n   {split.upper()}:")
                        print(f"     总图像数: {total}")
                        print(f"     成功提取: {success} ({success/total*100:.2f}%)")
                        print(f"     失败: {failed} ({failed/total*100:.2f}%)")

        # 显示关键点信息
        if 'keypoint_info' in data:
            print("\n4. 关键点信息:")
            kp_info = data['keypoint_info']
            for kp_type, info in kp_info.items():
                if isinstance(info, dict) and 'num_points' in info:
                    print(f"   {kp_type}: {info['num_points']} 个点 - {info.get('description', '')}")

        # 显示数据集信息
        if 'dataset_info' in data:
            print("\n5. 数据集信息:")
            ds_info = data['dataset_info']
            for key, value in ds_info.items():
                print(f"   {key}: {value}")

        # 详细查看第一个样本
        if detailed:
            print("\n6. 详细样本信息 (第一个样本):")
            for split in ['train', 'dev', 'test']:
                if split in data and len(data[split].get('keypoints', [])) > 0:
                    sample = data[split]['keypoints'][0]
                    print(f"\n   {split.upper()} 划分的第一个样本:")
                    print(f"     视频ID: {data[split]['video_ids'][0] if 'video_ids' in data[split] else 'N/A'}")
                    print(f"     图像路径: {data[split]['image_paths'][0] if 'image_paths' in data[split] else 'N/A'}")
                    print(f"     关键点:")
                    for kp_type in ['face', 'left_hand', 'right_hand', 'pose']:
                        if kp_type in sample:
                            kp = sample[kp_type]
                            if kp is not None:
                                print(f"       {kp_type}: shape {kp.shape}, dtype {kp.dtype}")
                                if detailed and kp.size > 0:
                                    print(f"         范围: x=[{kp[:, 0].min():.3f}, {kp[:, 0].max():.3f}], "
                                          f"y=[{kp[:, 1].min():.3f}, {kp[:, 1].max():.3f}]")
                            else:
                                print(f"       {kp_type}: None (未检测到)")
                    if 'image_shape' in sample:
                        print(f"       image_shape: {sample['image_shape']}")
                    break

    # 标准格式（只有 keypoints 列表）
    elif isinstance(data, dict) and 'keypoints' in data:
        print("\n2. 数据集结构 (标准格式):")
        num_samples = len(data['keypoints'])
        print(f"   总样本数: {num_samples}")

        if 'image_paths' in data:
            print(f"   图像路径数: {len(data['image_paths'])}")

        if 'stats' in data:
            print("\n3. 统计信息:")
            stats = data['stats']
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value}")

        # 详细查看第一个样本
        if detailed and num_samples > 0:
            print("\n4. 详细样本信息 (第一个样本):")
            sample = data['keypoints'][0]
            if 'image_paths' in data:
                print(f"   图像路径: {data['image_paths'][0]}")
            print(f"   关键点:")
            for kp_type in ['face', 'left_hand', 'right_hand', 'pose']:
                if kp_type in sample:
                    kp = sample[kp_type]
                    if kp is not None:
                        print(f"     {kp_type}: shape {kp.shape}, dtype {kp.dtype}")
                    else:
                        print(f"     {kp_type}: None (未检测到)")

    # 列表格式
    elif isinstance(data, list):
        print("\n2. 数据集结构 (列表格式):")
        print(f"   总样本数: {len(data)}")

        if detailed and len(data) > 0:
            print("\n3. 详细样本信息 (第一个样本):")
            sample = data[0]
            if isinstance(sample, dict):
                for kp_type in ['face', 'left_hand', 'right_hand', 'pose']:
                    if kp_type in sample:
                        kp = sample[kp_type]
                        if kp is not None:
                            print(f"   {kp_type}: shape {kp.shape}")
                        else:
                            print(f"   {kp_type}: None")

    else:
        print("\n2. 未知数据格式")
        print(f"   数据类型: {type(data)}")
        if isinstance(data, dict):
            print(f"   键: {list(data.keys())[:10]}")

    # 保存为 JSON（仅统计信息）
    if save_json:
        json_path = Path(save_json)
        json_data = {}

        if isinstance(data, dict):
            if 'train' in data:
                # PHOENIX 格式
                for split in ['train', 'dev', 'test']:
                    if split in data:
                        json_data[split] = {
                            'num_samples': len(data[split].get('keypoints', [])),
                            'num_videos': len(set(data[split].get('video_ids', [])))
                        }
                if 'stats' in data:
                    json_data['stats'] = data['stats']
            else:
                # 标准格式
                json_data = {
                    'num_samples': len(data.get('keypoints', [])),
                    'stats': data.get('stats', {})
                }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"\n统计信息已保存到: {json_path}")

    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(
        description='查看和分析关键点 pickle 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本查看
  python view_keypoints.py phoenix_keypoints_test.pkl

  # 详细查看
  python view_keypoints.py phoenix_keypoints_test.pkl --detailed

  # 保存统计信息为 JSON
  python view_keypoints.py phoenix_keypoints_test.pkl --save_json stats.json
        """
    )

    parser.add_argument(
        'file_path',
        type=str,
        help='关键点 pickle 文件路径'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='显示详细信息（包括样本数据）'
    )
    parser.add_argument(
        '--save_json',
        type=str,
        default=None,
        help='将统计信息保存为 JSON 文件'
    )

    args = parser.parse_args()

    view_keypoints_file(
        file_path=args.file_path,
        detailed=args.detailed,
        save_json=args.save_json
    )

if __name__ == "__main__":
    main()




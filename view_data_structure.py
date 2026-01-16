#!/usr/bin/env python
"""
查看 extract_phoenix_keypoints_distributed.py 生成的数据结构说明
（不实际加载文件，只显示数据结构说明）
"""
import argparse
from pathlib import Path


def print_structure_info(file_path):
    """
    打印数据结构说明
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return False

    file_size = file_path.stat().st_size / (1024*1024)

    print("="*70)
    print(f"文件: {file_path}")
    print("="*70)
    print(f"文件大小: {file_size:.2f} MB")
    print()

    print("="*70)
    print("数据结构说明")
    print("="*70)
    print("""
数据结构:
{
    "split/video_name": {
        "keypoint": tensor([num_frames, 143, 3]),  # torch.float64
        "name": "split/video_name",
        "gloss": "GLOSS TEXT",
        "num_frames": 63,
        "text": "German text description"
    },
    ...
}

字段说明:
  - keypoint: torch.Tensor, 形状 [num_frames, 143, 3]
    * num_frames: 该视频的帧数
    * 143: 关键点数量 (68面部 + 21左手 + 21右手 + 33姿态)
    * 3: 坐标维度 (x, y, z)
    * 数据类型: torch.float64

  - name: str, 视频名称（格式: "split/video_name"）

  - gloss: str, gloss 文本（从注释文件中的 'orth' 字段读取）

  - num_frames: int, 视频帧数

  - text: str, 文本描述（从注释文件中的 'translation' 字段读取）

加载数据示例:
  import pickle
  import torch

  # 加载数据
  with open('phoenix_keypoints.train', 'rb') as f:
      data = pickle.load(f)

  # 访问数据
  for video_key, video_data in data.items():
      keypoints = video_data['keypoint']  # [num_frames, 143, 3]
      name = video_data['name']
      gloss = video_data['gloss']
      num_frames = video_data['num_frames']
      text = video_data['text']

      print(f"视频: {name}")
      print(f"  帧数: {num_frames}")
      print(f"  关键点形状: {keypoints.shape}")
      print(f"  Gloss: {gloss}")
      print(f"  Text: {text}")
""")

    print("="*70)
    print("注意:")
    print("  由于数据中包含 torch.Tensor，加载时需要 torch 模块")
    print("  请确保已安装 torch: pip install torch")
    print("="*70)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='查看数据结构说明',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'file_path',
        type=str,
        help='要查看的文件路径（例如: phoenix_keypoints.train）'
    )

    args = parser.parse_args()

    print_structure_info(args.file_path)


if __name__ == "__main__":
    main()




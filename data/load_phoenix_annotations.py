"""
加载 PHOENIX 数据集的注释文件（gloss 和 text）
"""
import csv
from pathlib import Path
from typing import Dict, Optional


def load_phoenix_annotations(annotations_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    加载 PHOENIX 数据集的注释文件

    Args:
        annotations_dir: 注释文件目录路径

    Returns:
        annotations: Dict[split, Dict[video_name, {'orth': gloss, 'translation': text}]]
    """
    annotations_dir = Path(annotations_dir)
    annotations = {}

    splits = ['train', 'dev', 'test']

    for split in splits:
        csv_file = annotations_dir / f'PHOENIX-2014-T.{split}.corpus.csv'

        if not csv_file.exists():
            print(f"警告: 注释文件不存在，跳过: {csv_file}")
            annotations[split] = {}
            continue

        split_annotations = {}

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='|')
                for row in reader:
                    video_name = row['name'].strip()
                    gloss = row['orth'].strip()  # gloss
                    text = row['translation'].strip()  # text

                    split_annotations[video_name] = {
                        'orth': gloss,
                        'translation': text
                    }

            annotations[split] = split_annotations
            print(f"加载 {split} 注释: {len(split_annotations)} 个视频")

        except Exception as e:
            print(f"错误: 加载 {split} 注释文件失败: {e}")
            annotations[split] = {}

    return annotations


def get_video_annotation(
    annotations: Dict[str, Dict[str, Dict[str, str]]],
    split: str,
    video_name: str
) -> Optional[Dict[str, str]]:
    """
    获取指定视频的注释信息

    Args:
        annotations: 注释字典
        split: 划分名称 (train/dev/test)
        video_name: 视频名称

    Returns:
        annotation: {'orth': gloss, 'translation': text} 或 None
    """
    if split not in annotations:
        return None

    return annotations[split].get(video_name, None)


if __name__ == "__main__":
    # 测试
    annotations_dir = '/data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
    annotations = load_phoenix_annotations(annotations_dir)

    print("\n测试获取注释:")
    if 'dev' in annotations and annotations['dev']:
        example_name = list(annotations['dev'].keys())[0]
        print(f"视频: {example_name}")
        print(f"Gloss: {annotations['dev'][example_name]['orth']}")
        print(f"Text: {annotations['dev'][example_name]['translation']}")






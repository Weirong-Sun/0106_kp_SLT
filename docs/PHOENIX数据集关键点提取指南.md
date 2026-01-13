# PHOENIX-2014-T 数据集关键点提取指南

## 概述

本指南介绍如何使用 MediaPipe 从 PHOENIX-2014-T 数据集中提取全身关键点（面部、双手、姿态）。

## 数据集结构

PHOENIX-2014-T 数据集已经包含了提取好的图像帧，位于以下路径：

```
/data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/
└── PHOENIX-2014-T/
    └── features/
        └── fullFrame-210x260px/
            ├── train/          # 训练集视频文件夹
            ├── dev/            # 验证集视频文件夹
            └── test/           # 测试集视频文件夹
```

每个视频文件夹中包含该视频的所有图像帧（PNG格式），例如：
```
train/
  └── 03February_2010_Wednesday_tagesschau-2062/
      ├── images0001.png
      ├── images0002.png
      ├── images0003.png
      └── ...
```

## 提取的关键点

使用 MediaPipe 提取以下关键点：

- **面部关键点**: 68个点
- **左手关键点**: 21个点
- **右手关键点**: 21个点
- **姿态关键点**: 33个点
- **总计**: 143个关键点

## 使用方法

### 基本用法

处理所有划分（train, dev, test）：

```bash
python data/extract_phoenix_keypoints.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints.pkl
```

### 只处理特定划分

只处理训练集：

```bash
python data/extract_phoenix_keypoints.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_train.pkl \
    --splits train
```

处理训练集和验证集：

```bash
python data/extract_phoenix_keypoints.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_train_dev.pkl \
    --splits train dev
```

### 测试模式（限制样本数）

如果数据集很大，可以先处理少量样本进行测试：

```bash
python data/extract_phoenix_keypoints.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_test.pkl \
    --max_samples_per_split 10
```

这将限制每个划分只处理前10个视频文件夹。

### 调整检测参数

调整检测置信度（如果检测效果不好）：

```bash
python data/extract_phoenix_keypoints.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints.pkl \
    --min_detection_confidence 0.3 \
    --min_tracking_confidence 0.3
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset_path` | PHOENIX-2014-T 数据集根目录路径 | `/data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T` |
| `--output_path` | 输出 pickle 文件路径 | `phoenix_keypoints.pkl` |
| `--splits` | 要处理的划分列表 (train/dev/test) | `None` (处理所有划分) |
| `--max_samples_per_split` | 每个划分最多处理的视频数量 | `None` (处理全部) |
| `--min_detection_confidence` | 检测的最小置信度 (0.0-1.0) | `0.5` |
| `--min_tracking_confidence` | 跟踪的最小置信度 (0.0-1.0) | `0.5` |

## 输出数据格式

提取的关键点数据保存为 pickle 文件，格式如下：

```python
{
    'train': {
        'keypoints': [
            {
                'face': np.array([68, 3]),      # 68个面部关键点
                'left_hand': np.array([21, 3]), # 21个左手关键点
                'right_hand': np.array([21, 3]),# 21个右手关键点
                'pose': np.array([33, 3])       # 33个姿态关键点
            },
            ...
        ],
        'image_paths': [str, ...],  # 对应的图像路径
        'video_ids': [str, ...]     # 对应的视频ID
    },
    'dev': {...},  # 验证集数据，格式同train
    'test': {...}, # 测试集数据，格式同train
    'stats': {
        'train': {'total': int, 'success': int, 'failed': int},
        'dev': {...},
        'test': {...}
    },
    'keypoint_info': {
        'face': {'num_points': 68, 'description': 'Facial landmarks'},
        'left_hand': {'num_points': 21, 'description': 'Left hand landmarks'},
        'right_hand': {'num_points': 21, 'description': 'Right hand landmarks'},
        'pose': {'num_points': 33, 'description': 'Body pose landmarks'},
        'total_points': 143
    },
    'dataset_info': {
        'name': 'PHOENIX-2014-T',
        'source_path': str,
        'splits': ['train', 'dev', 'test']
    }
}
```

## 数据加载示例

```python
import pickle
import numpy as np

# 加载关键点数据
with open('phoenix_keypoints.pkl', 'rb') as f:
    data = pickle.load(f)

# 访问训练集数据
train_keypoints = data['train']['keypoints']
train_paths = data['train']['image_paths']
train_video_ids = data['train']['video_ids']

# 访问第一个样本
first_sample = train_keypoints[0]
print(f"面部关键点形状: {first_sample['face'].shape if first_sample['face'] is not None else None}")
print(f"左手关键点形状: {first_sample['left_hand'].shape if first_sample['left_hand'] is not None else None}")
print(f"右手关键点形状: {first_sample['right_hand'].shape if first_sample['right_hand'] is not None else None}")
print(f"姿态关键点形状: {first_sample['pose'].shape if first_sample['pose'] is not None else None}")

# 查看统计信息
print(f"\n训练集统计:")
print(f"  总图像数: {data['stats']['train']['total']}")
print(f"  成功提取: {data['stats']['train']['success']}")
print(f"  失败: {data['stats']['train']['failed']}")
```

## 处理流程

1. **扫描数据集**: 自动扫描 `features/fullFrame-210x260px/` 目录下的 train/dev/test 划分
2. **处理视频**: 对每个视频文件夹中的所有图像帧进行关键点提取
3. **保持结构**: 保持数据集的 train/dev/test 划分结构
4. **保存数据**: 将所有关键点数据保存为 pickle 文件

## 注意事项

1. **处理时间**: 数据集可能很大，完整处理可能需要较长时间。建议先用 `--max_samples_per_split` 参数进行测试。

2. **内存使用**: 如果数据集非常大，可能需要分批处理。可以考虑：
   - 分别处理 train/dev/test 划分
   - 使用 `--max_samples_per_split` 限制处理的视频数量

3. **检测失败**: 如果某些图像无法检测到关键点（例如人物不在画面中），这些图像会被跳过，但不会影响整体处理。

4. **视频ID**: 每个关键点样本都关联了对应的视频ID，方便后续按视频组织序列数据。

## 后续处理

提取关键点后，可以使用以下脚本进行后续处理：

1. **组织视频序列**:
   ```bash
   python data/prepare_video_sequences.py \
       --keypoints_path phoenix_keypoints.pkl \
       --output_path phoenix_video_sequences.pkl
   ```

2. **测试和可视化**:
   ```bash
   python data/test_keypoints.py \
       --data_path phoenix_keypoints.pkl \
       --num_samples 5 \
       --output_dir phoenix_keypoints_visualization
   ```

## 故障排除

### 问题1: 找不到数据集路径

**错误**: `数据集路径不存在: ...`

**解决**: 检查 `--dataset_path` 参数是否正确，确保路径指向 `PHOENIX-2014-T` 目录（不是 `PHOENIX-2014-T-release-v3`）。

### 问题2: 找不到图像帧目录

**错误**: `未找到图像帧目录: ...`

**解决**: 确保数据集结构正确，应该包含 `features/fullFrame-210x260px/` 目录。

### 问题3: 检测率很低

**解决**:
- 降低 `--min_detection_confidence` 参数（例如 0.3）
- 检查图像质量
- 确保图像中包含人物

### 问题4: 内存不足

**解决**:
- 分别处理各个划分
- 使用 `--max_samples_per_split` 限制处理的视频数量
- 分批处理数据

## 相关文档

- [数据提取说明](../data/README.md)
- [关键点提取脚本说明](../data/extract_body_keypoints.py)
- [项目主文档](../README.md)

---

**文档创建时间**: 2024年
**适用数据集**: PHOENIX-2014-T-release-v3
**提取工具**: MediaPipe



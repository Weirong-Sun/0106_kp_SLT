# PHOENIX 关键点数据集格式说明

## 文件信息

- **文件名**: `phoenix_keypoints.pkl`
- **文件大小**: 1692.49 MB
- **格式**: Python pickle 文件
- **数据集**: PHOENIX-2014-T
- **处理方式**: 分布式处理（4 个进程）

## 数据结构概览

```
phoenix_keypoints.pkl
│
├── train                    # 训练集
│   ├── keypoints           # List[dict], 827,354 个样本
│   ├── image_paths         # List[str], 827,354 个路径
│   └── video_ids           # List[str], 827,354 个视频ID
│
├── dev                      # 验证集
│   ├── keypoints           # List[dict], 55,775 个样本
│   ├── image_paths         # List[str], 55,775 个路径
│   └── video_ids           # List[str], 55,775 个视频ID
│
├── test                     # 测试集
│   ├── keypoints           # List[dict], 64,627 个样本
│   ├── image_paths         # List[str], 64,627 个路径
│   └── video_ids           # List[str], 64,627 个视频ID
│
├── stats                    # 统计信息
│   ├── train               # {'total': 827354, 'success': 827354, 'failed': 0}
│   ├── dev                 # {'total': 55775, 'success': 55775, 'failed': 0}
│   └── test                # {'total': 64627, 'success': 64627, 'failed': 0}
│
├── keypoint_info            # 关键点信息
│   ├── face                # {'num_points': 68, 'description': 'Facial landmarks'}
│   ├── left_hand           # {'num_points': 21, 'description': 'Left hand landmarks'}
│   ├── right_hand          # {'num_points': 21, 'description': 'Right hand landmarks'}
│   ├── pose                # {'num_points': 33, 'description': 'Body pose landmarks'}
│   └── total_points        # 143
│
└── dataset_info             # 数据集信息
    ├── name                # 'PHOENIX-2014-T'
    ├── source_path         # 数据集原始路径
    ├── splits              # ['train', 'dev', 'test']
    └── processing_info     # {'num_workers': 4, 'distributed': True}
```

## 数据汇总

| 划分 | 样本数 | 唯一视频数 | 成功率 |
|------|--------|-----------|--------|
| **TRAIN** | 827,354 | 7,096 | 100% |
| **DEV** | 55,775 | 519 | 100% |
| **TEST** | 64,627 | 642 | 100% |
| **总计** | **947,756** | **8,257** | **100%** |

## 关键点字典结构

每个样本的关键点数据是一个字典，包含以下键：

### 1. `face` - 面部关键点

- **类型**: `numpy.ndarray`
- **形状**: `(68, 3)`
- **数据类型**: `float32`
- **说明**: 68 个面部关键点的 3D 坐标 (x, y, z)
- **坐标范围**:
  - x, y: [0, 1] (归一化坐标，相对于图像宽度和高度)
  - z: [-2, 2] (深度信息，单位：像素)

**示例**:
```python
face_kp = sample['face']
# shape: (68, 3)
# 第0个点: (0.5003, 0.1090, 0.0304)  # (x, y, z)
```

### 2. `left_hand` - 左手关键点

- **类型**: `numpy.ndarray`
- **形状**: `(21, 3)`
- **数据类型**: `float32`
- **说明**: 21 个左手关键点的 3D 坐标 (x, y, z)
- **坐标范围**:
  - x, y: [0, 1] (归一化坐标)
  - z: [-2, 2] (深度信息)

**示例**:
```python
left_hand_kp = sample['left_hand']
# shape: (21, 3)
# 第0个点: (0.3996, 0.8778, 0.0000)  # 手腕
```

### 3. `right_hand` - 右手关键点

- **类型**: `numpy.ndarray`
- **形状**: `(21, 3)`
- **数据类型**: `float32`
- **说明**: 21 个右手关键点的 3D 坐标 (x, y, z)
- **坐标范围**:
  - x, y: [0, 1] (归一化坐标)
  - z: [-2, 2] (深度信息)

**示例**:
```python
right_hand_kp = sample['right_hand']
# shape: (21, 3)
# 第0个点: (0.7380, 0.8488, 0.0000)  # 手腕
```

### 4. `pose` - 姿态关键点

- **类型**: `numpy.ndarray`
- **形状**: `(33, 3)`
- **数据类型**: `float32`
- **说明**: 33 个身体姿态关键点的 3D 坐标 (x, y, z)
- **坐标范围**:
  - x, y: [0, 1] (归一化坐标)
  - z: [-2, 2] (深度信息)

**示例**:
```python
pose_kp = sample['pose']
# shape: (33, 3)
# 第0个点: (0.5295, 0.2087, -1.0955)  # 鼻子
```

### 5. `image_shape` - 图像尺寸

- **类型**: `numpy.ndarray`
- **形状**: `(2,)`
- **数据类型**: `int64`
- **说明**: 图像的高度和宽度 `[高度, 宽度]`
- **值**: `[260, 210]` (PHOENIX 数据集图像尺寸)

**示例**:
```python
image_shape = sample['image_shape']
# shape: (2,)
# 值: [260, 210]  # [高度, 宽度]
```

## 关键点数量

### 各部位关键点数量

| 部位 | 关键点数量 | 说明 |
|------|-----------|------|
| **面部 (face)** | 68 | 面部轮廓、眉毛、眼睛、鼻子、嘴巴 |
| **左手 (left_hand)** | 21 | 手腕 + 5根手指 × 4个关节 |
| **右手 (right_hand)** | 21 | 手腕 + 5根手指 × 4个关节 |
| **姿态 (pose)** | 33 | 身体主要关节和部位 |
| **总计** | **143** | 所有关键点 |

### 关键点完整性统计

基于前100个样本的统计：

| 划分 | 面部 | 左手 | 右手 | 姿态 |
|------|------|------|------|------|
| **TRAIN** | 100% | 72% | 95% | 100% |
| **DEV** | 100% | 81% | 99% | 100% |
| **TEST** | 100% | 82% | 99% | 100% |

**说明**:
- 面部和姿态检测率接近 100%
- 手部检测率较低（因为可能被遮挡或在画面外）
- 如果某个部位未检测到，对应值为 `None`

## Python 使用示例

### 基本加载和访问

```python
import pickle
import numpy as np

# 加载数据
with open('phoenix_keypoints.pkl', 'rb') as f:
    data = pickle.load(f)

# 访问训练集的第一个样本
sample_idx = 0
sample = data['train']['keypoints'][sample_idx]
image_path = data['train']['image_paths'][sample_idx]
video_id = data['train']['video_ids'][sample_idx]

print(f"视频ID: {video_id}")
print(f"图像路径: {image_path}")

# 获取各个部位的关键点
face_kp = sample['face']        # shape: (68, 3)
left_hand_kp = sample['left_hand']  # shape: (21, 3)
right_hand_kp = sample['right_hand']  # shape: (21, 3)
pose_kp = sample['pose']        # shape: (33, 3)
image_shape = sample['image_shape']  # shape: (2,)

# 检查是否检测到各个部位
print(f"面部检测: {face_kp is not None}")
print(f"左手检测: {left_hand_kp is not None}")
print(f"右手检测: {right_hand_kp is not None}")
print(f"姿态检测: {pose_kp is not None}")
```

### 组合所有关键点（用于模型输入）

```python
# 方法 1: 直接组合（如果所有部位都检测到）
if all(kp is not None for kp in [face_kp, left_hand_kp, right_hand_kp, pose_kp]):
    all_keypoints = np.concatenate([
        face_kp,        # (68, 3)
        left_hand_kp,   # (21, 3)
        right_hand_kp,  # (21, 3)
        pose_kp         # (33, 3)
    ], axis=0)  # shape: (143, 3)

# 方法 2: 处理缺失的关键点（使用零填充）
def combine_keypoints(sample):
    face = sample.get('face')
    left_hand = sample.get('left_hand')
    right_hand = sample.get('right_hand')
    pose = sample.get('pose')

    # 如果缺失，使用零填充
    if face is None:
        face = np.zeros((68, 3), dtype=np.float32)
    if left_hand is None:
        left_hand = np.zeros((21, 3), dtype=np.float32)
    if right_hand is None:
        right_hand = np.zeros((21, 3), dtype=np.float32)
    if pose is None:
        pose = np.zeros((33, 3), dtype=np.float32)

    # 组合
    all_kp = np.concatenate([face, left_hand, right_hand, pose], axis=0)
    return all_kp  # shape: (143, 3)

# 使用
all_keypoints = combine_keypoints(sample)
```

### 批量处理

```python
# 处理训练集的所有样本
train_keypoints = data['train']['keypoints']
train_image_paths = data['train']['image_paths']
train_video_ids = data['train']['video_ids']

# 转换为 NumPy 数组格式（用于模型训练）
def prepare_batch_keypoints(samples):
    """
    将多个样本的关键点转换为批量格式

    Args:
        samples: List[dict], 多个样本的关键点字典

    Returns:
        keypoints_array: np.ndarray, shape (N, 143, 3)
    """
    batch_keypoints = []
    for sample in samples:
        kp = combine_keypoints(sample)  # (143, 3)
        batch_keypoints.append(kp)
    return np.array(batch_keypoints)  # (N, 143, 3)

# 处理前1000个样本
batch_samples = train_keypoints[:1000]
batch_kp_array = prepare_batch_keypoints(batch_samples)
print(f"批量关键点形状: {batch_kp_array.shape}")  # (1000, 143, 3)
```

### 转换为像素坐标

```python
def normalized_to_pixel(normalized_coords, image_shape):
    """
    将归一化坐标转换为像素坐标

    Args:
        normalized_coords: np.ndarray, shape (N, 3), 归一化坐标
        image_shape: np.ndarray, shape (2,), [高度, 宽度]

    Returns:
        pixel_coords: np.ndarray, shape (N, 3), 像素坐标
    """
    h, w = image_shape
    pixel_coords = normalized_coords.copy()
    pixel_coords[:, 0] = normalized_coords[:, 0] * w  # x
    pixel_coords[:, 1] = normalized_coords[:, 1] * h  # y
    # z 坐标保持不变（深度）
    return pixel_coords

# 示例
face_kp_pixel = normalized_to_pixel(face_kp, image_shape)
print(f"面部关键点（像素坐标）: {face_kp_pixel.shape}")  # (68, 3)
print(f"第一个点（像素）: ({face_kp_pixel[0][0]:.1f}, {face_kp_pixel[0][1]:.1f})")
```

### 统计信息访问

```python
# 访问统计信息
stats = data['stats']
print(f"训练集统计: {stats['train']}")
print(f"验证集统计: {stats['dev']}")
print(f"测试集统计: {stats['test']}")

# 访问关键点信息
kp_info = data['keypoint_info']
print(f"面部关键点数: {kp_info['face']['num_points']}")
print(f"总关键点数: {kp_info['total_points']}")

# 访问数据集信息
ds_info = data['dataset_info']
print(f"数据集名称: {ds_info['name']}")
print(f"处理方式: {ds_info['processing_info']}")
```

## 数据格式特点

### 1. 坐标系统

- **归一化坐标**: 所有关键点使用归一化坐标 (x, y) ∈ [0, 1]
- **深度信息**: z 坐标表示深度信息，范围 [-2, 2]
- **图像尺寸**: PHOENIX 数据集图像尺寸为 210×260 像素

### 2. 数据完整性

- **所有样本**: 947,756 个样本全部成功提取（100% 成功率）
- **关键点完整性**:
  - 面部和姿态接近 100% 检测率
  - 手部检测率约 70-99%（取决于可见性）

### 3. 数据组织

- **按划分组织**: train, dev, test 三个划分
- **对应关系**: keypoints、image_paths、video_ids 三个列表长度相同，一一对应
- **视频分组**: 多个样本可能属于同一个视频（通过 video_ids 关联）

## 使用建议

### 1. 模型训练

推荐使用 `prepare_model_input.py` 脚本将数据转换为模型输入格式：

```bash
python data/prepare_model_input.py \
    --keypoints_file phoenix_keypoints.pkl \
    --output_dir model_input_data
```

这会生成：
- `train_keypoints.npz` - NumPy 格式，快速加载
- `dev_keypoints.npz` - NumPy 格式
- `test_keypoints.npz` - NumPy 格式

### 2. 数据可视化

使用可视化脚本查看关键点：

```bash
python visualize_phoenix_keypoints.py phoenix_keypoints.pkl \
    --num_samples 10 \
    --splits train
```

### 3. 数据分析

使用分析脚本查看数据集统计：

```bash
python view_keypoints.py phoenix_keypoints.pkl --detailed
python analyze_dataset_size.py --result_pkl phoenix_keypoints.pkl
```

## 相关文档

- [关键点划分依据说明.md](./关键点划分依据说明.md)
- [GPU分布式关键点提取指南.md](./GPU分布式关键点提取指南.md)
- [完整提取和整合工作流程.md](./完整提取和整合工作流程.md)

---

**文档创建时间**: 2024年
**数据集**: PHOENIX-2014-T
**格式版本**: v1.0




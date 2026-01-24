# skeleton 和 hierarchical_keypoint 训练的区别

## 核心区别

| 特性 | skeleton 训练 | hierarchical_keypoint 训练 |
|------|--------------|--------------------------|
| **训练任务** | 关键点 → 骨架图像生成 | 关键点自编码/重建 |
| **输入** | 关键点数据 (143, 3) | 关键点数据 (68, 3) |
| **输出** | 骨架图像 (256, 256) | 关键点数据 (68, 3) |
| **模型** | HierarchicalSkeletonTransformer | HierarchicalKeypointTransformer |
| **数据格式** | 字典列表 + 图像 | numpy 数组 |
| **关键点范围** | 全部关键点 (143点) | 仅面部关键点 (68点) |
| **训练目标** | 生成骨架图像 | 关键点表示学习 |

---

## 详细对比

### 1. 训练任务和目标

#### skeleton 训练

**任务类型**: 关键点到图像生成（Keypoint-to-Image Generation）

**目标**: 从关键点数据生成骨架图像

**输入 → 输出**:
```
关键点数据 (143, 3) → 骨架图像 (256, 256)
```

**应用场景**:
- 可视化关键点数据
- 图像生成任务
- 骨架重建

#### hierarchical_keypoint 训练

**任务类型**: 关键点自编码/重建（Keypoint Auto-encoding）

**目标**: 学习关键点的层次化表示

**输入 → 输出**:
```
关键点数据 (68, 3) → 关键点数据 (68, 3)
```

**应用场景**:
- 关键点表示学习
- 特征提取
- 降维和重建

---

### 2. 数据格式和输入

#### skeleton 训练

**数据格式**:
```python
# 输入数据
keypoints_data: List[dict]
# 每个字典包含:
{
    'face': np.array (68, 3),
    'left_hand': np.array (21, 3),
    'right_hand': np.array (21, 3),
    'pose': np.array (33, 3)
}

# 输出数据（ground truth）
images_data: np.array (num_samples, 256, 256)
# 或由 generate_skeleton_dataset() 生成
```

**数据加载** (`SkeletonDataset`):
```python
class SkeletonDataset(Dataset):
    def __init__(self, keypoints_data, images_data=None, image_size=256):
        # 1. 展平关键点为 (143, 3)
        # Format: [face(68), left_hand(21), right_hand(21), pose(33)]
        flattened = np.concatenate([face, left_hand, right_hand, pose], axis=0)
        # shape: (143, 3)

        # 2. 如果没有提供图像，从关键点生成
        if images_data is None:
            images_data = generate_skeleton_dataset(keypoints_data, image_size=256)

        # 3. 转换为 tensor
        self.keypoints = torch.FloatTensor(flattened_keypoints)  # (N, 143, 3)
        self.images = torch.FloatTensor(images_data)  # (N, 1, 256, 256)

    def __getitem__(self, idx):
        return self.keypoints[idx], self.images[idx]  # (143, 3), (1, 256, 256)
```

**关键点数量**: 143 个（全部关键点）
- 面部: 68 个
- 左手: 21 个
- 右手: 21 个
- 姿态: 33 个

#### hierarchical_keypoint 训练

**数据格式**:
```python
# 输入数据
keypoints_data: np.array (num_samples, 68, 3)
# 仅包含面部关键点
```

**数据加载** (`KeypointDataset`):
```python
class KeypointDataset(Dataset):
    def __init__(self, keypoints_data, normalize=True):
        # keypoints_data: (num_samples, 68, 3)
        self.keypoints = torch.FloatTensor(keypoints_data)
        # 归一化到 [0, 1]

    def __getitem__(self, idx):
        kp = self.keypoints[idx]  # (68, 3)
        return kp, kp  # 输入和目标相同（自编码）
```

**关键点数量**: 68 个（仅面部关键点）

---

### 3. 模型架构

#### skeleton 模型

**模型类**: `HierarchicalSkeletonTransformer`

**架构特点**:
- 编码器: 关键点数据 → 层次化表示
- 解码器: 层次化表示 → 图像

**输入维度**: (batch, 143, 3)
**输出维度**: (batch, 1, 256, 256)

**关键组件**:
- 关键点编码器（Transformer）
- 图像解码器（CNN，上采样）
- 全局 + 区域表示

#### hierarchical_keypoint 模型

**模型类**: `HierarchicalKeypointTransformer`

**架构特点**:
- 编码器: 关键点数据 → 层次化表示
- 解码器: 层次化表示 → 关键点数据

**输入维度**: (batch, 68, 3)
**输出维度**: (batch, 68, 3)

**关键组件**:
- 区域编码器（8个面部区域）
- 交互层（跨区域交互）
- 解码器（关键点重建）

---

### 4. 损失函数

#### skeleton 训练

**损失函数**: 加权 MSE 损失

**特点**:
- 对手部区域加权（weight = 2.0）
- 对面部区域加权（weight = 1.5）
- 其他区域权重 = 1.0

**代码**:
```python
def compute_weighted_loss(generated, target, image_size=256,
                          hand_weight=2.0, face_weight=1.5):
    # 创建权重掩码
    # 手部区域: 2.0
    # 面部区域: 1.5
    # 其他区域: 1.0
    loss = weighted_mse_loss(generated, target, weight_mask)
```

**原因**: 手部和面部在手语识别中更重要

#### hierarchical_keypoint 训练

**损失函数**: MSE 损失（标准）

**特点**:
- 直接比较输入和输出的关键点坐标
- 无区域加权

**代码**:
```python
criterion = nn.MSELoss()
loss = criterion(output, target)  # 都是关键点数据
```

---

### 5. 训练流程对比

#### skeleton 训练流程

```python
# 1. 加载关键点数据（字典格式）
with open(data_path, 'rb') as f:
    data = pickle.load(f)
keypoints_data = data['train']['keypoints']  # List[dict]

# 2. 创建数据集（会自动生成骨架图像）
dataset = SkeletonDataset(keypoints_data, image_size=256)
# 内部会调用: generate_skeleton_dataset(keypoints_data, image_size=256)

# 3. 训练
for keypoints, images in dataloader:
    # keypoints: (batch, 143, 3)
    # images: (batch, 1, 256, 256)
    generated_images = model(keypoints)
    loss = weighted_loss(generated_images, images)
```

#### hierarchical_keypoint 训练流程

```python
# 1. 加载关键点数据（numpy数组格式）
with open(data_path, 'rb') as f:
    data = pickle.load(f)
keypoints = data['keypoints']  # (num_samples, 68, 3)

# 2. 创建数据集
dataset = KeypointDataset(keypoints, normalize=True)

# 3. 训练
for src, tgt in dataloader:
    # src: (batch, 68, 3)
    # tgt: (batch, 68, 3) - 与src相同（自编码）
    output = model(src, tgt)
    loss = mse_loss(output, tgt)
```

---

### 6. 代码位置

#### skeleton 训练

**文件**: `training/skeleton/train.py`

**关键导入**:
```python
from models.skeleton.model import HierarchicalSkeletonTransformer
from utils.utils_skeleton import generate_skeleton_dataset
```

**模型文件**: `models/skeleton/model.py`

#### hierarchical_keypoint 训练

**文件**: `training/hierarchical_keypoint/train.py`

**关键导入**:
```python
from models.hierarchical_keypoint.model import HierarchicalKeypointTransformer
```

**模型文件**: `models/hierarchical_keypoint/model.py`

---

## 训练目标总结

### skeleton 训练

**目标**: 学习从关键点数据生成骨架图像的映射

**输入**: 关键点坐标 (143, 3)
**输出**: 骨架图像 (256, 256)

**应用**:
- 关键点可视化
- 图像生成
- 骨架重建

**特点**:
- 使用全部关键点（143个）
- 需要生成 ground truth 图像
- 使用加权损失函数

### hierarchical_keypoint 训练

**目标**: 学习关键点的层次化表示（自编码）

**输入**: 关键点坐标 (68, 3)
**输出**: 关键点坐标 (68, 3)

**应用**:
- 关键点特征提取
- 表示学习
- 降维

**特点**:
- 仅使用面部关键点（68个）
- 输入和输出相同（自编码）
- 使用标准MSE损失

---

## 使用场景

### 何时使用 skeleton 训练？

✅ 需要生成骨架图像
✅ 需要可视化关键点
✅ 处理全部关键点（面部+手部+姿态）
✅ 图像到图像的任务

### 何时使用 hierarchical_keypoint 训练？

✅ 需要学习关键点表示
✅ 仅处理面部关键点
✅ 特征提取任务
✅ 自编码/重建任务

---

## 训练命令示例

### skeleton 训练

```bash
python training/skeleton/train.py \
    --data_path phoenix_keypoints.pkl \
    --batch_size 16 \
    --epochs 100 \
    --image_size 256 \
    --use_weighted_loss \
    --hand_weight 2.0 \
    --face_weight 1.5
```

### hierarchical_keypoint 训练

```bash
python training/hierarchical_keypoint/train.py \
    --data_path keypoints_data.pkl \
    --batch_size 32 \
    --epochs 100 \
    --d_global 256 \
    --d_region 128
```

**注意**: `hierarchical_keypoint` 需要的数据格式与 `skeleton` 不同，需要仅包含面部关键点的数据。

---

## 总结

### 关键区别

1. **任务类型**:
   - skeleton: 关键点 → 图像生成
   - hierarchical_keypoint: 关键点 → 关键点（自编码）

2. **关键点范围**:
   - skeleton: 全部关键点 (143个)
   - hierarchical_keypoint: 仅面部 (68个)

3. **输出格式**:
   - skeleton: 图像 (256, 256)
   - hierarchical_keypoint: 关键点 (68, 3)

4. **损失函数**:
   - skeleton: 加权MSE损失
   - hierarchical_keypoint: 标准MSE损失

5. **数据格式**:
   - skeleton: 字典列表 + 图像
   - hierarchical_keypoint: numpy数组

### 关系

两者是**不同的训练任务**，服务于不同的目的：
- **skeleton**: 图像生成任务
- **hierarchical_keypoint**: 表示学习任务

可以根据具体需求选择合适的训练方式。

---

**文档创建时间**: 2024年
**训练任务**: skeleton (图像生成) vs hierarchical_keypoint (表示学习)
**关键区别**: 输出格式（图像 vs 关键点）和关键点范围（143 vs 68）






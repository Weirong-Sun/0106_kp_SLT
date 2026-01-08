# Models 目录

模型定义目录，包含所有神经网络模型架构。

## 目录结构

```
models/
├── hierarchical_keypoint/    # 层次化关键点模型
│   └── model.py
├── hierarchical_image/       # 层次化图像重构模型
│   └── model.py
├── skeleton/                 # 骨架重构模型
│   └── model.py
├── temporal/                 # 时序 Transformer 模型
│   └── model.py
├── alignment/                # 视频-语言对齐模型
│   └── model.py
└── __init__.py               # 统一导入接口
```

## 模型说明

### 核心模型（用于手语识别 Pipeline）

### 1. Skeleton Model (`skeleton/model.py`)

**用途**：从全身关键点重构骨架图像（用于手语视频）

**架构**：
- 输入：全身关键点 `[batch, 143, 3]`（68面部 + 21左手 + 21右手 + 33姿态）
- 编码器：4个区域编码器（面部、左手、右手、姿态）
- 输出：全局表征 `[batch, 256]` + 区域表征 `[batch, 4, 128]`
- 解码器：CNN 解码器生成骨架图像 `[batch, 1, 256, 256]`

**特点**：
- 专门处理全身关键点
- 支持加权损失（强调手部和面部区域）

### 2. Temporal Model (`temporal/model.py`)

**用途**：学习视频序列的时序表征

**架构**：
- 输入：视频序列 `[batch, seq_len, 143, 3]`
- 帧编码器：使用预训练的骨架模型编码每一帧
- 时序编码器：Transformer 编码器学习时序依赖
- 输出：
  - 全局表征：`[batch, 512]`（关注所有帧）
  - 局部表征：`[batch, 2, 512]`（关注不同时间窗口）

**特点**：
- 多尺度压缩表征（全局+局部）
- 支持冻结帧编码器
- 编码-解码重构训练

### 3. Alignment Model (`alignment/model.py`)

**用途**：将视频表征对齐到文本描述（使用 mBART）

**架构**：
- 输入：视频表征（全局+局部）
- 视频投影：将视频表征投影到 mBART 嵌入空间
- mBART 解码器：生成文本描述
- 输出：文本序列

**特点**：
- 使用 mBART 作为语言模型解码器
- 支持本地模型路径
- 支持冻结 mBART 参数

### 实验模型（早期实验，非手语识别 Pipeline）

### 4. Hierarchical Keypoint Model (`hierarchical_keypoint/model.py`)

**用途**：学习关键点坐标的层次化表征（用于68个面部关键点）

**架构**：
- 输入：关键点坐标 `[batch, 68, 3]`
- 编码器：区域编码 → 跨区域交互 → 全局聚合
- 输出：全局表征 `[batch, 256]` + 区域表征 `[batch, 8, 128]`
- 解码器：重构关键点坐标

**特点**：
- 8个预定义区域（面部轮廓、眉毛、鼻子、眼睛、嘴巴等）
- 全局和区域两个层次的表征

**注意**：这是早期实验模型，用于面部关键点（68点），与手语识别 pipeline 无关。

### 5. Hierarchical Image Model (`hierarchical_image/model.py`)

**用途**：从关键点坐标重构图像

**架构**：
- 输入：关键点坐标 `[batch, 68, 3]`
- 编码器：层次化编码器（同 hierarchical_keypoint）
- 解码器：CNN 解码器生成图像
- 输出：图像 `[batch, 1, 256, 256]`

**特点**：
- 使用层次化编码器提取表征
- CNN 解码器将表征转换为图像

**注意**：这是早期实验模型，用于面部关键点图像重构，与手语识别 pipeline 无关。

## 使用方式

### 直接导入
```python
from models import (
    HierarchicalKeypointTransformer,
    HierarchicalKeypointToImageTransformer,
    HierarchicalSkeletonTransformer,
    TemporalSkeletonTransformer,
    VideoLanguageAlignment
)
```

### 从子目录导入
```python
from models.hierarchical_keypoint.model import HierarchicalKeypointTransformer
from models.skeleton.model import HierarchicalSkeletonTransformer
```

### 示例
```python
from models.skeleton.model import HierarchicalSkeletonTransformer

# 创建模型
model = HierarchicalSkeletonTransformer(
    d_global=256,
    d_region=128,
    num_regions=4,
    image_size=256
)

# 前向传播
keypoints = torch.randn(1, 143, 3)  # [batch, 143, 3]
images = model(keypoints)  # [batch, 1, 256, 256]
```

## 模型参数说明

所有模型的通用参数：
- `d_global`: 全局表征维度（通常256或512）
- `d_region`: 区域表征维度（通常128）
- `nhead`: 注意力头数（通常8）
- `dropout`: Dropout 率（通常0.1）
- `dim_feedforward`: Feedforward 网络维度

## 注意事项

- 模型定义与训练脚本中的参数应保持一致
- 修改模型架构后，需要重新训练
- 检查点文件包含模型配置，加载时会自动使用


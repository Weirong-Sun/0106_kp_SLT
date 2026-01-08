# Config 目录

配置文件目录，集中管理所有模型和训练参数。

## 文件说明

### `config.py`

主配置文件，包含所有模型的训练和推理参数。

## 配置结构

配置文件按模型类型组织，每个模型包含：

```python
MODEL_NAME = {
    'model': {
        # 模型架构参数
        'd_global': 256,
        'd_region': 128,
        ...
    },
    'training': {
        # 训练参数
        'batch_size': 16,
        'epochs': 100,
        'lr': 1e-4,
        ...
    },
    'inference': {
        # 推理参数
        'checkpoint': 'path/to/checkpoint.pth',
        'num_samples': 10,
        ...
    }
}
```

## 配置项说明

### 1. SKELETON - 骨架模型配置

**模型参数**：
- `d_global`: 全局表征维度（256）
- `d_region`: 区域表征维度（128）
- `num_regions`: 区域数量（4：面部、左手、右手、姿态）
- `num_keypoints`: 关键点总数（143）
- `image_size`: 输出图像大小（256）

**训练参数**：
- `data_path`: 关键点数据文件路径
- `batch_size`: 批次大小
- `epochs`: 训练轮数
- `lr`: 学习率
- `use_weighted_loss`: 是否使用加权损失
- `hand_weight`: 手部区域权重
- `face_weight`: 面部区域权重

### 2. TEMPORAL - 时序模型配置

**模型参数**：
- `d_temporal`: 时序表征维度（512）
- `d_final`: 最终表征维度（512）
- `num_local_vars`: 局部变量数量（2）
- `num_temporal_layers`: 时序 Transformer 层数
- `freeze_frame_encoder`: 是否冻结帧编码器

**训练参数**：
- `video_data_path`: 视频序列数据路径
- `frame_encoder_checkpoint`: 预训练骨架模型路径
- `seq_len`: 序列长度
- `batch_size`: 批次大小
- `epochs`: 训练轮数

**推理参数**：
- `checkpoint`: 模型检查点路径
- `video_sequences`: 视频序列数据路径
- `num_samples`: 处理的样本数量
- `output_dir`: 输出目录

### 3. ALIGNMENT - 对齐模型配置

**模型参数**：
- `video_repr_dim`: 视频表征维度（1536 = 512 + 2*512）
- `mbart_model_path`: mBART 模型本地路径
- `d_model`: mBART 嵌入维度（1024）
- `freeze_mbart`: 是否冻结 mBART 参数

**训练参数**：
- `video_reprs_path`: 视频表征文件路径
- `text_data_path`: 文本数据文件路径
- `batch_size`: 批次大小（通常较小，如4）
- `epochs`: 训练轮数

**推理参数**：
- `checkpoint`: 模型检查点路径
- `video_reprs_path`: 视频表征文件路径
- `max_length`: 生成文本最大长度
- `num_beams`: Beam search 宽度

## 使用方式

### 1. 在脚本中使用

所有训练和推理脚本支持 `--config` 参数：

```bash
python training/skeleton/train.py --config config/config.py
```

脚本会自动从配置文件中读取对应模型的参数。

### 2. 在代码中使用

```python
import sys
sys.path.append('path/to/project')

from config.config import SKELETON, TEMPORAL, ALIGNMENT

# 访问配置
batch_size = SKELETON['training']['batch_size']
d_global = SKELETON['model']['d_global']
```

### 3. 命令行参数覆盖

即使使用配置文件，也可以通过命令行参数覆盖：

```bash
python training/skeleton/train.py \
    --config config/config.py \
    --batch_size 32 \
    --epochs 200
```

命令行参数的优先级高于配置文件。

## 路径配置

### 数据路径
```python
KEYPOINTS_DATA_PATH = "keypoints_data.pkl"
BODY_KEYPOINTS_DATA_PATH = "sign_language_keypoints.pkl"
VIDEO_SEQUENCES_PATH = "video_sequences.pkl"
TEXT_DATA_PATH = "text_data.json"
```

### 检查点路径
```python
CHECKPOINTS = {
    'skeleton': 'checkpoints_skeleton_hierarchical',
    'temporal': 'checkpoints_temporal',
    'alignment': 'checkpoints_alignment'
}
```

### 输出目录
```python
OUTPUT_DIRS = {
    'temporal_reprs': 'temporal_representations_all',
    'visualizations': 'visualizations'
}
```

## 修改配置

### 快速调整训练参数

```python
# 在 config/config.py 中修改
SKELETON['training']['batch_size'] = 32
SKELETON['training']['epochs'] = 200
SKELETON['training']['lr'] = 5e-5
```

### 修改模型架构

```python
# 修改模型维度
SKELETON['model']['d_global'] = 512
SKELETON['model']['d_region'] = 256
```

### 切换数据路径

```python
# 修改数据路径
SKELETON['training']['data_path'] = 'new_keypoints.pkl'
```

## 最佳实践

1. **备份配置**：修改前备份原始配置
2. **版本控制**：将配置文件纳入版本控制
3. **注释说明**：为重要参数添加注释
4. **环境分离**：可以为不同环境创建不同配置文件
5. **参数验证**：修改后验证参数合理性

## 注意事项

1. **路径格式**：使用相对路径或绝对路径，确保路径正确
2. **参数一致性**：确保模型参数与训练脚本一致
3. **依赖关系**：某些配置项依赖其他配置（如 `video_reprs_path` 依赖时序推理的输出）
4. **默认值**：所有参数都有默认值，但建议明确设置

## 创建自定义配置

可以创建新的配置文件：

```python
# my_config.py
from config.config import *

# 覆盖特定参数
SKELETON['training']['batch_size'] = 64
SKELETON['training']['epochs'] = 300
```

然后使用：
```bash
python run_pipeline.py --config-override my_config.py
```


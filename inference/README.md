# Inference 目录

推理脚本目录，包含所有模型的推理代码。

## 目录结构

```
inference/
├── hierarchical_keypoint/    # 层次化关键点模型推理
│   └── inference.py
├── hierarchical_image/       # 层次化图像模型推理
│   └── inference.py
├── skeleton/                 # 骨架模型推理
│   └── inference.py
├── temporal/                 # 时序模型推理
│   └── inference.py
└── alignment/                # 对齐模型推理
    └── inference.py
```

## 推理脚本说明

### 1. Skeleton Model Inference (`skeleton/inference.py`)

从全身关键点生成骨架图像，并提取表征。

**使用配置文件**：
```bash
python inference/skeleton/inference.py --config config/config.py
```

**命令行参数**：
```bash
python inference/skeleton/inference.py \
    --config config/config.py \
    --checkpoint checkpoints_skeleton_hierarchical/best_model.pth \
    --num_samples 10
```

**功能**：
- 生成骨架图像
- 提取全局和区域表征
- 可视化原始和重构的骨架对比

**输出**：
- 重构的骨架图像
- 表征统计信息
- 可视化对比图

### 2. Temporal Model Inference (`temporal/inference.py`)

从视频序列提取时序表征。

**使用配置文件**：
```bash
python inference/temporal/inference.py --config config/config.py
```

**命令行参数**：
```bash
python inference/temporal/inference.py \
    --config config/config.py \
    --checkpoint checkpoints_temporal/best_model_stage1.pth \
    --video_sequences video_sequences.pkl \
    --num_samples 152 \
    --output_dir temporal_representations_all
```

**功能**：
- 提取全局表征（`global_reprs`）
- 提取局部表征（`local_reprs`）
- 计算表征相似度矩阵
- 分析表征多样性

**输出**：
- `all_representations.npz`：包含所有表征
  - `global_reprs`: [num_samples, 512]
  - `local_reprs`: [num_samples, 2, 512]
  - `temporal_reprs`: [num_samples, seq_len, 512]
- `similarity_matrix.npy`：相似度矩阵
- `distance_matrix.npy`：距离矩阵
- 每个序列的单独表征文件

**输出文件格式**：
```python
# all_representations.npz
{
    'global_reprs': np.array([num_samples, 512]),      # 全局表征
    'local_reprs': np.array([num_samples, 2, 512]),    # 局部表征（2个局部变量）
    'temporal_reprs': np.array([num_samples, seq_len, 512])  # 时序表征
}
```

### 3. Alignment Model Inference (`alignment/inference.py`)

从视频表征生成文本描述。

**使用配置文件**：
```bash
python inference/alignment/inference.py --config config/config.py
```

**命令行参数**：
```bash
python inference/alignment/inference.py \
    --config config/config.py \
    --checkpoint checkpoints_alignment/best_model.pth \
    --video_reprs_path temporal_representations_all/all_representations.npz \
    --num_samples 10 \
    --output_path generation_results.json
```

**功能**：
- 从视频表征生成文本描述
- 与真实文本对比（如果提供）
- 保存生成结果

**输出**：
- 生成的文本描述（打印到控制台）
- `generation_results.json`：包含生成文本和真实文本（如果提供）

**输出文件格式**：
```json
[
    {
        "sample_id": 0,
        "generated_text": "生成的文本描述",
        "ground_truth": "真实文本描述"
    },
    ...
]
```

## 使用配置文件

所有推理脚本都支持从配置文件读取参数：

```bash
# 使用配置文件
python inference/temporal/inference.py --config config/config.py

# 使用配置文件 + 命令行参数覆盖
python inference/temporal/inference.py \
    --config config/config.py \
    --num_samples 50
```

## 完整推理流程

### 1. 提取时序表征
```bash
python inference/temporal/inference.py --config config/config.py
```

### 2. 生成文本描述
```bash
python inference/alignment/inference.py --config config/config.py
```

## 表征文件说明

### 时序表征文件 (`all_representations.npz`)

```python
import numpy as np

# 加载表征
data = np.load('temporal_representations_all/all_representations.npz')
global_reprs = data['global_reprs']      # [num_samples, 512]
local_reprs = data['local_reprs']        # [num_samples, 2, 512]
temporal_reprs = data['temporal_reprs']  # [num_samples, seq_len, 512]

# 组合表征（用于对齐模型）
# 总维度：512 + 2*512 = 1536
combined_repr = np.concatenate([
    global_reprs,
    local_reprs.reshape(num_samples, -1)
], axis=1)  # [num_samples, 1536]
```

## 注意事项

1. **检查点路径**：确保检查点文件存在
2. **输入数据**：确保输入数据格式正确
3. **输出目录**：推理脚本会自动创建输出目录
4. **GPU 内存**：如果遇到 OOM，减小 `num_samples` 或使用 CPU
5. **表征维度**：确保表征维度与对齐模型的配置一致

## 调试和验证

### 检查表征质量

运行时脚本会输出：
- 表征统计信息（均值、标准差、最小值、最大值）
- 相似度分析
- 表征多样性分析

如果相似度过高（>0.99），可能表示：
- 模型未充分训练
- 表征坍塌
- 需要更多训练轮次

### 可视化结果

- Skeleton 推理：生成可视化对比图
- Temporal 推理：保存相似度矩阵和距离矩阵
- Alignment 推理：保存生成文本结果


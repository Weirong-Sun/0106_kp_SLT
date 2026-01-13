# GPU 分布式关键点提取指南

## 概述

本指南介绍如何使用 GPU 优化版本进行 PHOENIX 数据集的关键点提取，以及如何整合生成的数据用于模型训练。

## 重要说明：MediaPipe 的 GPU 支持

### ⚠️ 重要提示

**MediaPipe 对静态图像处理主要使用 CPU**，即使绑定 GPU，加速效果可能不明显。但是：

1. **多进程并行**: 通过多进程并行处理，每个进程绑定到不同 GPU，可以：
   - 避免 GPU 上下文冲突
   - 更好地管理资源
   - 为未来可能的 GPU 加速做准备

2. **批处理优化**: GPU 版本包含批处理和进度跟踪优化

3. **实际加速**: 主要加速来自于多进程并行处理，而不是 GPU 本身

### MediaPipe GPU 加速情况

- **静态图像处理**: 主要使用 **CPU**
- **视频流处理**: 可能使用 **GPU**（OpenGL/Vulkan）
- **Python API**: 通常使用 **CPU** 模式

**结论**: 即使使用 GPU 绑定，MediaPipe 可能仍然使用 CPU，但多进程并行处理本身就能提供显著的加速。

## 使用方法

### 基本使用（4 个 GPU）

```bash
# 使用 4 个 GPU，每个 GPU 1 个进程（总共 4 个进程）
python data/extract_phoenix_keypoints_gpu.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_full.pkl \
    --num_gpus 4 \
    --num_workers_per_gpu 1
```

### 激进配置（每个 GPU 多个进程）

```bash
# 使用 4 个 GPU，每个 GPU 2 个进程（总共 8 个进程）
python data/extract_phoenix_keypoints_gpu.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_full.pkl \
    --num_gpus 4 \
    --num_workers_per_gpu 2
```

### 分批处理

```bash
# 只处理训练集（最大，约 70 小时）
python data/extract_phoenix_keypoints_gpu.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_train.pkl \
    --splits train \
    --num_gpus 4 \
    --num_workers_per_gpu 1

# 然后处理验证集和测试集
python data/extract_phoenix_keypoints_gpu.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_dev_test.pkl \
    --splits dev test \
    --num_gpus 4 \
    --num_workers_per_gpu 1
```

## 性能估算

### 假设条件

- **数据集**: 947,756 个图像
- **处理速度**:
  - 单进程: 约 1 图像/秒
  - 4进程并行: 约 3-4 图像/秒
  - 8进程并行: 约 6-8 图像/秒

### 时间估算

| 配置 | 进程数 | 估算速度 | 估算时间 | 说明 |
|------|--------|---------|---------|------|
| 单进程 | 1 | 1 图像/秒 | ~264 小时 (11 天) | 基准 |
| 4 GPU × 1 进程 | 4 | 3-4 图像/秒 | ~66-88 小时 (2.7-3.7 天) | **推荐** |
| 4 GPU × 2 进程 | 8 | 6-8 图像/秒 | ~33-44 小时 (1.4-1.8 天) | 激进 |

**注意**: 实际速度取决于：
- CPU 性能
- 内存带宽
- I/O 速度（图像加载）
- MediaPipe 处理时间
- 系统负载

## 数据整合

### 方法 1: 自动整合（推荐）

GPU 版本会自动整合分块处理的结果，生成统一的输出文件：

```bash
# 直接生成整合后的文件
python data/extract_phoenix_keypoints_gpu.py \
    --dataset_path /path/to/dataset \
    --output_path phoenix_keypoints_full.pkl \
    --num_gpus 4
```

输出文件 `phoenix_keypoints_full.pkl` 已经包含了所有划分的整合数据。

### 方法 2: 手动合并多个结果文件

如果分多次处理，可以合并多个结果文件：

```bash
# 合并多个结果文件
python data/merge_keypoint_results.py \
    --input_files result_train.pkl result_dev.pkl result_test.pkl \
    --output_path phoenix_keypoints_merged.pkl
```

### 方法 3: 准备模型输入数据

将关键点数据转换为模型可直接使用的格式：

```bash
# 准备模型输入数据（NumPy 格式）
python data/prepare_model_input.py \
    --keypoints_file phoenix_keypoints_full.pkl \
    --output_dir model_input_data
```

这会生成：
- `train_keypoints.npz` - 训练集（NumPy 格式）
- `dev_keypoints.npz` - 验证集（NumPy 格式）
- `test_keypoints.npz` - 测试集（NumPy 格式）
- 每个划分的统计信息 JSON 文件

## 模型输入格式

### 方法 1: 直接使用 pickle 文件

```python
import pickle

# 加载完整数据
with open('phoenix_keypoints_full.pkl', 'rb') as f:
    data = pickle.load(f)

# 使用训练集
train_keypoints = data['train']['keypoints']  # List of dicts
train_video_ids = data['train']['video_ids']
```

### 方法 2: 使用 NumPy 格式（推荐）

```python
import numpy as np

# 加载 NumPy 格式（更快，更小）
train_data = np.load('model_input_data/train_keypoints.npz')
train_keypoints = train_data['keypoints']  # [N, 143, 3]
train_video_ids = train_data['video_ids']
train_image_paths = train_data['image_paths']
```

### 数据格式说明

**Pickle 格式**（原始）:
```python
{
    'train': {
        'keypoints': [
            {
                'face': np.array([68, 3]) or None,
                'left_hand': np.array([21, 3]) or None,
                'right_hand': np.array([21, 3]) or None,
                'pose': np.array([33, 3]) or None
            },
            ...
        ],
        'image_paths': [str, ...],
        'video_ids': [str, ...]
    },
    ...
}
```

**NumPy 格式**（模型输入）:
```python
# train_keypoints.npz
{
    'keypoints': np.array([N, 143, 3]),  # N 个样本，143 个关键点，3 个坐标
    'image_paths': [str, ...],
    'video_ids': [str, ...]
}
```

其中 143 = 68 (face) + 21 (left_hand) + 21 (right_hand) + 33 (pose)

## 完整工作流程

### 步骤 1: 提取关键点（完整数据集）

```bash
# 使用 GPU 优化版本提取所有数据
python data/extract_phoenix_keypoints_gpu.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_full.pkl \
    --num_gpus 4 \
    --num_workers_per_gpu 1 \
    > extraction.log 2>&1 &

# 查看进度
tail -f extraction.log
```

### 步骤 2: 检查提取结果

```bash
# 检查进度
python check_extraction_progress.py phoenix_keypoints_full.pkl

# 查看详细信息
python view_keypoints.py phoenix_keypoints_full.pkl --detailed
```

### 步骤 3: 准备模型输入数据

```bash
# 转换为模型输入格式
python data/prepare_model_input.py \
    --keypoints_file phoenix_keypoints_full.pkl \
    --output_dir model_input_data
```

### 步骤 4: 验证数据

```bash
# 可视化一些样本
python visualize_phoenix_keypoints.py phoenix_keypoints_full.pkl \
    --num_samples 10 \
    --splits train
```

### 步骤 5: 在模型中使用

```python
# 在训练脚本中加载数据
import numpy as np

# 加载训练集
train_data = np.load('model_input_data/train_keypoints.npz')
train_keypoints = train_data['keypoints']  # [N, 143, 3]

# 加载验证集
dev_data = np.load('model_input_data/dev_keypoints.npz')
dev_keypoints = dev_data['keypoints']

# 加载测试集
test_data = np.load('model_input_data/test_keypoints.npz')
test_keypoints = test_data['keypoints']
```

## GPU 使用情况说明

### 当前实现

1. **GPU 绑定**: 每个进程绑定到不同的 GPU（通过 `CUDA_VISIBLE_DEVICES`）
2. **进程隔离**: 每个进程有独立的 MediaPipe 实例
3. **资源管理**: 避免多个进程竞争同一 GPU

### 实际 GPU 使用

- **MediaPipe**: 可能仍然使用 CPU（对静态图像处理）
- **PyTorch**: 如果代码中有 PyTorch 操作，可以使用 GPU
- **系统**: GPU 绑定有助于资源管理和避免冲突

### 加速来源

主要加速来自：
1. **多进程并行**: 4 个进程同时处理不同的图像批次
2. **批处理优化**: 更高效的内存使用和 I/O
3. **进程隔离**: 避免资源竞争

**理论加速比**: 4 进程 ≈ 3-4 倍（取决于硬件）

## 参数说明

### GPU 版本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_gpus` | 使用的 GPU 数量 | `4` |
| `--num_workers_per_gpu` | 每个 GPU 的工作进程数 | `1` |
| `--temp_dir` | 临时文件目录 | `temp_keypoints` |
| 其他参数 | 同普通版本 | - |

### 推荐配置

| 场景 | GPU 数 | 每GPU进程数 | 总进程数 | 说明 |
|------|--------|------------|---------|------|
| **推荐** | 4 | 1 | 4 | 平衡性能和资源 |
| **激进** | 4 | 2 | 8 | 最大化并行度 |
| **保守** | 2 | 1 | 2 | 如果资源有限 |

## 后台运行

```bash
# 使用 nohup 在后台运行
nohup python data/extract_phoenix_keypoints_gpu.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_full.pkl \
    --num_gpus 4 \
    --num_workers_per_gpu 1 \
    > extraction_full.log 2>&1 &

# 查看进程
ps aux | grep extract_phoenix_keypoints_gpu

# 查看日志
tail -f extraction_full.log

# 查看 GPU 使用情况
watch -n 5 nvidia-smi
```

## 监控和调试

### 监控 GPU 使用

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 查看 GPU 进程
nvidia-smi pmon
```

### 检查处理进度

```bash
# 定期检查文件大小（应该持续增长）
watch -n 60 'ls -lh phoenix_keypoints_full.pkl'

# 使用进度检查脚本
python check_extraction_progress.py phoenix_keypoints_full.pkl
```

### 如果进程崩溃

如果某个进程崩溃：
1. 检查日志文件中的错误信息
2. 临时文件可能保留在 `temp_keypoints/` 目录
3. 可以手动清理临时文件后重新运行

## 注意事项

1. **磁盘空间**: 完整数据集处理需要约 1.5-2 GB 磁盘空间
2. **内存使用**: 每个进程约 200-300 MB，4 个进程约 800 MB-1.2 GB
3. **临时文件**: 处理过程中会创建临时文件，确保有足够空间
4. **处理时间**: 完整数据集需要 1-4 天，建议后台运行
5. **结果验证**: 处理完成后建议验证结果完整性

## 与 CPU 版本对比

| 特性 | CPU 版本 | GPU 版本 |
|------|---------|---------|
| **处理速度** | 基准 | 3-4 倍（多进程） |
| **GPU 使用** | 否 | 是（绑定，但可能不使用） |
| **数据整合** | 自动 | 自动 |
| **模型输入准备** | 需要手动 | 提供工具脚本 |
| **推荐场景** | 小数据集/测试 | 完整数据集 |

## 相关脚本

1. **`extract_phoenix_keypoints_gpu.py`** - GPU 优化版本提取脚本
2. **`merge_keypoint_results.py`** - 合并多个结果文件
3. **`prepare_model_input.py`** - 准备模型输入数据
4. **`check_extraction_progress.py`** - 检查提取进度
5. **`view_keypoints.py`** - 查看关键点数据
6. **`visualize_phoenix_keypoints.py`** - 可视化关键点

## 故障排除

### 问题 1: GPU 绑定失败

**症状**: `CUDA_VISIBLE_DEVICES` 设置无效

**解决**:
- 检查 GPU 数量: `nvidia-smi --list-gpus`
- 确保 `num_gpus` 不超过实际 GPU 数量

### 问题 2: 内存不足

**症状**: OOM 错误

**解决**:
- 减少 `num_workers_per_gpu`（例如从 2 改为 1）
- 减少 `num_gpus`（例如从 4 改为 2）

### 问题 3: 处理速度慢

**解决**:
- 检查 CPU 核心数是否足够
- 检查磁盘 I/O 速度
- 考虑增加 `num_workers_per_gpu`

### 问题 4: 临时文件过多

**解决**:
- 处理完成后会自动清理临时文件
- 如果进程异常退出，手动清理 `temp_keypoints/` 目录

## 相关文档

- [分布式关键点提取分析.md](./分布式关键点提取分析.md)
- [数据集大小差异分析.md](./数据集大小差异分析.md)
- [PHOENIX数据集关键点提取指南.md](./PHOENIX数据集关键点提取指南.md)

---

**文档创建时间**: 2024年
**适用版本**: extract_phoenix_keypoints_gpu.py
**GPU 支持**: 4 块 GPU 分布式处理



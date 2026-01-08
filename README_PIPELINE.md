# Pipeline 使用说明

## 概述

`run_pipeline.py` 是一个自动化脚本，可以按照配置文件中定义的参数顺序执行整个训练和推理流程。

## 配置文件

所有参数都在 `config/config.py` 中定义，包括：
- 模型配置（网络结构、超参数）
- 训练配置（batch size、学习率、epochs）
- 推理配置（checkpoint路径、输出路径）
- 路径配置（数据路径、模型路径）

## 使用方法

### 1. 完整 Pipeline（默认）

运行所有阶段：
```bash
python run_pipeline.py
```

### 2. 运行特定阶段

只训练 skeleton 模型：
```bash
python run_pipeline.py --stage skeleton
```

只训练 temporal 模型：
```bash
python run_pipeline.py --stage temporal
```

只训练 alignment 模型：
```bash
python run_pipeline.py --stage alignment
```

### 3. 跳过已训练的模型

如果 checkpoint 已存在，跳过训练：
```bash
python run_pipeline.py --skip-trained
```

### 4. 训练但不运行推理

```bash
python run_pipeline.py --no-inference
```

### 5. 使用自定义配置

```bash
python run_pipeline.py --config-override my_config.py
```

## Pipeline 阶段

Pipeline 按以下顺序执行：

1. **Skeleton Model Training** (训练骨架重构模型)
   - 输入：全身关键点数据 (`sign_language_keypoints.pkl`)
   - 输出：骨架重构模型 checkpoint

2. **Temporal Model Training** (训练时序模型)
   - 输入：视频序列 (`video_sequences.pkl`) + 预训练的骨架模型
   - 输出：时序表征模型 checkpoint

3. **Temporal Model Inference** (提取时序表征)
   - 输入：训练好的时序模型
   - 输出：视频序列的时序表征 (`temporal_representations_all/all_representations.npz`)

4. **Alignment Model Training** (训练对齐模型)
   - 输入：时序表征 + 文本数据 (`text_data.json`)
   - 输出：对齐模型 checkpoint

5. **Alignment Model Inference** (生成文本描述)
   - 输入：训练好的对齐模型
   - 输出：生成的文本描述 (`generation_results.json`)

## 配置修改

编辑 `config/config.py` 来修改任何参数：

```python
# 修改 batch size
SKELETON['training']['batch_size'] = 32

# 修改学习率
TEMPORAL['training']['lr'] = 5e-5

# 修改 epochs
ALIGNMENT['training']['epochs'] = 50
```

## 示例工作流

### 第一次运行（完整流程）
```bash
# 1. 确保数据已准备好
# - sign_language_keypoints.pkl
# - video_sequences.pkl
# - text_data.json

# 2. 运行完整 pipeline
python run_pipeline.py
```

### 只重新训练 alignment 模型
```bash
# 如果 skeleton 和 temporal 模型已经训练好
python run_pipeline.py --stage alignment
```

### 快速测试（小 batch size，少 epochs）
```python
# 在 config/config.py 中临时修改
SKELETON['training']['batch_size'] = 4
SKELETON['training']['epochs'] = 5
TEMPORAL['training']['epochs'] = 5
ALIGNMENT['training']['epochs'] = 5

# 然后运行
python run_pipeline.py
```

## 注意事项

1. **数据准备**：确保所有必要的数据文件已准备好
2. **路径检查**：检查配置文件中的路径是否正确
3. **GPU 内存**：如果遇到 OOM 错误，减小 batch_size
4. **检查点**：训练过程中会自动保存最佳模型

## 故障排除

如果某个阶段失败：
1. 检查错误信息
2. 确认数据路径和格式正确
3. 检查 GPU 内存是否足够
4. 可以单独运行失败的阶段进行调试

例如，如果 temporal 训练失败：
```bash
python training/temporal/train.py --video_data_path video_sequences.pkl ...
```


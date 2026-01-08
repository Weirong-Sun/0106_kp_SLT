# Training 目录

训练脚本目录，包含所有模型的训练代码。

## 目录结构

```
training/
├── hierarchical_keypoint/    # 层次化关键点模型训练
│   └── train.py
├── hierarchical_image/       # 层次化图像模型训练
│   └── train.py
├── skeleton/                 # 骨架模型训练
│   └── train.py
├── temporal/                 # 时序模型训练
│   └── train.py
└── alignment/                # 对齐模型训练
    └── train.py
```

## 训练脚本说明

### 1. Skeleton Model Training (`skeleton/train.py`)

训练骨架重构模型，从全身关键点生成骨架图像。

**使用配置文件**：
```bash
python training/skeleton/train.py --config config/config.py
```

**命令行参数**（可选，会覆盖配置）：
```bash
python training/skeleton/train.py \
    --config config/config.py \
    --data_path sign_language_keypoints.pkl \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4
```

**主要参数**：
- `--data_path`: 关键点数据文件
- `--batch_size`: 批次大小（默认16）
- `--epochs`: 训练轮数（默认100）
- `--lr`: 学习率（默认1e-4）
- `--d_global`: 全局表征维度（默认256）
- `--d_region`: 区域表征维度（默认128）
- `--use_weighted_loss`: 使用加权损失（强调手部和面部）
- `--hand_weight`: 手部区域权重（默认2.0）
- `--face_weight`: 面部区域权重（默认1.5）

**输出**：
- 最佳模型：`checkpoints_skeleton_hierarchical/best_model.pth`

### 2. Temporal Model Training (`temporal/train.py`)

训练时序模型，学习视频序列的时序表征。

**使用配置文件**：
```bash
python training/temporal/train.py --config config/config.py
```

**命令行参数**：
```bash
python training/temporal/train.py \
    --config config/config.py \
    --video_data_path video_sequences.pkl \
    --frame_encoder_checkpoint checkpoints_skeleton_hierarchical/best_model.pth
```

**主要参数**：
- `--video_data_path`: 视频序列数据文件
- `--frame_encoder_checkpoint`: 预训练的骨架模型检查点
- `--batch_size`: 批次大小（默认8）
- `--seq_len`: 序列长度（默认6）
- `--stage1_epochs`: Stage 1 训练轮数（默认50）
- `--d_temporal`: 时序表征维度（默认512）
- `--d_final`: 最终表征维度（默认512）
- `--freeze_frame_encoder`: 冻结帧编码器

**训练策略**：
- Stage 1: 编码-解码重构训练（帧编码器冻结）

**输出**：
- 最佳模型：`checkpoints_temporal/best_model_stage1.pth`

### 3. Alignment Model Training (`alignment/train.py`)

训练视频-语言对齐模型，将视频表征对齐到文本描述。

**使用配置文件**：
```bash
python training/alignment/train.py --config config/config.py
```

**命令行参数**：
```bash
python training/alignment/train.py \
    --config config/config.py \
    --video_reprs_path temporal_representations_all/all_representations.npz \
    --text_data_path text_data.json
```

**主要参数**：
- `--video_reprs_path`: 视频表征文件（从时序模型推理得到）
- `--text_data_path`: 文本数据文件
- `--batch_size`: 批次大小（默认4）
- `--epochs`: 训练轮数（默认20）
- `--lr`: 学习率（默认1e-4）
- `--mbart_model_path`: mBART 模型本地路径
- `--freeze_mbart`: 冻结 mBART 参数

**输出**：
- 最佳模型：`checkpoints_alignment/best_model.pth`

## 训练流程

### 完整 Pipeline

推荐使用 `run_pipeline.py` 自动执行完整流程：

```bash
python run_pipeline.py
```

### 手动训练步骤

1. **训练骨架模型**：
   ```bash
   python training/skeleton/train.py --config config/config.py
   ```

2. **训练时序模型**（需要先完成步骤1）：
   ```bash
   python training/temporal/train.py --config config/config.py
   ```

3. **提取时序表征**（需要先完成步骤2）：
   ```bash
   python inference/temporal/inference.py --config config/config.py
   ```

4. **训练对齐模型**（需要先完成步骤3）：
   ```bash
   python training/alignment/train.py --config config/config.py
   ```

## 配置管理

所有训练参数都在 `config/config.py` 中统一管理：

```python
SKELETON = {
    'model': {...},      # 模型架构参数
    'training': {...},   # 训练参数
    'inference': {...}   # 推理参数
}
```

## 检查点格式

所有检查点包含：
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `epoch`: 训练轮数
- `val_loss`: 验证损失
- `model_config`: 模型配置（用于推理时重建模型）

## 注意事项

1. **数据准备**：确保训练数据已准备好
2. **GPU 内存**：如果遇到 OOM 错误，减小 `batch_size`
3. **检查点**：训练过程中会自动保存最佳模型
4. **恢复训练**：可以修改代码添加 `--resume` 参数来恢复训练
5. **配置优先**：使用 `--config` 参数时，配置文件的参数优先，命令行参数用于覆盖

## 监控训练

训练过程中会输出：
- 每个 epoch 的训练损失和验证损失
- 最佳模型的保存信息
- 模型结构信息（参数量、架构等）

可以使用 TensorBoard 或其他工具进行可视化（需要额外代码）。


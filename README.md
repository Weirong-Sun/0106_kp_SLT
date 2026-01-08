# Video-Language Alignment Pipeline

视频-语言对齐 Pipeline，用于从视频关键点学习表征并生成文本描述。

## 项目概述

本项目实现了一个完整的视频到文本的 pipeline，用于手语识别，包含以下阶段：

1. **数据提取**：从视频提取帧和全身关键点（面部+双手+姿态）
2. **骨架重构**：学习全身关键点的层次化表征
3. **时序建模**：学习视频序列的时序表征
4. **语言对齐**：将视频表征对齐到文本描述

**注意**：项目中还包含 `hierarchical_keypoint` 和 `hierarchical_image` 模型，这些是早期实验模型（用于68个面部关键点），与手语识别 pipeline 无关，保留用于参考。

## 项目结构

```
0106/
├── data/                    # 数据提取和预处理
│   ├── extract_video_frames.py          # 从视频提取帧
│   ├── extract_body_keypoints.py        # 提取全身关键点
│   ├── prepare_video_sequences.py        # 组织视频序列
│   ├── create_text_data_from_videos.py   # 生成文本数据
│   └── test_keypoints.py                 # 测试关键点提取
│
├── utils/                   # 工具函数
│   ├── utils_image.py                    # 图像处理工具
│   └── utils_skeleton.py                 # 骨架绘制工具
│
├── models/                  # 模型定义（按模型类型组织）
│   ├── hierarchical_keypoint/            # 层次化关键点模型
│   ├── hierarchical_image/               # 层次化图像重构模型
│   ├── skeleton/                         # 骨架重构模型
│   ├── temporal/                         # 时序 Transformer 模型
│   └── alignment/                        # 视频-语言对齐模型
│
├── training/                # 训练脚本（按模型类型组织）
│   ├── skeleton/                         # 骨架模型训练（核心）
│   ├── temporal/                         # 时序模型训练（核心）
│   ├── alignment/                        # 对齐模型训练（核心）
│   ├── hierarchical_keypoint/            # 实验模型训练（非手语）
│   └── hierarchical_image/               # 实验模型训练（非手语）
│
├── inference/               # 推理脚本（按模型类型组织）
│   ├── hierarchical_keypoint/            # 层次化关键点模型推理
│   ├── hierarchical_image/               # 层次化图像重构模型推理
│   ├── skeleton/                         # 骨架模型推理
│   ├── temporal/                         # 时序模型推理
│   └── alignment/                        # 对齐模型推理
│
├── config/                  # 配置文件
│   └── config.py                         # 统一配置文件
│
├── run_pipeline.py          # Pipeline 自动化脚本
├── PROJECT_STRUCTURE.md     # 项目结构详细说明
└── README.md                # 本文档
```

详细的目录说明请参考各目录下的 README.md 文件。

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保已安装 MediaPipe（用于关键点提取）
pip install mediapipe
```

### 2. 数据准备

```bash
# 步骤 1: 从视频提取帧
python data/extract_video_frames.py \
    --dataset_path /path/to/videos \
    --output_path extracted_frames

# 步骤 2: 提取关键点
python data/extract_body_keypoints.py \
    --input_path extracted_frames \
    --output_path sign_language_keypoints.pkl

# 步骤 3: 组织视频序列
python data/prepare_video_sequences.py \
    --keypoints_path sign_language_keypoints.pkl \
    --output_path video_sequences.pkl \
    --min_seq_len 4

# 步骤 4: 生成文本数据
python data/create_text_data_from_videos.py \
    --keypoints_path sign_language_keypoints.pkl \
    --output_path text_data.json
```

### 3. 训练 Pipeline

#### 方式 1：使用 Pipeline 脚本（推荐）

```bash
# 运行完整 pipeline
python run_pipeline.py

# 只训练特定阶段
python run_pipeline.py --stage skeleton
python run_pipeline.py --stage temporal
python run_pipeline.py --stage alignment

# 跳过已训练的模型
python run_pipeline.py --skip-trained
```

#### 方式 2：手动运行各阶段

```bash
# 1. 训练骨架模型
python training/skeleton/train.py --config config/config.py

# 2. 训练时序模型
python training/temporal/train.py --config config/config.py

# 3. 提取时序表征
python inference/temporal/inference.py --config config/config.py

# 4. 训练对齐模型
python training/alignment/train.py --config config/config.py
```

### 4. 推理和生成

```bash
# 生成文本描述
python inference/alignment/inference.py --config config/config.py
```

## Pipeline 流程详解

### 阶段 1: 数据提取

**目标**：从原始视频数据中提取关键点和序列

1. **视频 → 帧**：`extract_video_frames.py`
   - 从视频文件提取帧图像
   - 保存到 `extracted_frames/` 目录

2. **帧 → 关键点**：`extract_body_keypoints.py`
   - 使用 MediaPipe 提取全身关键点
   - 包括：面部（68点）、左手（21点）、右手（21点）、姿态（33点）
   - 总计 143 个关键点
   - 保存为 pickle 文件

3. **关键点 → 序列**：`prepare_video_sequences.py`
   - 根据视频 ID 或检测间隔组织序列
   - 过滤过短的序列
   - 保存视频序列数据

4. **生成文本**：`create_text_data_from_videos.py`
   - 从视频路径提取标题作为文本描述
   - 生成文本数据文件

### 阶段 1: 骨架模型训练

**目标**：学习全身关键点的层次化表征（手语识别核心模型）

**输入**：
- 全身关键点数据：`sign_language_keypoints.pkl`
- 关键点格式：143 个点 `[68 面部 + 21 左手 + 21 右手 + 33 姿态]`

**模型架构**：
- 4 个区域编码器（面部、左手、右手、姿态）
- 跨区域交互层
- 全局聚合层
- CNN 图像解码器

**输出**：
- 全局表征：`[batch, 256]`
- 区域表征：`[batch, 4, 128]`
- 重构的骨架图像：`[batch, 1, 256, 256]`

**检查点**：`checkpoints_skeleton_hierarchical/best_model.pth`

### 阶段 2: 时序模型训练

**目标**：学习视频序列的时序表征（手语识别核心模型）

**输入**：
- 视频序列：`video_sequences.pkl`
- 预训练的骨架模型（阶段 2 的输出）

**模型架构**：
- 帧编码器：使用预训练的骨架模型编码每一帧
- 时序融合层：合并全局和区域表征
- 时序 Transformer 编码器：学习时序依赖
- 多尺度压缩表征：
  - 全局变量：关注所有帧
  - 局部变量：关注不同时间窗口（如奇数/偶数帧）

**输出**：
- 全局表征：`[batch, 512]`（关注所有帧）
- 局部表征：`[batch, 2, 512]`（关注不同时间窗口）
- 总表征维度：1536 = 512 + 2×512

**检查点**：`checkpoints_temporal/best_model_stage1.pth`

**推理输出**：`temporal_representations_all/all_representations.npz`

### 阶段 3: 对齐模型训练

**目标**：将视频表征对齐到文本描述（手语识别核心模型）

**输入**：
- 视频表征：`temporal_representations_all/all_representations.npz`
- 文本数据：`text_data.json`

**模型架构**：
- 视频投影层：将视频表征投影到 mBART 嵌入空间
- mBART 解码器：生成文本描述

**输出**：
- 生成的文本描述

**检查点**：`checkpoints_alignment/best_model.pth`

## 配置管理

所有参数统一在 `config/config.py` 中管理：

```python
SKELETON = {
    'model': {...},      # 模型架构参数
    'training': {...},   # 训练参数
    'inference': {...}   # 推理参数
}

TEMPORAL = {...}
ALIGNMENT = {...}
```

### 修改配置

编辑 `config/config.py` 来调整参数：

```python
# 修改训练参数
SKELETON['training']['batch_size'] = 32
SKELETON['training']['epochs'] = 200

# 修改模型参数
SKELETON['model']['d_global'] = 512
```

详细配置说明请参考 `config/README.md`。

## 数据格式

### 关键点数据 (`sign_language_keypoints.pkl`)

```python
{
    'keypoints': [
        {
            'face': np.array([68, 3]),      # 68个面部关键点
            'left_hand': np.array([21, 3]), # 21个左手关键点
            'right_hand': np.array([21, 3]),# 21个右手关键点
            'pose': np.array([33, 3])       # 33个姿态关键点
        },
        ...
    ],
    'image_paths': [str, ...]  # 对应的图像路径
}
```

### 视频序列数据 (`video_sequences.pkl`)

```python
{
    'sequences': [
        np.array([[143, 3], ...]),  # 每个序列：多个帧，每帧143个关键点
        ...
    ]
}
```

### 视频表征 (`all_representations.npz`)

```python
{
    'global_reprs': np.array([num_samples, 512]),     # 全局表征
    'local_reprs': np.array([num_samples, 2, 512]),   # 局部表征
    'temporal_reprs': np.array([num_samples, seq_len, 512])  # 时序表征
}
```

### 文本数据 (`text_data.json`)

```json
{
    "texts": [
        "视频标题1",
        "视频标题2",
        ...
    ]
}
```

## 输出文件

### 模型检查点（核心模型）

- `checkpoints_skeleton_hierarchical/best_model.pth` - 骨架模型（核心）
- `checkpoints_temporal/best_model_stage1.pth` - 时序模型（核心）
- `checkpoints_alignment/best_model.pth` - 对齐模型（核心）

**实验模型检查点**（非手语识别）：
- `checkpoints_hierarchical/best_model.pth` - 层次化关键点模型（实验，68点面部）
- `checkpoints_hierarchical_image/best_model.pth` - 层次化图像模型（实验，68点面部）

### 推理输出

- `temporal_representations_all/all_representations.npz` - 视频表征
- `temporal_representations_all/similarity_matrix.npy` - 相似度矩阵
- `generation_results.json` - 生成的文本描述

### 可视化结果

- `visualizations_skeleton_hierarchical/` - 骨架重构可视化
- `test_keypoints_visualization/` - 关键点测试可视化

## 使用示例

### 完整 Pipeline 示例

```bash
# 1. 准备数据（假设数据已准备好）
# sign_language_keypoints.pkl
# video_sequences.pkl
# text_data.json

# 2. 运行完整 pipeline
python run_pipeline.py

# Pipeline 会自动执行：
# - 训练骨架模型
# - 训练时序模型
# - 提取时序表征
# - 训练对齐模型
# - 生成文本描述
```

### 单独运行示例

```bash
# 只训练骨架模型
python training/skeleton/train.py --config config/config.py

# 检查训练结果
python inference/skeleton/inference.py --config config/config.py --num_samples 5

# 训练时序模型（需要先有骨架模型）
python training/temporal/train.py --config config/config.py

# 提取表征
python inference/temporal/inference.py --config config/config.py

# 训练对齐模型
python training/alignment/train.py --config config/config.py

# 生成文本
python inference/alignment/inference.py --config config/config.py
```

## 参数说明

### 主要训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `batch_size` | 批次大小 | 16 (skeleton), 8 (temporal), 4 (alignment) |
| `epochs` | 训练轮数 | 100 (skeleton), 50 (temporal), 20 (alignment) |
| `lr` | 学习率 | 1e-4 |
| `d_global` | 全局表征维度 | 256 (skeleton), 512 (temporal) |
| `d_region` | 区域表征维度 | 128 |
| `seq_len` | 序列长度 | 6 |

### 模型架构参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_regions` | 区域数量 | 4 (skeleton) |
| `num_temporal_layers` | 时序 Transformer 层数 | 4 |
| `num_local_vars` | 局部变量数量 | 2 |
| `d_temporal` | 时序表征维度 | 512 |
| `d_final` | 最终表征维度 | 512 |

## 注意事项

1. **核心模型**：手语识别 Pipeline 只使用 Skeleton、Temporal、Alignment 三个模型
2. **实验模型**：`hierarchical_keypoint` 和 `hierarchical_image` 是早期实验模型（68点面部），与手语识别无关，保留用于参考
3. **数据准备**：确保数据格式正确，参考 `data/README.md`
4. **GPU 内存**：如果遇到 OOM 错误，减小 `batch_size`
5. **检查点**：训练过程中会自动保存最佳模型
6. **路径配置**：检查 `config/config.py` 中的路径是否正确
7. **依赖关系**：确保按顺序执行各阶段（skeleton → temporal → alignment）

## 故障排除

### 常见问题

1. **CUDA out of memory**
   - 解决：减小 `batch_size` 或使用 CPU

2. **找不到检查点文件**
   - 解决：检查路径配置，确保先训练前一阶段的模型

3. **数据格式错误**
   - 解决：参考数据格式说明，使用 `test_keypoints.py` 验证数据

4. **生成的文本为空或重复**
   - 解决：检查模型是否充分训练，可能需要增加训练轮数

## 文档索引

- **`data/README.md`** - 数据提取和预处理说明
- **`utils/README.md`** - 工具函数说明
- **`models/README.md`** - 模型架构说明（包含核心模型和实验模型）
- **`training/README.md`** - 训练脚本说明
- **`inference/README.md`** - 推理脚本说明
- **`config/README.md`** - 配置文件说明
- **`PROJECT_STRUCTURE.md`** - 项目结构详细说明
- **`README_PIPELINE.md`** - Pipeline 使用详细说明
- **`PIPELINE_CHECKLIST.md`** - Pipeline 检查清单

## 模型说明

### 核心模型（手语识别 Pipeline）

1. **Skeleton Model** - 骨架重构模型
   - 输入：143个全身关键点（面部+双手+姿态）
   - 输出：骨架图像 + 层次化表征

2. **Temporal Model** - 时序 Transformer 模型
   - 输入：视频序列的关键点
   - 输出：时序表征（全局+局部）

3. **Alignment Model** - 视频-语言对齐模型
   - 输入：视频时序表征
   - 输出：文本描述

### 实验模型（早期实验，非手语识别）

4. **Hierarchical Keypoint Model** - 层次化关键点模型
   - 输入：68个面部关键点
   - 用途：早期实验，与手语识别无关

5. **Hierarchical Image Model** - 层次化图像重构模型
   - 输入：68个面部关键点
   - 用途：早期实验，与手语识别无关

## 贡献

欢迎提交 Issue 和 Pull Request。

## 许可证

[添加许可证信息]

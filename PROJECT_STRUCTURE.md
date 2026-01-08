# 项目目录结构说明

本项目按照 pipeline 的方式组织代码，并按模型类型进一步细分，便于管理和定位。

## 目录结构

```
0106/
├── data/                    # 数据提取和预处理
│   ├── extract_video_frames.py          # 从视频提取帧
│   ├── extract_body_keypoints.py        # 提取全身关键点
│   ├── data_extraction.py                # 早期关键点提取（68点）
│   ├── prepare_video_sequences.py        # 组织视频序列
│   ├── create_text_data_example.py       # 创建文本数据示例
│   ├── create_text_data_from_videos.py   # 从视频创建文本数据
│   └── test_keypoints.py                 # 测试关键点提取
│
├── utils/                   # 工具函数
│   ├── utils_image.py                    # 图像处理工具（绘制关键点等）
│   └── utils_skeleton.py                 # 骨架绘制工具
│
├── models/                  # 模型定义（按模型类型组织）
│   ├── hierarchical_keypoint/            # 层次化关键点模型
│   │   └── model.py
│   ├── hierarchical_image/               # 层次化关键点到图像模型
│   │   └── model.py
│   ├── skeleton/                         # 层次化骨架重构模型
│   │   └── model.py
│   ├── temporal/                         # 时序 Transformer 模型
│   │   └── model.py
│   └── alignment/                        # 视频-语言对齐模型
│       └── model.py
│
├── training/                # 训练脚本（按模型类型组织）
│   ├── hierarchical_keypoint/            # 层次化关键点模型训练
│   │   └── train.py
│   ├── hierarchical_image/               # 层次化图像重构模型训练
│   │   └── train.py
│   ├── skeleton/                         # 层次化骨架模型训练
│   │   └── train.py
│   ├── temporal/                         # 时序模型训练
│   │   └── train.py
│   └── alignment/                        # 对齐模型训练
│       └── train.py
│
├── inference/               # 推理脚本（按模型类型组织）
│   ├── hierarchical_keypoint/            # 层次化关键点模型推理
│   │   └── inference.py
│   ├── hierarchical_image/               # 层次化图像模型推理
│   │   └── inference.py
│   ├── skeleton/                         # 层次化骨架模型推理
│   │   └── inference.py
│   ├── temporal/                         # 时序模型推理
│   │   └── inference.py
│   └── alignment/                        # 对齐模型推理
│       └── inference.py
│
├── config/                  # 配置文件
│   └── config.py                         # 项目配置
│
├── old/                     # 废弃的早期版本
│   ├── model.py, train.py, inference.py
│   ├── model_image.py, train_image.py, inference_image.py
│   └── model_skeleton.py, train_skeleton.py, inference_skeleton.py
│
├── checkpoints_*/           # 模型检查点
├── extracted_frames/        # 提取的视频帧
├── temporal_representations_all/  # 时序表征
├── visualizations_*/        # 可视化结果
└── README.md                # 项目说明
```

## Pipeline 流程

### 1. 数据提取阶段 (data/)
```bash
# 从视频提取帧
python data/extract_video_frames.py

# 提取关键点
python data/extract_body_keypoints.py

# 组织视频序列
python data/prepare_video_sequences.py

# 创建文本数据
python data/create_text_data_from_videos.py
```

### 2. 训练阶段 (training/)

#### 层次化关键点模型
```bash
python training/hierarchical_keypoint/train.py
```

#### 层次化图像重构模型
```bash
python training/hierarchical_image/train.py
```

#### 骨架重构模型
```bash
python training/skeleton/train.py
```

#### 时序模型
```bash
python training/temporal/train.py
```

#### 对齐模型
```bash
python training/alignment/train.py
```

### 3. 推理阶段 (inference/)

#### 提取时序表征
```bash
python inference/temporal/inference.py
```

#### 生成文本描述
```bash
python inference/alignment/inference.py
```

## 模型类型说明

1. **hierarchical_keypoint**: 层次化关键点坐标重构模型
   - 输入：关键点坐标 [batch, num_keypoints, 3]
   - 输出：重构的关键点坐标
   - 表征：全局 + 区域表征

2. **hierarchical_image**: 层次化关键点到图像重构模型
   - 输入：关键点坐标
   - 输出：重构的图像
   - 使用 CNN 解码器生成图像

3. **skeleton**: 层次化骨架重构模型（全身关键点）
   - 输入：全身关键点（面部+双手+姿态）[batch, 143, 3]
   - 输出：骨架图像
   - 用于手语视频处理

4. **temporal**: 时序 Transformer 模型
   - 输入：视频序列的关键点 [batch, seq_len, num_keypoints, 3]
   - 输出：时序表征（全局+局部）
   - 使用预训练的骨架模型作为帧编码器

5. **alignment**: 视频-语言对齐模型
   - 输入：视频时序表征
   - 输出：文本描述
   - 使用 mBART 作为语言解码器

## 注意事项

- 所有脚本已更新导入路径，支持从子目录运行
- 脚本会自动添加项目根目录到 Python 路径
- 建议从项目根目录运行脚本，或使用绝对路径
- 每个模型类型都有独立的目录，便于维护和扩展

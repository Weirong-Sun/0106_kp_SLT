# Data 目录

数据提取和预处理脚本。

## 文件说明

### 数据提取脚本

- **`extract_video_frames.py`** - 从视频文件中提取帧
  - 输入：视频文件目录
  - 输出：提取的帧图像（保存在 `extracted_frames/` 目录）
  - 用法：
    ```bash
    python data/extract_video_frames.py --dataset_path /path/to/videos --output_path extracted_frames
    ```

- **`extract_body_keypoints.py`** - 从图像中提取全身关键点（面部+双手+姿态）
  - 输入：图像文件目录或提取的帧
  - 输出：关键点数据（保存为 pickle 文件）
  - 提取内容：
    - 面部关键点：68个点
    - 左手关键点：21个点
    - 右手关键点：21个点
    - 姿态关键点：33个点
  - 用法：
    ```bash
    python data/extract_body_keypoints.py --dataset_path extracted_frames --output_path sign_language_keypoints.pkl
    ```

- **`extract_phoenix_keypoints.py`** - 专门用于处理 PHOENIX-2014-T 数据集的关键点提取
  - 输入：PHOENIX-2014-T 数据集路径（包含已提取的图像帧）
  - 输出：关键点数据（保存为 pickle 文件，保持 train/dev/test 划分）
  - 特点：
    - 自动识别数据集的 train/dev/test 划分
    - 保持数据集结构，每个关键点关联视频ID
    - 支持只处理特定划分或限制样本数（用于测试）
  - 提取内容：同 `extract_body_keypoints.py`（143个关键点）
  - 用法：
    ```bash
    # 处理所有划分
    python data/extract_phoenix_keypoints.py \
        --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
        --output_path phoenix_keypoints.pkl

    # 只处理训练集
    python data/extract_phoenix_keypoints.py \
        --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
        --output_path phoenix_keypoints_train.pkl \
        --splits train
    ```
  - 详细说明：参考 `docs/PHOENIX数据集关键点提取指南.md`

- **`extract_phoenix_keypoints_distributed.py`** - PHOENIX 数据集关键点提取（分布式版本）
  - 输入：同 `extract_phoenix_keypoints.py`
  - 输出：同 `extract_phoenix_keypoints.py`（格式完全兼容）
  - 特点：
    - 支持多进程并行处理，提高处理速度
    - 默认使用4个进程（对应4块GPU）
    - 可自定义工作进程数
    - **关键点输出与单进程版本完全一致**（见分析文档）
  - 用法：
    ```bash
    # 使用4个进程处理所有划分
    python data/extract_phoenix_keypoints_distributed.py \
        --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
        --output_path phoenix_keypoints.pkl \
        --num_workers 4

    # 使用8个进程只处理训练集
    python data/extract_phoenix_keypoints_distributed.py \
        --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
        --output_path phoenix_keypoints_train.pkl \
        --splits train \
        --num_workers 8
    ```
  - 性能：理论加速比 3-4倍（4进程）
  - 详细分析：参考 `docs/分布式关键点提取分析.md`

- **`extract_phoenix_keypoints_gpu.py`** - PHOENIX 数据集关键点提取（GPU 优化版本）
  - 输入：同 `extract_phoenix_keypoints.py`
  - 输出：同 `extract_phoenix_keypoints.py`（格式完全兼容）
  - 特点：
    - 支持多 GPU 绑定（每个进程绑定到不同的 GPU）
    - 自动整合分块处理的结果
    - 优化的批处理和进度跟踪
    - **注意**: MediaPipe 主要使用 CPU，但多进程并行能提供显著加速
  - 用法：
    ```bash
    # 使用 4 个 GPU，每个 GPU 1 个进程（推荐）
    python data/extract_phoenix_keypoints_gpu.py \
        --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
        --output_path phoenix_keypoints_full.pkl \
        --num_gpus 4 \
        --num_workers_per_gpu 1
    ```
  - 性能：理论加速比 3-4 倍（4 进程）
  - 详细说明：参考 `docs/GPU分布式关键点提取指南.md`

### 数据整合和准备脚本

- **`merge_keypoint_results.py`** - 合并多个关键点提取结果文件
  - 功能：合并分块处理的结果，去重，整合统计信息
  - 用法：
    ```bash
    python data/merge_keypoint_results.py \
        --input_files result1.pkl result2.pkl result3.pkl \
        --output_path merged_keypoints.pkl
    ```

- **`prepare_model_input.py`** - 准备模型输入数据
  - 功能：将关键点数据转换为模型可直接使用的格式
  - 输出：
    - NumPy 格式（`.npz`）- 模型输入，快速加载
    - Pickle 格式（`.pkl`）- 完整信息
    - 统计信息（`.json`）- 数据统计
  - 用法：
    ```bash
    python data/prepare_model_input.py \
        --keypoints_file phoenix_keypoints_full.pkl \
        --output_dir model_input_data
    ```
  - 输出格式：
    - `train_keypoints.npz` - 训练集 [N, 143, 3]
    - `dev_keypoints.npz` - 验证集
    - `test_keypoints.npz` - 测试集

- **`data_extraction.py`** - 早期版本：提取68个面部关键点
  - 输入：图像文件
  - 输出：面部关键点数据（`keypoints_data.pkl`）
  - 用法：
    ```bash
    python data/extract_video_frames.py --dataset_path /path/to/images --output_path keypoints_data.pkl
    ```

### 数据组织脚本

- **`prepare_video_sequences.py`** - 将关键点数据组织成视频序列
  - 输入：关键点 pickle 文件
  - 输出：视频序列数据（`video_sequences.pkl`）
  - 功能：
    - 从图像路径中提取视频ID，自动分组
    - 或根据检测间隔自动分组
    - 过滤过短的序列
  - 用法：
    ```bash
    python data/prepare_video_sequences.py --keypoints_path sign_language_keypoints.pkl --output_path video_sequences.pkl --min_seq_len 4
    ```

### 文本数据生成脚本

- **`create_text_data_from_videos.py`** - 从视频关键点数据生成文本描述
  - 输入：包含 `image_paths` 的关键点文件
  - 输出：文本数据 JSON 文件（`text_data.json`）
  - 功能：从图像路径中提取视频标题作为文本描述
  - 用法：
    ```bash
    python data/create_text_data_from_videos.py --keypoints_path sign_language_keypoints.pkl --output_path text_data.json
    ```

- **`create_text_data_example.py`** - 创建示例文本数据
  - 用于测试和示例
  - 生成简单的文本描述列表

### 测试和可视化脚本

- **`test_keypoints.py`** - 测试提取的关键点并可视化
  - 输入：关键点 pickle 文件
  - 输出：可视化图像（骨架图）
  - 功能：
    - 加载关键点数据
    - 绘制骨架图像
    - 分别显示面部、双手、姿态和完整骨架
  - 用法：
    ```bash
    python data/test_keypoints.py --data_path sign_language_keypoints.pkl --num_samples 5 --output_dir test_keypoints_visualization
    ```

## 数据格式

### 关键点数据格式 (pickle)
```python
{
    'keypoints': [
        {
            'face': np.array([68, 3]),      # 68个面部关键点，每个点有x,y,z坐标
            'left_hand': np.array([21, 3]), # 21个左手关键点
            'right_hand': np.array([21, 3]),# 21个右手关键点
            'pose': np.array([33, 3])       # 33个姿态关键点
        },
        ...
    ],
    'image_paths': [str, ...]  # 对应的图像路径（可选）
}
```

### 视频序列格式 (pickle)
```python
{
    'sequences': [
        np.array([[143, 3], ...]),  # 每个序列：多个帧，每帧143个关键点
        ...
    ],
    'metadata': {...}  # 元数据信息
}
```

### 文本数据格式 (JSON)
```json
{
    "texts": [
        "视频标题1",
        "视频标题2",
        ...
    ]
}
```

## 数据流程

1. **视频 → 帧**：`extract_video_frames.py`
2. **帧 → 关键点**：`extract_body_keypoints.py`
3. **关键点 → 序列**：`prepare_video_sequences.py`
4. **视频 → 文本**：`create_text_data_from_videos.py`

## 注意事项

- 确保安装了 MediaPipe：`pip install mediapipe`
- 关键点提取可能需要较长时间，建议批量处理
- 如果某些帧未检测到关键点，会用零填充
- 建议先运行 `test_keypoints.py` 检查数据质量


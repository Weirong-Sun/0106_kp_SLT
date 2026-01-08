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
    python data/extract_body_keypoints.py --input_path extracted_frames --output_path sign_language_keypoints.pkl
    ```

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


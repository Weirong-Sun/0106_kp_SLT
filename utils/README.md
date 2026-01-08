# Utils 目录

工具函数库，提供数据处理和可视化的辅助功能。

## 文件说明

### `utils_image.py`

用于处理面部关键点（68点）的工具函数。

**主要函数：**

- **`draw_keypoints_on_canvas(keypoints, image_size=256, ...)`**
  - 在画布上绘制关键点
  - 参数：
    - `keypoints`: 关键点数组 `[68, 3]`
    - `image_size`: 输出图像大小（默认256x256）
    - `point_radius`: 关键点半径
    - `line_thickness`: 连线粗细
    - `draw_lines`: 是否绘制连线
  - 返回：绘制好的图像 `[image_size, image_size]`

- **`generate_image_dataset(keypoints_data, output_dir, ...)`**
  - 批量生成关键点图像数据集
  - 用于生成训练图像数据
  - 支持数据增强选项

**用法示例：**
```python
from utils.utils_image import draw_keypoints_on_canvas
import numpy as np

# 绘制关键点
keypoints = np.array([[x, y, z], ...])  # [68, 3]
image = draw_keypoints_on_canvas(
    keypoints, 
    image_size=256,
    point_radius=3,
    draw_lines=True
)
```

### `utils_skeleton.py`

用于处理全身骨架关键点（143点）的工具函数。

**主要函数：**

- **`draw_face_skeleton(keypoints, canvas, ...)`**
  - 绘制面部骨架（68个点）
  - 包括面部轮廓、眉毛、眼睛、鼻子、嘴巴等

- **`draw_hand_skeleton(keypoints, canvas, ...)`**
  - 绘制手部骨架（21个点）
  - 包括手掌、手指、关节等
  - 增强手指和指尖的可见性

- **`draw_pose_skeleton(keypoints, canvas, ...)`**
  - 绘制身体姿态骨架（33个点）
  - 包括头部、肩膀、手臂、躯干、腿部等

- **`draw_full_skeleton(keypoints, image_size=256, ...)`**
  - 绘制完整骨架（143个点 = 68面部 + 21左手 + 21右手 + 33姿态）
  - 整合所有身体部位

- **`generate_skeleton_dataset(keypoints_data, output_dir, ...)`**
  - 批量生成骨架图像数据集
  - 用于训练骨架重构模型

**用法示例：**
```python
from utils.utils_skeleton import draw_full_skeleton
import numpy as np

# 绘制完整骨架
keypoints = {
    'face': np.array([68, 3]),
    'left_hand': np.array([21, 3]),
    'right_hand': np.array([21, 3]),
    'pose': np.array([33, 3])
}
image = draw_full_skeleton(keypoints, image_size=256)
```

## 关键点顺序

### 面部关键点 (68点)
- 0-16: 面部轮廓
- 17-21: 右眉毛
- 22-26: 左眉毛
- 27-35: 鼻子
- 36-41: 右眼
- 42-47: 左眼
- 48-67: 嘴巴

### 手部关键点 (21点/手)
- 0: 手腕
- 1-4: 拇指（4个关节）
- 5-8: 食指
- 9-12: 中指
- 13-16: 无名指
- 17-20: 小指

### 姿态关键点 (33点)
- 0-10: 头部和上身
- 11-16: 左手臂
- 17-22: 右手臂
- 23-32: 下身和腿部

## 注意事项

- 关键点坐标应该在 [0, 1] 范围内或已归一化
- 图像坐标系：左上角为原点 (0, 0)
- 使用 `draw_lines=False` 可以只显示关键点，不绘制连线
- 建议先测试单个样本再批量处理


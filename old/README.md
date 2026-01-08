# Old 目录

废弃的早期版本代码，保留用于参考和历史记录。

## 文件说明

此目录包含项目早期的模型和脚本版本，已被新的层次化版本替代。

### 废弃的文件

- **`model.py`** - 早期基础 Transformer 模型
  - 已被 `models/hierarchical_keypoint/model.py` 替代
  
- **`train.py`** - 早期训练脚本
  - 已被 `training/hierarchical_keypoint/train.py` 替代
  
- **`inference.py`** - 早期推理脚本
  - 已被 `inference/hierarchical_keypoint/inference.py` 替代

- **`model_image.py`** - 早期图像重构模型
  - 已被 `models/hierarchical_image/model.py` 替代
  
- **`train_image.py`** - 早期图像重构训练脚本
  - 已被 `training/hierarchical_image/train.py` 替代
  
- **`inference_image.py`** - 早期图像重构推理脚本
  - 已被 `inference/hierarchical_image/inference.py` 替代

- **`model_skeleton.py`** - 早期骨架重构模型
  - 已被 `models/skeleton/model.py` 替代
  
- **`train_skeleton.py`** - 早期骨架重构训练脚本
  - 已被 `training/skeleton/train.py` 替代
  
- **`inference_skeleton.py`** - 早期骨架重构推理脚本
  - 已被 `inference/skeleton/inference.py` 替代

## 保留原因

这些文件保留在此目录中：
1. **参考价值**：可以作为简单实现的参考
2. **历史记录**：记录项目的发展过程
3. **调试辅助**：如果新版本出现问题，可以对比旧版本

## 不建议使用

**请勿直接使用此目录中的文件进行训练或推理**，因为：
- 它们已被更好的版本替代
- 可能与当前的数据格式不兼容
- 缺少新版本的改进和优化

## 清理

如果确定不再需要，可以安全删除此目录。


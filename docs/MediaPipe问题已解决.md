# MediaPipe 问题已解决 ✓

## 问题总结

**原始问题**：
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

**环境信息**：
- Python 版本：3.11.0
- Conda 环境：0106_kp_SLT
- 原始 MediaPipe 版本：0.10.31（有问题）
- 修复后版本：0.10.9（正常）

## 问题原因

MediaPipe 0.10.31 在安装时出现了问题，导致：
- 模块可以导入
- 但缺少 `solutions` 属性
- 只有 `['Image', 'ImageFormat', 'tasks']` 属性，缺少关键的 `solutions` 模块

这可能是由于：
1. 安装包损坏
2. 版本兼容性问题
3. 安装过程中断

## 解决方案

```bash
# 1. 激活 conda 环境
conda activate 0106_kp_SLT

# 2. 卸载有问题的版本
pip uninstall mediapipe -y

# 3. 安装稳定版本
pip install --no-cache-dir mediapipe==0.10.9

# 4. 验证安装
python check_mediapipe.py
```

## 验证结果

✓ MediaPipe 0.10.9 安装成功
✓ `solutions` 模块可用
✓ 所有关键组件（face_mesh, hands, pose）正常
✓ 脚本可以正常运行

## 重要提示

1. **使用稳定版本**：推荐使用 `mediapipe==0.10.9` 而不是最新版本
2. **激活正确环境**：确保在 `0106_kp_SLT` conda 环境中运行
3. **验证安装**：使用 `python check_mediapipe.py` 或 `python diagnose_mediapipe.py` 验证

## 现在可以使用的功能

```bash
# 1. 检查 MediaPipe 状态
python check_mediapipe.py

# 2. 诊断环境
python diagnose_mediapipe.py

# 3. 提取 PHOENIX 数据集关键点
conda activate 0106_kp_SLT
python data/extract_phoenix_keypoints.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints.pkl
```

## 相关文档

- [MediaPipe安装和检查指南.md](./MediaPipe安装和检查指南.md)
- [MediaPipe环境问题排查指南.md](./MediaPipe环境问题排查指南.md)

---

**解决时间**: 2024年
**状态**: ✅ 已解决






# MediaPipe 安装和检查指南

## 问题描述

如果遇到以下错误：
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```
或
```
ModuleNotFoundError: No module named 'mediapipe'
```

这通常表示 MediaPipe 未正确安装或未安装。

## 检查 MediaPipe 是否已安装

### 方法 1: 使用 Python 命令检查

```bash
python -c "import mediapipe; print(f'MediaPipe 版本: {mediapipe.__version__}')"
```

**如果已安装**，会显示版本号，例如：
```
MediaPipe 版本: 0.10.9
```

**如果未安装**，会显示：
```
ModuleNotFoundError: No module named 'mediapipe'
```

### 方法 2: 使用 pip 检查

```bash
pip show mediapipe
```

**如果已安装**，会显示详细信息：
```
Name: mediapipe
Version: 0.10.9
Summary: MediaPipe Python Package
...
```

**如果未安装**，会显示：
```
WARNING: Package(s) not found: mediapipe
```

### 方法 3: 检查 MediaPipe 功能是否正常

```bash
python -c "import mediapipe as mp; print('MediaPipe 可用'); print(f'solutions 属性存在: {hasattr(mp, \"solutions\")}')"
```

**如果正常**，会显示：
```
MediaPipe 可用
solutions 属性存在: True
```

**如果有问题**，可能会显示：
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

## 安装 MediaPipe

### 基本安装

```bash
pip install mediapipe
```

### 安装特定版本

```bash
# 安装指定版本（推荐使用 >= 0.10.0）
pip install mediapipe>=0.10.0
```

### 从 requirements.txt 安装

如果项目中有 `requirements.txt` 文件，可以：

```bash
pip install -r requirements.txt
```

这会安装所有依赖，包括 mediapipe。

## 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "
import mediapipe as mp
print('✓ MediaPipe 导入成功')
print(f'  版本: {mp.__version__}')
print(f'  solutions 模块: {hasattr(mp, \"solutions\")}')

# 测试关键组件
try:
    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    print('✓ 所有关键组件可用')
except AttributeError as e:
    print(f'✗ 组件检查失败: {e}')
"
```

**成功输出示例**：
```
✓ MediaPipe 导入成功
  版本: 0.10.9
  solutions 模块: True
✓ 所有关键组件可用
```

## 常见问题解决

### 问题 1: 安装失败（编译错误）

**错误信息**：
```
error: subprocess-exited-with-error
× Building wheel for mediapipe (pyproject.toml) did not run successfully.
```

**解决方案**：

1. **更新 pip 和构建工具**：
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **安装系统依赖**（Linux）：
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev build-essential
   ```

3. **使用预编译的 wheel**：
   ```bash
   pip install --only-binary :all: mediapipe
   ```

### 问题 2: 版本不兼容

**错误信息**：
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

**可能原因**：
- MediaPipe 版本过旧
- 安装不完整

**解决方案**：

1. **卸载并重新安装**：
   ```bash
   pip uninstall mediapipe
   pip install mediapipe>=0.10.0
   ```

2. **检查 Python 版本兼容性**：
   ```bash
   python --version
   ```
   MediaPipe 需要 Python 3.8-3.11（某些版本可能支持 3.12）

### 问题 3: 虚拟环境问题

如果使用虚拟环境，确保：

1. **激活虚拟环境**：
   ```bash
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   ```

2. **在正确的环境中安装**：
   ```bash
   which python  # 检查当前使用的 Python
   pip install mediapipe
   ```

### 问题 4: 权限问题

如果遇到权限错误：

```bash
# 使用用户安装（推荐）
pip install --user mediapipe

# 或使用虚拟环境
python -m venv venv
source venv/bin/activate
pip install mediapipe
```

## 完整检查脚本

创建一个检查脚本 `check_mediapipe.py`：

```python
#!/usr/bin/env python
"""检查 MediaPipe 安装状态"""

def check_mediapipe():
    print("="*60)
    print("MediaPipe 安装检查")
    print("="*60)

    # 检查是否可导入
    try:
        import mediapipe as mp
        print("✓ MediaPipe 已安装")
        print(f"  版本: {mp.__version__}")
        print(f"  安装路径: {mp.__file__}")
    except ImportError:
        print("✗ MediaPipe 未安装")
        print("  请运行: pip install mediapipe")
        return False

    # 检查 solutions 模块
    try:
        has_solutions = hasattr(mp, 'solutions')
        print(f"✓ solutions 模块: {has_solutions}")

        if has_solutions:
            # 检查关键组件
            components = {
                'face_mesh': mp.solutions.face_mesh,
                'hands': mp.solutions.hands,
                'pose': mp.solutions.pose
            }
            print("✓ 关键组件检查:")
            for name, component in components.items():
                print(f"    - {name}: ✓")

            print("\n✓ MediaPipe 安装正常，可以使用！")
            return True
        else:
            print("✗ solutions 模块不可用，可能需要重新安装")
            return False

    except AttributeError as e:
        print(f"✗ 属性错误: {e}")
        print("  建议重新安装: pip uninstall mediapipe && pip install mediapipe")
        return False
    except Exception as e:
        print(f"✗ 检查过程中出错: {e}")
        return False

if __name__ == "__main__":
    success = check_mediapipe()
    exit(0 if success else 1)
```

运行检查脚本：
```bash
python check_mediapipe.py
```

## 快速检查命令总结

```bash
# 1. 检查是否安装
python -c "import mediapipe; print('已安装')" 2>/dev/null || echo "未安装"

# 2. 检查版本
python -c "import mediapipe; print(mediapipe.__version__)" 2>/dev/null || echo "未安装"

# 3. 检查功能
python -c "import mediapipe as mp; print('solutions:', hasattr(mp, 'solutions'))" 2>/dev/null || echo "未安装"

# 4. 使用 pip 检查
pip show mediapipe 2>/dev/null || echo "未安装"
```

## 安装后测试

安装完成后，可以运行一个简单的测试：

```python
import mediapipe as mp
import cv2
import numpy as np

# 创建测试图像
test_image = np.zeros((480, 640, 3), dtype=np.uint8)

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

# 处理图像
results = hands.process(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
print("MediaPipe 测试成功！")
```

## 相关资源

- [MediaPipe 官方文档](https://google.github.io/mediapipe/)
- [MediaPipe Python API](https://google.github.io/mediapipe/solutions/solutions.html)
- [项目 requirements.txt](../requirements.txt)

---

**文档创建时间**: 2024年
**适用版本**: MediaPipe >= 0.10.0
**Python 版本**: 3.8-3.11 (某些版本支持 3.12)






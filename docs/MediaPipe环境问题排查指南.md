# MediaPipe 环境问题排查指南

## 问题描述

即使安装了 MediaPipe 0.10.31，仍然遇到：
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

## 已解决的问题案例

### 案例：MediaPipe 0.10.31 缺少 solutions 模块

**症状**：
- Python 3.11.0 ✓
- MediaPipe 0.10.31 已安装 ✓
- 但 `hasattr(mp, 'solutions')` 返回 `False` ✗
- 模块内容只有 `['Image', 'ImageFormat', 'tasks']`，缺少 `solutions`

**原因**：
MediaPipe 0.10.31 在某些情况下安装不完整，可能是：
- 安装包损坏
- 版本兼容性问题
- 安装过程中断

**解决方案**：
```bash
# 1. 激活正确的 conda 环境
conda activate 0106_kp_SLT

# 2. 卸载有问题的版本
pip uninstall mediapipe -y

# 3. 安装稳定版本（推荐 0.10.9）
pip install --no-cache-dir mediapipe==0.10.9

# 4. 验证
python check_mediapipe.py
```

**结果**：MediaPipe 0.10.9 工作正常，所有组件可用。

## 常见原因

### 1. Python 版本不匹配

**问题**：MediaPipe 安装在了不同的 Python 环境中，但运行时使用了另一个环境。

**检查方法**：
```bash
# 检查当前 Python 版本和路径
python --version
python -c "import sys; print(sys.executable)"

# 检查 MediaPipe 安装位置
pip show mediapipe
python -c "import mediapipe; print(mediapipe.__file__)"
```

**解决方案**：
- 确保在正确的 Python 环境中安装和运行
- 如果使用 conda，确保激活了正确的环境

### 2. Conda 环境问题

**问题**：有多个 conda 环境，MediaPipe 安装在了错误的环境中。

**检查方法**：
```bash
# 列出所有 conda 环境
conda env list

# 检查当前激活的环境
conda info --envs
echo $CONDA_DEFAULT_ENV

# 检查每个环境中的 MediaPipe
conda activate <环境名>
python -c "import mediapipe; print('OK')"
```

**解决方案**：
```bash
# 激活正确的环境
conda activate 0106_kp_SLT  # 或你的环境名

# 在正确的环境中安装
pip install mediapipe>=0.10.0

# 验证安装
python -c "import mediapipe as mp; print(hasattr(mp, 'solutions'))"
```

### 3. MediaPipe 安装不完整

**问题**：安装过程中出错，导致安装不完整。

**检查方法**：
```bash
# 检查 MediaPipe 的属性和方法
python -c "
import mediapipe as mp
print('版本:', mp.__version__)
print('路径:', mp.__file__)
print('属性:', [x for x in dir(mp) if not x.startswith('_')])
print('solutions 存在:', hasattr(mp, 'solutions'))
"
```

**解决方案**：
```bash
# 完全卸载并重新安装
pip uninstall mediapipe -y
pip cache purge

# 推荐安装稳定版本 0.10.9（而不是最新版本）
pip install --no-cache-dir mediapipe==0.10.9

# 验证
python check_mediapipe.py
# 或
python -c "import mediapipe as mp; print(hasattr(mp, 'solutions'))"
```

**注意**：如果 0.10.31 有问题，可以尝试：
- `mediapipe==0.10.9`（推荐，稳定）
- `mediapipe==0.10.8`
- `mediapipe==0.10.7`

### 4. 命名冲突

**问题**：当前目录下有名为 `mediapipe.py` 的文件，导致导入冲突。

**检查方法**：
```bash
# 检查是否有 mediapipe.py 文件
find . -name "mediapipe.py" -type f

# 检查 Python 导入路径
python -c "import mediapipe; print(mediapipe.__file__)"
```

**解决方案**：
- 重命名或删除冲突的 `mediapipe.py` 文件
- 确保项目目录不在 Python 路径中（如果不需要）

### 5. Python 版本兼容性

**问题**：MediaPipe 0.10.31 可能不完全支持 Python 3.12。

**检查方法**：
```bash
python --version
python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}')"
```

**解决方案**：
- 使用 Python 3.8-3.11（推荐 3.10 或 3.11）
- 如果必须使用 3.12，尝试安装最新版本的 MediaPipe

## 完整排查步骤

### 步骤 1: 检查当前环境

```bash
# 1. 检查 Python 版本
python --version

# 2. 检查 Python 路径
python -c "import sys; print('Python 路径:', sys.executable)"

# 3. 检查是否在 conda 环境中
conda info --envs
echo $CONDA_DEFAULT_ENV
```

### 步骤 2: 检查 MediaPipe 安装

```bash
# 1. 检查是否安装
pip show mediapipe

# 2. 尝试导入
python -c "import mediapipe; print('导入成功')"

# 3. 检查版本和属性
python -c "
import mediapipe as mp
print('版本:', mp.__version__)
print('路径:', mp.__file__)
print('solutions 存在:', hasattr(mp, 'solutions'))
"
```

### 步骤 3: 如果使用 Conda

```bash
# 1. 列出所有环境
conda env list

# 2. 激活正确的环境
conda activate 0106_kp_SLT  # 替换为你的环境名

# 3. 确认 Python 版本
python --version

# 4. 安装 MediaPipe
pip install mediapipe>=0.10.0

# 5. 验证
python -c "import mediapipe as mp; print(hasattr(mp, 'solutions'))"
```

### 步骤 4: 重新安装 MediaPipe

```bash
# 1. 完全卸载
pip uninstall mediapipe -y

# 2. 清理缓存
pip cache purge

# 3. 重新安装（推荐使用稳定版本）
pip install --no-cache-dir mediapipe==0.10.9
# 如果 0.10.9 不可用，尝试：
# pip install --no-cache-dir mediapipe>=0.10.0

# 4. 验证
python check_mediapipe.py
```

### 步骤 5: 检查命名冲突

```bash
# 检查是否有冲突文件
find . -name "mediapipe.py" -o -name "mediapipe.pyc"

# 检查 Python 路径
python -c "import sys; print('\n'.join(sys.path))"
```

## 针对当前项目的解决方案

### 方案 1: 使用 Conda 环境（推荐）

```bash
# 1. 激活项目环境
conda activate 0106_kp_SLT

# 2. 检查 Python 版本（应该是 3.11）
python --version

# 3. 安装 MediaPipe（推荐使用稳定版本）
pip install --no-cache-dir mediapipe==0.10.9
# 如果遇到问题，可以尝试其他版本：
# pip install --no-cache-dir mediapipe==0.10.8

# 4. 验证
python check_mediapipe.py

# 5. 运行脚本
python data/extract_phoenix_keypoints.py --help
```

### 方案 2: 创建新的虚拟环境

```bash
# 1. 创建 Python 3.11 虚拟环境
python3.11 -m venv venv_phoenix

# 2. 激活环境
source venv_phoenix/bin/activate  # Linux/Mac
# 或
venv_phoenix\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证
python check_mediapipe.py
```

### 方案 3: 使用系统 Python 3.11

```bash
# 1. 找到 Python 3.11
which python3.11
# 或
/usr/bin/python3.11 --version

# 2. 使用 Python 3.11 安装
python3.11 -m pip install mediapipe>=0.10.0

# 3. 使用 Python 3.11 运行
python3.11 data/extract_phoenix_keypoints.py --help
```

## 诊断脚本

创建一个诊断脚本 `diagnose_mediapipe.py`：

```python
#!/usr/bin/env python
"""MediaPipe 环境诊断脚本"""
import sys
import os

print("="*60)
print("MediaPipe 环境诊断")
print("="*60)

# 1. Python 信息
print("\n1. Python 环境信息:")
print(f"   版本: {sys.version}")
print(f"   可执行文件: {sys.executable}")
print(f"   路径: {sys.path[:3]}...")

# 2. 检查 conda
print("\n2. Conda 环境:")
conda_env = os.environ.get('CONDA_DEFAULT_ENV', '未激活')
print(f"   当前环境: {conda_env}")
conda_prefix = os.environ.get('CONDA_PREFIX', '未设置')
print(f"   Conda 前缀: {conda_prefix}")

# 3. 检查 MediaPipe
print("\n3. MediaPipe 检查:")
try:
    import mediapipe as mp
    print(f"   ✓ 已安装")
    try:
        print(f"   版本: {mp.__version__}")
    except:
        print("   版本: 未知")
    print(f"   路径: {mp.__file__}")
    print(f"   solutions 存在: {hasattr(mp, 'solutions')}")

    if hasattr(mp, 'solutions'):
        print("   ✓ solutions 模块可用")
        try:
            mp.solutions.face_mesh
            mp.solutions.hands
            mp.solutions.pose
            print("   ✓ 所有关键组件可用")
        except Exception as e:
            print(f"   ✗ 组件检查失败: {e}")
    else:
        print("   ✗ solutions 模块不可用")
        print("   建议: pip uninstall mediapipe && pip install mediapipe>=0.10.0")

except ImportError as e:
    print(f"   ✗ 未安装: {e}")
    print("   建议: pip install mediapipe>=0.10.0")
except Exception as e:
    print(f"   ✗ 错误: {e}")

# 4. 检查命名冲突
print("\n4. 命名冲突检查:")
current_dir = os.getcwd()
mediapipe_files = []
for root, dirs, files in os.walk(current_dir):
    if 'mediapipe.py' in files:
        mediapipe_files.append(os.path.join(root, 'mediapipe.py'))
    if '__pycache__' in dirs:
        if 'mediapipe.cpython' in os.listdir(os.path.join(root, '__pycache__')):
            mediapipe_files.append(os.path.join(root, '__pycache__', 'mediapipe.cpython*'))

if mediapipe_files:
    print(f"   ⚠ 发现可能的冲突文件:")
    for f in mediapipe_files[:5]:
        print(f"      {f}")
    print("   建议: 重命名或删除这些文件")
else:
    print("   ✓ 未发现命名冲突")

print("\n" + "="*60)
```

运行诊断：
```bash
python diagnose_mediapipe.py
```

## 快速修复命令

如果确认是环境问题，使用以下命令快速修复：

```bash
# 1. 激活正确的 conda 环境
conda activate 0106_kp_SLT

# 2. 卸载并重新安装（使用稳定版本）
pip uninstall mediapipe -y
pip install --no-cache-dir mediapipe==0.10.9

# 3. 验证
python check_mediapipe.py

# 4. 测试
python data/extract_phoenix_keypoints.py --help
```

## 相关文档

- [MediaPipe安装和检查指南.md](./MediaPipe安装和检查指南.md)
- [项目 README.md](../README.md)

---

**文档创建时间**: 2024年
**适用场景**: MediaPipe 环境配置问题


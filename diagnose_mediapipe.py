#!/usr/bin/env python
"""
MediaPipe 环境诊断脚本
用于诊断 MediaPipe 安装和环境问题
"""
import sys
import os
from pathlib import Path

def diagnose_mediapipe():
    """诊断 MediaPipe 环境"""
    print("="*60)
    print("MediaPipe 环境诊断")
    print("="*60)

    # 1. Python 信息
    print("\n1. Python 环境信息:")
    print(f"   版本: {sys.version.split()[0]}")
    print(f"   完整版本: {sys.version}")
    print(f"   可执行文件: {sys.executable}")
    print(f"   Python 路径 (前3个):")
    for path in sys.path[:3]:
        print(f"      - {path}")

    # 2. 检查 conda
    print("\n2. Conda 环境:")
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '未激活')
    print(f"   当前环境: {conda_env}")
    conda_prefix = os.environ.get('CONDA_PREFIX', '未设置')
    if conda_prefix != '未设置':
        print(f"   Conda 前缀: {conda_prefix}")

    # 3. 检查 MediaPipe
    print("\n3. MediaPipe 检查:")
    try:
        import mediapipe as mp
        print(f"   ✓ 已安装")
        try:
            version = mp.__version__
            print(f"   版本: {version}")
        except AttributeError:
            print("   版本: 未知（无法获取版本信息）")

        try:
            file_path = mp.__file__
            print(f"   路径: {file_path}")
        except AttributeError:
            print("   路径: 未知")

        # 检查 solutions
        has_solutions = hasattr(mp, 'solutions')
        print(f"   solutions 存在: {has_solutions}")

        if has_solutions:
            print("   ✓ solutions 模块可用")
            # 检查关键组件
            components_ok = True
            try:
                mp.solutions.face_mesh
                print("      ✓ face_mesh")
            except Exception as e:
                print(f"      ✗ face_mesh: {e}")
                components_ok = False

            try:
                mp.solutions.hands
                print("      ✓ hands")
            except Exception as e:
                print(f"      ✗ hands: {e}")
                components_ok = False

            try:
                mp.solutions.pose
                print("      ✓ pose")
            except Exception as e:
                print(f"      ✗ pose: {e}")
                components_ok = False

            if components_ok:
                print("\n   ✓ 所有关键组件可用，MediaPipe 工作正常！")
                return True
            else:
                print("\n   ✗ 部分组件不可用，建议重新安装")
                return False
        else:
            print("   ✗ solutions 模块不可用")
            print("\n   可能原因:")
            print("     1. MediaPipe 版本过旧")
            print("     2. 安装不完整")
            print("     3. Python 版本不兼容")
            print("\n   建议:")
            print("     pip uninstall mediapipe -y")
            print("     pip install --no-cache-dir mediapipe>=0.10.0")
            return False

    except ImportError as e:
        print(f"   ✗ 未安装: {e}")
        print("\n   建议:")
        print("     pip install mediapipe>=0.10.0")
        return False
    except Exception as e:
        print(f"   ✗ 错误: {type(e).__name__}: {e}")
        print("\n   建议:")
        print("     pip uninstall mediapipe -y")
        print("     pip install --no-cache-dir mediapipe>=0.10.0")
        return False

    # 4. 检查命名冲突
    print("\n4. 命名冲突检查:")
    current_dir = Path.cwd()
    mediapipe_files = []

    # 检查当前目录
    if (current_dir / "mediapipe.py").exists():
        mediapipe_files.append(str(current_dir / "mediapipe.py"))

    # 检查项目目录
    project_root = current_dir
    for py_file in project_root.rglob("mediapipe.py"):
        if py_file.is_file():
            mediapipe_files.append(str(py_file))

    if mediapipe_files:
        print(f"   ⚠ 发现可能的冲突文件:")
        for f in mediapipe_files[:5]:
            print(f"      {f}")
        print("\n   建议: 重命名或删除这些文件，避免与 MediaPipe 模块冲突")
        return False
    else:
        print("   ✓ 未发现命名冲突")

    return True

def main():
    """主函数"""
    success = diagnose_mediapipe()

    print("\n" + "="*60)
    if success:
        print("诊断结果: ✓ MediaPipe 环境正常")
    else:
        print("诊断结果: ✗ MediaPipe 环境有问题")
        print("\n修复建议:")
        print("1. 如果使用 conda，确保激活了正确的环境:")
        print("   conda activate 0106_kp_SLT")
        print("\n2. 完全重新安装 MediaPipe:")
        print("   pip uninstall mediapipe -y")
        print("   pip install --no-cache-dir mediapipe>=0.10.0")
        print("\n3. 验证安装:")
        print("   python check_mediapipe.py")
        print("\n4. 查看详细文档:")
        print("   docs/MediaPipe环境问题排查指南.md")
    print("="*60)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())






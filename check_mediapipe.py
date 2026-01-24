#!/usr/bin/env python
"""
检查 MediaPipe 安装状态
用于诊断 MediaPipe 相关问题
"""
import sys

def check_mediapipe():
    """检查 MediaPipe 安装状态"""
    print("="*60)
    print("MediaPipe 安装检查")
    print("="*60)

    # 检查是否可导入
    try:
        import mediapipe as mp
        print("✓ MediaPipe 已安装")
        try:
            version = mp.__version__
            print(f"  版本: {version}")
        except AttributeError:
            print("  版本: 未知（无法获取版本信息）")

        try:
            file_path = mp.__file__
            print(f"  安装路径: {file_path}")
        except AttributeError:
            print("  安装路径: 未知")

    except ImportError as e:
        print("✗ MediaPipe 未安装")
        print(f"  错误: {e}")
        print("\n解决方案:")
        print("  运行: pip install mediapipe")
        print("  或: pip install mediapipe>=0.10.0")
        return False

    # 检查 solutions 模块
    print("\n检查 solutions 模块...")
    try:
        has_solutions = hasattr(mp, 'solutions')
        print(f"  solutions 属性存在: {has_solutions}")

        if not has_solutions:
            print("\n✗ solutions 模块不可用")
            print("  这可能是由于:")
            print("    1. MediaPipe 版本过旧")
            print("    2. 安装不完整")
            print("\n解决方案:")
            print("  pip uninstall mediapipe")
            print("  pip install mediapipe>=0.10.0")
            return False

        # 检查关键组件
        print("\n检查关键组件...")
        components_status = {}

        try:
            mp.solutions.face_mesh
            components_status['face_mesh'] = True
            print("  ✓ face_mesh: 可用")
        except AttributeError:
            components_status['face_mesh'] = False
            print("  ✗ face_mesh: 不可用")

        try:
            mp.solutions.hands
            components_status['hands'] = True
            print("  ✓ hands: 可用")
        except AttributeError:
            components_status['hands'] = False
            print("  ✗ hands: 不可用")

        try:
            mp.solutions.pose
            components_status['pose'] = True
            print("  ✓ pose: 可用")
        except AttributeError:
            components_status['pose'] = False
            print("  ✗ pose: 不可用")

        # 检查是否有失败的组件
        if not all(components_status.values()):
            print("\n✗ 部分组件不可用，建议重新安装 MediaPipe")
            return False

        print("\n" + "="*60)
        print("✓ MediaPipe 安装正常，所有组件可用！")
        print("="*60)
        return True

    except AttributeError as e:
        print(f"\n✗ 属性错误: {e}")
        print("\n解决方案:")
        print("  pip uninstall mediapipe")
        print("  pip install mediapipe>=0.10.0")
        return False
    except Exception as e:
        print(f"\n✗ 检查过程中出错: {e}")
        print(f"  错误类型: {type(e).__name__}")
        return False

def main():
    """主函数"""
    success = check_mediapipe()

    if not success:
        print("\n" + "="*60)
        print("安装建议")
        print("="*60)
        print("1. 更新 pip:")
        print("   pip install --upgrade pip")
        print("\n2. 安装 MediaPipe:")
        print("   pip install mediapipe>=0.10.0")
        print("\n3. 如果仍有问题，检查 Python 版本:")
        print("   python --version")
        print("   (MediaPipe 通常需要 Python 3.8-3.11)")
        print("\n4. 查看详细文档:")
        print("   docs/MediaPipe安装和检查指南.md")
        print("="*60)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())






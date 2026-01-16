#!/usr/bin/env python
"""
查看 phoenix_keypoints 关键点提取的进度
检查 .train, .dev, .test 文件的提取情况
"""
import pickle
import sys
from pathlib import Path
from datetime import datetime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch module not available, some information may be limited", file=sys.stderr)


def check_file_progress(file_path, split_name):
    """
    检查单个文件的提取进度

    Args:
        file_path: 文件路径
        split_name: 划分名称 (train/dev/test)

    Returns:
        progress_info: 字典，包含进度信息
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {
            'exists': False,
            'file_size': 0,
            'num_videos': 0,
            'total_frames': 0,
            'last_modified': None
        }

    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
    last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        num_videos = len(data)
        total_frames = 0

        # 计算总帧数
        for key, value in data.items():
            if isinstance(value, dict) and 'num_frames' in value:
                total_frames += value['num_frames']
            elif isinstance(value, dict) and 'keypoint' in value:
                kp = value['keypoint']
                if hasattr(kp, 'shape') and len(kp.shape) >= 1:
                    total_frames += kp.shape[0]

        return {
            'exists': True,
            'file_size': file_size,
            'num_videos': num_videos,
            'total_frames': total_frames,
            'last_modified': last_modified
        }

    except Exception as e:
        return {
            'exists': True,
            'file_size': file_size,
            'num_videos': 0,
            'total_frames': 0,
            'last_modified': last_modified,
            'error': str(e)
        }


def check_running_processes():
    """检查是否有正在运行的提取进程"""
    import subprocess
    import os

    processes = []

    # 检查提取脚本进程
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )

        lines = result.stdout.split('\n')
        for line in lines:
            if 'extract_phoenix_keypoints_distributed.py' in line or 'extract_phoenix_keypoints' in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    cmd = ' '.join(parts[10:])
                    processes.append({'pid': pid, 'command': cmd})

    except Exception as e:
        pass

    # 检查 PID 文件
    pid_file = Path('extract_keypoints.pid')
    if pid_file.exists():
        try:
            pid = pid_file.read_text().strip()
            # 检查进程是否还在运行
            try:
                os.kill(int(pid), 0)  # Signal 0 只检查进程是否存在
                processes.append({'pid': pid, 'source': 'PID file', 'status': 'running'})
            except ProcessLookupError:
                processes.append({'pid': pid, 'source': 'PID file', 'status': 'not running'})
            except Exception:
                pass
        except Exception:
            pass

    return processes


def check_log_files():
    """检查日志文件"""
    log_files = []
    log_dir = Path('.')

    for log_file in log_dir.glob('extract_keypoints_*.log'):
        file_size = log_file.stat().st_size / (1024 * 1024)  # MB
        last_modified = datetime.fromtimestamp(log_file.stat().st_mtime)

        # 读取最后几行
        last_lines = []
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                last_lines = lines[-10:] if len(lines) > 10 else lines
        except Exception:
            pass

        log_files.append({
            'path': str(log_file),
            'size': file_size,
            'last_modified': last_modified,
            'last_lines': ''.join(last_lines)
        })

    return sorted(log_files, key=lambda x: x['last_modified'], reverse=True)


def get_dataset_size(dataset_path):
    """获取原始数据集的视频数量"""
    dataset_path = Path(dataset_path)
    frames_dir = dataset_path / "features" / "fullFrame-210x260px"

    splits_info = {}

    for split in ['train', 'dev', 'test']:
        split_dir = frames_dir / split
        if split_dir.exists():
            video_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            splits_info[split] = len(video_dirs)
        else:
            splits_info[split] = 0

    return splits_info


def main():
    import argparse

    parser = argparse.ArgumentParser(description='查看 phoenix_keypoints 关键点提取的进度')
    parser.add_argument(
        '--base_path',
        type=str,
        default='phoenix_keypoints',
        help='输出文件基础路径（默认: phoenix_keypoints）'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T',
        help='数据集路径（用于对比完整数据集大小）'
    )

    args = parser.parse_args()

    print("="*70)
    print("PHOENIX 关键点提取进度检查")
    print("="*70)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 检查文件
    splits = ['train', 'dev', 'test']
    file_info = {}

    for split in splits:
        file_path = Path(f"{args.base_path}.{split}")
        file_info[split] = check_file_progress(file_path, split)

    # 显示文件状态
    print("="*70)
    print("文件状态")
    print("="*70)

    for split in splits:
        info = file_info[split]
        print(f"\n{split.upper()}:")

        if not info['exists']:
            print(f"  文件: {args.base_path}.{split} - ❌ 不存在")
        else:
            print(f"  文件: {args.base_path}.{split} - ✅ 存在")
            print(f"  文件大小: {info['file_size']:.2f} MB")
            print(f"  视频数量: {info['num_videos']}")
            print(f"  总帧数: {info['total_frames']}")
            print(f"  最后修改: {info['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}")

            if 'error' in info:
                print(f"  错误: {info['error']}")

    # 对比完整数据集
    print("\n" + "="*70)
    print("与完整数据集对比")
    print("="*70)

    try:
        dataset_size = get_dataset_size(args.dataset_path)

        for split in splits:
            extracted = file_info[split]['num_videos']
            total = dataset_size.get(split, 0)

            print(f"\n{split.upper()}:")
            print(f"  提取进度: {extracted} / {total} 视频")
            if total > 0:
                progress_pct = (extracted / total) * 100
                print(f"  完成度: {progress_pct:.1f}%")

                # 进度条
                bar_length = 30
                filled = int(bar_length * extracted / total)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"  进度: [{bar}] {progress_pct:.1f}%")
            else:
                print(f"  完成度: N/A")
    except Exception as e:
        print(f"无法获取完整数据集信息: {e}")

    # 检查运行进程
    print("\n" + "="*70)
    print("运行进程检查")
    print("="*70)

    processes = check_running_processes()
    if processes:
        print("\n找到运行中的提取进程:")
        for proc in processes:
            print(f"  PID: {proc.get('pid', 'N/A')}")
            if 'command' in proc:
                print(f"  命令: {proc['command'][:80]}...")
            if 'status' in proc:
                print(f"  状态: {proc['status']}")
    else:
        print("\n未找到运行中的提取进程")

    # 检查日志文件
    print("\n" + "="*70)
    print("日志文件")
    print("="*70)

    log_files = check_log_files()
    if log_files:
        print(f"\n找到 {len(log_files)} 个日志文件:")
        for log in log_files[:5]:  # 只显示最近5个
            print(f"\n  {log['path']}:")
            print(f"    大小: {log['size']:.2f} MB")
            print(f"    最后修改: {log['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}")

            if log['last_lines']:
                print(f"    最后几行:")
                for line in log['last_lines'].strip().split('\n')[-3:]:
                    if line.strip():
                        print(f"      {line[:80]}")
    else:
        print("\n未找到日志文件")

    # 总体统计
    print("\n" + "="*70)
    print("总体统计")
    print("="*70)

    total_videos = sum(info['num_videos'] for info in file_info.values())
    total_frames = sum(info['total_frames'] for info in file_info.values())
    total_size = sum(info['file_size'] for info in file_info.values())

    print(f"总视频数: {total_videos}")
    print(f"总帧数: {total_frames}")
    print(f"总文件大小: {total_size:.2f} MB")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()




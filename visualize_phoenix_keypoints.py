#!/usr/bin/env python
"""
可视化 PHOENIX 数据集关键点
将关键点绘制到画布上并保存或显示
"""
import pickle
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.utils_skeleton import (
    draw_full_skeleton,
    draw_face_skeleton,
    draw_hand_skeleton,
    draw_pose_skeleton
)

def visualize_phoenix_keypoints(
    pkl_path,
    output_dir="phoenix_keypoints_visualization",
    splits=None,
    num_samples=5,
    image_size=512,
    show_individual=True,
    show_combined=True,
    show_grid=True
):
    """
    可视化 PHOENIX 数据集关键点

    Args:
        pkl_path: PHOENIX 关键点 pickle 文件路径
        output_dir: 输出目录
        splits: 要可视化的划分列表，例如 ['train', 'dev', 'test']
        num_samples: 每个划分可视化的样本数
        image_size: 图像大小
        show_individual: 是否保存单独的组件图像
        show_combined: 是否保存组合图像
        show_grid: 是否保存网格图像
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f"错误: 文件不存在: {pkl_path}")
        return

    # 加载数据
    print("="*60)
    print("加载关键点数据...")
    print("="*60)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 确定要处理的划分
    if splits is None:
        splits = ['train', 'dev', 'test']

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n输出目录: {output_dir}")
    print(f"处理的划分: {splits}")
    print(f"每个划分样本数: {num_samples}")
    print(f"图像大小: {image_size}x{image_size}")

    # 处理每个划分
    all_samples = []

    for split in splits:
        if split not in data:
            print(f"\n警告: 划分 '{split}' 不存在，跳过")
            continue

        split_data = data[split]
        keypoints_list = split_data.get('keypoints', [])
        video_ids = split_data.get('video_ids', [])
        image_paths = split_data.get('image_paths', [])

        if len(keypoints_list) == 0:
            print(f"\n警告: 划分 '{split}' 没有数据，跳过")
            continue

        print(f"\n{'='*60}")
        print(f"处理 {split.upper()} 划分")
        print(f"{'='*60}")
        print(f"总样本数: {len(keypoints_list)}")

        # 限制样本数
        num_samples_to_show = min(num_samples, len(keypoints_list))
        print(f"可视化样本数: {num_samples_to_show}")

        # 创建划分输出目录
        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # 可视化每个样本
        for idx in range(num_samples_to_show):
            print(f"\n处理样本 {idx+1}/{num_samples_to_show}")
            kp_dict = keypoints_list[idx]

            # 获取元数据
            video_id = video_ids[idx] if idx < len(video_ids) else f"unknown_{idx}"
            image_path = image_paths[idx] if idx < len(image_paths) else "unknown"

            print(f"  视频ID: {video_id}")
            print(f"  图像路径: {os.path.basename(image_path)}")

            # 检查关键点可用性
            has_face = kp_dict.get('face') is not None
            has_left_hand = kp_dict.get('left_hand') is not None
            has_right_hand = kp_dict.get('right_hand') is not None
            has_pose = kp_dict.get('pose') is not None

            print(f"  关键点: 面部={has_face}, 左手={has_left_hand}, 右手={has_right_hand}, 姿态={has_pose}")

            # 绘制完整骨架
            full_skeleton = draw_full_skeleton(
                kp_dict,
                image_size=image_size,
                point_radius=3,
                line_thickness=2
            )

            # 保存完整骨架
            full_path = split_output_dir / f"sample_{idx:03d}_full_skeleton.png"
            cv2.imwrite(str(full_path), full_skeleton)
            print(f"  ✓ 保存: {full_path.name}")

            # 保存单独的组件
            if show_individual:
                components = {}

                # 面部
                face_canvas = np.ones((image_size, image_size), dtype=np.uint8) * 255
                if has_face:
                    draw_face_skeleton(face_canvas, kp_dict['face'], image_size,
                                     point_radius=3, line_thickness=2)
                face_path = split_output_dir / f"sample_{idx:03d}_face.png"
                cv2.imwrite(str(face_path), face_canvas)
                components['face'] = face_canvas

                # 双手
                hands_canvas = np.ones((image_size, image_size), dtype=np.uint8) * 255
                if has_left_hand:
                    draw_hand_skeleton(hands_canvas, kp_dict['left_hand'], image_size,
                                     point_radius=3, line_thickness=2)
                if has_right_hand:
                    draw_hand_skeleton(hands_canvas, kp_dict['right_hand'], image_size,
                                     point_radius=3, line_thickness=2)
                hands_path = split_output_dir / f"sample_{idx:03d}_hands.png"
                cv2.imwrite(str(hands_path), hands_canvas)
                components['hands'] = hands_canvas

                # 姿态
                pose_canvas = np.ones((image_size, image_size), dtype=np.uint8) * 255
                if has_pose:
                    draw_pose_skeleton(pose_canvas, kp_dict['pose'], image_size,
                                     point_radius=4, line_thickness=3)
                pose_path = split_output_dir / f"sample_{idx:03d}_pose.png"
                cv2.imwrite(str(pose_path), pose_canvas)
                components['pose'] = pose_canvas

                print(f"  ✓ 保存组件图像")

            # 创建组合图像
            if show_combined:
                # 组合：顶部是完整骨架，底部是各个组件
                if show_individual:
                    bottom_row = np.hstack([
                        components['face'],
                        components['hands'],
                        components['pose']
                    ])
                    bottom_width = bottom_row.shape[1]
                    top_row_resized = cv2.resize(full_skeleton, (bottom_width, image_size))
                    combined = np.vstack([top_row_resized, bottom_row])
                else:
                    combined = full_skeleton

                # 添加标签
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                color = 0  # 黑色

                cv2.putText(combined, f'{split.upper()} - Sample {idx+1}',
                          (10, 30), font, font_scale, color, thickness)
                cv2.putText(combined, f'Video: {video_id[:30]}',
                          (10, 60), font, font_scale*0.7, color, thickness)

                if show_individual:
                    cv2.putText(combined, 'Full Skeleton',
                              (10, image_size + 30), font, font_scale, color, thickness)
                    cv2.putText(combined, 'Face',
                              (10, image_size + 60), font, font_scale, color, thickness)
                    cv2.putText(combined, 'Hands',
                              (image_size + 10, image_size + 60), font, font_scale, color, thickness)
                    cv2.putText(combined, 'Pose',
                              (image_size * 2 + 10, image_size + 60), font, font_scale, color, thickness)

                combined_path = split_output_dir / f"sample_{idx:03d}_combined.png"
                cv2.imwrite(str(combined_path), combined)
                print(f"  ✓ 保存: {combined_path.name}")

            # 保存样本信息
            all_samples.append({
                'split': split,
                'idx': idx,
                'video_id': video_id,
                'image_path': image_path,
                'keypoints': kp_dict,
                'skeleton': full_skeleton
            })

        # 创建网格图像
        if show_grid and num_samples_to_show > 0:
            print(f"\n创建 {split.upper()} 划分的网格图像...")
            cols = min(3, num_samples_to_show)
            rows = (num_samples_to_show + cols - 1) // cols

            grid_height = rows * image_size
            grid_width = cols * image_size
            grid_image = np.ones((grid_height, grid_width), dtype=np.uint8) * 255

            font = cv2.FONT_HERSHEY_SIMPLEX
            for idx in range(num_samples_to_show):
                kp_dict = keypoints_list[idx]
                skeleton_img = draw_full_skeleton(
                    kp_dict,
                    image_size=image_size,
                    point_radius=3,
                    line_thickness=2
                )

                row = idx // cols
                col = idx % cols
                y_start = row * image_size
                x_start = col * image_size
                grid_image[y_start:y_start+image_size, x_start:x_start+image_size] = skeleton_img

                # 添加标签
                cv2.putText(grid_image, f'Sample {idx+1}',
                          (x_start + 10, y_start + 30), font, 1, 0, 2)

            grid_path = split_output_dir / f"grid_{split}_all_samples.png"
            cv2.imwrite(str(grid_path), grid_image)
            print(f"  ✓ 保存网格图像: {grid_path.name}")

    # 创建所有划分的汇总网格
    if show_grid and len(all_samples) > 0:
        print(f"\n创建所有划分的汇总网格...")
        total_samples = len(all_samples)
        cols = min(5, total_samples)
        rows = (total_samples + cols - 1) // cols

        grid_height = rows * image_size
        grid_width = cols * image_size
        grid_image = np.ones((grid_height, grid_width), dtype=np.uint8) * 255

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, sample in enumerate(all_samples):
            row = i // cols
            col = i % cols
            y_start = row * image_size
            x_start = col * image_size
            grid_image[y_start:y_start+image_size, x_start:x_start+image_size] = sample['skeleton']

            # 添加标签
            label = f"{sample['split'][0].upper()}-{sample['idx']+1}"
            cv2.putText(grid_image, label,
                      (x_start + 10, y_start + 30), font, 1, 0, 2)

        grid_path = output_dir / "grid_all_splits.png"
        cv2.imwrite(str(grid_path), grid_image)
        print(f"  ✓ 保存汇总网格: {grid_path.name}")

    print(f"\n{'='*60}")
    print("可视化完成!")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"总样本数: {len(all_samples)}")
    print(f"\n生成的文件:")
    print(f"  - 完整骨架图像: {output_dir}/<split>/sample_*_full_skeleton.png")
    if show_individual:
        print(f"  - 组件图像: {output_dir}/<split>/sample_*_face.png, hands.png, pose.png")
    if show_combined:
        print(f"  - 组合图像: {output_dir}/<split>/sample_*_combined.png")
    if show_grid:
        print(f"  - 网格图像: {output_dir}/<split>/grid_*.png")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='可视化 PHOENIX 数据集关键点',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本可视化（所有划分，每个划分5个样本）
  python visualize_phoenix_keypoints.py phoenix_keypoints_test.pkl

  # 只可视化训练集
  python visualize_phoenix_keypoints.py phoenix_keypoints_test.pkl --splits train

  # 可视化更多样本
  python visualize_phoenix_keypoints.py phoenix_keypoints_test.pkl --num_samples 10

  # 只生成完整骨架，不生成组件
  python visualize_phoenix_keypoints.py phoenix_keypoints_test.pkl --no-individual

  # 自定义输出目录和图像大小
  python visualize_phoenix_keypoints.py phoenix_keypoints_test.pkl \\
      --output_dir my_visualization \\
      --image_size 256
        """
    )

    parser.add_argument(
        'pkl_path',
        type=str,
        help='PHOENIX 关键点 pickle 文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='phoenix_keypoints_visualization',
        help='输出目录（默认: phoenix_keypoints_visualization）'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        choices=['train', 'dev', 'test'],
        default=None,
        help='要可视化的划分（默认: 所有划分）'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='每个划分可视化的样本数（默认: 5）'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=512,
        help='图像大小（默认: 512）'
    )
    parser.add_argument(
        '--no-individual',
        action='store_true',
        help='不生成单独的组件图像'
    )
    parser.add_argument(
        '--no-combined',
        action='store_true',
        help='不生成组合图像'
    )
    parser.add_argument(
        '--no-grid',
        action='store_true',
        help='不生成网格图像'
    )

    args = parser.parse_args()

    visualize_phoenix_keypoints(
        pkl_path=args.pkl_path,
        output_dir=args.output_dir,
        splits=args.splits,
        num_samples=args.num_samples,
        image_size=args.image_size,
        show_individual=not args.no_individual,
        show_combined=not args.no_combined,
        show_grid=not args.no_grid
    )

if __name__ == "__main__":
    main()




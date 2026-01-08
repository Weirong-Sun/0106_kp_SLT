"""
Extract frames from sign language video dataset
Converts videos to image frames for training dataset
"""
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

class VideoFrameExtractor:
    def __init__(self, output_dir="extracted_frames", fps_extract=1, image_format="jpg", quality=95):
        """
        Args:
            output_dir: Directory to save extracted frames
            fps_extract: Frames per second to extract (1 means extract 1 frame per second)
            image_format: Image format to save (jpg, png)
            quality: JPEG quality (1-100, only for jpg format)
        """
        self.output_dir = Path(output_dir)
        self.fps_extract = fps_extract
        self.image_format = image_format.lower()
        self.quality = quality
        
        # Supported video formats
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', 
                                '.MP4', '.AVI', '.MOV', '.MKV', '.FLV', '.WMV', '.WEBM'}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames_from_video(self, video_path, output_subdir=None):
        """
        Extract frames from a single video file
        
        Args:
            video_path: Path to video file
            output_subdir: Subdirectory name in output_dir (if None, uses video filename)
        
        Returns:
            num_frames: Number of frames extracted
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Warning: Video file not found: {video_path}")
            return 0
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open video: {video_path}")
            return 0
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame interval
        frame_interval = int(fps / self.fps_extract) if fps > 0 else 1
        if frame_interval < 1:
            frame_interval = 1
        
        # Create output subdirectory
        if output_subdir is None:
            output_subdir = video_path.stem
        
        output_path = self.output_dir / output_subdir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        frame_count = 0
        saved_count = 0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                # Save frame
                if self.image_format == 'jpg':
                    filename = f"frame_{frame_idx:06d}.jpg"
                    filepath = output_path / filename
                    cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                elif self.image_format == 'png':
                    filename = f"frame_{frame_idx:06d}.png"
                    filepath = output_path / filename
                    cv2.imwrite(str(filepath), frame)
                else:
                    print(f"Warning: Unsupported image format: {self.image_format}")
                    break
                
                saved_count += 1
                frame_idx += 1
            
            frame_count += 1
        
        cap.release()
        return saved_count
    
    def process_dataset(self, dataset_path, max_videos=None, preserve_structure=True):
        """
        Process all videos in the dataset directory
        
        Args:
            dataset_path: Path to dataset directory
            max_videos: Maximum number of videos to process (None for all)
            preserve_structure: If True, preserve directory structure in output
        
        Returns:
            stats: Dictionary with processing statistics
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Find all video files
        print(f"Scanning for videos in: {dataset_path}")
        video_files = []
        for ext in self.video_extensions:
            pattern = f'**/*{ext}'
            found_files = list(dataset_path.glob(pattern))
            video_files.extend(found_files)
        
        # Remove duplicates and sort
        video_files = sorted(list(set(video_files)))
        
        print(f"\nFound {len(video_files)} video files")
        
        if max_videos:
            video_files = video_files[:max_videos]
            print(f"Processing first {len(video_files)} videos (limited by max_videos)")
        else:
            print(f"Processing all {len(video_files)} videos")
        
        # Process videos
        total_frames = 0
        successful_videos = 0
        failed_videos = 0
        
        for video_path in tqdm(video_files, desc="Extracting frames"):
            try:
                # Determine output subdirectory
                if preserve_structure:
                    # Preserve relative path structure
                    relative_path = video_path.relative_to(dataset_path)
                    output_subdir = relative_path.parent / video_path.stem
                else:
                    output_subdir = None
                
                num_frames = self.extract_frames_from_video(video_path, output_subdir)
                if num_frames > 0:
                    total_frames += num_frames
                    successful_videos += 1
                else:
                    failed_videos += 1
            except Exception as e:
                print(f"\nError processing {video_path}: {e}")
                failed_videos += 1
        
        stats = {
            'total_videos': len(video_files),
            'successful_videos': successful_videos,
            'failed_videos': failed_videos,
            'total_frames': total_frames,
            'output_dir': str(self.output_dir)
        }
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='Extract frames from sign language video dataset')
    parser.add_argument('--dataset_path', type=str, 
                        default=r"D:\dataset\archive (18)\Indian Sign Language Video and Text dataset for sentences (ISLVT)",
                        help='Path to the video dataset directory')
    parser.add_argument('--output_dir', type=str, default='extracted_frames',
                        help='Directory to save extracted frames')
    parser.add_argument('--fps_extract', type=float, default=1.0,
                        help='Frames per second to extract (default: 1.0, meaning 1 frame per second)')
    parser.add_argument('--image_format', type=str, default='jpg', choices=['jpg', 'png'],
                        help='Image format to save (default: jpg)')
    parser.add_argument('--quality', type=int, default=95, choices=range(1, 101),
                        help='JPEG quality 1-100 (default: 95, only for jpg format)')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process (None for all)')
    parser.add_argument('--preserve_structure', action='store_true', default=True,
                        help='Preserve directory structure in output (default: True)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Video Frame Extraction")
    print("="*60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Extraction rate: {args.fps_extract} fps")
    print(f"Image format: {args.image_format}")
    if args.image_format == 'jpg':
        print(f"JPEG quality: {args.quality}")
    print(f"Preserve structure: {args.preserve_structure}")
    print("="*60)
    
    extractor = VideoFrameExtractor(
        output_dir=args.output_dir,
        fps_extract=args.fps_extract,
        image_format=args.image_format,
        quality=args.quality
    )
    
    stats = extractor.process_dataset(
        dataset_path=args.dataset_path,
        max_videos=args.max_videos,
        preserve_structure=args.preserve_structure
    )
    
    print("\n" + "="*60)
    print("Extraction Summary")
    print("="*60)
    print(f"Total videos found: {stats['total_videos']}")
    print(f"Successfully processed: {stats['successful_videos']}")
    print(f"Failed: {stats['failed_videos']}")
    print(f"Total frames extracted: {stats['total_frames']}")
    print(f"Average frames per video: {stats['total_frames']/stats['successful_videos']:.2f}" if stats['successful_videos'] > 0 else "N/A")
    print(f"Output directory: {stats['output_dir']}")
    print("="*60)
    print("\nDone!")

if __name__ == "__main__":
    main()


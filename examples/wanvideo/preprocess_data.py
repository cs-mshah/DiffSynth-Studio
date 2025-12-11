#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".gif",  # Read support only, output will be MP4
}


def list_videos(input_folder: Path) -> List[Path]:
    return sorted([p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def open_video_reader(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def get_video_meta(cap: cv2.VideoCapture) -> Tuple[int, float, int, int]:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return total_frames, fps, width, height


def create_video_writer(
    output_path: Path,
    fps: float,
    frame_size: Tuple[int, int],
) -> cv2.VideoWriter:
    """Create a video writer. Always uses MP4 format for output."""
    # Always use MP4 format for output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, max(fps, 1e-6), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {output_path}")
    return writer


def resize_with_padding(
    frame: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize frame to target size while maintaining aspect ratio.
    Pads with white pixels (255, 255, 255) to reach target size if needed.
    
    Args:
        frame: Input frame (H, W, C) in BGR format
        target_size: Target (width, height)
    
    Returns:
        Resized and padded frame of size target_size
    """
    target_width, target_height = target_size
    orig_height, orig_width = frame.shape[:2]
    
    # Calculate scale factor to fit within target while maintaining aspect ratio
    scale = min(target_width / orig_width, target_height / orig_height)
    
    # Calculate new dimensions
    new_width = int(round(orig_width * scale))
    new_height = int(round(orig_height * scale))
    
    # Resize maintaining aspect ratio
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create white canvas of target size
    if len(frame.shape) == 3:
        # Color image (BGR)
        canvas = np.ones((target_height, target_width, frame.shape[2]), dtype=frame.dtype) * 255
    else:
        # Grayscale
        canvas = np.ones((target_height, target_width), dtype=frame.dtype) * 255
    
    # Calculate padding offsets to center the resized image
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image on white canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    
    return canvas


def index_stream(
    cap: cv2.VideoCapture,
    indices: List[int],
    writer: cv2.VideoWriter,
    target_size: Optional[Tuple[int, int]] = None,
) -> int:
    """
    Read frames from cap by zero-based indices (monotonically increasing) and write to writer.
    If target_size is provided, frames are resized while maintaining aspect ratio and padded
    with white pixels to reach the target size.
    Returns number of frames written.
    """
    written = 0
    # Efficient approach: sequential read with a pointer
    target_iter = iter(indices)
    try:
        next_target = next(target_iter)
    except StopIteration:
        return 0

    current_index = 0
    ok, frame = cap.read()
    while ok:
        if current_index == next_target:
            if target_size is not None:
                frame = resize_with_padding(frame, target_size)
            writer.write(frame)
            written += 1
            try:
                next_target = next(target_iter)
            except StopIteration:
                break
        current_index += 1
        ok, frame = cap.read()
    return written


def downsample_to_fps(
    input_path: Path,
    output_path: Path,
    target_fps: float,
    target_resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    cap = open_video_reader(input_path)
    try:
        total_frames, orig_fps, width, height = get_video_meta(cap)
        if total_frames <= 0:
            raise RuntimeError(f"Video has zero frames: {input_path}")
        if orig_fps <= 0:
            # Fallback if FPS is unknown; treat as 30
            orig_fps = 30.0

        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")

        # Determine output frame size
        output_size = target_resolution if target_resolution is not None else (width, height)

        # If target_fps >= orig_fps, simply copy frames but write at orig_fps (no reduction needed)
        if target_fps >= orig_fps - 1e-6:
            frame_indices = list(range(total_frames))
            writer = create_video_writer(output_path, orig_fps, output_size)
            try:
                written = index_stream(cap, frame_indices, writer, target_resolution)
            finally:
                writer.release()
            if written == 0:
                raise RuntimeError(f"No frames written for {input_path}")
            return total_frames, written

        # Compute stride to approximately maintain duration when changing FPS.
        # Dropping frames by this stride and writing at target_fps keeps duration similar.
        stride = max(int(round(orig_fps / target_fps)), 1)
        frame_indices = list(range(0, total_frames, stride))
        # Ensure at least 1 frame
        if len(frame_indices) == 0:
            frame_indices = [0]

        writer = create_video_writer(output_path, target_fps, output_size)
        try:
            written = index_stream(cap, frame_indices, writer, target_resolution)
        finally:
            writer.release()
        if written == 0:
            raise RuntimeError(f"No frames written for {input_path}")
        return total_frames, written
    finally:
        cap.release()


def downsample_to_num_frames(
    input_path: Path,
    output_path: Path,
    target_num_frames: int,
    sampling: str,
    target_resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    cap = open_video_reader(input_path)
    try:
        total_frames, orig_fps, width, height = get_video_meta(cap)
        if total_frames <= 0:
            raise RuntimeError(f"Video has zero frames: {input_path}")
        if orig_fps <= 0:
            orig_fps = 30.0
        print(f'{input_path} orig_fps: {orig_fps}')
        target_num_frames = int(target_num_frames)
        if target_num_frames <= 0:
            raise ValueError("target_num_frames must be > 0")

        # Determine output frame size
        output_size = target_resolution if target_resolution is not None else (width, height)

        if target_num_frames >= total_frames:
            # Nothing to reduce, copy as-is
            frame_indices = list(range(total_frames))
        else:
            if sampling == "uniform":
                # Uniformly sample indices across the full range
                frame_indices = np.linspace(0, total_frames - 1, num=target_num_frames, endpoint=True)
                frame_indices = np.rint(frame_indices).astype(int)
                # Ensure strictly increasing and within bounds
                frame_indices = np.clip(frame_indices, 0, total_frames - 1)
                frame_indices = np.unique(frame_indices)
                # If uniqueness reduced the count, append as needed
                while len(frame_indices) < target_num_frames:
                    frame_indices = np.append(frame_indices, min(total_frames - 1, frame_indices[-1] + 1))
                frame_indices = frame_indices.tolist()
            elif sampling == "prefix":
                frame_indices = list(range(target_num_frames))
            else:
                raise ValueError("sampling must be one of ['uniform', 'prefix']")

        # Write at original fps; duration may shorten if fewer frames are written (intentional).
        writer = create_video_writer(output_path, orig_fps, output_size)
        try:
            written = index_stream(cap, frame_indices, writer, target_resolution)
        finally:
            writer.release()
        if written == 0:
            raise RuntimeError(f"No frames written for {input_path}")
        return total_frames, written
    finally:
        cap.release()


def parse_resolution(resolution_str: str) -> Tuple[int, int]:
    """Parse resolution string in format 'WIDTHxHEIGHT' (e.g., '512x512' or '1280x720')."""
    try:
        parts = resolution_str.lower().split("x")
        if len(parts) != 2:
            raise ValueError("Resolution must be in format WIDTHxHEIGHT (e.g., 512x512)")
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")
        return (width, height)
    except ValueError as e:
        raise ValueError(f"Invalid resolution format '{resolution_str}': {e}")


def process_folder(
    input_folder: Path,
    output_folder: Path,
    target_fps: Optional[float],
    target_num_frames: Optional[int],
    sampling: str,
    overwrite: bool,
    target_resolution: Optional[Tuple[int, int]] = None,
) -> None:
    videos = list_videos(input_folder)
    if not videos:
        print(f"No videos found in {input_folder}", file=sys.stderr)
        return
    safe_mkdir(output_folder)

    for video_path in videos:
        # Always output as MP4, regardless of input format
        out_path = output_folder / (video_path.stem + ".mp4")
        if out_path.exists() and not overwrite:
            print(f"Skipping existing: {out_path}")
            continue
        try:
            if target_fps is not None:
                initial_frames, output_frames = downsample_to_fps(video_path, out_path, float(target_fps), target_resolution)
            else:
                assert target_num_frames is not None
                initial_frames, output_frames = downsample_to_num_frames(video_path, out_path, int(target_num_frames), sampling, target_resolution)
            print(f"Processed: {video_path.name} -> {out_path} (frames: {initial_frames} -> {output_frames})")
        except Exception as e:
            print(f"Error processing {video_path}: {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce the number of frames in all videos within a folder.\n"
            "- If --fps is provided, videos are reduced to target FPS (duration preserved approximately).\n"
            "- If --num-frames is provided, videos are reduced by uniform or prefix sampling (duration may shorten).\n"
            "- Optionally resize videos to a target resolution using --resolution."
        )
    )
    parser.add_argument("--input-folder", type=Path, required=True, help="Folder containing input videos")
    parser.add_argument("--output-folder", type=Path, required=True, help="Folder to write processed videos")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fps", type=float, help="Target FPS to downsample to (e.g., 8)")
    group.add_argument("--num-frames", type=int, help="Target total number of frames to keep (e.g., 64)")

    parser.add_argument(
        "--sampling",
        type=str,
        default="uniform",
        choices=["uniform", "prefix"],
        help="Sampling strategy when using --num-frames",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Target resolution in format WIDTHxHEIGHT (e.g., 512x512 or 1280x720). Optional.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    args = parser.parse_args()
    if not args.input_folder.exists() or not args.input_folder.is_dir():
        parser.error(f"--input-folder must be an existing directory: {args.input_folder}")
    
    # Parse resolution if provided
    target_resolution = None
    if args.resolution is not None:
        target_resolution = parse_resolution(args.resolution)
    
    args.target_resolution = target_resolution
    return args


def main() -> None:
    args = parse_args()
    process_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        target_fps=args.fps,
        target_num_frames=args.num_frames,
        sampling=args.sampling,
        overwrite=args.overwrite,
        target_resolution=args.target_resolution,
    )


if __name__ == "__main__":
    main()



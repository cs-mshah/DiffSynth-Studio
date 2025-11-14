#!/usr/bin/env python3
import argparse
import sys
import os
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


def pick_fourcc_for_extension(ext: str) -> int:
    ext = ext.lower()
    # Reasonable defaults; these are widely supported
    if ext in {".mp4", ".m4v", ".mov"}:
        return cv2.VideoWriter_fourcc(*"mp4v")
    if ext in {".avi"}:
        return cv2.VideoWriter_fourcc(*"XVID")
    if ext in {".webm"}:
        return cv2.VideoWriter_fourcc(*"VP90")
    # Fallback
    return cv2.VideoWriter_fourcc(*"mp4v")


def create_video_writer(
    output_path: Path,
    fps: float,
    frame_size: Tuple[int, int],
) -> cv2.VideoWriter:
    fourcc = pick_fourcc_for_extension(output_path.suffix)
    writer = cv2.VideoWriter(str(output_path), fourcc, max(fps, 1e-6), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {output_path}")
    return writer


def index_stream(cap: cv2.VideoCapture, indices: List[int], writer: cv2.VideoWriter) -> int:
    """
    Read frames from cap by zero-based indices (monotonically increasing) and write to writer.
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
) -> None:
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

        # If target_fps >= orig_fps, simply copy frames but write at orig_fps (no reduction needed)
        if target_fps >= orig_fps - 1e-6:
            frame_indices = list(range(total_frames))
            writer = create_video_writer(output_path, orig_fps, (width, height))
            try:
                written = index_stream(cap, frame_indices, writer)
            finally:
                writer.release()
            if written == 0:
                raise RuntimeError(f"No frames written for {input_path}")
            return

        # Compute stride to approximately maintain duration when changing FPS.
        # Dropping frames by this stride and writing at target_fps keeps duration similar.
        stride = max(int(round(orig_fps / target_fps)), 1)
        frame_indices = list(range(0, total_frames, stride))
        # Ensure at least 1 frame
        if len(frame_indices) == 0:
            frame_indices = [0]

        writer = create_video_writer(output_path, target_fps, (width, height))
        try:
            written = index_stream(cap, frame_indices, writer)
        finally:
            writer.release()
        if written == 0:
            raise RuntimeError(f"No frames written for {input_path}")
    finally:
        cap.release()


def downsample_to_num_frames(
    input_path: Path,
    output_path: Path,
    target_num_frames: int,
    sampling: str,
) -> None:
    cap = open_video_reader(input_path)
    try:
        total_frames, orig_fps, width, height = get_video_meta(cap)
        if total_frames <= 0:
            raise RuntimeError(f"Video has zero frames: {input_path}")
        if orig_fps <= 0:
            orig_fps = 30.0

        target_num_frames = int(target_num_frames)
        if target_num_frames <= 0:
            raise ValueError("target_num_frames must be > 0")

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
        writer = create_video_writer(output_path, orig_fps, (width, height))
        try:
            written = index_stream(cap, frame_indices, writer)
        finally:
            writer.release()
        if written == 0:
            raise RuntimeError(f"No frames written for {input_path}")
    finally:
        cap.release()


def process_folder(
    input_folder: Path,
    output_folder: Path,
    target_fps: Optional[float],
    target_num_frames: Optional[int],
    sampling: str,
    overwrite: bool,
) -> None:
    videos = list_videos(input_folder)
    if not videos:
        print(f"No videos found in {input_folder}", file=sys.stderr)
        return
    safe_mkdir(output_folder)

    for video_path in videos:
        out_path = output_folder / video_path.name
        if out_path.exists() and not overwrite:
            print(f"Skipping existing: {out_path}")
            continue
        try:
            if target_fps is not None:
                downsample_to_fps(video_path, out_path, float(target_fps))
            else:
                assert target_num_frames is not None
                downsample_to_num_frames(video_path, out_path, int(target_num_frames), sampling)
            print(f"Processed: {video_path.name} -> {out_path}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce the number of frames in all videos within a folder.\n"
            "- If --fps is provided, videos are reduced to target FPS (duration preserved approximately).\n"
            "- If --num-frames is provided, videos are reduced by uniform or prefix sampling (duration may shorten)."
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
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    args = parser.parse_args()
    if not args.input_folder.exists() or not args.input_folder.is_dir():
        parser.error(f"--input-folder must be an existing directory: {args.input_folder}")
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
    )


if __name__ == "__main__":
    main()



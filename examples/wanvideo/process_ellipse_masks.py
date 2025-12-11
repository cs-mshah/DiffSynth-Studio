#!/usr/bin/env python3
import argparse
import shutil
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
}


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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, max(fps, 1e-6), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {output_path}")
    return writer


def find_tightest_ellipse(mask_frame: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """
    Find the tightest bounding ellipse to the positive region (where mask = 1).
    
    Args:
        mask_frame: Binary mask frame (H, W) where 1 = object region
        
    Returns:
        Ellipse parameters ((center_x, center_y), (width, height), angle) or None if no positive region found
    """
    # Convert to binary: 1 where mask > 0, 0 otherwise
    binary = (mask_frame > 0).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Combine all contours into one
    all_points = np.vstack(contours)
    
    if len(all_points) < 5:  # Need at least 5 points for ellipse fitting
        return None
    
    # Fit ellipse to all positive points
    ellipse = cv2.fitEllipse(all_points)
    
    return ellipse


def create_ellipse_mask(
    frame_shape: Tuple[int, int],
    ellipse: Tuple[Tuple[float, float], Tuple[float, float], float],
) -> np.ndarray:
    """
    Create a binary mask with the ellipse filled.
    
    Args:
        frame_shape: (height, width) of the frame
        ellipse: Ellipse parameters from cv2.fitEllipse
        
    Returns:
        Binary mask (H, W) with 1 inside ellipse, 0 outside
    """
    mask = np.zeros(frame_shape, dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)  # -1 means filled
    mask = (mask > 0).astype(np.uint8)
    return mask


def process_video_triplet(
    original_path: Path,
    vace_video_path: Path,
    vace_mask_path: Path,
    output_folder: Path,
    overwrite: bool,
) -> None:
    """
    Process a video triplet:
    1. Copy original video as-is
    2. Create new vace_video with gray 127 in ellipse region
    3. Create new vace_mask with ellipse mask
    """
    # Output paths
    orig_output = output_folder / original_path.name
    vace_output = output_folder / vace_video_path.name
    mask_output = output_folder / vace_mask_path.name
    
    # Check if outputs exist
    if not overwrite and orig_output.exists() and vace_output.exists() and mask_output.exists():
        print(f"Skipping existing triplet: {original_path.stem}")
        return
    
    # Open all three videos
    orig_cap = open_video_reader(original_path)
    vace_cap = open_video_reader(vace_video_path)
    mask_cap = open_video_reader(vace_mask_path)
    
    try:
        # Get metadata (assume all videos have same properties)
        total_frames, fps, width, height = get_video_meta(mask_cap)
        if total_frames <= 0:
            raise RuntimeError(f"Video has zero frames: {vace_mask_path}")
        if fps <= 0:
            fps = 30.0
        
        frame_size = (width, height)
        
        # Create writers for vace_video and mask
        vace_writer = create_video_writer(vace_output, fps, frame_size)
        mask_writer = create_video_writer(mask_output, fps, frame_size)
        
        try:
            frame_count = 0
            while True:
                # Read frames
                orig_ok, orig_frame = orig_cap.read()
                vace_ok, vace_frame = vace_cap.read()
                mask_ok, mask_frame = mask_cap.read()
                
                if not (orig_ok and vace_ok and mask_ok):
                    break
                
                # Convert mask to grayscale if needed
                if len(mask_frame.shape) == 3:
                    mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                else:
                    mask_gray = mask_frame
                
                # Find tightest ellipse
                ellipse = find_tightest_ellipse(mask_gray)
                
                if ellipse is not None:
                    # Create ellipse mask
                    ellipse_mask = create_ellipse_mask((height, width), ellipse)
                    
                    # Create new vace_video: gray 127 where ellipse mask is 1, original video pixels where 0
                    # Use original video frame as base, then apply gray 127 in ellipse region
                    if len(orig_frame.shape) == 3:
                        # Color image
                        new_vace_frame = orig_frame.copy()
                        # Set gray (127, 127, 127) in BGR where ellipse mask is 1
                        ellipse_mask_3d = ellipse_mask[:, :, np.newaxis]
                        new_vace_frame = np.where(ellipse_mask_3d == 1, 127, new_vace_frame)
                    else:
                        # Grayscale
                        new_vace_frame = np.where(ellipse_mask == 1, 127, orig_frame)
                    
                    # Write ellipse mask (convert to 0/255 for video)
                    mask_frame_out = (ellipse_mask * 255).astype(np.uint8)
                    if len(mask_frame.shape) == 3:
                        # Convert to 3-channel if original was 3-channel
                        mask_frame_out = cv2.cvtColor(mask_frame_out, cv2.COLOR_GRAY2BGR)
                else:
                    # No positive region found, create empty mask and use original video frame
                    new_vace_frame = orig_frame.copy()
                    mask_frame_out = np.zeros((height, width), dtype=np.uint8)
                    if len(mask_frame.shape) == 3:
                        mask_frame_out = cv2.cvtColor(mask_frame_out, cv2.COLOR_GRAY2BGR)
                
                # Write frames
                vace_writer.write(new_vace_frame)
                mask_writer.write(mask_frame_out)
                frame_count += 1
            
            if frame_count == 0:
                raise RuntimeError(f"No frames processed for triplet: {original_path.stem}")
            
        finally:
            vace_writer.release()
            mask_writer.release()
        
        # Copy original video
        shutil.copy2(original_path, orig_output)
        
        print(f"Processed: {original_path.stem} ({frame_count} frames)")
        
    finally:
        orig_cap.release()
        vace_cap.release()
        mask_cap.release()


def find_video_triplets(input_folder: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Find all video triplets in the input folder.
    Returns list of (original_path, vace_video_path, vace_mask_path) tuples.
    """
    videos = sorted([p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])
    
    triplets = []
    processed = set()
    
    for video_path in videos:
        if video_path in processed:
            continue
        
        stem = video_path.stem
        
        # Check if this is a base video (not _vace_video or _vace_video_mask)
        if stem.endswith("_vace_video") or stem.endswith("_vace_video_mask"):
            continue
        
        # Look for corresponding vace files
        vace_video_path = input_folder / f"{stem}_vace_video{video_path.suffix}"
        vace_mask_path = input_folder / f"{stem}_vace_video_mask{video_path.suffix}"
        
        if vace_video_path.exists() and vace_mask_path.exists():
            triplets.append((video_path, vace_video_path, vace_mask_path))
            processed.add(video_path)
            processed.add(vace_video_path)
            processed.add(vace_mask_path)
    
    return triplets


def process_folder(
    input_folder: Path,
    output_folder: Path,
    overwrite: bool,
) -> None:
    triplets = find_video_triplets(input_folder)
    
    if not triplets:
        print(f"No video triplets found in {input_folder}", file=sys.stderr)
        return
    
    safe_mkdir(output_folder)
    
    print(f"Found {len(triplets)} video triplet(s)")
    
    for orig_path, vace_path, mask_path in triplets:
        try:
            process_video_triplet(orig_path, vace_path, mask_path, output_folder, overwrite)
        except Exception as e:
            print(f"Error processing triplet {orig_path.stem}: {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process video triplets (original, vace_video, vace_video_mask) and create "
            "new versions where the mask is replaced with the tightest bounding ellipse "
            "to the positive region in each frame.\n"
            "- Original video is copied as-is\n"
            "- New vace_video has gray 127 in the ellipse region\n"
            "- New vace_video_mask contains the ellipse mask"
        )
    )
    parser.add_argument("--input-folder", type=Path, required=True, help="Folder containing input video triplets")
    parser.add_argument("--output-folder", type=Path, required=True, help="Folder to write processed videos")
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
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    # python examples/wanvideo/process_ellipse_masks.py --input-folder data/raw/example4 --output-folder data/raw/example4/ellipse --overwrite
    main()


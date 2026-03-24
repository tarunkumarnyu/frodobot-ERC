#!/usr/bin/env python3
"""Step 1: Extract frames from corridor video and filter blurry ones."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from topnav.graph_builder import filter_blurry_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", help="Path to corridor video")
    p.add_argument("--fps", type=int, default=2, help="Frames per second to extract")
    p.add_argument("--output", default="data/frames", help="Output directory")
    p.add_argument("--blur-threshold", type=float, default=50.0)
    args = p.parse_args()

    raw_dir = Path(args.output + "_raw")
    clean_dir = Path(args.output)
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Extract with ffmpeg
    print(f"Extracting frames at {args.fps} FPS...")
    subprocess.run([
        "ffmpeg", "-i", args.video, "-vf", f"fps={args.fps}",
        str(raw_dir / "frame_%05d.jpg")
    ], check=True)

    raw_count = len(list(raw_dir.glob("*.jpg")))
    print(f"Extracted {raw_count} frames")

    # Filter blurry
    print(f"Filtering blurry frames (threshold={args.blur_threshold})...")
    clean = filter_blurry_frames(raw_dir, threshold=args.blur_threshold)

    import shutil
    for f in clean:
        shutil.copy(f, clean_dir / f.name)

    print(f"Kept {len(clean)}/{raw_count} clean frames in {clean_dir}")


if __name__ == "__main__":
    main()

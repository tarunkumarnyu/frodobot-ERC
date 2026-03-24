#!/usr/bin/env python3
"""Extract frames from FrodoBot H5 dataset files.

Usage:
    python scripts/extract_h5_frames.py /path/to/test_outdoor_1.h5 --output data/frames
"""

import argparse
import io
from pathlib import Path

import h5py
from PIL import Image
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5_files", nargs="+", help="H5 dataset file(s)")
    p.add_argument("--output", default="data/frames", help="Output directory")
    p.add_argument("--skip", type=int, default=1, help="Extract every Nth frame")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0

    for h5_path in args.h5_files:
        print(f"\nProcessing {h5_path}...")
        f = h5py.File(h5_path, "r")

        if "front_frames" in f and "data" in f["front_frames"]:
            frames = f["front_frames"]["data"]
            n = len(frames)
            print(f"  {n} frames found")

            for i in tqdm(range(0, n, args.skip), desc="  Extracting"):
                try:
                    frame_bytes = bytes(frames[i])
                    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                    img.save(out_dir / f"frame_{frame_idx:06d}.jpg")
                    frame_idx += 1
                except Exception as e:
                    print(f"  Skip frame {i}: {e}")

        f.close()

    print(f"\nExtracted {frame_idx} frames to {out_dir}")


if __name__ == "__main__":
    main()

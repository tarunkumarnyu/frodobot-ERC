#!/usr/bin/env python3
"""Record a topomap by driving the robot manually.

Drive the robot through the corridor while this script captures frames.
These frames become the topomap that ViNT follows.

Usage:
    # In terminal 1: drive with WASD
    python simple_control.py

    # In terminal 2: record topomap
    python scripts/04_record_topomap.py --output data/topomap
"""

import argparse
import base64
import io
import os
import time

import requests
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/topomap", help="Output directory")
    p.add_argument("--sdk-url", default="http://localhost:8000")
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between captures")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Recording topomap to {args.output}")
    print(f"Drive the robot in another terminal. Press Ctrl+C to stop.\n")

    i = 0
    try:
        while True:
            try:
                resp = requests.get(f"{args.sdk_url}/v2/front", timeout=3)
                b64 = resp.json().get("front_frame")
                if b64:
                    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                    if img.size[0] > 10:
                        path = os.path.join(args.output, f"frame_{i:05d}.jpg")
                        img.save(path)
                        print(f"\r  Frame {i}: saved", end="", flush=True)
                        i += 1
            except Exception as e:
                print(f"\r  Error: {e}", end="")

            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

    print(f"\n\nRecorded {i} frames to {args.output}")
    print(f"Now run: python scripts/05_vint_navigate.py --topomap {args.output}")


if __name__ == "__main__":
    main()

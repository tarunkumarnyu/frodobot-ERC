#!/usr/bin/env python3
"""Step 2: Extract CosPlace features and build topological graph."""

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from topnav.feature_extractor import CosPlaceExtractor
from topnav.graph_builder import TopologicalGraph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="data/frames", help="Clean frames directory")
    p.add_argument("--output", default="data/graph.pkl", help="Output graph file")
    p.add_argument("--loop-threshold", type=float, default=0.90)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    frame_dir = Path(args.frames)
    frame_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    print(f"Found {len(frame_paths)} frames")

    # Extract features
    print("Loading CosPlace...")
    extractor = CosPlaceExtractor(device=args.device)

    print("Extracting features...")
    images = [Image.open(p) for p in tqdm(frame_paths, desc="Loading")]
    features = extractor.extract_batch(images)
    print(f"Features: {features.shape}")

    # Build graph
    print("Building topological graph...")
    graph = TopologicalGraph()
    graph.build(frame_paths, features, loop_closure_threshold=args.loop_threshold)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    graph.save(args.output)
    print(f"Saved graph to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 3: Run live navigation to a target image."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from topnav.feature_extractor import CosPlaceExtractor
from topnav.graph_builder import TopologicalGraph
from topnav.direction_estimator import DirectionEstimator
from topnav.navigator import Navigator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="data/graph.pkl", help="Graph file")
    p.add_argument("--target", required=True, help="Target image path")
    p.add_argument("--sdk-url", default="http://localhost:8000")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # Load models
    print("Loading CosPlace...")
    extractor = CosPlaceExtractor(device=args.device)

    print("Loading LightGlue...")
    dir_est = DirectionEstimator(device=args.device)

    # Load graph
    print("Loading graph...")
    graph = TopologicalGraph()
    graph.load(args.graph)
    print(f"Graph: {graph.num_nodes} nodes")

    # Find target
    print(f"Locating target: {args.target}")
    target_desc = extractor.extract_from_path(Path(args.target))
    target_node, target_score = graph.find_target(target_desc)
    print(f"Target → node {target_node} (score={target_score:.3f})")
    print(f"Target frame: {graph.frame_paths[target_node]}")

    # Navigate
    nav = Navigator(graph, extractor, dir_est, target_node, sdk_url=args.sdk_url)
    success = nav.run(max_steps=args.max_steps)
    print(f"\nNavigation {'SUCCEEDED' if success else 'FAILED'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build topological graph directly from H5 dataset (no disk extraction needed).

Reads frames from H5, extracts CosPlace features in memory, builds graph.

Usage:
    python scripts/build_graph_from_h5.py /path/to/test_outdoor_2.h5 --output data/graph.pkl
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from topnav.feature_extractor import CosPlaceExtractor
from topnav.graph_builder import TopologicalGraph


def load_frames_from_h5(h5_path: str, skip: int = 5, max_frames: int = 500):
    """Load frames from H5 file into memory."""
    f = h5py.File(h5_path, "r")
    frames_data = f["front_frames"]["data"]
    n = len(frames_data)
    print(f"  H5 has {n} frames, using every {skip}th (max {max_frames})")

    images = []
    frame_names = []
    for i in tqdm(range(0, n, skip), desc="  Loading frames"):
        if len(images) >= max_frames:
            break
        try:
            img = Image.open(io.BytesIO(bytes(frames_data[i]))).convert("RGB")
            images.append(img)
            frame_names.append(f"{Path(h5_path).stem}_frame_{i:05d}")
        except Exception:
            continue

    f.close()
    return images, frame_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5_file", help="H5 dataset file")
    p.add_argument("--output", default="data/graph.pkl")
    p.add_argument("--skip", type=int, default=5, help="Use every Nth frame")
    p.add_argument("--max-frames", type=int, default=500)
    p.add_argument("--loop-threshold", type=float, default=0.90)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # Load frames
    print(f"Loading frames from {args.h5_file}...")
    images, frame_names = load_frames_from_h5(
        args.h5_file, skip=args.skip, max_frames=args.max_frames)
    print(f"Loaded {len(images)} frames")

    # Extract features
    print("Loading CosPlace...")
    extractor = CosPlaceExtractor(device=args.device)

    print("Extracting features...")
    features = extractor.extract_batch(images, batch_size=16)
    print(f"Features: {features.shape}")

    # Build graph
    print("Building graph...")
    graph = TopologicalGraph()
    # Use frame names as paths (they're virtual since we loaded from H5)
    graph.frame_paths = frame_names
    graph.features = features
    n = len(frame_names)
    graph.adjacency = {i: [] for i in range(n)}

    # Sequential edges
    for i in range(n - 1):
        graph.adjacency[i].append(i + 1)
        graph.adjacency[i + 1].append(i)

    # Loop closures
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(features)
    n_loops = 0
    for i in range(n):
        for j in range(i + 10, n):
            if sim[i, j] > args.loop_threshold:
                graph.adjacency[i].append(j)
                graph.adjacency[j].append(i)
                n_loops += 1

    print(f"Graph: {n} nodes, {n - 1} sequential + {n_loops} loop closure edges")

    # Save graph (also save images as numpy for later use)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    import pickle
    # Save compact: features + adjacency + thumbnail for each frame
    thumbnails = [np.array(img.resize((160, 120))) for img in images]
    pickle.dump({
        "frame_names": frame_names,
        "features": features,
        "adjacency": graph.adjacency,
        "thumbnails": np.array(thumbnails),
        "h5_source": args.h5_file,
    }, open(args.output, "wb"))

    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"Saved to {args.output} ({size_mb:.1f} MB)")

    # Print similarity stats
    print(f"\nSimilarity stats:")
    print(f"  Mean self-sim: {np.diag(sim).mean():.3f}")
    print(f"  Mean neighbor-sim: {np.mean([sim[i, i+1] for i in range(n-1)]):.3f}")
    print(f"  Min neighbor-sim: {np.min([sim[i, i+1] for i in range(n-1)]):.3f}")


if __name__ == "__main__":
    main()

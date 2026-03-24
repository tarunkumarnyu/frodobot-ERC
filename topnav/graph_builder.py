"""Build topological graph from extracted features."""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def filter_blurry_frames(frame_dir: Path, threshold: float = 50.0) -> list[Path]:
    """Filter out blurry frames using Laplacian variance."""
    frames = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    clean = []
    for f in frames:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if cv2.Laplacian(img, cv2.CV_64F).var() > threshold:
            clean.append(f)
    return clean


class TopologicalGraph:
    """Topological navigation graph built from CosPlace features."""

    def __init__(self):
        self.frame_paths: list[str] = []
        self.features: np.ndarray = np.array([])
        self.adjacency: dict[int, list[int]] = {}

    @property
    def num_nodes(self) -> int:
        return len(self.frame_paths)

    def build(self, frame_paths: list[Path], features: np.ndarray,
              loop_closure_threshold: float = 0.90,
              min_loop_gap: int = 10):
        """Build graph from features.

        Args:
            frame_paths: ordered list of frame file paths
            features: (N, D) feature matrix
            loop_closure_threshold: cosine similarity for loop closure edges
            min_loop_gap: minimum node gap to consider loop closure
        """
        self.frame_paths = [str(p) for p in frame_paths]
        self.features = features
        n = len(frame_paths)
        self.adjacency = {i: [] for i in range(n)}

        # Sequential edges
        for i in range(n - 1):
            self.adjacency[i].append(i + 1)
            self.adjacency[i + 1].append(i)

        # Loop closure edges
        sim = cosine_similarity(features)
        for i in range(n):
            for j in range(i + min_loop_gap, n):
                if sim[i, j] > loop_closure_threshold:
                    if j not in self.adjacency[i]:
                        self.adjacency[i].append(j)
                        self.adjacency[j].append(i)

        n_loop = sum(len(v) for v in self.adjacency.values()) - 2 * (n - 1)
        print(f"Graph: {n} nodes, {n - 1} sequential + {n_loop // 2} loop closure edges")

    def localize(self, query_desc: np.ndarray, current_node: int = None,
                 local_threshold: float = 0.80) -> tuple[int, float]:
        """Find the best matching node for a query descriptor.

        Uses local search first (neighbors of current node) for speed,
        falls back to global search if local match is weak.
        """
        if current_node is not None and current_node in self.adjacency:
            # Local search: current + 2-hop neighbors
            candidates = set([current_node])
            for n in self.adjacency[current_node]:
                candidates.add(n)
                for nn in self.adjacency.get(n, []):
                    candidates.add(nn)

            cand_list = list(candidates)
            local_feats = self.features[cand_list]
            sims = cosine_similarity([query_desc], local_feats)[0]
            best_idx = cand_list[int(np.argmax(sims))]
            best_score = float(sims.max())

            if best_score > local_threshold:
                return best_idx, best_score

        # Global fallback
        sims = cosine_similarity([query_desc], self.features)[0]
        return int(np.argmax(sims)), float(sims.max())

    def find_target(self, target_desc: np.ndarray) -> tuple[int, float]:
        """Find the graph node closest to a target image descriptor."""
        sims = cosine_similarity([target_desc], self.features)[0]
        return int(np.argmax(sims)), float(sims.max())

    def shortest_path(self, start: int, goal: int) -> list[int] | None:
        """BFS shortest path."""
        if start == goal:
            return [start]
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            for neighbor in self.adjacency.get(node, []):
                if neighbor == goal:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def save(self, path: str):
        pickle.dump({
            "frame_paths": self.frame_paths,
            "features": self.features,
            "adjacency": self.adjacency,
        }, open(path, "wb"))

    def load(self, path: str):
        data = pickle.load(open(path, "rb"))
        self.frame_paths = data["frame_paths"]
        self.features = data["features"]
        self.adjacency = data["adjacency"]

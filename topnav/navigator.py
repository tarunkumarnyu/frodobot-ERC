"""Main navigation controller — CosPlace localization + LightGlue direction."""

from __future__ import annotations

import base64
import csv
import io
import time
from collections import deque
from typing import Optional

import numpy as np
import requests
from PIL import Image

from .graph_builder import TopologicalGraph
from .feature_extractor import CosPlaceExtractor
from .direction_estimator import DirectionEstimator


class Navigator:
    """Full navigation pipeline: localize → plan → steer → drive."""

    def __init__(
        self,
        graph: TopologicalGraph,
        extractor: CosPlaceExtractor,
        direction_est: DirectionEstimator,
        target_node: int,
        sdk_url: str = "http://localhost:8000",
    ):
        self.graph = graph
        self.extractor = extractor
        self.dir_est = direction_est
        self.target_node = target_node
        self.sdk_url = sdk_url
        self.current_node: Optional[int] = None
        self.log: list[dict] = []

        # Tuning parameters
        self.FORWARD_SPEED = 0.35
        self.JUNCTION_SPEED = 0.20
        self.ROTATION_SPEED = 0.25
        self.ARRIVAL_DISTANCE = 3       # nodes
        self.ARRIVAL_SCORE = 0.85
        self.LOOKAHEAD = 3              # nodes ahead
        self.JUNCTION_THRESHOLD = 5     # node gap = junction
        self.ALIGN_THRESHOLD = 0.15
        self.CONTROL_INTERVAL = 0.5     # seconds
        self.STUCK_THRESHOLD = 10       # steps at same node

        # Temporal smoothing
        self.node_history: deque[int] = deque(maxlen=5)

    def get_frame(self) -> Optional[Image.Image]:
        """Get front camera frame from SDK."""
        try:
            resp = requests.get(f"{self.sdk_url}/v2/front", timeout=3)
            data = resp.json()
            b64 = data.get("front_frame")
            if b64:
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                if img.size[0] > 10:
                    return img
        except Exception:
            pass
        return None

    def send_control(self, linear: float, angular: float):
        try:
            requests.post(
                f"{self.sdk_url}/control-legacy",
                json={"command": {"linear": round(linear, 3),
                                  "angular": round(angular, 3), "lamp": 0}},
                timeout=1.0,
            )
        except Exception:
            pass

    def stop(self):
        self.send_control(0, 0)

    def localize(self, img: Image.Image) -> tuple[int, float]:
        """Localize with temporal smoothing."""
        desc = self.extractor.extract(img)
        node, score = self.graph.localize(desc, self.current_node)

        self.node_history.append(node)

        # Reject wild jumps
        if len(self.node_history) >= 3:
            median_node = sorted(self.node_history)[len(self.node_history) // 2]
            if abs(node - median_node) > 15:
                node = median_node

        self.current_node = node
        return node, score

    def is_junction(self, current: int, next_node: int) -> bool:
        return abs(next_node - current) > self.JUNCTION_THRESHOLD

    def run(self, max_steps: int = 500) -> bool:
        """Run navigation loop. Returns True if target reached."""
        print(f"\n{'='*60}")
        print(f"NAVIGATING TO NODE {self.target_node}")
        print(f"Target frame: {self.graph.frame_paths[self.target_node]}")
        print(f"{'='*60}\n")

        for step in range(max_steps):
            img = self.get_frame()
            if img is None:
                time.sleep(0.2)
                continue

            # 1. Localize
            node, score = self.localize(img)
            distance = abs(node - self.target_node)

            # 2. Check arrival
            if distance <= self.ARRIVAL_DISTANCE and score > self.ARRIVAL_SCORE:
                target_path = self.graph.frame_paths[self.target_node]
                aligned, ang, n = self.dir_est.is_aligned(img, target_path)
                if n > 20:
                    print(f"\n*** ARRIVED! Node {node}, matches={n} ***")
                    self.stop()
                    self._save_log()
                    return True

            # 3. Plan path
            path = self.graph.shortest_path(node, self.target_node)
            if path is None or len(path) < 2:
                print(f"[{step:3d}] Node {node:4d} | No path — rotating")
                self.send_control(0, self.ROTATION_SPEED)
                time.sleep(self.CONTROL_INTERVAL)
                continue

            # 4. Pick waypoint
            lookahead = min(self.LOOKAHEAD, len(path) - 1)
            next_node = path[lookahead]

            # 5. Stuck detection
            if len(self.log) >= self.STUCK_THRESHOLD:
                recent = [l["node"] for l in self.log[-self.STUCK_THRESHOLD:]]
                if max(recent) - min(recent) <= 1:
                    print(f"[{step:3d}] STUCK! Rotating 90°...")
                    self.send_control(0, 0.5)
                    time.sleep(2.0)
                    self.stop()
                    time.sleep(0.5)
                    self.current_node = None  # force global relocalization
                    continue

            # 6. Control
            if self.is_junction(node, next_node):
                target_frame = self.graph.frame_paths[next_node]
                angular, n_matches = self.dir_est.estimate_direction(img, target_frame)

                if n_matches < 5:
                    # Try further lookahead
                    alt = path[min(lookahead + 2, len(path) - 1)]
                    target_frame = self.graph.frame_paths[alt]
                    angular, n_matches = self.dir_est.estimate_direction(img, target_frame)

                aligned = abs(angular) < self.ALIGN_THRESHOLD
                linear = self.JUNCTION_SPEED if aligned else 0.0
                action = f"JUNCTION {'ALIGNED' if aligned else 'TURNING'} ang={angular:.2f} m={n_matches}"
            else:
                # Straight section
                if step % 5 == 0:
                    target_frame = self.graph.frame_paths[next_node]
                    angular, n_matches = self.dir_est.estimate_direction(img, target_frame)
                    if n_matches < 5:
                        angular = 0.0
                    angular *= 0.3
                else:
                    angular = 0.0

                linear = self.JUNCTION_SPEED if distance < 10 else self.FORWARD_SPEED
                action = f"FORWARD speed={linear:.2f} correction={angular:.2f}"

            self.send_control(linear, angular)

            print(f"[{step:3d}] Node {node:4d} score={score:.3f} "
                  f"dist={distance:3d} → {action}")

            self.log.append({
                "step": step, "node": node, "score": round(score, 3),
                "distance": distance, "next_node": next_node,
                "is_junction": self.is_junction(node, next_node),
            })

            time.sleep(self.CONTROL_INTERVAL)

        print("\nMax steps reached.")
        self.stop()
        self._save_log()
        return False

    def _save_log(self, path: str = "nav_log.csv"):
        if not self.log:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.log[0].keys())
            writer.writeheader()
            writer.writerows(self.log)
        print(f"Log saved to {path}")

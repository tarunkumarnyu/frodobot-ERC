"""ViNT-based navigation — goal-conditioned visual navigation transformer.

Give it a goal image and it drives there. No graph, no localization, no BFS.
The model directly outputs waypoints from (context_frames, goal_image).
"""

from __future__ import annotations

import base64
import io
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import yaml
from PIL import Image

# Add ViNT training code to path
VINT_ROOT = Path(__file__).resolve().parents[1]
VINT_TRAIN = None

# Search for visualnav-transformer in common locations
for candidate in [
    Path.home() / "visualnav-transformer" / "train",
    VINT_ROOT / "third_party" / "visualnav-transformer" / "train",
    VINT_ROOT.parent / "visualnav-transformer" / "train",
]:
    if candidate.exists():
        VINT_TRAIN = candidate
        break

if VINT_TRAIN and str(VINT_TRAIN) not in sys.path:
    sys.path.insert(0, str(VINT_TRAIN))


class ViNTNavigator:
    """End-to-end goal-conditioned navigation using ViNT."""

    def __init__(
        self,
        weights_path: str,
        config_path: str,
        sdk_url: str = "http://localhost:8000",
        device: str = "auto",
        max_v: float = 0.3,
        max_w: float = 0.4,
        rate: float = 4.0,
        waypoint_index: int = 2,
        close_threshold: float = 3.0,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.sdk_url = sdk_url
        self.max_v = max_v
        self.max_w = max_w
        self.rate = rate
        self.waypoint_index = waypoint_index
        self.close_threshold = close_threshold

        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.context_size = self.config["context_size"]
        self.image_size = self.config["image_size"]  # [width, height]
        self.normalize = self.config.get("normalize", True)
        self.len_traj_pred = self.config.get("len_traj_pred", 5)

        # Load model
        from vint_train.models.vint.vint import ViNT as ViNTModel
        self.model = ViNTModel(
            context_size=self.context_size,
            len_traj_pred=self.len_traj_pred,
            learn_angle=self.config.get("learn_angle", True),
            obs_encoding_size=self.config.get("obs_encoding_size", 512),
            mha_num_attention_heads=self.config.get("mha_num_attention_heads", 4),
            mha_num_attention_layers=self.config.get("mha_num_attention_layers", 4),
            mha_ff_dim_factor=self.config.get("mha_ff_dim_factor", 4),
        )
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)
        self.model.eval().to(self.device)

        # Context queue
        self.context_queue: deque[Image.Image] = deque(maxlen=self.context_size + 1)

    def get_frame(self) -> Optional[Image.Image]:
        try:
            resp = requests.get(f"{self.sdk_url}/v2/front", timeout=3)
            b64 = resp.json().get("front_frame")
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

    def _transform_img(self, img: Image.Image) -> torch.Tensor:
        """Resize and normalize image for ViNT."""
        img = img.resize((self.image_size[0], self.image_size[1]))
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(img_np).permute(2, 0, 1).float()

    def _build_obs_tensor(self) -> torch.Tensor:
        """Build observation tensor from context queue."""
        tensors = [self._transform_img(img) for img in self.context_queue]
        return torch.cat(tensors, dim=0).unsqueeze(0).to(self.device)

    def _build_goal_tensor(self, goal_img: Image.Image) -> torch.Tensor:
        return self._transform_img(goal_img).unsqueeze(0).to(self.device)

    def _waypoint_to_control(self, waypoint: np.ndarray) -> tuple[float, float]:
        """Convert ViNT waypoint [dx, dy, hx, hy] to (linear, angular)."""
        if self.normalize:
            waypoint[:2] *= (self.max_v / self.rate)

        dx, dy = waypoint[0], waypoint[1]
        eps = 1e-8
        dt = 1.0 / self.rate

        if abs(dx) < eps and abs(dy) < eps:
            linear = 0.0
            angular = float(np.arctan2(waypoint[3], waypoint[2])) / dt
        elif abs(dx) < eps:
            linear = 0.0
            angular = float(np.sign(dy)) * np.pi / (2 * dt)
        else:
            linear = dx / dt
            angular = float(np.arctan(dy / dx)) / dt

        linear = float(np.clip(linear, 0.0, self.max_v))
        angular = float(np.clip(angular, -self.max_w, self.max_w))
        return linear, angular

    def predict(self, goal_img: Image.Image) -> tuple[float, float, float, np.ndarray]:
        """Run one prediction step.

        Returns (linear, angular, distance, all_waypoints).
        """
        if len(self.context_queue) < self.context_size + 1:
            return 0.0, 0.0, float("inf"), np.zeros((self.len_traj_pred, 4))

        obs = self._build_obs_tensor()
        goal = self._build_goal_tensor(goal_img)

        with torch.no_grad():
            dist_pred, waypoints = self.model(obs, goal)

        dist = float(dist_pred.cpu().numpy()[0, 0])
        wp_np = waypoints.cpu().numpy()[0]  # (5, 4)
        chosen = wp_np[self.waypoint_index].copy()
        linear, angular = self._waypoint_to_control(chosen)

        return linear, angular, dist, wp_np

    def navigate_to_image(
        self,
        goal_image_path: str,
        max_steps: int = 300,
        log_interval: int = 5,
    ) -> bool:
        """Navigate to a goal image. Returns True if reached."""
        goal_img = Image.open(goal_image_path).convert("RGB")
        self.context_queue.clear()

        print(f"\n{'='*60}")
        print(f"ViNT NAVIGATION")
        print(f"Goal: {goal_image_path}")
        print(f"Max steps: {max_steps}")
        print(f"{'='*60}\n")

        period = 1.0 / self.rate

        for step in range(max_steps):
            t0 = time.time()

            frame = self.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            self.context_queue.append(frame)

            linear, angular, dist, waypoints = self.predict(goal_img)

            # Check arrival
            if dist < self.close_threshold and len(self.context_queue) >= self.context_size + 1:
                print(f"\n*** ARRIVED! Distance={dist:.2f} ***")
                self.stop()
                return True

            self.send_control(linear, angular)

            dt = (time.time() - t0) * 1000
            if step % log_interval == 0:
                wp = waypoints[self.waypoint_index]
                print(f"[{step:3d}] dist={dist:.2f} cmd=({linear:.2f},{angular:.2f}) "
                      f"wp=({wp[0]:.2f},{wp[1]:.2f}) {dt:.0f}ms")

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

        print("\nMax steps reached.")
        self.stop()
        return False

    def navigate_topomap(
        self,
        topomap_dir: str,
        goal_node: int = -1,
        max_steps: int = 500,
        radius: int = 4,
    ) -> bool:
        """Navigate along a topomap (sequence of images).

        This is the standard ViNT deployment: the robot follows a
        recorded path by matching against nearby topomap images.
        """
        # Load topomap
        img_dir = Path(topomap_dir)
        topomap_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        if not topomap_paths:
            print(f"No images in {topomap_dir}")
            return False

        topomap = [Image.open(p).convert("RGB") for p in topomap_paths]
        n = len(topomap)
        if goal_node < 0:
            goal_node = n - 1

        print(f"\n{'='*60}")
        print(f"ViNT TOPOMAP NAVIGATION")
        print(f"Topomap: {n} images from {topomap_dir}")
        print(f"Goal node: {goal_node}")
        print(f"{'='*60}\n")

        closest_node = 0
        period = 1.0 / self.rate
        self.context_queue.clear()

        for step in range(max_steps):
            t0 = time.time()

            frame = self.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            self.context_queue.append(frame)

            if len(self.context_queue) < self.context_size + 1:
                time.sleep(period)
                continue

            obs = self._build_obs_tensor()

            # Search nearby topomap nodes for best match
            start = max(closest_node - radius, 0)
            end = min(closest_node + radius + 1, goal_node + 1)

            best_dist = float("inf")
            best_wp = None
            best_node = closest_node

            for i in range(start, end):
                goal = self._build_goal_tensor(topomap[i])
                with torch.no_grad():
                    dist_pred, waypoints = self.model(obs, goal)
                d = float(dist_pred.cpu().numpy()[0, 0])
                if d < best_dist:
                    best_dist = d
                    best_wp = waypoints.cpu().numpy()[0]
                    best_node = i

            # Advance closest node
            if best_dist < self.close_threshold:
                closest_node = min(best_node + 1, goal_node)
            else:
                closest_node = best_node

            # Check if reached goal
            if closest_node >= goal_node:
                print(f"\n*** REACHED GOAL NODE {goal_node}! ***")
                self.stop()
                return True

            # Compute control from best waypoint
            chosen = best_wp[self.waypoint_index].copy()
            if self.normalize:
                chosen[:2] *= (self.max_v / self.rate)

            dx, dy = chosen[0], chosen[1]
            eps = 1e-8
            dt = 1.0 / self.rate
            if abs(dx) < eps and abs(dy) < eps:
                linear, angular = 0.0, 0.0
            elif abs(dx) < eps:
                linear, angular = 0.0, float(np.sign(dy)) * np.pi / (2 * dt)
            else:
                linear = dx / dt
                angular = float(np.arctan(dy / dx)) / dt

            linear = float(np.clip(linear, 0.0, self.max_v))
            angular = float(np.clip(angular, -self.max_w, self.max_w))

            self.send_control(linear, angular)

            dt_ms = (time.time() - t0) * 1000
            if step % 5 == 0:
                dist_to_goal = goal_node - closest_node
                print(f"[{step:3d}] node={closest_node}/{goal_node} "
                      f"dist={best_dist:.2f} gap={dist_to_goal} "
                      f"cmd=({linear:.2f},{angular:.2f}) {dt_ms:.0f}ms")

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

        print("\nMax steps reached.")
        self.stop()
        return False

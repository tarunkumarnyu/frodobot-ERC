"""LightGlue-based direction estimation at junctions."""

from __future__ import annotations

import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, numpy_image_to_torch
from PIL import Image


class DirectionEstimator:
    """Uses SuperPoint + LightGlue to estimate relative direction to a target view."""

    def __init__(self, max_keypoints: int = 1024, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        img_np = np.array(pil_img.convert("RGB"))
        return numpy_image_to_torch(img_np).to(self.device)

    def _load_image(self, path: str) -> torch.Tensor:
        return load_image(path).to(self.device)

    def estimate_direction(
        self,
        current_img: Image.Image,
        target_frame_path: str,
    ) -> tuple[float, int]:
        """Estimate angular direction from current view to target view.

        Returns:
            angular: steering command [-1, 1]. Negative = turn right, positive = turn left.
            n_matches: number of keypoint matches (confidence indicator).
        """
        img0 = self._pil_to_tensor(current_img)
        img1 = self._load_image(target_frame_path)

        with torch.no_grad():
            feats0 = self.extractor.extract(img0.unsqueeze(0))
            feats1 = self.extractor.extract(img1.unsqueeze(0))
            matches01 = self.matcher({"image0": feats0, "image1": feats1})

        matches = matches01["matches"][0].cpu().numpy()

        if len(matches) < 5:
            return 0.0, len(matches)

        kp0 = feats0["keypoints"][0].cpu().numpy()
        kp1 = feats1["keypoints"][0].cpu().numpy()

        matched_kp0 = kp0[matches[:, 0]]
        matched_kp1 = kp1[matches[:, 1]]

        img_width = img0.shape[-1]
        center_x = img_width / 2.0

        # Weighted displacement: how much do target keypoints shift vs current
        dx_values = matched_kp1[:, 0] - matched_kp0[:, 0]
        weights = 1.0 - np.abs(matched_kp0[:, 0] - center_x) / center_x
        weights = np.clip(weights, 0.1, 1.0)

        weighted_dx = np.average(dx_values, weights=weights)
        angular = np.clip(weighted_dx / (img_width * 0.3), -1.0, 1.0) * 0.5

        return float(angular), len(matches)

    def is_aligned(self, current_img: Image.Image, target_frame_path: str,
                   threshold: float = 0.15, min_matches: int = 15) -> tuple[bool, float, int]:
        """Check if current view is roughly aligned with target."""
        angular, n_matches = self.estimate_direction(current_img, target_frame_path)
        aligned = abs(angular) < threshold and n_matches > min_matches
        return aligned, angular, n_matches

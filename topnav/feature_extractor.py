"""CosPlace feature extraction for place recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class CosPlaceExtractor:
    """Extracts CosPlace global descriptors for place recognition."""

    def __init__(self, backbone: str = "ResNet18", dim: int = 512,
                 device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = torch.hub.load(
            "gmberton/cosplace", "get_trained_model",
            backbone=backbone, fc_output_dim=dim, trust_repo=True,
        )
        self.model.eval().to(self.device)
        self.dim = dim

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract a single descriptor from a PIL image. Returns (dim,)."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            desc = self.model(tensor)
        return desc.cpu().numpy().flatten()

    def extract_batch(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Extract descriptors for a list of images. Returns (N, dim)."""
        descs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([self.transform(img.convert("RGB")) for img in batch])
            tensors = tensors.to(self.device)
            with torch.no_grad():
                d = self.model(tensors)
            descs.append(d.cpu().numpy())
        return np.concatenate(descs, axis=0)

    def extract_from_path(self, path: Path) -> np.ndarray:
        """Extract descriptor from an image file."""
        return self.extract(Image.open(path))

"""
Feature Normalizer
==================
Lightweight wrapper around a persisted sklearn StandardScaler.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import torch


class FeatureNormalizer:
    """Applies a persisted StandardScaler to torch tensors."""

    def __init__(self, scaler_path: Path) -> None:
        self.scaler_path = Path(scaler_path)
        with open(self.scaler_path, "rb") as f:
            scaler = pickle.load(f)
        self.mean = torch.tensor(scaler.mean_, dtype=torch.float32)
        self.scale = torch.tensor(scaler.scale_, dtype=torch.float32)
        self.var = torch.tensor(getattr(scaler, "var_", self.scale ** 2), dtype=torch.float32)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return self.transform(features)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean.to(features.device)) / self.scale.to(features.device)

    def inverse_transform(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.scale.to(features.device) + self.mean.to(features.device)


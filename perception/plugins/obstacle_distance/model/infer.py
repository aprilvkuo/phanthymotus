"""Public model interface used by both the judgeflow entry point and the
phanthymotus MCP plugin.

The contract the leaderboard is assumed to call (documented in README.md):
    from model.infer import ObstacleDistancePredictor
    pred = ObstacleDistancePredictor()          # loads weights / juicefs
    distance_m = pred.predict(rgb_uint8_hwc)    # np.ndarray (H,W,3), RGB, 0-255

predict() returns the Nearest Obstacle Distance in meters (float). If no
obstacle is detected in the frontal ROI, returns cfg.max_depth (i.e. "clear").
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig, RuntimeConfig
from .model import build_model
from .weights import load_state_dict


class ObstacleDistancePredictor:
    def __init__(
        self,
        cfg: ModelConfig | None = None,
        rt: RuntimeConfig | None = None,
        force_download: bool = False,
    ):
        self.cfg = cfg or ModelConfig()
        self.rt = rt or RuntimeConfig()
        self.device = self.rt.device
        self.model = build_model(self.cfg, pretrained=False)
        sd = load_state_dict(self.rt, force_download=force_download)
        self.trained = sd is not None
        if self.trained:
            self.model.load_state_dict(sd)
        else:
            print("[predictor] WARN: running UNTRAINED baseline — train and upload weights before submission.")
        self.model.to(self.device).eval()

        # Precompute ROI pixel slice.
        H, W = self.cfg.img_size
        self.roi = (
            int(self.cfg.roi_top * H),
            int(self.cfg.roi_bottom * H),
            int(self.cfg.roi_left * W),
            int(self.cfg.roi_right * W),
        )
        m = np.array(self.cfg.mean, dtype=np.float32).reshape(1, 1, 3)
        s = np.array(self.cfg.std, dtype=np.float32).reshape(1, 1, 3)
        self._mean = m
        self._std = s

    @torch.no_grad()
    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        # image_rgb: (H,W,3) uint8
        img = np.ascontiguousarray(image_rgb, dtype=np.float32) / 255.0
        img = (img - self._mean) / self._std
        img = np.transpose(img, (2, 0, 1))  # (3,H,W)
        t = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return F.interpolate(t, size=self.cfg.img_size, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> float:
        """Nearest Obstacle Distance (meters) for a single RGB frame."""
        x = self._preprocess(image_rgb)
        depth = self.model(x)[0, 0].cpu().numpy().astype(np.float32)  # (H,W) meters
        return self._nearest_from_depth(depth)

    def _nearest_from_depth(self, depth: np.ndarray) -> float:
        t, b, l, r = self.roi
        roi = depth[t:b, l:r]
        valid = roi[(roi > self.cfg.min_depth) & (roi < self.cfg.max_depth)]
        if valid.size == 0:
            return float(self.cfg.max_depth)  # no obstacle in path -> clear
        return float(np.percentile(valid, self.cfg.nearest_percentile))

    def count_parameters(self) -> int:
        return self.model.count_parameters()

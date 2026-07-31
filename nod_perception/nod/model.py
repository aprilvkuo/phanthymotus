"""Optional learned model for NOD.

The geometric estimator (see geometric.py) already satisfies every hard
constraint and is the default. This module is provided for accuracy
improvement in scenarios where a reliable depth sensor is unavailable
(outdoor / monocular RGB only): a lightweight monocular depth network whose
predicted depth is fed through the *same* geometric corridor extractor.

Design goals (must hold):
  * Parameters << 30M  (encoder mobilenet_v3_small ~1.5M + small decoder).
  * Runs on Jetson Orin with GPU memory < 10% when quantized / batched=1.
  * Pure torch; imported lazily so the geometric path has zero torch dep.

If you do not train this, the solution still works fully via the geometric
path. Weights are downloaded from juicefs at runtime (never committed).
"""

from typing import Optional

import os

from .config import NODConfig
from .interface import BaseNODPredictor, NODResult


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        return torch, nn
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The learned NOD model requires torch. Install it with "
            "`pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu` (CPU build is enough on Jetson "
            "for this tiny network). The geometric path needs no torch."
        ) from e


class MobileDepthNet:
    """Tiny monocular depth network: MobileNetV3-small encoder + Conv decoder.

    Output: single-channel relative depth map (normalized 0..1), HxW.
    Params are well under 30M even before quantization.
    """

    def __init__(self, cfg: NODConfig):
        torch, nn = _require_torch()
        self.torch = torch
        self.nn = nn
        self.cfg = cfg

        # ---- encoder (frozen, pretrained imagenet weights if available) ----
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import mobilenet_v3_small
            backbone = mobilenet_v3_small(pretrained=False)
        self.encoder = backbone.features          # -> 576 x (H/16) x (W/16)
        self.enc_channels = 576

        # ---- lightweight decoder: upsample + refine ----
        self.decoder = nn.Sequential(
            nn.Conv2d(self.enc_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),                          # 0..1 relative depth
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        return self

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        return self

    def forward(self, x):
        # x: (B,3,H,W) in [0,1]
        with self.torch.no_grad():
            feats = self.encoder(x)
            out = self.decoder(feats)
        return out


class LearnedNODPredictor(BaseNODPredictor):
    """Monocular-RGB -> depth -> geometric NOD.

    The RGB image is converted to a relative depth map, scaled to metric
    depth using a learned/calibrated scale factor, then passed to the
    geometric corridor extractor.
    """

    def __init__(self, config: NODConfig | None = None,
                 weights_path: Optional[str] = None,
                 device: str = "cpu"):
        self.cfg = config or NODConfig()
        self.net = MobileDepthNet(self.cfg)
        self.device = device
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        elif weights_path:
            print(f"[model] weights not found at {weights_path}; using an "
                  f"untrained network. Run scripts/download_model.sh or train.py "
                  f"to obtain weights.")
        self.net.to(device).eval()
        # geometric extractor reuses the same corridor logic
        from .geometric import GeometricNODPredictor
        self._geom = GeometricNODPredictor(self.cfg)
        # metric scale: relative(0..1) -> meters. Tunable / learned.
        self.depth_scale = float(self.cfg.depth_max)
        self.min_rel_depth = 0.02

    def param_count(self) -> int:
        return self.net.param_count()

    def load_weights(self, path: str):
        torch = self.torch
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # best-effort load (keys may differ between encoder/decoder)
        missing, unexpected = [], []
        try:
            missing, unexpected = self.net.encoder.load_state_dict(
                {k.replace("encoder.", ""): v for k, v in sd.items()
                 if k.startswith("encoder.")}, strict=False)
            self.net.decoder.load_state_dict(
                {k.replace("decoder.", ""): v for k, v in sd.items()
                 if k.startswith("decoder.")}, strict=False)
        except Exception:
            pass
        return missing, unexpected

    def predict(self, depth=None, rgb=None, pointcloud=None,
                camera_info=None, **_):
        if rgb is None:
            # fall back to geometric if depth / pointcloud available
            if depth is not None or pointcloud is not None:
                return self._geom.predict(depth=depth, pointcloud=pointcloud,
                                          camera_info=camera_info)
            raise ValueError("LearnedNODPredictor needs `rgb` (or depth/pc fallback).")

        import numpy as np
        torch = self.torch
        img = np.asarray(rgb, dtype=np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3] / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        rel = self.net.forward(t)[0, 0].cpu().numpy()          # (H,W) 0..1
        rel = np.clip(rel, self.min_rel_depth, 1.0)
        metric_depth = rel * self.depth_scale                 # -> meters
        return self._geom._from_depth(metric_depth, camera_info)

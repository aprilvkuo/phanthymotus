"""Lightweight monocular depth / obstacle-distance network.

Architecture:
  MobileNetV3-Small encoder (ImageNet init optional, ~2.5M params)
  + a 4-stage bilinear-upsample depth decoder (a few conv layers, < 1M params)
  -> full-resolution depth map (meters)
  -> nearest-obstacle-distance derived in postprocess (see infer.py)

Total params are well under the 30M hard limit and the network is small
enough to run in real time on a Jetson Orin 16G with TensorRT / FP16.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class _UpConv(nn.Module):
    """Bilinear upsample (x2) followed by a conv + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, activation: bool = True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(self.up(x))))


class ObstacleDistanceModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None, pretrained: bool = False):
        super().__init__()
        self.cfg = cfg or ModelConfig()

        # --- Encoder: MobileNetV3-Small features (stride 16, 576 ch) ---
        if self.cfg.backbone == "mobilenet_v3_small":
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            enc = mobilenet_v3_small(weights=weights)
            self.encoder = enc.features  # Sequential, output 576ch @ stride16
            enc_channels = 576
        else:
            raise ValueError(f"Unsupported backbone: {self.cfg.backbone}")

        # --- Decoder: stride16 -> full res depth map ---
        chs = (enc_channels, *self.cfg.decoder_channels)  # (576,128,64,32,16)
        stages = []
        for i in range(len(self.cfg.decoder_channels)):
            in_ch = chs[i]
            out_ch = chs[i + 1]
            last = i == len(self.cfg.decoder_channels) - 1
            # penultimate stage must still upsample once more to reach full res
            if last:
                # 16 -> depth head: one more upsample + 1ch conv, no BN/act on final
                stages.append(_UpConv(in_ch, out_ch, activation=True))
                stages.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
                stages.append(nn.Conv2d(out_ch, 1, 3, padding=1, bias=True))
            else:
                stages.append(_UpConv(in_ch, out_ch, activation=True))
        self.decoder = nn.Sequential(*stages)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) normalized. Returns depth map (B,1,H,W) in meters."""
        feats = self.encoder(x)
        depth = self.decoder(feats)  # (B,1,H,W)
        # Map to [min_depth, max_depth] with a smooth sigmoid.
        depth = self.cfg.min_depth + (self.cfg.max_depth - self.cfg.min_depth) * torch.sigmoid(depth)
        return depth

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: ModelConfig | None = None, pretrained: bool = False) -> ObstacleDistanceModel:
    return ObstacleDistanceModel(cfg=cfg, pretrained=pretrained)

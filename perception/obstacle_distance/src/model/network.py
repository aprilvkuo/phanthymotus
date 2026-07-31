"""深度估计网络(src/model/network.py)。

DepthNet: MobileNetV3-small 编码器 + 轻量密集深度解码器。
输出 [N,1,H,W] 单通道深度图(米)，已 clamp 到 depth_range。
参数量远小于 5M（满足 <30M 硬约束）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class DepthNet(nn.Module):
    """单目密集深度估计网络。

    Args:
        encoder_name: 仅支持 mobilenet_v3_small(基线)。
        depth_range: (min, max) 深度范围(米)。
        pretrained: 编码器是否加载 ImageNet 预训练权重。
    """

    SUPPORTED_ENCODERS = ("mobilenet_v3_small",)

    def __init__(
        self,
        encoder_name: str = "mobilenet_v3_small",
        depth_range: tuple = (0.1, 30.0),
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if encoder_name not in self.SUPPORTED_ENCODERS:
            raise ValueError(f"不支持的编码器: {encoder_name}")
        self.encoder_name = encoder_name
        self.depth_min = float(depth_range[0])
        self.depth_max = float(depth_range[1])

        backbone = models.mobilenet_v3_small(pretrained=pretrained)
        self.encoder = backbone.features  # [N, 576, H/16, W/16]
        enc_channels = 576

        # 轻量解码器：16x 上采样 + 两级 3x3 卷积 + 1x1 输出单通道深度
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),
            nn.Conv2d(enc_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
        )
        self._init_decoder()

    def _init_decoder(self) -> None:
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """前向。

        Args:
            rgb: [N,3,H,W] 已归一化张量。
        Returns:
            [N,1,H,W] 深度图(米)，clamp 到 [depth_min, depth_max]。
        """
        feat = self.encoder(rgb)
        raw = self.decoder(feat)  # [N,1,H,W]
        # sigmoid 映射到 [0,1]，再线性缩放到深度范围，保证输出物理合理
        depth = self.depth_min + (self.depth_max - self.depth_min) * torch.sigmoid(raw)
        depth = depth.clamp(self.depth_min, self.depth_max)
        return depth

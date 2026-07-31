"""数据集(src/data/dataset.py)。

NODDataset: __getitem__ 返回 (rgb_np uint8 HWC, depth_np float32 HWC米)。
source 可插拔：synthetic(默认) 或 real(占位)。
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    from config import Config
except ImportError:  # pragma: no cover
    from ..config import Config
try:
    from data.synthetic import SyntheticSceneGenerator
except ImportError:  # pragma: no cover
    from .synthetic import SyntheticSceneGenerator

logger = logging.getLogger(__name__)


class NODDataset:
    """NOD 训练/评测数据集(占位基线)。"""

    def __init__(
        self,
        config: Optional[Config] = None,
        source: str = "synthetic",
        split: str = "train",
        length: int = 1000,
    ) -> None:
        self.config = config or Config.default()
        self.source = source
        self.split = split
        self.length = length
        if source == "synthetic":
            self._gen = SyntheticSceneGenerator(
                size=self.config.input_size,
                depth_range=(self.config.depth_min, self.config.depth_max),
                horizon_row=self.config.passable_region.horizon_row,
                corridor_x=tuple(self.config.passable_region.corridor_x),
                seed=self.config.seed,
            )
        elif source == "real":
            # 【占位】接入真实采集数据管线时保持 __getitem__ 契约不变
            raise NotImplementedError(
                "真实数据加载为占位：请接入采集数据管线(相机/标注)，"
                "保持 __getitem__ 返回 (rgb_np uint8 HWC, depth_np float32 HWC米)。"
            )
        else:
            raise ValueError(f"未知数据源: {source}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple:
        """返回 (rgb_np uint8 HWC, depth_np float32 HWC米)。"""
        if self.source == "synthetic":
            # 用 index 派生确定性随机，避免每 epoch 重复
            gen = SyntheticSceneGenerator(
                size=self.config.input_size,
                depth_range=(self.config.depth_min, self.config.depth_max),
                horizon_row=self.config.passable_region.horizon_row,
                corridor_x=tuple(self.config.passable_region.corridor_x),
                seed=self.config.seed + index,
            )
            rgb, depth, _nod_gt = gen.generate()
            return rgb, depth
        raise NotImplementedError(self.source)

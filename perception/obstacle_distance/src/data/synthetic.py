"""合成数据生成(src/data/synthetic.py)。

SyntheticSceneGenerator: 用 numpy/opencv 程序化生成随机几何场景，
得到真值深度图(米)与 RGB。不依赖 Pyrender/OpenGL。
【占位】可替换为真实采集数据；输出契约固定为 (rgb, depth_m, nod_gt)。
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SyntheticSceneGenerator:
    """程序化合成单目前向场景。

    生成：
        - rgb:   [H,W,3] uint8
        - depth_m: [H,W,1] float32，单位米，范围 [depth_min, depth_max]
        - nod_gt: 可通行区域最小深度(米)；无障碍为 +inf
    """

    def __init__(
        self,
        size: int = 224,
        depth_range: Tuple[float, float] = (0.1, 30.0),
        horizon_row: float = 0.55,
        corridor_x: Tuple[float, float] = (0.25, 0.75),
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.size = size
        self.depth_min = float(depth_range[0])
        self.depth_max = float(depth_range[1])
        self.horizon_row = horizon_row
        self.corridor_x = corridor_x
        self.rng = rng or np.random.default_rng(seed)

    # ---------------- 主入口 ----------------
    def generate(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """生成一帧合成场景。

        Returns:
            (rgb uint8 HWC, depth_m float32 HWC, nod_gt float)。
        """
        h = w = self.size
        horizon = int(round(self.horizon_row * h))
        depth = self._ground_depth(h, horizon)  # [H,W]
        rgb = self._sky_ground_rgb(depth, horizon)

        num_obs = int(self.rng.integers(0, 4))  # 0~3 个障碍物
        for _ in range(num_obs):
            depth, rgb = self._add_obstacle(depth, rgb, horizon)

        nod_gt = self._compute_nod_gt(depth, horizon)
        depth = depth.astype(np.float32)[..., None]  # HWC
        rgb = rgb.astype(np.uint8)
        return rgb, depth, nod_gt

    # ---------------- 内部 ----------------
    def _ground_depth(self, h: int, horizon: int) -> np.ndarray:
        depth = np.full((h, h), self.depth_max, dtype=np.float32)
        if horizon >= h:
            return depth
        ys = np.arange(horizon, h)
        # 透视：近(图像底部)深度小，远(地平线)深度大
        t = (ys - horizon) / max(1, (h - 1 - horizon))  # 0@horizon -> 1@bottom
        ramp = self.depth_min + (self.depth_max - self.depth_min) * (1.0 - t)
        depth[horizon:, :] = ramp[:, None]
        return depth

    def _sky_ground_rgb(self, depth: np.ndarray, horizon: int) -> np.ndarray:
        h, w = depth.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        # 天空渐变(上蓝下淡)
        if horizon > 0:
            sky_t = np.linspace(0, 1, horizon)[:, None]
            sky = np.array([135, 206, 235]) * (1 - sky_t) + np.array(
                [200, 220, 235]
            ) * sky_t
            rgb[:horizon, :] = sky[:, None, :]
        # 地面：灰棕，按深度加雾
        if horizon < h:
            fog = np.clip(
                (depth[horizon:, :] - self.depth_min)
                / (self.depth_max - self.depth_min),
                0,
                1,
            )
            ground_base = np.array([110, 100, 90], dtype=np.float32)
            fog_color = np.array([200, 205, 210], dtype=np.float32)
            ground = (
                ground_base[None, None, :] * (1 - fog[:, :, None])
                + fog_color[None, None, :] * fog[:, :, None]
            )
            rgb[horizon:, :] = ground
        return np.clip(rgb, 0, 255)

    def _add_obstacle(
        self, depth: np.ndarray, rgb: np.ndarray, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = depth.shape
        x0 = int(round(self.corridor_x[0] * w))
        x1 = int(round(self.corridor_x[1] * w))
        bw = int(self.rng.integers(w * 0.06, w * 0.20))
        bh = int(self.rng.integers(h * 0.08, h * 0.30))
        if x1 - bw <= x0:
            return depth, rgb
        cx = int(self.rng.integers(x0, x1 - bw))
        if horizon + bh // 2 >= h - bh // 2:
            return depth, rgb
        cy = int(self.rng.integers(horizon + bh // 2, h - bh // 2))
        bx0, bx1 = cx, min(w, cx + bw)
        by0, by1 = cy - bh // 2, min(h, cy + bh // 2)
        if bx1 <= bx0 or by1 <= by0:
            return depth, rgb
        # 障碍物深度(比背后地面更近)
        obs_depth = float(self.rng.uniform(self.depth_min + 0.5, 18.0))
        ys_local = np.arange(by0, by1)
        t = (ys_local - by0) / max(1, (by1 - by0))
        local_depth = obs_depth * (1.0 - 0.3 * t)  # 底比顶近 30%
        patch = depth[by0:by1, bx0:bx1].copy()
        local_depth_map = np.tile(local_depth[:, None], (1, bx1 - bx0))
        depth[by0:by1, bx0:bx1] = np.minimum(patch, local_depth_map)
        # 障碍物颜色 + 明暗(左亮右暗)
        color = self.rng.integers(40, 220, size=3).astype(np.float32)
        shade = np.linspace(1.0, 0.6, bx1 - bx0)[None, :]
        obs_rgb = color[None, None, :] * shade[:, :, None]
        rgb[by0:by1, bx0:bx1] = np.clip(obs_rgb, 0, 255)
        return depth, rgb

    def _compute_nod_gt(self, depth: np.ndarray, horizon: int) -> float:
        h, w = depth.shape
        x0 = int(round(self.corridor_x[0] * w))
        x1 = int(round(self.corridor_x[1] * w))
        mask = np.zeros((h, w), dtype=bool)
        mask[horizon:, x0:x1] = True
        region = depth[mask]
        if region.size == 0:
            return float("inf")
        return float(np.min(region))

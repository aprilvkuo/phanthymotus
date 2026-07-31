"""NOD 模型封装(src/model/nod_model.py)。

NODModel: 干净接口 predict(rgb_np) -> (distance_m, has_obstacle)。
内部：预处理 -> DepthNet -> 可通行区域 mask -> 最小深度 -> clamp。
支持 backend 切换（torch 完整 / onnx 占位 / trt 占位）。
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

try:  # 兼容 `from model...`(src 在路径) 与 `import src`(根在路径) 两种用法
    from config import Config
except ImportError:  # pragma: no cover
    from ..config import Config
try:
    from model.network import DepthNet
except ImportError:  # pragma: no cover
    from .network import DepthNet

logger = logging.getLogger(__name__)

# 后端常量
BACKEND_TORCH = "torch"
BACKEND_ONNX = "onnx"
BACKEND_TRT = "trt"


class NODModel:
    """最近可通行障碍物距离模型。"""

    def __init__(
        self, config: Optional[Config] = None, backend: Optional[str] = None
    ) -> None:
        self.config = config or Config.default()
        self.backend = backend or self.config.backend
        self.device = self.config.device
        self.depth_min = self.config.depth_min
        self.depth_max = self.config.depth_max

        # 最近一次结果缓存（属性暴露）
        self.last_distance: float = float("inf")
        self.last_has_obstacle: bool = False

        self._net = None
        self._onnx_session = None
        self._torch = None

        self._load_backend()

    # ---------------- 后端加载 ----------------
    def _load_backend(self) -> None:
        if self.backend == BACKEND_TORCH:
            self._load_torch()
        elif self.backend == BACKEND_ONNX:
            self._load_onnx()
        elif self.backend == BACKEND_TRT:
            self._load_trt()
        else:
            raise ValueError(f"不支持的后端: {self.backend}")

    def _load_torch(self) -> None:
        import torch

        self._torch = torch
        net = DepthNet(
            encoder_name=self.config.encoder_name,
            depth_range=(self.depth_min, self.depth_max),
            pretrained=False,
        )
        ckpt = self.config.ckpt_path_abs
        if os.path.isfile(ckpt):
            try:
                state = torch.load(ckpt, map_location="cpu")
                net.load_state_dict(state)
                logger.info("已加载权重: %s", ckpt)
            except Exception as e:  # pragma: no cover
                logger.warning("权重加载失败，使用随机初始化: %s", e)
        else:
            logger.warning("未找到权重(%s)，使用随机初始化(占位基线)", ckpt)
        net.eval()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            net.to(self.device)
        self._net = net

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        self._onnx_session = ort.InferenceSession(
            self.config.onnx_path_abs,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def _load_trt(self) -> None:  # pragma: no cover - 仅在 Jetson 可用
        raise NotImplementedError(
            "TensorRT 后端仅在 Jetson 侧构建/运行，请先 build_trt.py 生成 models/model.trt"
        )

    # ---------------- 推理 ----------------
    def _torch_device(self) -> str:
        if self._torch is None:
            return "cpu"
        if self.device.startswith("cuda") and self._torch.cuda.is_available():
            return self.device
        return "cpu"

    def _preprocess(self, rgb_np: np.ndarray):
        try:
            from preprocess import preprocess
        except ImportError:  # pragma: no cover
            from ..preprocess import preprocess

        cfg = self.config
        return preprocess(
            rgb_np,
            size=cfg.input_size,
            mean=cfg.imagenet_mean,
            std=cfg.imagenet_std,
            pad_value=cfg.imagenet_mean,
            to_chw=True,
        )

    def _infer(self, rgb_np: np.ndarray) -> np.ndarray:
        """返回 [H,W] 米 深度图(numpy float32)。"""
        tensor = self._preprocess(rgb_np)  # [1,3,size,size]
        if self.backend == BACKEND_TORCH:
            torch = self._torch
            with torch.no_grad():
                out = self._net(tensor.to(self._torch_device()))
            depth = out[0, 0].cpu().numpy().astype(np.float32)
        elif self.backend == BACKEND_ONNX:
            inp = tensor.numpy() if hasattr(tensor, "numpy") else tensor
            out = self._onnx_session.run(["output_depth"], {"input_rgb": inp})
            depth = out[0][0, 0].astype(np.float32)
        else:
            raise NotImplementedError(self.backend)
        return depth

    # ---------------- NOD 计算 ----------------
    def _passable_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """可通行区域：地平线以下 + 前向走廊带。返回 bool mask [H,W]。"""
        h, w = shape
        horizon = int(round(self.config.passable_region.horizon_row * h))
        x0 = int(round(self.config.passable_region.corridor_x[0] * w))
        x1 = int(round(self.config.passable_region.corridor_x[1] * w))
        mask = np.zeros((h, w), dtype=bool)
        mask[horizon:, x0:x1] = True
        return mask

    def _compute_nod(self, depth_map: np.ndarray) -> Tuple[float, bool]:
        """由深度图计算 NOD。

        无障碍(可通行区域为空，或最小深度达到远裁剪)返回 (+inf, False)；
        严禁返回 0 表示无障碍。
        """
        mask = self._passable_mask(depth_map.shape)
        region = depth_map[mask]
        if region.size == 0:
            return float("inf"), False
        min_depth = float(np.min(region))
        if (not np.isfinite(min_depth)) or min_depth >= (
            self.config.obstacle_distance_threshold
        ):
            return float("inf"), False
        min_depth = float(np.clip(min_depth, self.depth_min, self.depth_max))
        return min_depth, True

    def predict(self, rgb_np: np.ndarray) -> Tuple[float, bool]:
        """预测最近可通行障碍物距离。

        Args:
            rgb_np: RGB / HWC / uint8 图像。
        Returns:
            (distance_m, has_obstacle): 距离(米，无障碍为 +inf)；是否存在障碍物。
        """
        depth_map = self._infer(rgb_np)
        distance, has = self._compute_nod(depth_map)
        self.last_distance = distance
        self.last_has_obstacle = has
        return distance, has

    # 属性暴露
    @property
    def distance_m(self) -> float:
        return self.last_distance

    @property
    def has_obstacle(self) -> bool:
        return self.last_has_obstacle

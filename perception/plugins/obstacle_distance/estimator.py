"""最近障碍物距离估计的场景路由与失败策略。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backends import DepthBackend, DetectorBackend
from .postprocess import (
    center_corridor_distance,
    indoor_distance,
    infer_scene,
    outdoor_distance,
)
from .types import DistanceEstimate, Scene


@dataclass(frozen=True)
class EstimatorConfig:
    """不依赖模型权重的推理配置。"""

    min_distance_m: float = 0.05
    max_distance_m: float = 80.0
    fallback_distance_m: float = 0.5
    outdoor_compensation_m: float = 1.7

    def __post_init__(self) -> None:
        if not 0 < self.min_distance_m < self.max_distance_m:
            raise ValueError("距离边界配置无效")
        if not self.min_distance_m <= self.fallback_distance_m <= self.max_distance_m:
            raise ValueError("降级距离必须位于有效范围内")
        if self.outdoor_compensation_m < 0:
            raise ValueError("保险杠补偿不能为负数")


class ObstacleDistanceEstimator:
    """组合深度模型、检测模型和评测定义。"""

    def __init__(
        self,
        depth_backend: DepthBackend,
        detector_backend: DetectorBackend,
        config: EstimatorConfig | None = None,
    ):
        self._depth_backend = depth_backend
        self._detector_backend = detector_backend
        self._config = config or EstimatorConfig()

    def estimate(
        self,
        image: np.ndarray,
        source_name: str | Path,
        scene: str | Scene | None = None,
    ) -> DistanceEstimate:
        resolved_scene = infer_scene(source_name, scene)
        try:
            depth = self._depth_backend.predict(image, resolved_scene)
            if resolved_scene is Scene.INDOOR:
                distance = indoor_distance(depth)
                return self._result(distance, resolved_scene, confidence=0.7)

            detections = self._detector_backend.detect(image)
            distance = outdoor_distance(
                depth,
                detections,
                compensation_m=self._config.outdoor_compensation_m,
            )
            if distance is not None:
                confidence = max(
                    (detection.confidence for detection in detections),
                    default=0.5,
                )
                return self._result(distance, resolved_scene, confidence=confidence)

            corridor_depth = center_corridor_distance(depth)
            distance = max(
                self._config.min_distance_m,
                corridor_depth - self._config.outdoor_compensation_m,
            )
            return self._result(
                distance,
                resolved_scene,
                confidence=0.25,
                degraded=True,
                reason="检测器未找到有效障碍物，已使用中心通行区深度",
            )
        except Exception as exc:
            return DistanceEstimate(
                distance_m=self._config.fallback_distance_m,
                scene=resolved_scene,
                confidence=0.0,
                degraded=True,
                reason=f"推理降级: {exc}",
            )

    def _result(
        self,
        distance: float,
        scene: Scene,
        *,
        confidence: float,
        degraded: bool = False,
        reason: str = "",
    ) -> DistanceEstimate:
        if not math.isfinite(distance):
            raise ValueError("模型输出非有限距离")
        bounded = min(
            self._config.max_distance_m,
            max(self._config.min_distance_m, float(distance)),
        )
        return DistanceEstimate(
            distance_m=bounded,
            scene=scene,
            confidence=min(1.0, max(0.0, float(confidence))),
            degraded=degraded,
            reason=reason,
        )

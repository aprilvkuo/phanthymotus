"""障碍物距离模块的轻量数据类型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Scene(str, Enum):
    """评测场景类型。"""

    INDOOR = "indoor"
    OUTDOOR = "outdoor"


@dataclass(frozen=True)
class Detection:
    """图像坐标系中的目标检测框。"""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class DistanceEstimate:
    """一次距离估计及其运行状态。"""

    distance_m: float
    scene: Scene
    confidence: float
    degraded: bool = False
    reason: str = ""

"""将 metric depth 和检测结果转换为最近障碍物距离。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import Detection, Scene


_OUTDOOR_CLASSES = {
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction vehicle",
    "debris",
    "motorcycle",
    "person",
    "pushable pullable",
    "traffic cone",
    "truck",
}

_CLASS_ALIASES = {
    "human.pedestrian.adult": "person",
    "movable_object.barrier": "barrier",
    "movable_object.debris": "debris",
    "movable_object.pushable_pullable": "pushable pullable",
    "movable_object.trafficcone": "traffic cone",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.truck": "truck",
}


def infer_scene(source_name: str | Path, override: str | Scene | None = None) -> Scene:
    """根据显式配置或输入格式推断场景。"""

    if override is not None:
        try:
            return override if isinstance(override, Scene) else Scene(override.lower())
        except ValueError as exc:
            raise ValueError(f"无法识别场景配置: {override}") from exc

    suffix = Path(source_name).suffix.lower()
    if suffix == ".png":
        return Scene.INDOOR
    if suffix in {".jpg", ".jpeg"}:
        return Scene.OUTDOOR
    raise ValueError(f"无法识别场景: 不支持的图片格式 {suffix or '<none>'}")


def indoor_distance(
    depth: np.ndarray,
    *,
    percentile: float = 1.0,
    min_depth_m: float = 0.05,
    max_depth_m: float = 100.0,
) -> float:
    """计算中心三分之一、上方五分之八区域的有效深度 P1。"""

    depth_map = _depth_map(depth)
    height, width = depth_map.shape
    x1 = width // 3
    x2 = (2 * width) // 3 + 1
    y2 = (5 * height) // 8 + 1
    values = _valid_values(
        depth_map[0:y2, x1:x2],
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    if values.size == 0:
        raise ValueError("室内 ROI 中没有有效深度")
    return float(np.percentile(values, percentile))


def outdoor_distance(
    depth: np.ndarray,
    detections: Iterable[Detection],
    *,
    compensation_m: float = 0.0,
    confidence_threshold: float = 0.25,
    percentile: float = 5.0,
) -> float | None:
    """在允许类别的检测框内计算最近表面距离。"""

    depth_map = _depth_map(depth)
    distances: list[float] = []
    for detection in detections:
        if detection.confidence < confidence_threshold:
            continue
        if _normalize_class(detection.class_name) not in _OUTDOOR_CLASSES:
            continue
        crop = _inner_box_crop(depth_map, detection)
        values = _valid_values(crop, min_depth_m=0.05, max_depth_m=200.0)
        if values.size == 0:
            continue
        surface_depth = float(np.percentile(values, percentile))
        distances.append(max(0.05, surface_depth - max(0.0, compensation_m)))

    return min(distances) if distances else None


def center_corridor_distance(
    depth: np.ndarray,
    *,
    percentile: float = 5.0,
) -> float:
    """检测器无结果时，使用中心通行区域给出保守距离。"""

    depth_map = _depth_map(depth)
    height, width = depth_map.shape
    crop = depth_map[height // 4 : height, width // 3 : (2 * width) // 3 + 1]
    values = _valid_values(crop, min_depth_m=0.05, max_depth_m=200.0)
    if values.size == 0:
        raise ValueError("中心通行区域中没有有效深度")
    return float(np.percentile(values, percentile))


def _depth_map(depth: np.ndarray) -> np.ndarray:
    depth_map = np.asarray(depth, dtype=np.float32)
    if depth_map.ndim != 2 or not all(size > 0 for size in depth_map.shape):
        raise ValueError("深度图必须是非空二维数组")
    return depth_map


def _valid_values(
    values: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    mask = np.isfinite(flat) & (flat > min_depth_m) & (flat <= max_depth_m)
    return flat[mask]


def _normalize_class(class_name: str) -> str:
    normalized = " ".join(class_name.strip().lower().replace("_", " ").split())
    return _CLASS_ALIASES.get(class_name.strip().lower(), normalized)


def _inner_box_crop(depth: np.ndarray, detection: Detection) -> np.ndarray:
    height, width = depth.shape
    x1 = max(0, min(width, math.floor(detection.x1)))
    y1 = max(0, min(height, math.floor(detection.y1)))
    x2 = max(0, min(width, math.ceil(detection.x2)))
    y2 = max(0, min(height, math.ceil(detection.y2)))
    if x2 <= x1 or y2 <= y1:
        return depth[0:0, 0:0]

    margin_x = int((x2 - x1) * 0.1)
    margin_y = int((y2 - y1) * 0.1)
    inner_x1, inner_x2 = x1 + margin_x, x2 - margin_x
    inner_y1, inner_y2 = y1 + margin_y, y2 - margin_y
    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        return depth[y1:y2, x1:x2]
    return depth[inner_y1:inner_y2, inner_x1:inner_x2]

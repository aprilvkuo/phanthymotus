from pathlib import Path

import numpy as np
import pytest
from plugins.obstacle_distance.postprocess import (
    indoor_distance,
    infer_scene,
    outdoor_distance,
)
from plugins.obstacle_distance.types import Detection, Scene


def test_infer_scene_uses_image_format_and_accepts_override() -> None:
    assert infer_scene(Path("frame.png")) is Scene.INDOOR
    assert infer_scene(Path("frame.jpg")) is Scene.OUTDOOR
    assert infer_scene(Path("frame.jpeg")) is Scene.OUTDOOR
    assert infer_scene(Path("frame.jpg"), "indoor") is Scene.INDOOR


def test_infer_scene_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="无法识别场景"):
        infer_scene(Path("frame.bmp"))


def test_indoor_distance_uses_inclusive_roi_and_filters_invalid_depth() -> None:
    depth = np.full((480, 640), 5.0, dtype=np.float32)
    depth[100, 100] = 0.1
    depth[10, 213] = np.nan
    depth[20, 426] = 1.0
    depth[300, 426] = 2.0
    depth[301, 426] = 0.2

    distance = indoor_distance(depth)

    valid_roi = depth[0:301, 213:427]
    valid_roi = valid_roi[np.isfinite(valid_roi) & (valid_roi > 0.05)]
    assert distance == pytest.approx(float(np.percentile(valid_roi, 1)))
    assert distance > 0.9


def test_outdoor_distance_ignores_invalid_classes_and_uses_nearest_box() -> None:
    depth = np.full((100, 100), 20.0, dtype=np.float32)
    depth[20:80, 20:60] = 7.0
    depth[30:90, 60:95] = 4.0
    detections = [
        Detection("car", 0.9, 20, 20, 60, 80),
        Detection("person", 0.8, 60, 30, 95, 90),
        Detection("traffic light", 0.99, 0, 0, 20, 20),
    ]

    distance = outdoor_distance(depth, detections, compensation_m=1.5)

    assert distance == pytest.approx(2.5)


def test_outdoor_distance_returns_none_without_usable_detection() -> None:
    depth = np.full((20, 20), 5.0, dtype=np.float32)
    detections = [Detection("traffic light", 0.9, 0, 0, 10, 10)]

    assert outdoor_distance(depth, detections) is None

import numpy as np
import pytest

from plugins.obstacle_distance.estimator import (
    EstimatorConfig,
    ObstacleDistanceEstimator,
)
from plugins.obstacle_distance.types import Detection, Scene


class FakeDepthBackend:
    def __init__(self, depth: np.ndarray):
        self.depth = depth
        self.calls: list[Scene] = []

    def predict(self, image: np.ndarray, scene: Scene) -> np.ndarray:
        self.calls.append(scene)
        return self.depth


class FakeDetectorBackend:
    def __init__(self, detections: list[Detection]):
        self.detections = detections
        self.call_count = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.call_count += 1
        return self.detections


def test_estimator_routes_png_to_indoor_without_detector() -> None:
    depth = FakeDepthBackend(np.full((48, 64), 2.5, dtype=np.float32))
    detector = FakeDetectorBackend([])
    estimator = ObstacleDistanceEstimator(depth, detector)

    result = estimator.estimate(np.zeros((48, 64, 3), dtype=np.uint8), "frame.png")

    assert result.scene is Scene.INDOOR
    assert result.distance_m == pytest.approx(2.5)
    assert result.degraded is False
    assert detector.call_count == 0


def test_estimator_uses_detection_and_bumper_compensation_outdoor() -> None:
    depth_map = np.full((40, 60), 8.0, dtype=np.float32)
    depth = FakeDepthBackend(depth_map)
    detector = FakeDetectorBackend([Detection("car", 0.8, 10, 10, 50, 35)])
    estimator = ObstacleDistanceEstimator(
        depth,
        detector,
        EstimatorConfig(outdoor_compensation_m=1.7),
    )

    result = estimator.estimate(np.zeros((40, 60, 3), dtype=np.uint8), "frame.jpg")

    assert result.distance_m == pytest.approx(6.3)
    assert result.confidence == pytest.approx(0.8)
    assert result.degraded is False


def test_estimator_uses_center_corridor_when_detector_has_no_target() -> None:
    depth_map = np.full((40, 60), 9.0, dtype=np.float32)
    depth_map[20:35, 25:35] = 4.0
    estimator = ObstacleDistanceEstimator(
        FakeDepthBackend(depth_map),
        FakeDetectorBackend([]),
        EstimatorConfig(outdoor_compensation_m=1.0),
    )

    result = estimator.estimate(np.zeros((40, 60, 3), dtype=np.uint8), "frame.jpeg")

    assert result.distance_m == pytest.approx(3.0)
    assert result.degraded is True
    assert "检测器未找到" in result.reason


def test_estimator_returns_finite_conservative_value_on_backend_failure() -> None:
    class BrokenDepthBackend:
        def predict(self, image: np.ndarray, scene: Scene) -> np.ndarray:
            raise RuntimeError("engine failed")

    estimator = ObstacleDistanceEstimator(
        BrokenDepthBackend(),
        FakeDetectorBackend([]),
        EstimatorConfig(fallback_distance_m=0.5),
    )

    result = estimator.estimate(np.zeros((20, 20, 3), dtype=np.uint8), "frame.png")

    assert result.distance_m == pytest.approx(0.5)
    assert result.degraded is True
    assert "engine failed" in result.reason

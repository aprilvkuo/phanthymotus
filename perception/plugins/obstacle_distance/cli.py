"""单图最近障碍物距离命令行入口。"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .backends import TransformersDepthBackend, UltralyticsDetectorBackend
from .estimator import EstimatorConfig, ObstacleDistanceEstimator
from .model_store import ensure_model_file
from .types import Scene


_DEFAULT_DETECTOR_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
)


def build_default_estimator() -> ObstacleDistanceEstimator:
    """根据环境变量创建默认零样本估计器。"""

    model_dir = Path(
        os.environ.get("OBSTACLE_MODEL_DIR", "/models/obstacle_distance")
    )
    detector_path = Path(
        os.environ.get(
            "OBSTACLE_DETECTOR_MODEL",
            str(model_dir / "yolov8n.pt"),
        )
    )
    if not detector_path.is_file():
        detector_path = ensure_model_file(
            os.environ.get("OBSTACLE_DETECTOR_MODEL_URL", _DEFAULT_DETECTOR_URL),
            detector_path,
            os.environ.get("OBSTACLE_DETECTOR_MODEL_SHA256") or None,
        )

    depth_references = {
        Scene.INDOOR: os.environ.get(
            "OBSTACLE_DEPTH_INDOOR_MODEL",
            TransformersDepthBackend.DEFAULT_MODEL_REFERENCES[Scene.INDOOR],
        ),
        Scene.OUTDOOR: os.environ.get(
            "OBSTACLE_DEPTH_OUTDOOR_MODEL",
            TransformersDepthBackend.DEFAULT_MODEL_REFERENCES[Scene.OUTDOOR],
        ),
    }
    depth_backend = TransformersDepthBackend(
        depth_references,
        device=os.environ.get("OBSTACLE_DEPTH_DEVICE") or None,
    )
    detector_backend = UltralyticsDetectorBackend(
        str(detector_path),
        confidence=float(os.environ.get("OBSTACLE_DETECTION_CONFIDENCE", "0.25")),
        device=os.environ.get("OBSTACLE_DETECTOR_DEVICE") or None,
    )
    config = EstimatorConfig(
        outdoor_compensation_m=float(
            os.environ.get("OBSTACLE_OUTDOOR_COMPENSATION_M", "1.7")
        ),
        fallback_distance_m=float(
            os.environ.get("OBSTACLE_FALLBACK_DISTANCE_M", "0.5")
        ),
    )
    return ObstacleDistanceEstimator(depth_backend, detector_backend, config)


def predict_distance(
    image_path: str | Path,
    *,
    estimator: ObstacleDistanceEstimator | None = None,
    scene: str | Scene | None = None,
) -> float:
    """读取单张图片并返回单位为米的最近障碍物距离。"""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {path}")
    with Image.open(path) as pil_image:
        rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    active_estimator = estimator or build_default_estimator()
    return float(active_estimator.estimate(bgr, str(path), scene).distance_m)


def main(
    argv: Sequence[str] | None = None,
    *,
    estimator_factory: Callable[[], ObstacleDistanceEstimator] = build_default_estimator,
) -> int:
    parser = argparse.ArgumentParser(description="估计正前方最近障碍物距离")
    parser.add_argument("image", help="PNG、JPG 或 JPEG 图片路径")
    parser.add_argument(
        "--scene",
        choices=[scene.value for scene in Scene],
        help="覆盖按图片格式推断的场景",
    )
    args = parser.parse_args(argv)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            estimator = estimator_factory()
            distance = predict_distance(
                args.image,
                estimator=estimator,
                scene=args.scene,
            )
        print(f"{distance:.6f}")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

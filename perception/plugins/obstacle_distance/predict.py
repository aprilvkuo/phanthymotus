#!/usr/bin/env python3
"""Judgeflow entry point for the Nearest Obstacle Distance model.

This is the stable interface the leaderboard is expected to call. It wraps the
`model` package so the internal implementation can evolve without breaking the
submission contract.

Assumed judgeflow contract (see README.md for the full assumption list):
    import predict
    distance_m = predict.predict(rgb_uint8_hwc)   # np.ndarray (H,W,3) RGB 0-255
    distance_m = predict.predict_from_path("frame.jpg")

CLI:
    python predict.py --image frame.jpg
    python predict.py --image frame.jpg --model-dir /models/obstacle_distance
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

try:
    from model import ModelConfig, ObstacleDistancePredictor, RuntimeConfig
except ImportError:  # when imported as plugins.obstacle_distance.predict
    from .model import ModelConfig, ObstacleDistancePredictor, RuntimeConfig


_PREDICTOR: ObstacleDistancePredictor | None = None


def _get_predictor(model_dir: str | None = None) -> ObstacleDistancePredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        rt = RuntimeConfig()
        if model_dir:
            rt.model_dir = model_dir
        _PREDICTOR = ObstacleDistancePredictor(rt=rt)
    return _PREDICTOR


def predict(image_rgb: np.ndarray, model_dir: str | None = None) -> float:
    """Nearest Obstacle Distance in meters for one RGB frame."""
    return _get_predictor(model_dir).predict(image_rgb)


def predict_from_path(path: str, model_dir: str | None = None) -> float:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return predict(np.asarray(img), model_dir=model_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Nearest Obstacle Distance inference")
    ap.add_argument("--image", required=True, help="path to an RGB image")
    ap.add_argument("--model-dir", default=None, help="directory containing weights")
    ap.add_argument("--json", action="store_true", help="output JSON")
    args = ap.parse_args(argv)

    dist = predict_from_path(args.image, model_dir=args.model_dir)
    if args.json:
        print(json.dumps({"nearest_obstacle_distance_m": dist}))
    else:
        print(f"nearest_obstacle_distance_m = {dist:.3f}")


if __name__ == "__main__":
    main()

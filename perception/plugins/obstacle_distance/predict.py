#!/usr/bin/env python3
"""Judgeflow entry point for the Nearest Obstacle Distance (NOD) model.

This is the stable interface the leaderboard is expected to call. It wraps the
`model` package so the internal implementation can evolve without breaking the
submission contract.

No training needed: the model uses an open-source pretrained monocular depth
backbone (MiDaS small by default). See model/config.py / README.md.

Assumed judgeflow contract (see README.md for the full assumption list):
    import predict
    value = predict.predict(rgb_uint8_hwc)        # np.ndarray (H,W,3) RGB 0-255
    value = predict.predict_from_path("frame.jpg")

Units: relative inverse-depth units by default (metric_mode="relative").
Set metric_mode="pinhole_ground" in model/config.py (with camera params) for
a rule-based meter estimate. The return type is always a float.

CLI:
    python predict.py --image frame.jpg
    python predict.py --image frame.jpg --model-dir /models/obstacle_distance
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    from model import ModelConfig, load_predictor
except ImportError:  # when imported as plugins.obstacle_distance.predict
    from .model import ModelConfig, load_predictor


_PREDICTOR = None


def _default_weights_path(model_dir: str | None) -> str | None:
    if model_dir:
        p = os.path.join(model_dir, "obstacle_distance_backbone.pt")
        if os.path.exists(p):
            return p
    # phanthymotus production mount
    p = os.path.join("/models/obstacle_distance", "obstacle_distance_backbone.pt")
    if os.path.exists(p):
        return p
    return None


def _get_predictor(model_dir: str | None = None):
    global _PREDICTOR
    if _PREDICTOR is None:
        cfg = ModelConfig()
        wp = _default_weights_path(model_dir)
        _PREDICTOR = load_predictor(cfg, weights_path=wp)
    return _PREDICTOR


def predict(image_rgb: np.ndarray, model_dir: str | None = None) -> float:
    """Nearest Obstacle Distance for one RGB frame (see module docstring)."""
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
        print(json.dumps({"nearest_obstacle_distance": dist}))
    else:
        print(f"nearest_obstacle_distance = {dist:.4f}")


if __name__ == "__main__":
    main()

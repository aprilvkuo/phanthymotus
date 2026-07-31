#!/usr/bin/env python3
"""Local self-check for the Nearest Obstacle Distance model.

Validates, WITHOUT the real leaderboard data:
  1. the open-source backbone loads and has < 30M params (hard limit),
  2. a forward pass + predict() runs end-to-end on a real image,
  3. the produced depth map has structure (nearer pixels -> larger inverse
     depth) in the central frontal ROI.

Pass --pinhole to also exercise the rule-based metric (meters) path.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from model import ModelConfig, ObstacleDistancePredictor


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of inference runs")
    ap.add_argument("--image", default="/tmp/od_test/synth.png",
                    help="image to run on (optional)")
    ap.add_argument("--pinhole", action="store_true",
                    help="also exercise metric_mode='pinhole_ground'")
    ap.add_argument("--backbone", default="midas_small")
    args = ap.parse_args(argv)

    cfg = ModelConfig(backbone=args.backbone)
    pred = ObstacleDistancePredictor(cfg=cfg)

    nparams = pred.num_params
    print(f"backbone          : {args.backbone}")
    print(f"parameter count   : {nparams:,}  (limit < 30,000,000 -> "
          f"{'OK' if nparams < 30_000_000 else 'FAIL'})")
    assert nparams < 30_000_000, "model exceeds 30M param limit!"

    from PIL import Image
    img = np.asarray(Image.open(args.image).convert("RGB"))

    import time
    t0 = time.time()
    for _ in range(args.n):
        inv = pred.predict_depth(img)
    dt = (time.time() - t0) / args.n

    H, W = inv.shape
    y0, y1 = int(H * cfg.horizon_row_frac), int(H * cfg.roi_bottom_frac)
    c0, c1 = int(W * (0.5 - cfg.roi_width_frac / 2)), int(W * (0.5 + cfg.roi_width_frac / 2))
    roi = inv[y0:y1, c0:c1]
    print(f"image             : {img.shape}  depth map {inv.shape}")
    print(f"avg infer time    : {dt*1000:.1f} ms/frame (CPU; Jetson+TRT is faster)")
    print(f"ROI inv-depth     : min={roi.min():.3f} mean={roi.mean():.3f} max={roi.max():.3f}")
    print(f"NOD (relative)    : {pred.predict(img):.4f}")
    # Sanity: a real depth map should not be flat.
    assert roi.std() > 1e-3, "depth map looks flat — backbone may have failed"
    print("OK: end-to-end inference + depth structure check passed.")

    if args.pinhole:
        cfg2 = ModelConfig(backbone=args.backbone, metric_mode="pinhole_ground",
                           cam_height_m=0.6, horizon_row_frac=0.45)
        pred2 = ObstacleDistancePredictor(cfg=cfg2)
        print(f"NOD (pinhole m)   : {pred2.predict(img):.3f} m  (rule-based, camera-dependent)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

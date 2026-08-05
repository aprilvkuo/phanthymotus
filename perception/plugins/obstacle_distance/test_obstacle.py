#!/usr/bin/env python3
"""
test_obstacle.py — Quick self-test for the Obstacle-Distance (NOD) plugin.

Models after perception/test_kws.py. Run INSIDE the perception container
(or anywhere with the `model` package importable):

  # 1) auto-generate a synthetic scene and test (no camera needed)
  python test_obstacle.py

  # 2) test on a real front-camera image
  python test_obstacle.py --image /data/frame.jpg

  # 3) rule-based meter estimate (needs camera params in model/config.py)
  python test_obstacle.py --image /data/frame.jpg --mode pinhole_ground

It will:
  1. Load the open-source pretrained depth backbone (MiDaS small by default)
  2. Assert the param-count hard limit (< 30M)
  3. Run depth inference and the central-ROI rule to get the NOD
  4. Print depth-map statistics + the NOD value for a quick sanity check
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Make the sibling `model` package importable whether run as a script or module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from PIL import Image
import numpy as np

from model import ModelConfig, ObstacleDistancePredictor, load_predictor


def synth_scene(path: str) -> str:
    """Write a tiny synthetic 'robot corridor with a near box' scene; return path."""
    import random  # noqa: F401  (kept for future scene randomization)
    W, H = 320, 256
    img = np.ones((H, W, 3), dtype=np.float32) * 0.8  # bright sky/wall
    img[H // 2:, :] = 0.55  # ground band
    d = Image.fromarray((img * 255).astype("uint8"))
    dr = ImageDraw.Draw(d)
    # near obstacle: dark rectangle in lower-center
    dr.rectangle([W // 2 - 40, H - 70, W // 2 + 40, H - 20], fill=(30, 30, 30))
    # a second nearer object on the left
    dr.rectangle([40, H - 55, 90, H - 25], fill=(20, 20, 20))
    d.save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description="Test Obstacle-Distance NOD plugin")
    ap.add_argument("--image", default=None, help="RGB image path (synthesized if omitted)")
    ap.add_argument("--model-dir", default=None, help="weights dir (optional)")
    ap.add_argument("--backbone", default="midas_small", help="midas_small | lite_mono")
    ap.add_argument("--mode", default="relative",
                    help="relative | pinhole_ground (meters)")
    ap.add_argument("--synth-out", default="/tmp/obstacle_synth.png",
                    help="where to write the synthetic scene if --image is omitted")
    args = ap.parse_args()

    if args.image is None:
        import inspect
        global ImageDraw
        from PIL import ImageDraw
        args.image = synth_scene(args.synth_out)
        print(f"[info] no --image given; using synthetic scene: {args.image}")

    cfg = ModelConfig(backbone=args.backbone, metric_mode=args.mode)
    t0 = time.time()
    if args.model_dir:
        wp = os.path.join(args.model_dir, "obstacle_distance_backbone.pt")
        predictor = load_predictor(cfg, weights_path=wp if os.path.exists(wp) else None)
    else:
        predictor = ObstacleDistancePredictor(cfg)
    t_load = time.time() - t0

    # Hard constraint check
    n = predictor.num_params
    print(f"backbone      : {args.backbone}")
    print(f"params        : {n:,}  (< 30M: {n < 30_000_000})")
    if n >= 30_000_000:
        print("ERROR: param count exceeds the 30M hard limit!")
        sys.exit(1)

    img = np.asarray(Image.open(args.image).convert("RGB"))
    print(f"input image   : {args.image}  shape={img.shape}")

    # Depth map
    t1 = time.time()
    inv = predictor.predict_depth(img)
    t_inf = time.time() - t1
    print(f"depth map     : {inv.shape}  min/mean/max = "
          f"{inv.min():.2f}/{inv.mean():.2f}/{inv.max():.2f}  std={inv.std():.2f}")

    # NOD
    t2 = time.time()
    nod = predictor.predict(img)
    t_pred = time.time() - t2
    unit = "m" if args.mode == "pinhole_ground" else "rel(inv-depth)"
    print(f"NOD ({unit})    : {nod:.4f}")
    print(f"timing        : load={t_load:.2f}s  depth={t_inf*1000:.1f}ms  "
          f"nod={t_pred*1000:.1f}ms")

    # Sanity: a structured depth map should have non-trivial variation in the ROI
    H, W = inv.shape
    y0, y1 = int(H * 0.45), int(H * 0.92)
    c0, c1 = int(W * 0.25), int(W * 0.75)
    roi = inv[y0:y1, c0:c1]
    if roi.std() < 1e-3:
        print("WARN: ROI depth is nearly flat — backbone may be random / mis-loaded.")
        sys.exit(2)
    print("\nOK: model loaded, depth structured, NOD computed.")
    print("Example judgeflow call:")
    print(f"  import predict; v = predict.predict_from_path('{args.image}')")


if __name__ == "__main__":
    main()

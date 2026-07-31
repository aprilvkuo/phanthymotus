#!/usr/bin/env python3
"""Local sanity benchmark for the metric pipeline.

We do NOT have the real leaderboard data here, so this script validates that:
  1. the model builds and has < 30M params,
  2. a forward pass + predict() runs end-to-end,
  3. the Nearest-Obstacle-Distance metric (MAE / RMSE) computes correctly
     against a synthetic ground truth.

Replace `make_synthetic()` with a real data loader (depth + frontal-obstacle
label) to turn this into the actual training/eval harness. The metric math is
identical to what the leaderboard should report.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from model import ModelConfig, ObstacleDistancePredictor


def make_synthetic(rng: np.random.Generator, h: int = 256, w: int = 320):
    """Synthetic RGB + ground-truth nearest distance.

    We put a bright 'obstacle' blob in the central frontal ROI at a random
    distance, and a far background everywhere else. The true nearest distance
    is exactly the value we embed.
    """
    img = np.full((h, w, 3), 120, dtype=np.uint8)          # grey background
    true_dist = float(rng.uniform(0.3, 8.0))
    # closer obstacle -> larger, brighter blob (a crude stand-in for real depth)
    size = int(40 * (1.0 / true_dist))
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - size), min(h, cy + size)
    x0, x1 = max(0, cx - size), min(w, cx + size)
    img[y0:y1, x0:x1] = rng.integers(200, 255, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
    return img, true_dist


def metric(preds: list[float], gts: list[float]):
    preds = np.asarray(preds, dtype=np.float64)
    gts = np.asarray(gts, dtype=np.float64)
    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    return {"mae_m": mae, "rmse_m": rmse, "n": int(len(preds))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="number of synthetic samples")
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    cfg = ModelConfig()
    pred = ObstacleDistancePredictor(cfg=cfg, rt=None)
    if args.model_dir:
        from model import RuntimeConfig
        pred = ObstacleDistancePredictor(cfg=cfg, rt=RuntimeConfig())
        pred.rt.model_dir = args.model_dir

    print(f"parameter count = {pred.count_parameters():,}  (limit < 30,000,000)")
    assert pred.count_parameters() < 30_000_000, "model exceeds 30M param limit!"

    preds, gts = [], []
    for _ in range(args.n):
        img, gt = make_synthetic(rng, *cfg.img_size)
        preds.append(pred.predict(img))
        gts.append(gt)

    res = metric(preds, gts)
    print("synthetic benchmark (UNTESTED baseline — expects trained weights for real numbers):")
    print(f"  {res}")
    print("OK: end-to-end inference + metric pipeline run successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

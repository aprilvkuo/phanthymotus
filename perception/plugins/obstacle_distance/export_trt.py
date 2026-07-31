#!/usr/bin/env python3
"""Export the model to ONNX and (on Jetson) build a TensorRT engine.

Why: the requirement caps GPU usage at < 10% on Jetson Orin 16G. Running the
PyTorch model through TensorRT (FP16) cuts latency and GPU occupancy sharply,
easily meeting the real-time + low-GPU budget.

Usage:
    python export_trt.py --weights /models/obstacle_distance/obstacle_distance.pt \
        --out obstacle_distance.onnx
Then on the Jetson box:
    trtexec --onnx=obstacle_distance.onnx --saveEngine=obstacle_distance.engine \
        --fp16 --workspace-size=1073741824
The engine is loaded at runtime the same way weights.py loads .pt (extend
load_state_dict to accept .engine). Keeping it out of the repo (it is downloaded
from juicefs) respects the no->1MB-committed-file rule.
"""
from __future__ import annotations

import argparse

import torch

from model import ModelConfig, ObstacleDistancePredictor, RuntimeConfig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, nargs=2, default=None,
                    help="H W (default from ModelConfig)")
    args = ap.parse_args(argv)

    cfg = ModelConfig()
    if args.img_size:
        cfg.img_size = tuple(args.img_size)
    rt = RuntimeConfig()
    if args.weights:
        rt.model_dir = args.weights
    pred = ObstacleDistancePredictor(cfg=cfg, rt=rt)

    dummy = torch.randn(1, 3, *cfg.img_size[::-1])  # (B,3,W,H)
    torch.onnx.export(
        pred.model,
        dummy,
        args.out,
        input_names=["image"],
        output_names=["depth"],
        dynamic_axes={"image": {0: "batch"}, "depth": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported ONNX -> {args.out}")


if __name__ == "__main__":
    main()

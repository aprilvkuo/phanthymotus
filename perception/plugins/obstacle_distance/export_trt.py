#!/usr/bin/env python3
"""Export the open-source depth backbone to ONNX (and, on Jetson, to TensorRT).

Why: the requirement caps GPU usage at < 10% on Jetson Orin 16G. Running the
backbone through TensorRT (FP16) cuts latency and GPU occupancy sharply, easily
meeting the real-time + low-GPU budget. The post-processing (ROI nearest
distance) is cheap and stays in PyTorch.

Usage:
    python export_trt.py --out obstacle_distance.onnx
Then on the Jetson box:
    trtexec --onnx=obstacle_distance.onnx --saveEngine=obstacle_distance.engine \\
        --fp16 --workspace-size=1073741824
The engine is loaded at runtime (extend load_predictor to accept .engine). Keep
engine files out of the repo; download them from juicefs at runtime to respect
the no->1MB-committed-file rule.
"""
from __future__ import annotations

import argparse

import torch

from model import ModelConfig, MonocularDepthBackbone


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="midas_small")
    ap.add_argument("--size", type=int, default=384,
                    help="square input size the backbone expects")
    args = ap.parse_args(argv)

    cfg = ModelConfig(backbone=args.backbone)
    backbone = MonocularDepthBackbone(cfg)
    model = backbone.model.eval()

    dummy = torch.randn(1, 3, args.size, args.size)  # MiDaS small input is 3x384x384, [-1,1]
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["image"],
        output_names=["inv_depth"],
        dynamic_axes={"image": {0: "batch"}, "inv_depth": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported ONNX -> {args.out}")


if __name__ == "__main__":
    main()

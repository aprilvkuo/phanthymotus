"""导出 ONNX(src/export_onnx.py)。

固定输入 [1,3,224,224] 名 input_rgb；输出 [1,1,224,224] 名 output_depth。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# 将 src 加入路径
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from config import Config
    from model.network import DepthNet
except ImportError:  # pragma: no cover
    from .config import Config
    from .model.network import DepthNet

logger = logging.getLogger(__name__)


def export(config: Config) -> str:
    import torch

    model = DepthNet(
        encoder_name=config.encoder_name,
        depth_range=(config.depth_min, config.depth_max),
        pretrained=False,
    )
    ckpt = config.ckpt_path_abs
    if os.path.isfile(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, config.input_size, config.input_size)
    os.makedirs(os.path.dirname(config.onnx_path_abs) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        config.onnx_path_abs,
        input_names=["input_rgb"],
        output_names=["output_depth"],
        opset_version=13,
        dynamic_axes=None,
    )
    logger.info("已导出 ONNX: %s", config.onnx_path_abs)
    return config.onnx_path_abs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt_path", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config) if args.config else Config.default()
    cfg.override_from_args(args)
    export(cfg)


if __name__ == "__main__":
    main()

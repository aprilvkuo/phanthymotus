"""构建 TensorRT 引擎(src/build_trt.py)。

从 ONNX 构建 FP16 引擎并序列化 models/model.trt。
【仅在 Jetson 侧运行】：依赖 tensorrt / CUDA。
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
except ImportError:  # pragma: no cover
    from .config import Config

logger = logging.getLogger(__name__)


def build(config: Config) -> str:
    import tensorrt as trt

    onnx_path = config.onnx_path_abs
    trt_path = config.trt_path_abs
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"未找到 ONNX: {onnx_path}，请先 export_onnx.py")

    logger.info("构建 TensorRT(FP16) 引擎: %s", trt_path)
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: %s", parser.get_error(i))
        raise RuntimeError("ONNX 解析失败")

    config_trt = builder.create_builder_config()
    config_trt.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_serialized_network(network, config_trt)
    if engine is None:
        raise RuntimeError("TensorRT 引擎构建失败")

    os.makedirs(os.path.dirname(trt_path) or ".", exist_ok=True)
    with open(trt_path, "wb") as f:
        f.write(engine)
    logger.info("已保存 TRT 引擎: %s", trt_path)
    return trt_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config) if args.config else Config.default()
    build(cfg)


if __name__ == "__main__":
    main()

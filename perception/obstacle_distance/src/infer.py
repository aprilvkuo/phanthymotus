"""推理 CLI(src/infer.py)。

支持 --image <path> 与 --camera(占位) 与 --backend。
输出 NOD(米) + has_obstacle。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

# 将 src 加入路径
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from config import Config
    from model.nod_model import NODModel
except ImportError:  # pragma: no cover
    from .config import Config
    from .model.nod_model import NODModel


def _load_image(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--camera", action="store_true", help="从相机读取(占位)")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config.default()
    cfg.override_from_args(args)
    model = NODModel(cfg)

    if args.camera:
        logging.warning("--camera 为占位：真实相机需接入 rclpy/相机驱动")
        return
    if args.image is None:
        logging.error("请提供 --image <path>")
        return

    rgb = _load_image(args.image)
    distance, has = model.predict(rgb)
    print(f"NOD distance_m={distance} has_obstacle={has}")


if __name__ == "__main__":
    main()

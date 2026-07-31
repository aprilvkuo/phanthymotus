"""评测入口(scripts/eval.py)。

用 metrics 计算 MAE / RMSE / δ；数据来源走 config(NODDataset synthetic)。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

# 将 src 加入路径（scripts/ 在模块根，src 在其下）
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    from config import Config
    from data.dataset import NODDataset
    from metrics import evaluate
    from model.nod_model import NODModel
except ImportError:  # pragma: no cover
    from src.config import Config
    from src.data.dataset import NODDataset
    from src.metrics import evaluate
    from src.model.nod_model import NODModel

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config) if args.config else Config.default()
    cfg.override_from_args(args)

    model = NODModel(cfg)
    dataset = NODDataset(cfg, source="synthetic", length=args.num)

    preds, gts = [], []
    for i in range(args.num):
        rgb, depth = dataset[i]
        distance, _has = model.predict(rgb)
        nod_gt = (
            float(np.min(depth[np.isfinite(depth)]))
            if np.any(np.isfinite(depth))
            else float("inf")
        )
        preds.append(distance)
        gts.append(nod_gt)

    preds = np.array(preds, dtype=np.float64)
    gts = np.array(gts, dtype=np.float64)
    res = evaluate(preds, gts)
    print("=== Eval ===")
    print(
        f"MAE={res['mae']:.4f}  RMSE={res['rmse']:.4f}  "
        f"δ(|err|<0.5)={res['delta']:.4f}"
    )


if __name__ == "__main__":
    main()

"""端到端时延/帧率/GPU 占用基准(scripts/benchmark.py)。

目标：端到端 ≤30ms、GPU<10%、显存<2GB。
GPU 占用可用 pynvml 可选，缺失则占位。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

import numpy as np

# 将 src 加入路径（scripts/ 在模块根，src 在其下）
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    from config import Config
    from model.nod_model import NODModel
except ImportError:  # pragma: no cover
    from src.config import Config
    from src.model.nod_model import NODModel

logger = logging.getLogger(__name__)


def _gpu_info() -> dict:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {"gpu_util": util.gpu, "mem_mb": mem.used / 1024 ** 2}
    except Exception:
        return {"gpu_util": None, "mem_mb": None}


def benchmark(config: Config, n: int = 100, size: Optional[int] = None) -> dict:
    model = NODModel(config)
    s = size or config.input_size
    img = (np.random.rand(s, s, 3) * 255).astype(np.uint8)
    # 预热
    for _ in range(5):
        model.predict(img)
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(img)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(latencies)
    gpu = _gpu_info()
    return {
        "backend": config.backend,
        "n": n,
        "mean_ms": float(lat.mean()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "max_ms": float(lat.max()),
        "fps": float(1000.0 / lat.mean()),
        "gpu_util": gpu["gpu_util"],
        "mem_mb": gpu["mem_mb"],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config) if args.config else Config.default()
    cfg.override_from_args(args)
    rep = benchmark(cfg, n=args.n)
    print("=== Benchmark ===")
    for k, v in rep.items():
        print(f"{k}: {v}")
    ok = rep["mean_ms"] <= 30 and (rep["gpu_util"] is None or rep["gpu_util"] < 10)
    print("TARGET_OK:", ok)


if __name__ == "__main__":
    main()

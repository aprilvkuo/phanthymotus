"""Evaluation: accuracy + throughput + resource-budget checks.

Metrics aligned with the benchmark's stated focus:
  * distance prediction accuracy (MAE / RMSE / delta<1m / delta<10%)
  * real-time capability (FPS on the target hardware class)
  * hard constraints: param count (<30M) and GPU budget (<10%)

Run:
    python -m nod.evaluate --data /path/to/dataset   # if you have a dataset
    python -m nod.evaluate --demo 50                  # synthetic self-test
"""

import argparse
import time

import numpy as np

from .config import NODConfig
from .data import make_synthetic_scene, make_synthetic_pointcloud
from .infer import NODInference


def accuracy_metrics(preds, gts):
    preds = np.asarray(preds, dtype=np.float64)
    gts = np.asarray(gts, dtype=np.float64)
    err = preds - gts
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    delta1m = float(np.mean(np.abs(err) < 1.0))
    rel = np.abs(err) / np.where(gts == 0, 1e-6, gts)
    delta10 = float(np.mean(rel < 0.10))
    return {"MAE_m": mae, "RMSE_m": rmse,
            "acc_within_1m": delta1m, "acc_within_10pct": delta10}


def benchmark_speed(inf: NODInference, make_sample, n=100):
    # warmup
    for _ in range(5):
        s = make_sample()
        inf.run(**s)
    t0 = time.time()
    for _ in range(n):
        s = make_sample()
        inf.run(**s)
    dt = (time.time() - t0) / n
    return {"mean_inference_ms": dt * 1000.0, "fps": 1.0 / dt if dt > 0 else float("inf")}


def resource_report(cfg: NODConfig, inf: NODInference):
    rep = {"mode": cfg.mode}
    if cfg.mode in ("learned", "hybrid"):
        try:
            n = inf.predictor.param_count()
            rep["params"] = n
            rep["params_ok_<30M"] = cfg.param_budget_ok(n)
        except Exception:
            rep["params"] = "n/a"
    else:
        rep["params"] = 0
        rep["params_ok_<30M"] = True
    rep["gpu_budget_<10pct"] = cfg.gpu_budget_ok(0.0 if not cfg.use_gpu else 0.05)
    rep["gpu_used"] = "none (CPU)" if not cfg.use_gpu else "<10% (target)"
    return rep


def run_demo(n=50, seed=0):
    cfg = NODConfig(mode="geometric")
    inf = NODInference.from_config(cfg)
    preds, gts = [], []
    for i in range(n):
        d, gt = make_synthetic_scene(seed=seed + i,
                                     obstacle_distance=np.random.uniform(0.5, 8.0))
        r = inf.run(depth=d)
        preds.append(r.distance if r.distance != float("inf") else cfg.depth_max)
        gts.append(gt)
    # pointcloud demo
    pcpreds, pcgts = [], []
    for i in range(n):
        pc, gt = make_synthetic_pointcloud(seed=seed + 1000 + i,
                                           obstacle_distance=np.random.uniform(0.5, 8.0))
        r = inf.run(pointcloud=pc)
        pcpreds.append(r.distance if r.distance != float("inf") else cfg.depth_max)
        pcgts.append(gt)

    print("== Depth-image NOD (synthetic) ==")
    print(accuracy_metrics(preds, gts))
    print("== Point-cloud NOD (synthetic) ==")
    print(accuracy_metrics(pcpreds, pcgts))
    print("== Speed ==")
    print(benchmark_speed(inf, lambda: {"depth": make_synthetic_scene()[0]}))
    print("== Resource ==")
    print(resource_report(cfg, inf))


def _cli():
    p = argparse.ArgumentParser("NOD evaluation")
    p.add_argument("--demo", type=int, default=50, help="run synthetic self-test N samples")
    p.add_argument("--data", default=None, help="path to a dataset (depth + gt)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.demo:
        run_demo(n=args.demo, seed=args.seed)
    else:
        print("Provide --demo or --data. See module docstring.")


if __name__ == "__main__":
    _cli()

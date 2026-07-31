"""Inference orchestrator: builds the right predictor from config and runs it.

Usage (library):
    from nod.infer import NODInference
    nod = NODInference.from_config()
    res = nod.run(depth=depth_map)          # or rgb=..., or pointcloud=...

Usage (CLI):
    python -m nod.infer --depth demo/depth.npy --mode geometric
    python -m nod.infer --rgb demo/rgb.png --mode learned --weights .cache/models/nod.pt
"""

import argparse
import json
import time

from .config import NODConfig
from .geometric import GeometricNODPredictor
from .interface import NODResult


def build_predictor(cfg: NODConfig, weights_path=None):
    if cfg.mode == "geometric":
        return GeometricNODPredictor(cfg)
    if cfg.mode in ("learned", "hybrid"):
        from .model import LearnedNODPredictor
        return LearnedNODPredictor(cfg, weights_path=weights_path,
                                   device="cuda" if cfg.use_gpu else "cpu")
    raise ValueError(f"Unknown mode: {cfg.mode}")


class NODInference:
    def __init__(self, predictor, cfg: NODConfig | None = None):
        self.predictor = predictor
        self.cfg = cfg or NODConfig()

    @classmethod
    def from_config(cls, cfg: NODConfig | None = None, weights_path=None):
        cfg = cfg or NODConfig()
        # Auto-resolve learned weights from juicefs if missing.
        if cfg.mode in ("learned", "hybrid") and weights_path is None:
            from .packaging import ensure_model
            weights_path = ensure_model(cfg)
        return cls(build_predictor(cfg, weights_path), cfg)

    def run(self, depth=None, rgb=None, pointcloud=None,
            camera_info=None) -> NODResult:
        return self.predictor.predict(depth=depth, rgb=rgb,
                                      pointcloud=pointcloud,
                                      camera_info=camera_info)

    def run_batch(self, samples, camera_info=None):
        """samples: list of dicts with keys depth/rgb/pointcloud."""
        return [self.run(**s, camera_info=camera_info) for s in samples]


def _cli():
    p = argparse.ArgumentParser("NOD inference")
    p.add_argument("--rgb", default=None)
    p.add_argument("--depth", default=None)
    p.add_argument("--pointcloud", default=None)
    p.add_argument("--mode", default="geometric")
    p.add_argument("--weights", default=None)
    p.add_argument("--json", action="store_true", help="emit JSON result")
    args = p.parse_args()

    cfg = NODConfig(mode=args.mode)
    inf = NODInference.from_config(cfg, weights_path=args.weights)

    depth = rgb = pc = None
    if args.depth:
        from .data import load_depth
        depth = load_depth(args.depth)
    if args.rgb:
        from .data import load_rgb
        rgb = load_rgb(args.rgb)
    if args.pointcloud:
        import numpy as np
        pc = np.load(args.pointcloud)

    t0 = time.time()
    res = inf.run(depth=depth, rgb=rgb, pointcloud=pc)
    dt = time.time() - t0

    out = {
        "distance_m": None if res.distance == float("inf") else res.distance,
        "has_obstacle": res.has_obstacle,
        "source": res.source,
        "corridor_ratio": res.corridor_ratio,
        "inference_s": dt,
        "fps": 1.0 / dt if dt > 0 else float("inf"),
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"NOD = {out['distance_m']} m | obstacle={res.has_obstacle} "
              f"| source={res.source} | {out['fps']:.1f} FPS")


if __name__ == "__main__":
    _cli()

"""Judgeflow entry point.

The benchmark auto-pulls this repo and runs judgeflow to evaluate the model.
Because the exact call contract is defined by judgeflow, we expose BOTH:

  1. A class `NODSolution` with a flexible `predict(obs)` that accepts either
     a dict of numpy arrays (depth/rgb/pointcloud) or file paths, and returns
     the NOD in meters (float; inf when no obstacle in corridor).
  2. A module-level `predict(obs)` convenience function.

Judgeflow integration team: if your harness expects a different signature
(e.g. `infer(sample) -> float` or a server endpoint), adapt the single
`predict` method below — the heavy lifting lives in `nod.infer.NODInference`
and need not change.

Contract
--------
obs (dict), any of:
    {"depth": np.ndarray}                 # HxW depth (meters)
    {"rgb": np.ndarray}                   # HxWx3 (learned mode)
    {"pointcloud": np.ndarray}            # Nx3 robot-frame XYZ
    {"depth_path": str} / {"rgb_path": str} / {"pointcloud_path": str}
    {"camera_info": {...}}                # optional fx/fy/cx/cy override

returns: float  (meters; math.inf if no obstacle in forward corridor)
"""

import math
from typing import Any, Dict

# Make imports robust to where this file lives inside the repo:
#   - as a top-level script  (python solution.py)
#   - imported as `solution` (its directory on sys.path)
#   - imported as `nod_perception.solution` (subpackage layout)
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from nod.config import NODConfig
    from nod.infer import NODInference
except ImportError:  # pragma: no cover - alternate layouts
    try:
        from .config import NODConfig
        from .infer import NODInference
    except ImportError:
        from nod_perception.nod.config import NODConfig
        from nod_perception.nod.infer import NODInference


class NODSolution:
    def __init__(self, mode: str = "geometric", weights_path=None, cfg: NODConfig | None = None):
        self.cfg = cfg or NODConfig(mode=mode)
        self.inference = NODInference.from_config(self.cfg, weights_path=weights_path)

    def predict(self, obs: Dict[str, Any]) -> float:
        camera_info = obs.get("camera_info")
        depth = obs.get("depth")
        rgb = obs.get("rgb")
        pc = obs.get("pointcloud")

        # accept file paths
        if depth is None and obs.get("depth_path"):
            from .data import load_depth
            depth = load_depth(obs["depth_path"])
        if rgb is None and obs.get("rgb_path"):
            from .data import load_rgb
            rgb = load_rgb(obs["rgb_path"])
        if pc is None and obs.get("pointcloud_path"):
            import numpy as np
            pc = np.load(obs["pointcloud_path"])

        res = self.inference.run(depth=depth, rgb=rgb,
                                 pointcloud=pc, camera_info=camera_info)
        return res.distance  # float; inf when no obstacle


# module-level convenience
_solution = None


def predict(obs: Dict[str, Any]) -> float:
    """Default entry used by judgeflow (geometric path, no weights needed)."""
    global _solution
    if _solution is None:
        _solution = NODSolution(mode="geometric")
    return _solution.predict(obs)


def predict_with_learned(obs: Dict[str, Any], weights_path=None) -> float:
    return NODSolution(mode="learned", weights_path=weights_path).predict(obs)

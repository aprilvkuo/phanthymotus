"""Self-contained tests (no external data needed).

Run:  python -m pytest tests/ -q
  or:  python tests/test_geometric.py
"""

import math
import os
import sys

import numpy as np

# allow running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nod.config import NODConfig
from nod.geometric import GeometricNODPredictor
from nod.data import make_synthetic_scene, make_synthetic_pointcloud
from nod.infer import NODInference
from nod.packaging import package_check


def test_depth_nearest_obstacle():
    d, gt = make_synthetic_scene(obstacle_distance=3.0)
    p = GeometricNODPredictor(NODConfig())
    r = p.predict(depth=d)
    assert r.has_obstacle is True
    assert abs(r.distance - gt) < 0.2
    assert r.source == "geometric-depth"


def test_depth_no_obstacle():
    # all far background -> no obstacle within presence_distance
    d, _ = make_synthetic_scene(obstacle_distance=20.0, background_distance=25.0)
    p = GeometricNODPredictor(NODConfig())
    r = p.predict(depth=d)
    assert r.has_obstacle is False


def test_pointcloud_nearest_obstacle():
    pc, gt = make_synthetic_pointcloud(obstacle_distance=4.0)
    p = GeometricNODPredictor(NODConfig())
    r = p.predict(pointcloud=pc)
    assert r.has_obstacle is True
    assert abs(r.distance - gt) < 0.3
    assert r.source == "geometric-pointcloud"


def test_pointcloud_no_obstacle():
    pc, _ = make_synthetic_pointcloud(obstacle_distance=20.0, fov=0.2)
    # shrink corridor so the far slab is outside it
    cfg = NODConfig(fov_half_h=0.1)
    p = GeometricNODPredictor(cfg)
    r = p.predict(pointcloud=pc)
    assert r.has_obstacle is False


def test_orchestrator_end_to_end():
    inf = NODInference.from_config(NODConfig(mode="geometric"))
    d, gt = make_synthetic_scene(obstacle_distance=2.5)
    r = inf.run(depth=d)
    assert abs(r.distance - gt) < 0.2


def test_resource_budget_geometric():
    cfg = NODConfig(mode="geometric")
    inf = NODInference.from_config(cfg)
    rep = {"params": 0, "params_ok_<30M": cfg.param_budget_ok(0),
           "gpu_used": "none (CPU)"}
    assert rep["params_ok_<30M"] is True
    # geometric uses no GPU
    assert cfg.gpu_budget_ok(0.0) is True


def test_package_check_no_oversized():
    rep = package_check(NODConfig())
    assert rep["ok"] is True, f"oversized files: {rep['oversized_files']}"


def test_inf_distance_when_empty():
    d = np.full((480, 640), np.inf, dtype=np.float32)
    p = GeometricNODPredictor(NODConfig())
    r = p.predict(depth=d)
    assert r.distance == math.inf
    assert r.has_obstacle is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

"""Geometric Nearest Obstacle Distance estimator (default, constraint-safe).

Why this is the safe default for the benchmark:
  * Trainable parameters : 0   -> trivially satisfies the <30M rule.
  * Compute             : CPU-only, O(H*W) -> satisfies the <10% GPU rule and
                          is easily real-time (hundreds of FPS on a 640x480 map).
  * Accuracy            : exact (uses true geometric depth), not statistical.

It works on two input modalities exposed by the Perception stack:
  1. A depth image (camera_depth)  -> forward-corridor min-depth.
  2. A point cloud  (camera_pointcloud) -> forward-corridor min range.

Both define a "navigable corridor" = the central frustum directly in front of
the robot. The minimum valid range inside that corridor is the NOD.
"""

import numpy as np

from .interface import BaseNODPredictor, NODResult
from .config import NODConfig


class GeometricNODPredictor(BaseNODPredictor):
    def __init__(self, config: NODConfig | None = None):
        self.cfg = config or NODConfig()

    # -- public API ---------------------------------------------------- #
    def predict(self, depth=None, rgb=None, pointcloud=None,
                camera_info=None, **_):
        if pointcloud is not None:
            return self._from_pointcloud(np.asarray(pointcloud), camera_info)
        if depth is not None:
            return self._from_depth(np.asarray(depth), camera_info)
        raise ValueError(
            "GeometricNODPredictor requires `depth` (depth image) "
            "or `pointcloud` (Nx3 / HxWx3 in robot frame)."
        )

    # -- depth-image path ---------------------------------------------- #
    def _from_depth(self, depth, camera_info):
        cfg = self.cfg
        depth = depth.astype(np.float32, copy=False)
        if depth.ndim == 3:
            depth = depth[..., 0]
        H, W = depth.shape

        fx = _get(camera_info, "fx", cfg.fx)
        fy = _get(camera_info, "fy", cfg.fy)
        cx = _get(camera_info, "cx", cfg.cx)
        cy = _get(camera_info, "cy", cfg.cy)

        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)

        if cfg.use_row_band:
            v_min = int(cfg.v_min_frac * H)
            v_max = int(cfg.v_max_frac * H)
            mask_v = (v >= v_min) & (v < v_max)
            mask_h = np.ones(W, dtype=bool)
        else:
            ang_h = np.arctan2(u - cx, fx)          # (W,)
            ang_v = np.arctan2(v - cy, fy)          # (H,)
            mask_h = np.abs(ang_h) < cfg.fov_half_h
            mask_v = np.abs(ang_v) < cfg.fov_half_v

        corridor = np.zeros((H, W), dtype=bool)
        corridor[np.ix_(mask_v, mask_h)] = True

        valid = np.isfinite(depth) & (depth > cfg.depth_min) & (depth < cfg.depth_max)
        keep = corridor & valid

        if not np.any(keep):
            return NODResult(distance=float("inf"), has_obstacle=False,
                             source="geometric-depth", corridor_ratio=0.0)

        d = depth[keep]
        min_depth = float(np.min(d))
        near_mask = d < cfg.presence_distance
        ratio = float(np.count_nonzero(near_mask)) / float(d.size)
        has = (min_depth < cfg.presence_distance) and (ratio >= cfg.min_near_pixels_ratio)
        return NODResult(distance=min_depth, has_obstacle=has,
                         source="geometric-depth", corridor_ratio=ratio)

    # -- point-cloud path ---------------------------------------------- #
    def _from_pointcloud(self, pc, camera_info):
        cfg = self.cfg
        pc = pc.reshape(-1, 3).astype(np.float32, copy=False)
        # Robot frame assumption: x=forward, y=left, z=up.
        fwd = pc[:, 0]
        lat = pc[:, 1]
        up = pc[:, 2]

        horiz = np.sqrt(fwd * fwd + lat * lat)
        # guard divide-by-zero on the forward axis
        with np.errstate(divide="ignore", invalid="ignore"):
            ang = np.abs(np.arctan2(lat, np.where(fwd == 0, 1e-6, fwd)))

        keep = (
            (fwd > cfg.depth_min)
            & (fwd < cfg.depth_max)
            & (ang < cfg.fov_half_h)
            & (up > cfg.pc_ground_min)
            & (up < cfg.pc_ground_max)
        )

        if not np.any(keep):
            return NODResult(distance=float("inf"), has_obstacle=False,
                             source="geometric-pointcloud", corridor_ratio=0.0)

        r = horiz[keep]
        min_d = float(np.min(r))
        near = r < cfg.presence_distance
        ratio = float(np.count_nonzero(near)) / float(r.size)
        has = (min_d < cfg.presence_distance) and (ratio >= cfg.min_near_pixels_ratio)
        return NODResult(distance=min_d, has_obstacle=has,
                         source="geometric-pointcloud", corridor_ratio=ratio)


def _get(camera_info, key, default):
    if camera_info and key in camera_info:
        return float(camera_info[key])
    return default

"""Data loading + a self-contained synthetic generator.

No external dataset is bundled (the benchmark supplies its own). To make the
solution runnable and testable out-of-the-box we provide:

  * load_rgb / load_depth : read PNG/JPG/NPY/TXT into numpy.
  * make_synthetic_scene  : build a depth map with a configurable wall /
                             obstacle at a known distance -> ground-truth NOD.
                             Used by tests and the demo, so CI needs no data.
"""

import os
import numpy as np


def load_depth(path: str) -> np.ndarray:
    """Load a depth map. Supports .npy and image formats (scaled to meters)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except ImportError:
        from PIL import Image
        img = np.asarray(Image.open(path))
    if img is None:
        raise FileNotFoundError(path)
    arr = img.astype(np.float32)
    # If 16-bit PNG, assume millimeter-encoded depth (common for RealSense/etc.)
    if arr.dtype == np.float32 and arr.max() > 1000 and ext in (".png", ".jpg", ".jpeg"):
        arr = arr / 1000.0
    return arr


def load_rgb(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    try:
        import cv2
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def make_synthetic_scene(height=480, width=640, obstacle_distance=3.0,
                         obstacle_width_frac=0.3, background_distance=12.0,
                         noise=0.02, seed=0):
    """Create a depth map with a single frontal obstacle (a wall slab) at a
    known distance, plus far background. Ground-truth NOD = obstacle_distance
    (when the slab lies inside the forward corridor).

    Returns (depth_map, gt_nod).
    """
    rng = np.random.default_rng(seed)
    depth = np.full((height, width), background_distance, dtype=np.float32)
    cxs, cxe = int((0.5 - obstacle_width_frac / 2) * width), \
               int((0.5 + obstacle_width_frac / 2) * width)
    # Put the slab in the central vertical band (body height)
    cys, cye = int(0.35 * height), int(0.65 * height)
    slab = obstacle_distance + noise * rng.standard_normal((cye - cys, cxe - cxs)).astype(np.float32)
    depth[cys:cye, cxs:cxe] = np.clip(slab, 0.1, None)
    return depth, float(obstacle_distance)


def make_synthetic_pointcloud(obstacle_distance=3.0, n_ground=2000,
                              n_obstacle=400, fov=0.6, seed=0):
    """Build an (N,3) point cloud in robot frame (x=forward, y=left, z=up)
    with a frontal obstacle plane at x=obstacle_distance."""
    rng = np.random.default_rng(seed)
    # ground points spread forward, clearly on/below the floor (z<=0) so the
    # floor-exclusion filter (pc_ground_min) removes them as non-obstacles.
    gx = rng.uniform(0.2, 12.0, n_ground)
    gy = rng.uniform(-2.0, 2.0, n_ground)
    gz = rng.uniform(-0.10, 0.0, n_ground)
    ground = np.stack([gx, gy, gz], -1)
    # obstacle: a vertical slab at x=obstacle_distance, within corridor angle
    ang = rng.uniform(-fov, fov, n_obstacle)
    r = obstacle_distance + rng.normal(0, 0.02, n_obstacle)
    oy = r * np.sin(ang)
    ox = r * np.cos(ang)
    oz = rng.uniform(0.0, 1.0, n_obstacle)   # body height
    obstacle = np.stack([ox, oy, oz], -1)
    pc = np.concatenate([ground, obstacle], 0).astype(np.float32)
    return pc, float(obstacle_distance)

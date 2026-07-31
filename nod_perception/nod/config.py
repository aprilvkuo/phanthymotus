"""Central configuration for the Nearest Obstacle Distance (NOD) benchmark solution.

All hard limits below come directly from the benchmark requirements
(具身智能-空间感知-障碍物距离度量评测榜):

  * Hardware: NVIDIA Jetson Orin 16G (~100 TOPS, Ampere, ~RTX3090)
  * Model size:  < 30M parameters
  * GPU usage:   < 10%  (of the 16G Orin)
  * Real-time:   must run online as the Perception module's NOD output
  * Submission:  fork 4paradigm/phanthymotus -> personal PUBLIC repo,
                 commit id + repo URL. Model weights go on juicefs
                 (http://172.28.4.81:34567/), NOT committed (no file > 1MB).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NODConfig:
    # ------------------------------------------------------------------ #
    # Hard benchmark constraints (do not relax without re-checking)
    # ------------------------------------------------------------------ #
    MAX_PARAMS: int = 30_000_000
    MAX_GPU_MEM_FRACTION: float = 0.10   # GPU memory budget < 10%
    TARGET_DEVICE: str = "jetson-orin-16g"
    REAL_TIME_FPS: float = 10.0          # design target (>= 10 Hz)

    # ------------------------------------------------------------------ #
    # Sensor / camera model (front RGB-D camera, matches the Perception
    # stack outputs: camera_rgb / camera_depth / camera_pointcloud).
    # Override per robot platform via camera_info at inference time.
    # ------------------------------------------------------------------ #
    depth_width: int = 640
    depth_height: int = 480
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 240.0

    # ------------------------------------------------------------------ #
    # Forward "navigable corridor" definition
    # ------------------------------------------------------------------ #
    # Horizontal half-angle around the optical axis that still counts as
    # "directly in front / can block passage". ~35 deg covers the body width.
    fov_half_h: float = 0.61            # radians (~35 deg)
    # Vertical half-angle. Keeps the central band that maps to the robot's
    # body height; sky / ceiling are excluded.
    fov_half_v: float = 0.40            # radians (~23 deg)
    # If your camera intrinsics are unknown, fall back to a row band:
    use_row_band: bool = False
    v_min_frac: float = 0.30            # keep rows in [0.30, 0.70] of H
    v_max_frac: float = 0.70

    # ------------------------------------------------------------------ #
    # Depth validity + obstacle decision
    # ------------------------------------------------------------------ #
    depth_min: float = 0.10             # m, below this = invalid / too close
    depth_max: float = 30.0             # m, beyond this = "no relevant obstacle"
    presence_distance: float = 15.0     # m, NOD < this => obstacle present
    min_near_pixels_ratio: float = 0.001  # fraction of corridor pixels that are near

    # Point-cloud frame assumption (robot frame): x=forward, y=left, z=up
    # The drivable floor (z ~ 0) is NOT an obstacle, so exclude it. Only
    # surfaces rising above this height count as "blocking passage".
    pc_ground_min: float = 0.05         # m (floor exclusion; curbs/edges above this count)
    pc_ground_max: float = 1.20         # m (top of obstacle blocking wheeled/legged robot)

    # ------------------------------------------------------------------ #
    # Model selection
    # ------------------------------------------------------------------ #
    # "geometric" (default, 0 params, CPU) | "learned" | "hybrid"
    mode: str = "geometric"

    # juicefs model host + filename (weights are NOT committed)
    model_repo_url: str = "http://172.28.4.81:34567/"
    model_filename: str = "nod_depth_refiner.pt"
    # Local cache dir for downloaded weights (git-ignored)
    model_cache_dir: str = ".cache/models"

    # Learned model hyper-params (kept small to stay < 30M)
    learned_encoder: str = "mobilenet_v3_small"   # ~1.5M params
    learned_features: int = 64
    use_gpu: bool = False               # geometric needs none; learned uses minimal

    def param_budget_ok(self, n_params: int) -> bool:
        return n_params <= self.MAX_PARAMS

    def gpu_budget_ok(self, frac: float) -> bool:
        return frac <= self.MAX_GPU_MEM_FRACTION

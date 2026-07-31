"""Generate a demo depth scene so the pipeline can be tried without a dataset.

Run:
    python demo/generate_demo.py
Produces demo/depth_scene.npy + prints the ground-truth NOD.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nod.data import make_synthetic_scene
import numpy as np


def main(out_dir="demo"):
    os.makedirs(out_dir, exist_ok=True)
    for dist in (1.5, 3.0, 6.0):
        depth, gt = make_synthetic_scene(obstacle_distance=dist, seed=int(dist * 10))
        path = os.path.join(out_dir, f"depth_{dist:.1f}m.npy")
        np.save(path, depth)
        print(f"wrote {path}  (gt NOD={gt} m)")


if __name__ == "__main__":
    main()

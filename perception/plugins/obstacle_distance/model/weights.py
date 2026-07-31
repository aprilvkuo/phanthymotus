"""Weights / checkpoint handling for the open-source backbone.

The model is NOT trained (project decision): it uses a public pretrained
monocular depth checkpoint (MiDaS small). There are two ways the judgeflow
image obtains the weights:

  1. Online (default): torch.hub downloads the MiDaS repo + weights at first
     load. Works whenever the image has internet access to GitHub/HF.

  2. Offline (recommended for a reproducible leaderboard run): vendor the repo
     and bake the checkpoint files into the image, then set:
         MIDAS_REPO_DIR=/opt/MiDaS            # vendored repo (local path)
         TORCH_HOME=/opt/torch-hub            # pre-populated with checkpoints/
     This script helps you collect the cached checkpoints for (2).

The MiDaS small checkpoint that gets downloaded is:
    ~/.cache/torch/hub/checkpoints/midas_v21_small_256.pt   (~85 MB)
plus a dependency:
    ~/.cache/torch/hub/checkpoints/tf_efficientnet_lite3-*.pth
"""
from __future__ import annotations

import os
import shutil
import argparse


def find_torch_hub_dir() -> str:
    import torch
    return torch.hub.get_dir()


def find_checkpoints() -> list[str]:
    ckpt_dir = os.path.join(find_torch_hub_dir(), "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    return [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".pt") or f.endswith(".pth")
    ]


def mirror_checkpoints(dest_dir: str) -> list[str]:
    """Copy cached MiDaS/efficientnet checkpoints into `dest_dir`.

    Use `dest_dir` as the image's TORCH_HOME/checkpoints so the judgeflow can
    load fully offline. Returns the list of copied files.
    """
    os.makedirs(dest_dir, exist_ok=True)
    copied = []
    for src in find_checkpoints():
        dst = os.path.join(dest_dir, os.path.basename(src))
        shutil.copyfile(src, dst)
        copied.append(dst)
    if not copied:
        print("No cached checkpoints found. Run a single inference first so "
              "torch.hub downloads midas_v21_small_256.pt, then re-run this.")
    else:
        for c in copied:
            print(f"mirrored -> {c} ({os.path.getsize(c)/1e6:.1f} MB)")
        print(f"\nBake '{dest_dir}' into the image and set TORCH_HOME to its parent.")
    return copied


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="weights_cache",
                    help="directory to copy cached checkpoints into")
    args = ap.parse_args(argv)
    mirror_checkpoints(args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Weight loading + juicefs download.

Constraint from the requirement: do NOT commit any file > 1MB. Trained weights
live on juicefs and are downloaded by the judgeflow image at runtime via
http://172.28.4.81:34567/<filename>. This module handles that download and the
local caching under the model_dir.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from .config import RuntimeConfig


def weights_path(cfg: RuntimeConfig | None = None) -> Path:
    cfg = cfg or RuntimeConfig()
    return Path(cfg.model_dir) / cfg.weights_filename


def download_weights(cfg: RuntimeConfig | None = None, force: bool = False) -> Path:
    """Download the checkpoint from juicefs into model_dir. Returns local path.

    Raises on failure so callers can decide whether to fall back to an
    untrained baseline (clearly marked as non-submission-ready).
    """
    cfg = cfg or RuntimeConfig()
    dest = weights_path(cfg)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{cfg.juicefs_base.rstrip('/')}/{cfg.weights_filename}"
    print(f"[weights] downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"[weights] downloaded {dest.stat().st_size} bytes")
    return dest


def load_state_dict(cfg: RuntimeConfig | None = None, force_download: bool = False):
    """Return a state_dict loaded from local/juicefs weights, or None if unavailable."""
    cfg = cfg or RuntimeConfig()
    dest = weights_path(cfg)
    if not dest.exists():
        try:
            dest = download_weights(cfg, force=force_download)
        except Exception as e:  # network/internal — fall back gracefully
            print(f"[weights] WARN: could not load weights ({e}); using untrained baseline.")
            return None
    try:
        sd = torch_load(dest)
        return sd
    except Exception as e:
        print(f"[weights] WARN: failed to read weights ({e}); using untrained baseline.")
        return None


def torch_load(dest):
    import torch
    return torch.load(dest, map_location="cpu")

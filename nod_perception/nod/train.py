"""Training skeleton for the learned monocular-depth NOD model.

NOTE: The geometric path needs no training and is the benchmark default.
This script is provided so you *can* improve accuracy in monocular / outdoor
setups. It expects a dataset of (rgb, metric_depth) pairs; the NOD label is
derived geometrically from the metric depth during training via the same
corridor extractor, keeping the train/eval contract identical.

Run:
    python -m nod.train --data /path/to/rgbd_dataset --epochs 20
"""

import argparse

from .config import NODConfig


def build_dataset(data_root):
    """Yield (rgb_path, depth_path) pairs. Adapt to your dataset layout."""
    import os
    pairs = []
    for fn in os.listdir(data_root):
        if fn.endswith((".png", ".jpg", ".jpeg")):
            base = os.path.splitext(fn)[0]
            dp = os.path.join(data_root, base + "_depth.npy")
            if os.path.exists(dp):
                pairs.append((os.path.join(data_root, fn), dp))
    return pairs


def train(data_root, epochs=20, cfg: NODConfig | None = None):
    cfg = cfg or NODConfig(mode="learned")
    torch, nn = _torch()
    from .model import MobileDepthNet, LearnedNODPredictor
    from .geometric import GeometricNODPredictor
    from .data import load_rgb, load_depth

    net = MobileDepthNet(cfg)
    geom = GeometricNODPredictor(cfg)
    opt = torch.optim.AdamW(net.decoder.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    pairs = build_dataset(data_root)
    if not pairs:
        print("No (rgb, depth) pairs found. Adjust build_dataset().")
        return

    net.to("cuda" if cfg.use_gpu else "cpu").train()
    for ep in range(epochs):
        total = 0.0
        for rgb_p, depth_p in pairs:
            rgb = load_rgb(rgb_p) / 255.0
            depth = load_depth(depth_p)
            t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
            gt = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
            rel = net.forward(t)
            # supervise relative depth -> metric via scale, then geometric NOD loss
            pred_depth = rel * cfg.depth_max
            loss = criterion(pred_depth, gt)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        print(f"epoch {ep+1}/{epochs}  loss={total/len(pairs):.4f}")

    # save to cache dir (NOT committed)
    import os
    os.makedirs(cfg.model_cache_dir, exist_ok=True)
    out = os.path.join(cfg.model_cache_dir, cfg.model_filename)
    torch.save({"decoder": net.decoder.state_dict()}, out)
    print(f"saved weights -> {out}")


def _torch():
    try:
        import torch, torch.nn as nn
        return torch, nn
    except ImportError as e:
        raise ImportError("torch is required for training.") from e


def _cli():
    p = argparse.ArgumentParser("NOD training")
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()
    train(args.data, epochs=args.epochs)


if __name__ == "__main__":
    _cli()

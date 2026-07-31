"""训练脚本(src/train.py)。

loss = L1 + lambda * 尺度不变项(占位)。保存 ckpt.pt。
断言 param_count < 30_000_000。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

# 将 src 加入路径，支持 `python src/train.py` 与 `python -m src.train`
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from config import Config
    from data.dataset import NODDataset
    from model.network import DepthNet
except ImportError:  # pragma: no cover
    from .config import Config
    from .data.dataset import NODDataset
    from .model.network import DepthNet

logger = logging.getLogger(__name__)


def _scale_invariant_loss(pred, gt, mask):
    """尺度不变损失(Eigen et al.)，仅对有效(有限)像素。"""
    p = (pred + 1e-3).clamp(min=1e-3).log()
    g = (gt + 1e-3).clamp(min=1e-3).log()
    diff = (p - g) * mask
    n = mask.sum().clamp(min=1)
    si = (diff ** 2).sum() / n - (diff.sum() ** 2) / (n ** 2)
    return si


def train(config: Config) -> None:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = DepthNet(
        encoder_name=config.encoder_name,
        depth_range=(config.depth_min, config.depth_max),
        pretrained=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("参数量: %d (%.2f M)", n_params, n_params / 1e6)
    # 硬约束：模型参数 < 30M
    assert n_params < 30_000_000, f"参数量 {n_params} 超过 30M 硬约束"

    device = (
        config.device
        if (config.device.startswith("cuda") and torch.cuda.is_available())
        else "cpu"
    )
    model.to(device)
    model.train()

    dataset = NODDataset(config, source="synthetic", length=config.train.num_samples)
    loader = DataLoader(
        dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=0
    )

    opt = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    lam = config.train.lambda_scale_invariant

    for epoch in range(config.train.epochs):
        running = 0.0
        for rgb, depth in loader:
            try:
                from preprocess import preprocess
            except ImportError:  # pragma: no cover
                from .preprocess import preprocess

            x = preprocess(
                rgb.numpy() if hasattr(rgb, "numpy") else rgb,
                size=config.input_size,
                mean=config.imagenet_mean,
                std=config.imagenet_std,
                pad_value=config.imagenet_mean,
                to_chw=True,
            )
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(device)
            depth = depth.to(device).permute(0, 3, 1, 2)  # HWC -> CHW
            valid = torch.isfinite(depth) & (depth > 0)
            pred = model(x)
            l1 = (torch.abs(pred - depth) * valid).sum() / valid.sum().clamp(min=1)
            loss = l1 + lam * _scale_invariant_loss(pred, depth, valid)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
        logger.info("epoch %d loss=%.4f", epoch, running / max(1, len(loader)))

    os.makedirs(os.path.dirname(config.ckpt_path_abs) or ".", exist_ok=True)
    torch.save(model.state_dict(), config.ckpt_path_abs)
    logger.info("已保存权重: %s", config.ckpt_path_abs)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--input_size", type=int, default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config) if args.config else Config.default()
    cfg.override_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()

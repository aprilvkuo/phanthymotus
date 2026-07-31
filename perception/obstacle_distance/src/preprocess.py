"""图像预处理(src/preprocess.py)。

纯 numpy 实现，不依赖 torch 也能运行；torch 路径为可选封装。
约定：输入 RGB / HWC / uint8；输出 letterbox 到 input_size，pad 填 ImageNet 均值后归一化。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _resize(img: np.ndarray, size_wh: tuple) -> np.ndarray:
    """等比缩放；优先 cv2，缺失时退化为最近邻(numpy)，保证可运行。"""
    try:
        import cv2

        return cv2.resize(img, size_wh, interpolation=cv2.INTER_LINEAR)
    except Exception:
        h, w = img.shape[:2]
        nh, nw = size_wh
        ys = np.linspace(0, h - 1, nh).round().astype(int)
        xs = np.linspace(0, w - 1, nw).round().astype(int)
        return img[np.ix_(ys, xs)]


def letterbox_resize(
    img: np.ndarray, size: int, pad_value: Sequence[float]
) -> np.ndarray:
    """等比缩放并居中填充到 (size, size)，pad 填 pad_value(ImageNet 均值)。

    Args:
        img: HWC uint8 或 float 图像。
        size: 目标边长。
        pad_value: 各通道填充值（0~1 比例，与 img 同量纲）。
    Returns:
        尺寸 (size, size) 的图像，dtype 同 img。
    """
    if img.ndim == 2:
        img = img[:, :, None]
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = _resize(img, (new_h, new_w))

    canvas = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    if img.dtype == np.uint8:
        pv = np.array([int(round(v * 255)) for v in pad_value], dtype=np.uint8)
    else:
        pv = np.array(pad_value, dtype=img.dtype)
    canvas[:] = pv

    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def normalize(
    img: np.ndarray, mean: Sequence[float], std: Sequence[float]
) -> np.ndarray:
    """将 uint8 [0,255] 归一化到 (img/255 - mean)/std。返回 float32。"""
    arr = img.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (arr - mean) / std


def to_tensor(img: np.ndarray):
    """HWC -> CHW 并增加 batch 维。torch 可用时返回 torch.Tensor([1,C,H,W])。"""
    arr = img.transpose(2, 0, 1)[None]  # [1, C, H, W]
    try:
        import torch

        return torch.from_numpy(arr.copy())
    except Exception:
        return arr


def preprocess(
    img: np.ndarray,
    size: int,
    mean: Sequence[float],
    std: Sequence[float],
    pad_value: Sequence[float],
    to_chw: bool = True,
):
    """完整预处理：letterbox + normalize + (可选)to_tensor。

    Returns:
        torch.Tensor [1,3,size,size] 或 numpy [1,3,size,size]，float32。
    """
    lb = letterbox_resize(img, size, pad_value)
    norm = normalize(lb, mean, std)
    if to_chw:
        return to_tensor(norm)
    return norm

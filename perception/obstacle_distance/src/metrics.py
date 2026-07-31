"""评测指标(src/metrics.py)。

mae / rmse / delta 纯 numpy 实现(可替换为 torch)。
delta: |pred-gt| < thr 的占比；gt=inf(无障碍)时 pred=inf 判为正确。
"""
from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.isfinite(gt)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - gt[valid])))


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.isfinite(gt)
    if valid.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[valid] - gt[valid]) ** 2)))


def delta(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> float:
    """δ=|pred-gt|<thr 的占比。

    处理无障碍样本：gt=inf 且 pred=inf 视为正确；gt 有限但 pred=inf 视为错误。
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    correct = np.zeros(gt.shape, dtype=bool)
    finite_gt = np.isfinite(gt)
    correct[finite_gt] = np.abs(pred[finite_gt] - gt[finite_gt]) < thr
    inf_gt = ~finite_gt
    correct[inf_gt] = ~np.isfinite(pred[inf_gt])
    if gt.size == 0:
        return float("nan")
    return float(np.mean(correct))


def evaluate(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> dict:
    """一次性返回 MAE / RMSE / δ。"""
    return {
        "mae": mae(pred, gt),
        "rmse": rmse(pred, gt),
        "delta": delta(pred, gt, thr),
    }

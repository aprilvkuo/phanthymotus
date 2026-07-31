"""指标纯 numpy 测试（无需 torch）。"""
import numpy as np
from metrics import mae, rmse, delta


def test_mae():
    assert abs(mae(np.array([1.0, 2.0]), np.array([1.5, 1.0])) - 0.75) < 1e-9


def test_rmse():
    pred = np.array([0.0, 0.0])
    gt = np.array([0.0, 2.0])
    assert abs(rmse(pred, gt) - 1.0) < 1e-9


def test_delta_basic():
    pred = np.array([1.0, 5.0])
    gt = np.array([1.2, 5.4])  # 误差 0.2, 0.4 均 < 0.5
    assert abs(delta(pred, gt) - 1.0) < 1e-9


def test_delta_handles_inf_gt():
    pred = np.array([float("inf"), 3.0])
    gt = np.array([float("inf"), 3.4])  # 第一个无障碍正确，第二个误差 0.4 < 0.5
    assert abs(delta(pred, gt) - 1.0) < 1e-9


def test_delta_mismatch():
    pred = np.array([float("inf")])
    gt = np.array([2.0])  # gt 有障碍物但预测无障碍 -> 错误
    assert delta(pred, gt) == 0.0

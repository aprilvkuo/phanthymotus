"""NODModel 测试（torch 相关，无 torch 环境自动跳过）。"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from model.nod_model import NODModel  # noqa: E402
from src.config import Config  # noqa: E402


def _model() -> NODModel:
    cfg = Config.default()
    cfg.device = "cpu"
    return NODModel(cfg, backend="torch")


def test_predict_finite_on_random_image():
    m = _model()
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dist, has = m.predict(img)
    assert np.isfinite(dist)
    assert has is True


def test_compute_nod_no_obstacle_returns_inf():
    m = _model()
    depth = np.full((224, 224), 30.0, dtype=np.float32)  # 全部在远裁剪 -> 开阔
    dist, has = m._compute_nod(depth)
    assert not np.isfinite(dist)
    assert has is False
    assert dist == float("inf")


def test_compute_nod_obstacle_present():
    m = _model()
    depth = np.full((224, 224), 10.0, dtype=np.float32)
    dist, has = m._compute_nod(depth)
    assert np.isfinite(dist)
    assert has is True

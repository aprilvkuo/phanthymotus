"""数据集测试（torch 相关自动跳过；合成数据纯 numpy 可运行）。"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src.config import Config  # noqa: E402
from src.data.dataset import NODDataset  # noqa: E402
from src.data.synthetic import SyntheticSceneGenerator  # noqa: E402


def test_synthetic_shape_and_range():
    gen = SyntheticSceneGenerator(size=224, seed=0)
    rgb, depth, nod = gen.generate()
    assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
    assert depth.shape == (224, 224, 1) and depth.dtype == np.float32
    assert depth.min() >= 0.1 - 1e-3
    assert depth.max() <= 30.0 + 1e-3
    assert np.isfinite(nod) or nod == float("inf")


def test_dataset_getitem_contract():
    cfg = Config.default()
    ds = NODDataset(cfg, source="synthetic", length=10)
    rgb, depth = ds[0]
    assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
    assert depth.shape == (224, 224, 1) and depth.dtype == np.float32
    assert depth.min() >= 0.1 - 1e-3


def test_dataset_real_is_placeholder():
    cfg = Config.default()
    with pytest.raises(NotImplementedError):
        NODDataset(cfg, source="real")

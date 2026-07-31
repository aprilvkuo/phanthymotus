"""参数量约束测试（torch 相关自动跳过）。"""
import pytest

torch = pytest.importorskip("torch")
from model.network import DepthNet  # noqa: E402


def _count() -> int:
    net = DepthNet(
        encoder_name="mobilenet_v3_small",
        depth_range=(0.1, 30.0),
        pretrained=False,
    )
    return sum(p.numel() for p in net.parameters())


def test_param_count_under_30m():
    n = _count()
    assert n < 30_000_000, f"参数量 {n} 超过 30M 硬约束"


def test_param_count_ideally_under_5m():
    n = _count()
    assert n < 5_000_000, f"参数量 {n} 宜 <5M(基线目标)"

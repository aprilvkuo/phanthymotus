"""pytest 公共夹具(tests/conftest.py)。"""
import os
import sys

# 将 perception/obstacle_distance 与 src 加入路径，使 `import src` 与 `from model...` 均可用
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # perception/obstacle_distance
_SRC = os.path.join(_ROOT, "src")
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.config import Config  # noqa: E402


@pytest.fixture
def config() -> Config:
    """默认配置（字段与 default.yaml 一致）。"""
    return Config.default()

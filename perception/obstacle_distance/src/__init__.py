"""obstacle_distance 源码包。

顶层 API 暴露（包根 src，外部既可 `import src` 也可把 src 加入 path 后 `from model...`）：
    from src import Config, NODModel, DepthNet
    # 或把 src 加入 PYTHONPATH 后：
    from model.nod_model import NODModel
"""
try:
    from config import Config
    from model.network import DepthNet
    from model.nod_model import NODModel
except ImportError:  # pragma: no cover
    from .config import Config
    from .model.network import DepthNet
    from .model.nod_model import NODModel

__all__ = ["Config", "NODModel", "DepthNet"]

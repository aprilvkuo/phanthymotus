"""模型子包。"""
try:
    from model.network import DepthNet
    from model.nod_model import NODModel
except ImportError:  # pragma: no cover
    from .network import DepthNet
    from .nod_model import NODModel

__all__ = ["DepthNet", "NODModel"]

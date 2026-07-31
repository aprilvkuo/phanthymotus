"""数据子包。"""
try:
    from data.synthetic import SyntheticSceneGenerator
    from data.dataset import NODDataset
except ImportError:  # pragma: no cover
    from .synthetic import SyntheticSceneGenerator
    from .dataset import NODDataset

__all__ = ["SyntheticSceneGenerator", "NODDataset"]

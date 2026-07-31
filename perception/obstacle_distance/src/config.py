"""配置单一来源(src/config.py)。

提供 Config.load(path) 与 override_from_args(args)。
所有超参禁止硬编码，统一来自 configs/default.yaml。
跨文件统一使用本模块读取配置；相对路径以配置文件所在目录为根解析。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("需要 pyyaml，请先 pip install pyyaml") from exc


@dataclass
class PassableRegionConfig:
    """可通行区域定义（归一化比例）。"""

    horizon_row: float = 0.55
    corridor_x: List[float] = field(default_factory=lambda: [0.25, 0.75])


@dataclass
class TrainConfig:
    """训练超参（占位基线）。"""

    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1.0e-3
    lambda_scale_invariant: float = 0.5
    num_samples: int = 2000


@dataclass
class Config:
    """全局配置。字段与 default.yaml 一一对应。"""

    input_size: int = 224
    encoder_name: str = "mobilenet_v3_small"
    depth_range: List[float] = field(default_factory=lambda: [0.1, 30.0])
    passable_region: PassableRegionConfig = field(default_factory=PassableRegionConfig)
    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    obstacle_distance_threshold: float = 30.0
    ckpt_path: str = "models/ckpt.pt"
    onnx_path: str = "models/model.onnx"
    trt_path: str = "models/model.trt"
    backend: str = "torch"
    device: str = "cuda"
    seed: int = 42
    train: TrainConfig = field(default_factory=TrainConfig)

    # 运行时解析的项目根目录（含 configs/ 的目录）
    root_dir: str = field(default="", repr=False)

    # -------------------- 路径解析 --------------------
    def _resolve(self, p: str) -> str:
        if not p:
            return p
        if os.path.isabs(p):
            return p
        base = self.root_dir or os.getcwd()
        return os.path.normpath(os.path.join(base, p))

    @property
    def ckpt_path_abs(self) -> str:
        return self._resolve(self.ckpt_path)

    @property
    def onnx_path_abs(self) -> str:
        return self._resolve(self.onnx_path)

    @property
    def trt_path_abs(self) -> str:
        return self._resolve(self.trt_path)

    @property
    def depth_min(self) -> float:
        return float(self.depth_range[0])

    @property
    def depth_max(self) -> float:
        return float(self.depth_range[1])

    # -------------------- 加载 --------------------
    @classmethod
    def load(cls, path: str) -> "Config":
        """从 YAML 加载配置，并记录项目根目录用于相对路径解析。"""
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        root_dir = os.path.dirname(os.path.abspath(path))
        cfg = cls.from_dict(raw)
        cfg.root_dir = root_dir
        return cfg

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        raw = dict(raw)
        pr = raw.pop("passable_region", None) or {}
        train = raw.pop("train", None) or {}
        known = {f.name for f in fields(cls)}
        flat = {k: v for k, v in raw.items() if k in known}
        return cls(
            passable_region=PassableRegionConfig(**pr),
            train=TrainConfig(**train),
            **flat,
        )

    @classmethod
    def default(cls) -> "Config":
        """不读取文件的默认配置（以 cwd 为根）。"""
        return cls(root_dir=os.getcwd())

    # -------------------- 命令行覆盖 --------------------
    def override_from_args(self, args: Any) -> None:
        """用 argparse.Namespace 或 dict 覆盖已知字段（忽略 None 与未知键）。"""
        if args is None:
            return
        mapping = vars(args) if hasattr(args, "__dict__") else dict(args)
        known = {f.name for f in fields(self)} | {"passable_region", "train"}
        for key, val in mapping.items():
            if val is None or key not in known:
                continue
            if key == "passable_region" and isinstance(val, dict):
                self.passable_region = PassableRegionConfig(**val)
            elif key == "train" and isinstance(val, dict):
                self.train = TrainConfig(**val)
            else:
                setattr(self, key, val)

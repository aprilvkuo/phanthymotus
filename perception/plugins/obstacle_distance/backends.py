"""深度估计与目标检测模型的懒加载适配层。"""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np

from .model_store import ensure_model_file
from .types import Detection, Scene


class DepthBackend(Protocol):
    """Metric depth 后端协议。"""

    def predict(self, image: np.ndarray, scene: Scene) -> np.ndarray:
        """返回与输入图像同宽高、单位为米的二维深度图。"""


class DetectorBackend(Protocol):
    """目标检测后端协议。"""

    def detect(self, image: np.ndarray) -> list[Detection]:
        """返回输入图像坐标系中的检测框。"""


class TransformersDepthBackend:
    """Hugging Face Depth Anything V2 Metric Small 后端。"""

    DEFAULT_MODEL_REFERENCES: ClassVar[Mapping[Scene, str]] = {
        Scene.INDOOR: "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        Scene.OUTDOOR: "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    }

    def __init__(
        self,
        model_references: Mapping[Scene, str] | None = None,
        *,
        device: str | None = None,
    ):
        self._model_references = {
            **self.DEFAULT_MODEL_REFERENCES,
            **(model_references or {}),
        }
        self._requested_device = device
        self._active_scene: Scene | None = None
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str | None = None

    def predict(self, image: np.ndarray, scene: Scene) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("模型输入必须是 H×W×3 图像")
        self._ensure_loaded(scene)
        torch = self._torch
        rgb_image = np.ascontiguousarray(image[:, :, ::-1])
        inputs = self._processor(images=rgb_image, return_tensors="pt")
        inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}

        with torch.inference_mode():
            prediction = self._model(**inputs).predicted_depth
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
        return prediction.float().cpu().numpy().astype(np.float32, copy=False)

    def _ensure_loaded(self, scene: Scene) -> None:
        if self._active_scene is scene and self._model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        reference = self._model_references[scene]
        local_only = Path(reference).exists()
        self._processor = AutoImageProcessor.from_pretrained(
            reference,
            local_files_only=local_only,
        )
        self._model = AutoModelForDepthEstimation.from_pretrained(
            reference,
            local_files_only=local_only,
        )
        self._device = self._requested_device or (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self._model.to(self._device).eval()
        self._torch = torch
        self._active_scene = scene


class UltralyticsDetectorBackend:
    """YOLO 检测结果到内部 Detection 类型的适配器。"""

    def __init__(
        self,
        model_path: str,
        *,
        confidence: float = 0.25,
        image_size: int = 640,
        device: str | None = None,
        model_url: str | None = None,
        model_sha256: str | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ):
        self._model_path = model_path
        self._confidence = confidence
        self._image_size = image_size
        self._device = device
        self._model_url = model_url
        self._model_sha256 = model_sha256
        self._model_factory = model_factory
        self._model: Any = None

    def detect(self, image: np.ndarray) -> list[Detection]:
        model = self._ensure_loaded()
        arguments: dict[str, Any] = {
            "conf": self._confidence,
            "imgsz": self._image_size,
            "verbose": False,
        }
        if self._device:
            arguments["device"] = self._device
        results = model(image, **arguments)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = boxes.xyxy.tolist()
        classes = boxes.cls.tolist()
        confidences = boxes.conf.tolist()
        return [
            Detection(
                class_name=str(result.names[int(class_id)]),
                confidence=float(confidence),
                x1=float(coordinates[0]),
                y1=float(coordinates[1]),
                x2=float(coordinates[2]),
                y2=float(coordinates[3]),
            )
            for coordinates, class_id, confidence in zip(
                xyxy,
                classes,
                confidences,
                strict=True,
            )
        ]

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        if self._model_factory is None:
            from ultralytics import YOLO

            self._model_factory = YOLO
        model_path = Path(self._model_path)
        if self._model_url:
            model_path = ensure_model_file(
                self._model_url,
                model_path,
                self._model_sha256,
            )
        self._model = self._model_factory(str(model_path))
        return self._model

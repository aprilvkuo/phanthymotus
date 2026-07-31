import hashlib
from pathlib import Path

import numpy as np
import pytest

from plugins.obstacle_distance.backends import UltralyticsDetectorBackend
from plugins.obstacle_distance.model_store import ensure_model_file


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_ensure_model_file_downloads_and_reuses_valid_cache(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "models" / "model.bin"
    source.write_bytes(b"model-v1")

    result = ensure_model_file(source.as_uri(), destination, _sha256(b"model-v1"))
    source.write_bytes(b"model-v2")
    cached = ensure_model_file(source.as_uri(), destination, _sha256(b"model-v1"))

    assert result == destination
    assert cached == destination
    assert destination.read_bytes() == b"model-v1"


def test_ensure_model_file_rejects_bad_checksum_without_publishing_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "model.bin"
    source.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="SHA256"):
        ensure_model_file(source.as_uri(), destination, _sha256(b"expected"))

    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_failed_refresh_keeps_existing_model(tmp_path: Path) -> None:
    destination = tmp_path / "model.bin"
    destination.write_bytes(b"known-good")
    missing = tmp_path / "missing.bin"

    with pytest.raises(Exception):
        ensure_model_file(
            missing.as_uri(),
            destination,
            _sha256(b"replacement"),
        )

    assert destination.read_bytes() == b"known-good"


def test_ultralytics_backend_lazily_maps_detection_results() -> None:
    class FakeBoxes:
        xyxy = np.array([[1.0, 2.0, 10.0, 20.0]], dtype=np.float32)
        cls = np.array([2.0], dtype=np.float32)
        conf = np.array([0.75], dtype=np.float32)

    class FakeResult:
        boxes = FakeBoxes()
        names = {2: "car"}

    class FakeModel:
        def __call__(self, image, **kwargs):
            assert image.shape == (24, 32, 3)
            assert kwargs["verbose"] is False
            return [FakeResult()]

    loaded_paths: list[str] = []

    def model_factory(path: str):
        loaded_paths.append(path)
        return FakeModel()

    backend = UltralyticsDetectorBackend(
        "model.pt",
        confidence=0.3,
        model_factory=model_factory,
    )
    detections = backend.detect(np.zeros((24, 32, 3), dtype=np.uint8))

    assert loaded_paths == ["model.pt"]
    assert len(detections) == 1
    assert detections[0].class_name == "car"
    assert detections[0].confidence == pytest.approx(0.75)
    assert detections[0].x2 == pytest.approx(10.0)

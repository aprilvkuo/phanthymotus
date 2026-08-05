from pathlib import Path

from PIL import Image
from plugins.obstacle_distance import cli as cli_module
from plugins.obstacle_distance.cli import (
    _DEFAULT_DETECTOR_SHA256,
    build_default_estimator,
    main,
    predict_distance,
)
from plugins.obstacle_distance.types import DistanceEstimate, Scene


class FakeEstimator:
    def estimate(self, image, source_name, scene=None):
        assert image.shape == (12, 16, 3)
        assert source_name.endswith(".png")
        return DistanceEstimate(2.43, Scene.INDOOR, 0.9)


def test_predict_distance_reads_image_and_returns_float(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (16, 12), (10, 20, 30)).save(image_path)

    result = predict_distance(image_path, estimator=FakeEstimator())

    assert result == 2.43


def test_cli_prints_only_distance_to_stdout(tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (16, 12), (10, 20, 30)).save(image_path)

    exit_code = main(
        [str(image_path)],
        estimator_factory=FakeEstimator,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "2.430000\n"
    assert captured.err == ""


def test_cli_returns_nonzero_for_missing_image(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [str(tmp_path / "missing.jpg")],
        estimator_factory=FakeEstimator,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "图片不存在" in captured.err


def test_cli_strict_mode_returns_nonzero_for_backend_failure(
    tmp_path: Path,
    capsys,
) -> None:
    class BackendFailingEstimator:
        def estimate(self, image, source_name, scene=None, *, raise_on_error=False):
            assert raise_on_error is True
            raise RuntimeError("engine failed")

    image_path = tmp_path / "frame.png"
    Image.new("RGB", (16, 12), (10, 20, 30)).save(image_path)

    exit_code = main(
        [str(image_path), "--fail-on-backend-error"],
        estimator_factory=BackendFailingEstimator,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "engine failed" in captured.err


def test_custom_detector_path_is_not_overwritten_by_default_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    custom_model = tmp_path / "custom.pt"
    monkeypatch.setenv("OBSTACLE_DETECTOR_MODEL", str(custom_model))
    monkeypatch.delenv("OBSTACLE_DETECTOR_MODEL_URL", raising=False)
    monkeypatch.delenv("OBSTACLE_DETECTOR_MODEL_SHA256", raising=False)
    captured: dict = {}

    class CapturingDetector:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured.update(kwargs)

    monkeypatch.setattr(cli_module, "UltralyticsDetectorBackend", CapturingDetector)
    build_default_estimator()

    assert captured["model_path"] == str(custom_model)
    assert captured["model_url"] is None
    assert captured["model_sha256"] is None


def test_default_detector_download_uses_pinned_checksum(monkeypatch) -> None:
    monkeypatch.delenv("OBSTACLE_DETECTOR_MODEL", raising=False)
    monkeypatch.delenv("OBSTACLE_DETECTOR_MODEL_URL", raising=False)
    monkeypatch.delenv("OBSTACLE_DETECTOR_MODEL_SHA256", raising=False)
    captured: dict = {}

    class CapturingDetector:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured.update(kwargs)

    monkeypatch.setattr(cli_module, "UltralyticsDetectorBackend", CapturingDetector)
    build_default_estimator()

    assert captured["model_url"] is not None
    assert captured["model_sha256"] == _DEFAULT_DETECTOR_SHA256

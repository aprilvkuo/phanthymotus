from pathlib import Path

from PIL import Image

from plugins.obstacle_distance.cli import main, predict_distance
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

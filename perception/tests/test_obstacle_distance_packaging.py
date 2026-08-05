from pathlib import Path

from plugins.obstacle_distance.cli import _DEFAULT_DETECTOR_SHA256

ROOT = Path(__file__).resolve().parents[2]
PERCEPTION = ROOT / "perception"


def test_dockerfiles_install_obstacle_distance_runtime_dependencies() -> None:
    for dockerfile in [
        PERCEPTION / "Dockerfile",
        PERCEPTION / "Dockerfile.jetson",
    ]:
        source = dockerfile.read_text()
        assert "transformers==4.45.2" in source
        assert "safetensors" in source
        assert "pillow" in source.lower()


def test_service_mounts_external_model_storage_and_sets_cache_paths() -> None:
    source = (PERCEPTION / "deploy" / "service.yml").read_text()

    assert "/opt/embodied/models:/models" in source
    assert "OBSTACLE_MODEL_DIR=/models/obstacle_distance" in source
    assert "HF_HOME=/models/huggingface" in source


def test_docs_describe_cli_models_and_license() -> None:
    readme = (PERCEPTION / "README.md").read_text()

    assert "judge_obstacle_distance.py" in readme
    assert "OBSTACLE_DEPTH_INDOOR_MODEL" in readme
    assert (PERCEPTION / "plugins" / "obstacle_distance" / "MODELS.md").is_file()
    assert (
        PERCEPTION / "plugins" / "obstacle_distance" / "THIRD_PARTY_NOTICES.md"
    ).is_file()


def test_repository_does_not_contain_model_weights_larger_than_one_mb() -> None:
    model_suffixes = {".engine", ".onnx", ".pth", ".pt", ".safetensors"}
    offending = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".venv" not in path.parts
        and path.suffix.lower() in model_suffixes
        and path.stat().st_size > 1024 * 1024
    ]

    assert offending == []


def test_default_detector_download_has_pinned_sha256() -> None:
    assert _DEFAULT_DETECTOR_SHA256 == (
        "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
    )


def test_one_click_script_is_executable_and_pinned_to_jetson_dockerfile() -> None:
    script = ROOT / "deploy" / "obstacle_distance.sh"

    assert script.is_file()
    assert script.stat().st_mode & 0o111
    assert "--variant jetson" in script.read_text()
    build_script = (ROOT / "deploy" / "build_perception.sh").read_text()
    assert 'DOCKERFILE="${REPO_ROOT}/perception/Dockerfile.jetson"' in build_script
    assert "resolve_image_destination" in script.read_text()
    assert "resolve_image_destination" in build_script


def test_docs_describe_one_click_jetson_deploy_test() -> None:
    readme = (PERCEPTION / "README.md").read_text()

    assert "deploy/obstacle_distance.sh deploy-test" in readme
    assert "Dockerfile.jetson" in readme
    assert "PYTHONPATH=perception uv run --no-project" in readme
    assert "fps: 3" in readme

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "obstacle_distance.sh"


def run_script(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_lists_supported_commands() -> None:
    result = run_script("--help")

    assert result.returncode == 0
    for command in ["deploy-test", "build", "start", "status", "logs", "stop"]:
        assert command in result.stdout


def test_deploy_test_requires_image() -> None:
    result = run_script("deploy-test", "--dry-run")

    assert result.returncode != 0
    assert "--image" in result.stderr


def test_missing_value_never_consumes_dry_run_option() -> None:
    result = run_script(
        "start",
        "--model-dir",
        "--dry-run",
        "--image-ref",
        "test/perception:jetson",
    )

    assert result.returncode == 2
    assert "--model-dir 缺少参数值" in result.stderr
    assert "docker" not in result.stdout


def test_deploy_test_rejects_missing_image(tmp_path: Path) -> None:
    result = run_script(
        "deploy-test",
        "--dry-run",
        "--image",
        str(tmp_path / "missing.png"),
    )

    assert result.returncode != 0
    assert "图片不存在" in result.stderr


def test_deploy_test_rejects_unsupported_image(tmp_path: Path) -> None:
    image = tmp_path / "frame.bmp"
    image.write_bytes(b"image")

    result = run_script("deploy-test", "--dry-run", "--image", str(image))

    assert result.returncode != 0
    assert "PNG、JPG 或 JPEG" in result.stderr


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--scene", "warehouse", "场景"),
        ("--mirror", "unknown", "镜像源"),
    ],
)
def test_rejects_unsupported_enum_values(
    tmp_path: Path,
    option: str,
    value: str,
    message: str,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"image")

    result = run_script(
        "deploy-test",
        "--dry-run",
        "--image",
        str(image),
        option,
        value,
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_build_is_pinned_to_jetson_variant() -> None:
    result = run_script("build", "--dry-run", "--mirror", "tuna")

    assert result.returncode == 0
    assert "build_perception.sh" in result.stdout
    assert "--variant jetson" in result.stdout
    assert "--mirror tuna" in result.stdout


@pytest.mark.parametrize(
    ("registry", "namespace", "expected_prefix"),
    [
        ("registry.example", "team", "registry.example/team/perception:"),
        ("", "team", "local/team/perception:"),
        ("registry.example", "", "registry.example/phanthy-motus/perception:"),
    ],
)
def test_partial_registry_configuration_matches_build_script(
    tmp_path: Path,
    registry: str,
    namespace: str,
    expected_prefix: str,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"image")

    result = run_script(
        "deploy-test",
        "--dry-run",
        "--image",
        str(image),
        env={
            "REGISTRY": registry,
            "REGISTRY_USER": "",
            "REGISTRY_PASSWORD": "",
            "IMAGE_NAMESPACE": namespace,
        },
    )

    assert result.returncode == 0
    assert expected_prefix in result.stdout


@pytest.mark.parametrize(
    ("filename", "scene", "container_path"),
    [
        ("frame.png", "indoor", "/data/input.png"),
        ("frame.jpg", "outdoor", "/data/input.jpg"),
    ],
)
def test_deploy_test_renders_gpu_inference_command(
    tmp_path: Path,
    filename: str,
    scene: str,
    container_path: str,
) -> None:
    image = tmp_path / filename
    image.write_bytes(b"image")
    model_dir = tmp_path / "models"

    result = run_script(
        "deploy-test",
        "--dry-run",
        "--image",
        str(image),
        "--scene",
        scene,
        "--image-ref",
        "test/perception:jetson",
        "--model-dir",
        str(model_dir),
    )

    assert result.returncode == 0
    assert "--runtime nvidia" in result.stdout
    assert f"{image}:{container_path}:ro" in result.stdout
    assert f"{model_dir}:/models" in result.stdout
    assert "HF_HOME=/models/huggingface" in result.stdout
    assert "OBSTACLE_MODEL_DIR=/models/obstacle_distance" in result.stdout
    assert "OBSTACLE_DEPTH_DEVICE=cuda:0" in result.stdout
    assert "OBSTACLE_DETECTOR_DEVICE=0" in result.stdout
    assert "python3 -m plugins.obstacle_distance.cli" in result.stdout
    assert "--fail-on-backend-error" in result.stdout
    assert f"{container_path} --scene {scene}" in result.stdout
    assert "test/perception:jetson" in result.stdout


def test_auto_scene_uses_image_extension_without_scene_override(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpeg"
    image.write_bytes(b"image")

    result = run_script(
        "deploy-test",
        "--dry-run",
        "--image",
        str(image),
        "--scene",
        "auto",
        "--image-ref",
        "test/perception:jetson",
    )

    assert result.returncode == 0
    inference_line = next(
        line
        for line in result.stdout.splitlines()
        if "plugins.obstacle_distance.cli" in line
    )
    assert "/data/input.jpeg" in inference_line
    assert "--scene" not in inference_line


def test_optional_model_environment_is_forwarded(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"image")

    result = run_script(
        "deploy-test",
        "--dry-run",
        "--image",
        str(image),
        "--image-ref",
        "test/perception:jetson",
        env={"OBSTACLE_DEPTH_INDOOR_MODEL": "/models/custom/indoor"},
    )

    assert result.returncode == 0
    assert "--env OBSTACLE_DEPTH_INDOOR_MODEL" in result.stdout


def test_start_uses_full_jetson_service_runtime(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"

    result = run_script(
        "start",
        "--dry-run",
        "--image-ref",
        "test/perception:jetson",
        "--model-dir",
        str(model_dir),
    )

    assert result.returncode == 0
    for fragment in [
        "docker run -d",
        "--runtime nvidia",
        "--network host",
        "--ipc host",
        "--pid host",
        "--privileged",
        "--restart unless-stopped",
        "/dev:/dev",
        f"{model_dir}:/models",
        "--name embodied-perception-obstacle",
        "test/perception:jetson",
    ]:
        assert fragment in result.stdout
    assert "docker rm" not in result.stdout


def test_stop_never_removes_container() -> None:
    result = run_script("stop", "--dry-run")

    assert result.returncode == 0
    assert "docker stop embodied-perception-obstacle" in result.stdout
    assert "docker rm" not in result.stdout


def test_status_and_logs_use_custom_container_name() -> None:
    status = run_script(
        "status",
        "--dry-run",
        "--container-name",
        "custom-obstacle",
    )
    logs = run_script(
        "logs",
        "--dry-run",
        "--no-follow",
        "--container-name",
        "custom-obstacle",
    )

    assert status.returncode == 0
    assert "docker container inspect custom-obstacle" in status.stdout
    assert logs.returncode == 0
    assert "docker logs --tail 100 custom-obstacle" in logs.stdout
    assert "--follow" not in logs.stdout


def test_following_logs_is_the_default() -> None:
    result = run_script("logs", "--dry-run")

    assert result.returncode == 0
    assert (
        "docker logs --tail 100 --follow embodied-perception-obstacle" in result.stdout
    )

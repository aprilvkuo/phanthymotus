# Obstacle Distance One-Click Deploy and Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe Jetson one-click command that builds with `perception/Dockerfile.jetson`, verifies CUDA, runs one real obstacle-distance inference, and manages the optional Perception service.

**Architecture:** A single strict-mode Bash command owns argument parsing and Docker orchestration while reusing `deploy/build_perception.sh --variant jetson` for image construction. Pytest executes the real script in `--dry-run` mode to validate rendered commands without Docker or GPU access.

**Tech Stack:** Bash, Docker with NVIDIA runtime, NVIDIA Jetson, Python 3.12, pytest.

## Global Constraints

- Docker image construction must use `perception/Dockerfile.jetson` through `deploy/build_perception.sh --variant jetson`; no CPU fallback.
- Do not delete or replace existing containers, images, model directories, or model files.
- Store models outside Git under `/opt/embodied/models` by default.
- Keep model files larger than 1 MB out of the repository.
- All new code comments and operator-facing errors must be Chinese.
- Run tests with `PYTHONPATH=perception` and `uv --no-project` to avoid generating a repository lockfile.

---

### Task 1: Define the deployment command contract with failing tests

**Files:**
- Create: `perception/tests/test_obstacle_distance_deploy_script.py`
- Create: `deploy/obstacle_distance.sh`

**Interfaces:**
- Consumes: `bash deploy/obstacle_distance.sh <subcommand> [options]`
- Produces: `deploy-test`, `build`, `start`, `status`, `logs`, and `stop` command behavior.

- [ ] **Step 1: Write failing tests for help and argument validation**

Create subprocess helpers that execute the real script and tests equivalent to:

```python
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "obstacle_distance.sh"


def run_script(*args: str, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_lists_supported_commands():
    result = run_script("--help")
    assert result.returncode == 0
    for command in ["deploy-test", "build", "start", "status", "logs", "stop"]:
        assert command in result.stdout


def test_deploy_test_requires_existing_supported_image(tmp_path):
    missing = run_script("deploy-test", "--dry-run")
    assert missing.returncode != 0
    assert "--image" in missing.stderr

    unsupported = tmp_path / "frame.bmp"
    unsupported.write_bytes(b"image")
    result = run_script("deploy-test", "--dry-run", "--image", str(unsupported))
    assert result.returncode != 0
    assert "PNG、JPG 或 JPEG" in result.stderr
```

- [ ] **Step 2: Run the target tests and verify RED**

Run:

```bash
PYTHONPATH=perception uv run --no-project --python 3.12 \
  --with pytest --with numpy --with pillow \
  pytest perception/tests/test_obstacle_distance_deploy_script.py -q
```

Expected: FAIL because `deploy/obstacle_distance.sh` does not exist.

- [ ] **Step 3: Add the minimal Bash entry point**

Implement strict mode, usage output, subcommand validation, common option
parsing, required image checks, extension checks, and scene/mirror enums. The
parser must recognize:

```text
--image --scene --mirror --model-dir --image-ref --rebuild
--container-name --dry-run --no-follow --help
```

Do not call Docker yet. Make the file executable.

- [ ] **Step 4: Run the target tests and verify GREEN**

Run the Task 1 test command. Expected: all Task 1 tests pass.

---

### Task 2: Implement dry-run build and inference orchestration

**Files:**
- Modify: `perception/tests/test_obstacle_distance_deploy_script.py`
- Modify: `deploy/obstacle_distance.sh`

**Interfaces:**
- Consumes: validated arguments from Task 1.
- Produces: shell-escaped build, CUDA probe, and inference commands in dry-run; the same argument arrays execute in live mode.

- [ ] **Step 1: Write failing command-rendering tests**

Add tests equivalent to:

```python
def test_build_is_pinned_to_jetson_variant():
    result = run_script("build", "--dry-run", "--mirror", "tuna")
    assert result.returncode == 0
    assert "build_perception.sh" in result.stdout
    assert "--variant jetson" in result.stdout
    assert "--mirror tuna" in result.stdout


@pytest.mark.parametrize(
    ("filename", "scene", "container_path"),
    [("frame.png", "indoor", "/data/input.png"),
     ("frame.jpg", "outdoor", "/data/input.jpg")],
)
def test_deploy_test_renders_gpu_inference_command(
    tmp_path, filename, scene, container_path
):
    image = tmp_path / filename
    image.write_bytes(b"image")
    result = run_script(
        "deploy-test", "--dry-run", "--image", str(image),
        "--scene", scene, "--image-ref", "test/perception:jetson",
        "--model-dir", str(tmp_path / "models"),
    )
    assert result.returncode == 0
    assert "--runtime nvidia" in result.stdout
    assert f"{image}:{container_path}:ro" in result.stdout
    assert "HF_HOME=/models/huggingface" in result.stdout
    assert "OBSTACLE_DEPTH_DEVICE=cuda:0" in result.stdout
    assert "python3 -m plugins.obstacle_distance.cli" in result.stdout
    assert f"{container_path} --scene {scene}" in result.stdout
```

Add a test that `scene=auto` omits `--scene`, and a test that an exported
`OBSTACLE_DEPTH_INDOOR_MODEL` is rendered as a forwarded `--env` argument.

- [ ] **Step 2: Run the new tests and verify RED**

Expected: FAIL because no orchestration command is rendered.

- [ ] **Step 3: Implement image resolution and command arrays**

Add functions with these responsibilities:

```text
resolve_image_ref      derive release.<date>.<commit>-jetson or use override
print_command          print shell-escaped command arguments
run_command            execute or print based on --dry-run
ensure_jetson_image    reuse an existing image or invoke build_perception.sh
append_model_env       add fixed and optional model environment variables
run_cuda_probe         execute torch.cuda availability check in the image
run_single_image_test  mount the image, run CLI, and report distance/time
```

Dry-run must skip host architecture, Docker daemon, NVIDIA runtime, image
existence, and model-directory permission checks. A custom `--image-ref`
bypasses automatic building and is treated as an already supplied image.

- [ ] **Step 4: Run target tests and verify GREEN**

Expected: all deployment-script tests pass.

---

### Task 3: Implement live preflight and safe service lifecycle

**Files:**
- Modify: `perception/tests/test_obstacle_distance_deploy_script.py`
- Modify: `deploy/obstacle_distance.sh`

**Interfaces:**
- Consumes: resolved image, model directory, container name, and optional forwarded model variables.
- Produces: safe live Docker operations with nonzero failures and no automatic deletion.

- [ ] **Step 1: Write failing lifecycle safety tests**

Add dry-run tests asserting:

```python
def test_start_uses_full_jetson_service_runtime(tmp_path):
    result = run_script(
        "start", "--dry-run", "--image-ref", "test/perception:jetson",
        "--model-dir", str(tmp_path / "models"),
    )
    assert result.returncode == 0
    for fragment in [
        "docker run -d", "--runtime nvidia", "--network host",
        "--ipc host", "--pid host", "--privileged",
        "--restart unless-stopped", "/dev:/dev",
    ]:
        assert fragment in result.stdout
    assert "docker rm" not in result.stdout


def test_stop_never_removes_container():
    result = run_script("stop", "--dry-run")
    assert result.returncode == 0
    assert "docker stop embodied-perception-obstacle" in result.stdout
    assert "docker rm" not in result.stdout
```

Also cover `status`, following/non-following `logs`, and custom container name.

- [ ] **Step 2: Run lifecycle tests and verify RED**

Expected: FAIL because lifecycle commands are not rendered.

- [ ] **Step 3: Implement scoped preflight and lifecycle operations**

Live `build`, `deploy-test`, and `start` check ARM64, Docker daemon, NVIDIA
runtime, and model-directory usability. `status`, `logs`, and `stop` check only
the Docker client/daemon. `start` fails when the named container already
exists. `stop` calls only `docker stop`. No path invokes `docker rm`.

The live inference path captures only stdout from the model CLI, propagates a
nonzero status, validates the result as a finite numeric value, and reports
elapsed seconds without hiding stderr diagnostics.

- [ ] **Step 4: Run deployment-script tests and verify GREEN**

Expected: all deployment-script tests pass.

---

### Task 4: Document and package the workflow

**Files:**
- Modify: `perception/README.md`
- Modify: `perception/tests/test_obstacle_distance_packaging.py`

**Interfaces:**
- Consumes: final script command interface.
- Produces: copy-paste Jetson instructions and packaging regression checks.

- [ ] **Step 1: Write failing documentation assertions**

Extend the packaging tests:

```python
def test_docs_describe_one_click_jetson_deploy_test():
    readme = (PERCEPTION / "README.md").read_text()
    assert "deploy/obstacle_distance.sh deploy-test" in readme
    assert "Dockerfile.jetson" in readme
    assert "PYTHONPATH=perception uv run --no-project" in readme
```

Add an assertion that `deploy/obstacle_distance.sh` is executable.

- [ ] **Step 2: Run the packaging tests and verify RED**

Expected: FAIL because README usage is absent.

- [ ] **Step 3: Update the README**

Document one-click indoor/outdoor use, image building, service commands,
online/offline models, unit tests, Jetson-only verification, and that the
configured `fps: 3` is a processing cap rather than a measured benchmark.

- [ ] **Step 4: Run packaging and full tests and verify GREEN**

Run:

```bash
PYTHONPATH=perception uv run --no-project --python 3.12 \
  --with pytest --with numpy --with pillow \
  pytest perception/tests -q
```

Expected: all tests pass.

---

### Task 5: Final verification, commit, and push

**Files:**
- Verify all files changed by Tasks 1-4.

**Interfaces:**
- Consumes: completed implementation.
- Produces: verified commit on `codex/obstacle-distance` and matching GitHub branch.

- [ ] **Step 1: Run shell and repository checks**

```bash
bash -n deploy/obstacle_distance.sh
./deploy/obstacle_distance.sh --help
git diff --check
git status --short
find . -type f -size +1M -not -path './.git/*' -not -path './.venv/*'
```

Confirm no model weights or generated `uv.lock` were added.

- [ ] **Step 2: Run a dry-run smoke test**

Create a temporary PNG outside the repository and run `deploy-test --dry-run`
with an explicit Jetson image reference. Confirm the rendered build and Docker
commands use `--variant jetson`, NVIDIA runtime, read-only image mount, and
persistent model mount.

- [ ] **Step 3: Record Jetson-only gaps honestly**

Do not claim a Docker build, CUDA inference, FPS, or GPU utilization result
unless it was run on the Jetson server. Report these as server verification
steps when unavailable locally.

- [ ] **Step 4: Commit intended files**

```bash
git add deploy/obstacle_distance.sh \
  perception/tests/test_obstacle_distance_deploy_script.py \
  perception/tests/test_obstacle_distance_packaging.py \
  perception/README.md \
  docs/superpowers/plans/2026-08-05-obstacle-distance-deploy-test.md
git commit -m "feat: add one-click obstacle deployment test"
```

- [ ] **Step 5: Push and verify the remote commit**

```bash
git push origin codex/obstacle-distance
git ls-remote origin refs/heads/codex/obstacle-distance
git rev-parse HEAD
```

Expected: local and remote commit IDs match.

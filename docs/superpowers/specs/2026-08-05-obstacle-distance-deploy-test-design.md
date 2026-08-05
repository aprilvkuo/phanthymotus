# Obstacle Distance One-Click Deploy and Test Design

## 1. Goal

Provide one Jetson-oriented command that validates the runtime environment,
builds or reuses the perception image, verifies CUDA access, and performs a
real single-image nearest-obstacle inference. Keep long-running Perception
service lifecycle operations available as explicit subcommands.

The feature must not delete existing images, containers, or model files, and
must not commit model weights to Git.

## 2. Selected Approach

Add one Bash entry point at `deploy/obstacle_distance.sh`. A single command is
preferred over multiple scripts because image selection, model mounts,
NVIDIA runtime arguments, and error handling stay consistent across build,
test, and service-management operations.

The script reuses `deploy/build_perception.sh --variant jetson` for image
construction. That variant is required to resolve to
`perception/Dockerfile.jetson`; the one-click workflow must never fall back to
the CPU `perception/Dockerfile`. It invokes the existing in-container module entry point
`python3 -m plugins.obstacle_distance.cli` for single-image inference, so the
Docker image does not need to contain repository tests or the top-level judge
wrapper.

## 3. Command Interface

### 3.1 One-click deploy and test

```bash
./deploy/obstacle_distance.sh deploy-test \
  --image /absolute/path/indoor.png \
  --scene indoor
```

`deploy-test` performs these steps in order:

1. Validate the host, Docker, NVIDIA runtime, arguments, and model directory.
2. Resolve the deterministic image reference for the current Git commit.
3. Reuse the image when it already exists, unless `--rebuild` is supplied.
4. Otherwise call `deploy/build_perception.sh --variant jetson` with the
   selected mirror. This must build from `perception/Dockerfile.jetson`.
5. Run a small PyTorch CUDA probe inside the image.
6. Mount the input image and persistent model directory read-only/read-write as
   appropriate.
7. Run one real inference and propagate its exit status.
8. Print the image reference, predicted distance, and wall-clock duration.

### 3.2 Service lifecycle

```bash
./deploy/obstacle_distance.sh build
./deploy/obstacle_distance.sh start
./deploy/obstacle_distance.sh status
./deploy/obstacle_distance.sh logs
./deploy/obstacle_distance.sh stop
```

- `build` builds or reuses the current Jetson image. `--rebuild` forces a new
  build.
- `start` starts the full Perception service in a named container using host
  networking and the NVIDIA runtime. It fails if a container with the same
  name already exists; it never removes or replaces that container silently.
- `status` prints the container status and exits nonzero when the container
  does not exist.
- `logs` follows the container logs. `--no-follow` prints the latest logs and
  returns.
- `stop` stops the named container but does not remove it.

The default container name is `embodied-perception-obstacle`. It can be
overridden with `--container-name`.

### 3.3 Options

| Option | Applies to | Default | Meaning |
|---|---|---|---|
| `--image PATH` | `deploy-test` | required | Absolute or relative PNG/JPG/JPEG input path |
| `--scene VALUE` | `deploy-test` | `auto` | `auto`, `indoor`, or `outdoor` |
| `--mirror VALUE` | build operations | `tuna` | `tencent`, `tuna`, or `none` |
| `--model-dir PATH` | test/start | `/opt/embodied/models` | Persistent host model directory |
| `--image-ref REF` | all image operations | derived | Explicit Docker image override |
| `--rebuild` | build operations | false | Force image construction |
| `--container-name NAME` | service operations | `embodied-perception-obstacle` | Service container name |
| `--dry-run` | mutating/inference operations | false | Print resolved commands without executing them |
| `--no-follow` | `logs` | false | Do not follow log output |
| `--help` | all | n/a | Print usage |

Unknown options, missing option values, unsupported scenes or mirrors, and a
missing image cause an explanatory error and a nonzero exit status.

`--dry-run` still validates command-line values and the input image, but skips
host architecture, Docker daemon, NVIDIA runtime, image existence, and model
directory permission checks. It prints shell-escaped commands and performs no
build, container, directory, download, or inference operation.

## 4. Image Resolution and Build Behavior

Without `--image-ref`, the script uses the same tag convention as the existing
Jetson build script:

```text
<registry>/<namespace>/perception:release.<YYMMDD>.<short-commit>-jetson
```

When registry credentials are not fully configured in `deploy/.env`, the
repository and namespace resolve to:

```text
local/phanthy-motus/perception
```

When all existing registry settings are present, the script resolves the same
registry and namespace as `build_perception.sh`. `--image-ref` bypasses derived
image naming and lets operators use a known image directly. `deploy-test`
checks whether the resolved image exists locally before deciding to build.

## 5. Environment and Model Handling

The Jetson preflight requires:

- Host architecture `aarch64` or `arm64`.
- `docker` available and the Docker daemon reachable.
- NVIDIA runtime advertised by Docker.
- A usable model directory.
- A readable PNG, JPG, or JPEG image for `deploy-test`.

The script mounts the host model directory at `/models` and sets:

```text
HF_HOME=/models/huggingface
OBSTACLE_MODEL_DIR=/models/obstacle_distance
OBSTACLE_DEPTH_DEVICE=cuda:0
OBSTACLE_DETECTOR_DEVICE=0
```

If the following variables are already set on the host, they are forwarded to
the container without changing their values:

- `OBSTACLE_DEPTH_INDOOR_MODEL`
- `OBSTACLE_DEPTH_OUTDOOR_MODEL`
- `OBSTACLE_DETECTOR_MODEL`
- `OBSTACLE_DETECTOR_MODEL_URL`
- `OBSTACLE_DETECTOR_MODEL_SHA256`
- `OBSTACLE_OUTDOOR_COMPENSATION_M`
- `OBSTACLE_FALLBACK_DISTANCE_M`
- `OBSTACLE_DETECTION_CONFIDENCE`

The first online inference may download weights into the persistent mount.
Offline use requires the operator to provide local model paths through the
forwarded variables.

## 6. Container Execution

Single-image inference uses an ephemeral container with `--rm`, NVIDIA runtime,
and only the required model and image mounts. The image is mounted read-only.
The output from the inference module remains the predicted floating-point
distance; wrapper progress and timing messages are clearly labeled.

The long-running service uses:

- NVIDIA runtime
- host network, IPC, and PID namespaces
- privileged mode and `/dev:/dev`
- `/opt/embodied/models` or the configured host model directory mounted at
  `/models`
- restart policy `unless-stopped`
- ports already exposed through host networking

No command removes an existing service container automatically.

## 7. Error Handling

The script uses strict Bash mode and returns nonzero when any required step
fails. Error messages identify the failed boundary:

- unsupported host architecture
- Docker command or daemon unavailable
- NVIDIA runtime unavailable
- inaccessible image or model directory
- image build failure
- CUDA unavailable inside the image
- model download/load failure
- inference process failure
- service container absent or already present

Temporary timing files are created with `mktemp` and removed through a trap.
No broad or recursive deletion is performed.

## 8. Testing Strategy

Add `perception/tests/test_obstacle_distance_deploy_script.py`. Tests invoke the
real Bash entry point in `--dry-run` mode and use temporary input images, so
they do not require Docker or a GPU.

Coverage includes:

- help output and supported subcommands
- required image validation
- accepted and rejected scene values
- accepted and rejected mirror values
- indoor and outdoor container image mounts
- NVIDIA runtime and persistent model mounts
- CUDA-related environment variables
- explicit image reference handling
- fixed use of the Jetson build variant and `perception/Dockerfile.jetson`
- forwarded optional model variables
- safe lifecycle behavior, including no silent `docker rm`
- executable file mode and README usage examples

The full obstacle-distance test suite runs with:

```bash
PYTHONPATH=perception uv run --no-project --python 3.12 \
  --with pytest --with numpy --with pillow \
  pytest perception/tests -q
```

Shell syntax is checked with `bash -n deploy/obstacle_distance.sh`. A local
dry-run smoke test validates the resolved command. Real CUDA inference, FPS,
and GPU utilization remain Jetson-only verification steps and are reported as
such.

## 9. Documentation

Update `perception/README.md` with:

- the one-click Jetson command
- supported service lifecycle commands
- online and offline model behavior
- the unit-test command with `PYTHONPATH=perception`
- the distinction between the 3 FPS processing cap and measured performance

## 10. Delivery

After implementation and verification, commit the script, tests, and README to
`codex/obstacle-distance` and push the branch to
`https://github.com/aprilvkuo/phanthymotus.git`. Do not create a pull request
unless explicitly requested.

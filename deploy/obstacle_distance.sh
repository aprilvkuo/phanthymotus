#!/usr/bin/env bash
# Jetson 最近障碍物距离一键部署与测试入口。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_SCRIPT="${SCRIPT_DIR}/build_perception.sh"

usage() {
    cat <<'EOF'
用法：
  ./deploy/obstacle_distance.sh deploy-test --image PATH [选项]
  ./deploy/obstacle_distance.sh build [选项]
  ./deploy/obstacle_distance.sh start [选项]
  ./deploy/obstacle_distance.sh status [选项]
  ./deploy/obstacle_distance.sh logs [--no-follow] [选项]
  ./deploy/obstacle_distance.sh stop [选项]

命令：
  deploy-test  检查环境、构建或复用 Jetson 镜像并执行单图推理
  build        构建或复用 Jetson 镜像
  start        启动完整 Perception 服务
  status       查看服务容器状态
  logs         查看服务容器日志
  stop         停止服务容器（不删除）

选项：
  --image PATH             PNG、JPG 或 JPEG 测试图片
  --scene VALUE            auto、indoor 或 outdoor，默认 auto
  --mirror VALUE           tencent、tuna 或 none，默认 tuna
  --model-dir PATH         宿主机模型目录，默认 /opt/embodied/models
  --image-ref REF          使用指定 Docker 镜像
  --rebuild                强制重新构建镜像
  --container-name NAME    服务容器名称
  --dry-run                只打印命令，不访问 Docker 或 GPU
  --no-follow              logs 命令不持续跟踪
  -h, --help               显示帮助
EOF
}

die() {
    printf '错误：%s\n' "$*" >&2
    exit 2
}

require_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "${value}" ]] || die "${option} 缺少参数值"
}

print_argument() {
    local value="$1"
    if [[ "${value}" =~ ^[a-zA-Z0-9_./:=+@%,-]+$ ]]; then
        printf '%s' "${value}"
        return
    fi
    local escaped
    escaped="$(printf '%s' "${value}" | sed "s/'/'\\\\''/g")"
    printf "'%s'" "${escaped}"
}

print_command() {
    printf '+ '
    local argument
    for argument in "$@"; do
        print_argument "${argument}"
        printf ' '
    done
    printf '\n'
}

run_command() {
    if ${DRY_RUN}; then
        print_command "$@"
        return 0
    fi
    "$@"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"
}

check_docker() {
    ${DRY_RUN} && return 0
    require_command docker
    docker info >/dev/null 2>&1 || die "Docker daemon 不可用，请检查 Docker 服务和当前用户权限"
}

check_jetson_runtime() {
    ${DRY_RUN} && return 0
    local architecture
    architecture="$(uname -m)"
    case "${architecture}" in
        aarch64|arm64) ;;
        *) die "当前主机不是 Jetson ARM64：${architecture}" ;;
    esac
    check_docker
    if ! docker info --format '{{json .Runtimes}}' | grep -qi nvidia; then
        die "Docker 未配置 NVIDIA runtime"
    fi
}

prepare_model_dir() {
    ${DRY_RUN} && return 0
    mkdir -p "${MODEL_DIR}" 2>/dev/null \
        || die "无法创建模型目录：${MODEL_DIR}，请预先创建并授予当前用户权限"
    [[ -d "${MODEL_DIR}" && -w "${MODEL_DIR}" ]] \
        || die "模型目录不可写：${MODEL_DIR}"
}

resolve_image_ref() {
    if [[ -n "${IMAGE_REF}" ]]; then
        CUSTOM_IMAGE_REF=true
        return
    fi

    local env_file="${SCRIPT_DIR}/.env"
    if [[ -f "${env_file}" ]]; then
        # 与现有构建脚本使用相同的 Registry 配置。
        set +u
        # shellcheck disable=SC1090
        source "${env_file}"
        set -u
    fi

    local registry="local"
    local namespace="phanthy-motus"
    if [[ -n "${REGISTRY:-}" && -n "${REGISTRY_USER:-}" \
        && -n "${REGISTRY_PASSWORD:-}" && -n "${IMAGE_NAMESPACE:-}" ]]; then
        registry="${REGISTRY}"
        namespace="${IMAGE_NAMESPACE}"
    fi

    local commit
    commit="$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)"
    IMAGE_REF="${registry}/${namespace}/perception:release.$(date +%y%m%d).${commit}-jetson"
}

ensure_jetson_image() {
    if ${CUSTOM_IMAGE_REF}; then
        if ! ${DRY_RUN} && ! docker image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
            die "指定镜像在本机不存在：${IMAGE_REF}"
        fi
        printf '[image] 使用指定镜像：%s\n' "${IMAGE_REF}"
        return 0
    fi

    if ! ${DRY_RUN} && ! ${REBUILD} && docker image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
        printf '[image] 复用已有镜像：%s\n' "${IMAGE_REF}"
        return 0
    fi

    run_command "${BUILD_SCRIPT}" --variant jetson --mirror "${MIRROR}"
    if ! ${DRY_RUN} && ! docker image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
        die "Jetson 镜像构建结束后未找到：${IMAGE_REF}"
    fi
}

build_model_env_args() {
    MODEL_ENV_ARGS=(
        --env "HF_HOME=/models/huggingface"
        --env "OBSTACLE_MODEL_DIR=/models/obstacle_distance"
        --env "OBSTACLE_DEPTH_DEVICE=cuda:0"
        --env "OBSTACLE_DETECTOR_DEVICE=0"
    )

    local name
    for name in \
        OBSTACLE_DEPTH_INDOOR_MODEL \
        OBSTACLE_DEPTH_OUTDOOR_MODEL \
        OBSTACLE_DETECTOR_MODEL \
        OBSTACLE_DETECTOR_MODEL_URL \
        OBSTACLE_DETECTOR_MODEL_SHA256 \
        OBSTACLE_OUTDOOR_COMPENSATION_M \
        OBSTACLE_FALLBACK_DISTANCE_M \
        OBSTACLE_DETECTION_CONFIDENCE; do
        if env | grep -q "^${name}="; then
            MODEL_ENV_ARGS+=(--env "${name}")
        fi
    done
}

run_cuda_probe() {
    run_command docker run --rm --runtime nvidia \
        "${IMAGE_REF}" \
        python3 -c \
        'import sys, torch; ok=torch.cuda.is_available(); print("CUDA:", ok); sys.exit(0 if ok else 1)'
}

resolve_input_path() {
    local directory
    directory="$(cd "$(dirname "${IMAGE_PATH}")" && pwd -P)"
    IMAGE_PATH="${directory}/$(basename "${IMAGE_PATH}")"
    case "${IMAGE_PATH}" in
        *.png|*.PNG) CONTAINER_IMAGE="/data/input.png" ;;
        *.jpg|*.JPG) CONTAINER_IMAGE="/data/input.jpg" ;;
        *.jpeg|*.JPEG) CONTAINER_IMAGE="/data/input.jpeg" ;;
    esac
}

run_single_image_test() {
    local command=(
        docker run --rm --runtime nvidia
        --volume "${MODEL_DIR}:/models"
        --volume "${IMAGE_PATH}:${CONTAINER_IMAGE}:ro"
        "${MODEL_ENV_ARGS[@]}"
        "${IMAGE_REF}"
        python3 -m plugins.obstacle_distance.cli
        "${CONTAINER_IMAGE}"
    )
    if [[ "${SCENE}" != "auto" ]]; then
        command+=(--scene "${SCENE}")
    fi

    if ${DRY_RUN}; then
        print_command "${command[@]}"
        return 0
    fi

    local started="${SECONDS}"
    local distance
    local status
    set +e
    distance="$("${command[@]}")"
    status=$?
    set -e
    [[ ${status} -eq 0 ]] || die "单图推理失败，退出码：${status}"
    if [[ ! "${distance}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        die "单图推理未返回有效距离：${distance}"
    fi
    printf '[image] %s\n' "${IMAGE_REF}"
    printf '[result] distance_m=%s\n' "${distance}"
    printf '[result] elapsed_seconds=%s\n' "$((SECONDS - started))"
}

start_service() {
    if ! ${DRY_RUN} && docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        die "容器已存在，不会自动替换：${CONTAINER_NAME}"
    fi
    local command=(
        docker run -d
        --name "${CONTAINER_NAME}"
        --runtime nvidia
        --network host
        --ipc host
        --pid host
        --privileged
        --restart unless-stopped
        --volume /dev:/dev
        --volume "${MODEL_DIR}:/models"
        --env ROS_DOMAIN_ID=42
        --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp
        "${MODEL_ENV_ARGS[@]}"
        "${IMAGE_REF}"
    )
    run_command "${command[@]}"
}

show_status() {
    if ${DRY_RUN}; then
        print_command docker container inspect "${CONTAINER_NAME}"
        return 0
    fi
    docker container inspect \
        --format 'name={{.Name}} status={{.State.Status}} image={{.Config.Image}}' \
        "${CONTAINER_NAME}" \
        || die "容器不存在：${CONTAINER_NAME}"
}

show_logs() {
    local command=(docker logs --tail 100)
    if ! ${NO_FOLLOW}; then
        command+=(--follow)
    fi
    command+=("${CONTAINER_NAME}")
    run_command "${command[@]}"
}

stop_service() {
    run_command docker stop "${CONTAINER_NAME}"
}

COMMAND="${1:-}"
if [[ -z "${COMMAND}" || "${COMMAND}" == "-h" || "${COMMAND}" == "--help" ]]; then
    usage
    exit 0
fi
shift

case "${COMMAND}" in
    deploy-test|build|start|status|logs|stop) ;;
    *) die "不支持的命令：${COMMAND}" ;;
esac

IMAGE_PATH=""
SCENE="auto"
MIRROR="tuna"
MODEL_DIR="/opt/embodied/models"
IMAGE_REF=""
REBUILD=false
CONTAINER_NAME="embodied-perception-obstacle"
DRY_RUN=false
NO_FOLLOW=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            require_value "$1" "${2:-}"
            IMAGE_PATH="$2"
            shift 2
            ;;
        --scene)
            require_value "$1" "${2:-}"
            SCENE="$2"
            shift 2
            ;;
        --mirror)
            require_value "$1" "${2:-}"
            MIRROR="$2"
            shift 2
            ;;
        --model-dir)
            require_value "$1" "${2:-}"
            MODEL_DIR="$2"
            shift 2
            ;;
        --image-ref)
            require_value "$1" "${2:-}"
            IMAGE_REF="$2"
            shift 2
            ;;
        --container-name)
            require_value "$1" "${2:-}"
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-follow)
            NO_FOLLOW=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) die "不支持的选项：$1" ;;
    esac
done

case "${SCENE}" in
    auto|indoor|outdoor) ;;
    *) die "不支持的场景：${SCENE}，可选 auto、indoor、outdoor" ;;
esac

case "${MIRROR}" in
    tencent|tuna|none) ;;
    *) die "不支持的镜像源：${MIRROR}，可选 tencent、tuna、none" ;;
esac

if [[ "${COMMAND}" == "deploy-test" ]]; then
    [[ -n "${IMAGE_PATH}" ]] || die "deploy-test 必须提供 --image PATH"
    [[ -f "${IMAGE_PATH}" ]] || die "图片不存在：${IMAGE_PATH}"
    case "${IMAGE_PATH}" in
        *.png|*.PNG|*.jpg|*.JPG|*.jpeg|*.JPEG) ;;
        *) die "测试图片必须是 PNG、JPG 或 JPEG：${IMAGE_PATH}" ;;
    esac
fi

case "${MODEL_DIR}" in
    /*) ;;
    *) MODEL_DIR="${PWD}/${MODEL_DIR}" ;;
esac

CUSTOM_IMAGE_REF=false
MODEL_ENV_ARGS=()
CONTAINER_IMAGE=""
resolve_image_ref
build_model_env_args

case "${COMMAND}" in
    build)
        check_jetson_runtime
        ensure_jetson_image
        ;;
    deploy-test)
        check_jetson_runtime
        prepare_model_dir
        resolve_input_path
        ensure_jetson_image
        run_cuda_probe
        run_single_image_test
        ;;
    start)
        check_jetson_runtime
        prepare_model_dir
        ensure_jetson_image
        start_service
        ;;
    status)
        check_docker
        show_status
        ;;
    logs)
        check_docker
        show_logs
        ;;
    stop)
        check_docker
        stop_service
        ;;
esac

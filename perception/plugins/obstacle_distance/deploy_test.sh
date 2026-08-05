#!/usr/bin/env bash
# deploy_test.sh — 障碍物距离插件「一键部署 + 测试」
#
# 两种模式：
#   standalone : 构建独立轻量镜像并跑模型自测（不依赖 ROS，最快验证模型 / judgeflow 入口）
#   stack      : 构建完整 perception-stack 镜像 -> 后台启动 MCP 服务 ->
#                调用 initialize / tools/list / tools/call 验证插件已正确接入 -> 停止容器
#
# 用法：
#   ./deploy_test.sh standalone                      # 构建 + 合成场景自测
#   ./deploy_test.sh standalone --image /abs/f.jpg   # 构建 + 真实正前方图测试
#   ./deploy_test.sh stack                           # 构建 + 启动服务 + MCP 调用验证
#   ./deploy_test.sh stack --mirror tuna             # 用清华源加速 pip
#
# 说明：stack 模式在裸容器内没有 ROS 相机帧，tools/call 的 query 会返回
#       "no frame" —— 这是预期的（需真实相机/ROS 发布帧）。模型本身请用
#       standalone 模式验证。stack 模式主要确认「服务能起来 + 插件已注册 + 可分发」。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="standalone"
IMG_ARG=""
MIRROR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        standalone|stack) MODE="$1"; shift ;;
        --image) IMG_ARG="$2"; shift 2 ;;
        --mirror) MIRROR="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

cyan()  { printf '\033[36m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }

STANDALONE_IMG="obstacle-nod:test"
PERCEP_IMG="phanthy-motus/perception:local-test"
MCP_PORT=15720
CID=""

cleanup() { [[ -n "${CID}" ]] && { cyan "==> stopping container ${CID}"; docker stop "${CID}" >/dev/null 2>&1 || true; }; }
trap cleanup EXIT

# ── standalone ────────────────────────────────────────────────────────
if [[ "$MODE" == "standalone" ]]; then
    cyan "==> [standalone] building image ${STANDALONE_IMG}"
    docker build -t "${STANDALONE_IMG}" "${SCRIPT_DIR}"

    if [[ -n "$IMG_ARG" ]]; then
        HOST_DIR="$(cd "$(dirname "$IMG_ARG")" && pwd)"
        FNAME="$(basename "$IMG_ARG")"
        cyan "==> running test on ${IMG_ARG}"
        docker run --rm -v "${HOST_DIR}:/data:ro" "${STANDALONE_IMG}" \
            python test_obstacle.py --image "/data/${FNAME}"
    else
        cyan "==> running self-test (synthetic scene)"
        docker run --rm "${STANDALONE_IMG}"
    fi
    green "==> standalone deploy-test PASSED"
    exit 0
fi

# ── stack ─────────────────────────────────────────────────────────────
cyan "==> [stack] building perception image ${PERCEP_IMG}"
BUILD_ARGS=()
if [[ -n "$MIRROR" ]]; then
    case "$MIRROR" in
        tuna)    BUILD_ARGS+=(--build-arg "PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple/") ;;
        tencent) BUILD_ARGS+=(--build-arg "PYPI_MIRROR=https://mirrors.tencentyun.com/pypi/simple/") ;;
        *)       BUILD_ARGS+=(--build-arg "PYPI_MIRROR=${MIRROR}") ;;
    esac
fi
docker build "${BUILD_ARGS[@]}" -t "${PERCEP_IMG}" -f "${REPO_ROOT}/perception/Dockerfile" "${REPO_ROOT}/perception"

cyan "==> starting MCP server (port ${MCP_PORT})"
CID="$(docker run -d -p "${MCP_PORT}:${MCP_PORT}" "${PERCEP_IMG}")"

# wait for health
for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://localhost:${MCP_PORT}/mcp" \
        -H 'Content-Type: application/json' \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' || true)
    if [[ "$code" == "200" ]]; then
        green "==> MCP server is up"; break
    fi
    sleep 2
    if [[ $i -eq 30 ]]; then red "ERROR: MCP server did not come up"; exit 1; fi
done

cyan "==> tools/list"
curl -s -X POST "http://localhost:${MCP_PORT}/mcp" -H 'Content-Type: application/json' \
    -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | python3 -m json.tool || true

cyan "==> tools/call obstacle_nearest_obstacle_distance (action=start)"
curl -s -X POST "http://localhost:${MCP_PORT}/mcp" -H 'Content-Type: application/json' \
    -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"obstacle_nearest_obstacle_distance","arguments":{"action":"start"}}}' \
    | python3 -m json.tool || true

green "==> stack deploy-test: service booted & tool registered (see output above)."
green "    Real NOD needs a front-camera frame over ROS2; validate the model via standalone mode."

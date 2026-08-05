#!/usr/bin/env bash
# 一键部署测试脚本 (one-click deploy + test) for the NOD benchmark solution.
#
# 在 fork 根目录执行：
#   bash deploy_and_test.sh            # 默认：本机 venv 方式部署并测试
#   bash deploy_and_test.sh docker     # Docker 方式（build + 容器内测试）
#   bash deploy_and_test.sh help       # 查看说明
#
# 该脚本会进入 nod_perception/ 子目录，自动建 venv / 装依赖，
# 然后运行：单元测试 + 合成自测 + predict() 烟雾测试。
set -e

case "${1:-}" in
  help|-h|--help)
    sed -n '2,11p' "$0"; exit 0 ;;
esac

# 定位到 nod_perception/（兼容从根目录或子目录调用）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/nod_perception" ]; then
  APP="$SCRIPT_DIR/nod_perception"
else
  APP="$SCRIPT_DIR"
fi
cd "$APP"
echo "== working dir: $APP =="

MODE="${1:-native}"
case "$MODE" in
  docker)
    echo "== [docker] build + 容器内一键测试 =="
    docker compose build
    docker compose run --rm nod-test bash scripts/test_server.sh
    ;;
  native|*)
    echo "== [native] venv 部署 + 测试 =="
    bash scripts/test_server.sh
    ;;
esac

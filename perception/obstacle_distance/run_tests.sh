#!/usr/bin/env bash
#
# 本地一键测试脚本：从脚本自身目录出发，安装依赖并运行 pytest。
#
# 用法:
#   bash run_tests.sh
#
# 说明:
#   - 默认按 requirements.txt 安装全部依赖（含 torch 等）后跑测试。
#   - torch 相关用例在缺少权重/无 GPU 时会自动跳过；纯 numpy 用例始终可跑。
#   - 若希望“不安装 torch”以加速本地验证，可将下方 -r requirements.txt 改为
#     -r requirements.txt 并手动剔除 torch 行，或临时指定 --no-torch 依赖清单；
#     本脚本保持默认走 requirements.txt，确保与 CI 行为一致。
#
set -euo pipefail

cd "$(dirname "$0")"

echo "==> 安装依赖 (requirements.txt)"
python3 -m pip install -q -r requirements.txt

echo "==> 运行测试"
python3 -m pytest tests/ -q -rs

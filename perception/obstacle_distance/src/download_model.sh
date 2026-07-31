#!/usr/bin/env bash
# 从 juicefs(172.28.4.81:34567) 拉取模型权重/引擎到 models/
# 用法: bash src/download_model.sh [version] [target]
set -euo pipefail

BASE_URL="${JUICEFS_URL:-http://172.28.4.81:34567}"
VERSION="${1:-latest}"
TARGET="${2:-models}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/${TARGET}"
mkdir -p "${OUT_DIR}"

FILES=("model.onnx" "model.trt" "ckpt.pt")

echo ">> 从 ${BASE_URL} 拉取版本 ${VERSION} 到 ${OUT_DIR}"
for f in "${FILES[@]}"; do
  url="${BASE_URL}/${VERSION}/${f}"
  dest="${OUT_DIR}/${f}"
  if curl -fSL "${url}" -o "${dest}.tmp"; then
    mv "${dest}.tmp" "${dest}"
    if command -v sha256sum >/dev/null 2>&1; then
      echo "   ${f}: $(sha256sum "${dest}" | cut -d' ' -f1)"
    fi
  else
    echo "   跳过(不存在): ${url}"
    rm -f "${dest}.tmp"
  fi
done
echo ">> 完成。缺失文件为占位，请确认 juicefs 上已发布对应版本。"

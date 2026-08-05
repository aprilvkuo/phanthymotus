#!/usr/bin/env bash
# docker_test.sh — Build & run the standalone Obstacle-Distance test image.
#
# Usage:
#   ./docker_test.sh                 # build + run self-test (synthetic scene)
#   ./docker_test.sh --image /abs/path/frame.jpg
#                                   # build (if needed) + run on a real image
#   ./docker_test.sh --no-build --image /abs/path/frame.jpg
#                                   # skip build, just run
#
# The script runs the container with the host image dir mounted read-only at
# /data so --image /data/xxx.jpg resolves inside the container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="obstacle-nod:test"

BUILD=true
IMG_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD=false; shift ;;
        --image)    IMG_ARG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if ${BUILD}; then
    echo "==> Building ${IMAGE} ..."
    docker build -t "${IMAGE}" "${SCRIPT_DIR}"
fi

if [ -n "${IMG_ARG}" ]; then
    HOST_DIR="$(cd "$(dirname "${IMG_ARG}")" && pwd)"
    FNAME="$(basename "${IMG_ARG}")"
    echo "==> Running on ${IMG_ARG} (mounted at /data/${FNAME}) ..."
    docker run --rm -v "${HOST_DIR}:/data:ro" "${IMAGE}" \
        python test_obstacle.py --image "/data/${FNAME}"
else
    echo "==> Running self-test (synthetic scene) ..."
    docker run --rm "${IMAGE}"
fi

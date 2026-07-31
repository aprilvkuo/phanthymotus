#!/usr/bin/bash
# Download the learned-model weights from juicefs (run inside the Jetson image,
# or locally for testing). Weights are NOT committed — they are pulled at runtime.
#
# Usage: bash scripts/download_model.sh [filename]
set -e
HOST="http://172.28.4.81:34567"
FILENAME="${1:-nod_depth_refiner.pt}"
CACHE_DIR=".cache/models"
mkdir -p "$CACHE_DIR"
OUT="$CACHE_DIR/$FILENAME"
echo "Downloading $HOST/$FILENAME -> $OUT"
curl -fsSL "$HOST/$FILENAME" -o "$OUT"
echo "Done."

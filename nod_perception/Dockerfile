# NOD benchmark solution — containerized test/run image.
#
# The DEFAULT geometric path is CPU-only and needs just numpy, so a plain
# python:slim image is enough to test it on any x86_64 Linux server.
#
# For the ACTUAL Jetson Orin deployment (arm64 + CUDA), swap the base image
# for an L4T one, e.g.:
#   FROM nvcr.io/nvidia/l4t-base:r36.4.0
# (and, if you enable the optional learned model, use l4t-pytorch + a
#  Jetson-built torch). The geometric path itself runs unchanged on arm64.
FROM python:3.11-slim

WORKDIR /app

# curl is handy for the optional juicefs weight download at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the unit-test suite. Override at runtime, e.g.:
#   docker run --rm nod-benchmark python -m nod.evaluate --demo 50
#   docker run --rm nod-benchmark bash scripts/test_server.sh
CMD ["python", "tests/test_geometric.py"]

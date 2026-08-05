# ── Standalone Obstacle-Distance (NOD) model — CPU test/build image ──
# Lightweight image to test the judgeflow entry (predict.py / benchmark.py)
# WITHOUT the ROS2 stack. Use this to validate the model on a plain server.
#
# Build (from the obstacle_distance/ dir):
#   docker build -t obstacle-nod:test .
#
# Run the self-test (synthetic scene, no data needed):
#   docker run --rm obstacle-nod:test
#
# Run on a real image:
#   docker run --rm -v /path/to/images:/data obstacle-nod:test \
#       python test_obstacle.py --image /data/frame.jpg
#
# Or the judgeflow entry directly:
#   docker run --rm -v /path/to/images:/data obstacle-nod:test \
#       python predict.py --image /data/frame.jpg --json
#
# Offline weights: mount a pre-populated torch hub cache and point TORCH_HOME:
#   docker run --rm -e TORCH_HOME=/cache -v /host/cache:/cache obstacle-nod:test
# ───────────────────────────────────────────────────────────────────
FROM python:3.11-slim

ARG PYPI_MIRROR=https://pypi.org/simple/
ENV PYPI_MIRROR=${PYPI_MIRROR}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ffmpeg && rm -rf /var/lib/apt/lists/*

# CPU torch (keeps image small; Jetson/TensorRT handled by the perception image)
RUN pip3 install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url ${PYPI_MIRROR} \
        torch torchvision

RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
        timm pillow numpy

WORKDIR /work
COPY requirements.txt /work/requirements.txt
COPY model/ /work/model/
COPY predict.py benchmark.py test_obstacle.py export_trt.py /work/

# Default: run the self-test (synthetic scene) so `docker run` validates the build.
CMD ["python", "test_obstacle.py"]

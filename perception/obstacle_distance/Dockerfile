# NOD 推理镜像（占位 / 榜单自动拉取）
#
# 基础镜像：Jetson JetPack ML 镜像（含 CUDA/cuDNN/TensorRT）。
# ROS2 侧请使用对应 ros2 基础镜像并安装 rclpy。
# 注：开发/CI 可用通用 CUDA 镜像替代；Jetson 设备请用 L4T 镜像。
FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# 系统依赖（ROS2 占位，按需启用）
# RUN apt-get update && apt-get install -y ros-humble-desktop python3-rosdep ...

WORKDIR /app
COPY perception/obstacle_distance /app/perception/obstacle_distance
WORKDIR /app/perception/obstacle_distance

# Python 依赖（Jetson）
COPY perception/obstacle_distance/requirements_jetson.txt /app/requirements_jetson.txt
RUN pip install --no-cache-dir -r /app/requirements_jetson.txt || true

# 拉取模型权重/引擎（juicefs）
RUN bash src/download_model.sh latest models || true

# 推理入口（默认 torch 后端；Jetson 可改 trt）
ENV NOD_BACKEND=torch
CMD ["python", "src/infer.py", "--image", "/app/sample.jpg"]

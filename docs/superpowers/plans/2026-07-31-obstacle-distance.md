# Nearest Obstacle Distance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 PhanthyMotus Perception Stack 中增加单目最近障碍物距离估计能力，并提供榜单可调用的单图 CLI。

**Architecture:** 核心包由纯 Python/NumPy 的场景识别、ROI/P1 后处理和结果融合组成，通过 `DepthBackend`、`DetectorBackend` 协议隔离重型模型。默认后端使用 Hugging Face Transformers 的 Depth Anything V2 Metric Small，并在无人车场景使用 YOLOv8n 过滤有效障碍物；ROS2 插件和 CLI 共用同一个 `ObstacleDistanceEstimator`。

**Tech Stack:** Python 3.12（开发测试）、Python 3.8（JetPack 5.1.1 容器兼容）、NumPy、OpenCV、PyTorch、Transformers、Ultralytics、ROS2 Humble、pytest。

## Global Constraints

- 推理输入为单张 PNG（29 mm 等效焦距）或 JPG/JPEG（33 mm 等效焦距）。
- 输出为单位米的单个有限浮点数。
- 室内 ROI 为原图列 `[1/3, 2/3]`、行 `[0, 5/8]`，使用有效深度 P1。
- 无人车只保留车辆、行人、自行车、摩托车等允许类别。
- Depth Anything V2 Small 24.8M 参数与 YOLOv8n 3.157M 参数合计约 27.96M。
- Git 仓库不得加入任何大于 1 MB 的模型文件。
- 模型通过配置 URL 下载到 `/models`，下载采用临时文件和原子替换。
- 核心单元测试不得依赖 ROS2、CUDA、Torch 或真实模型下载。

---

### Task 1: 核心类型、场景识别和 ROI/P1 后处理

**Files:**
- Create: `perception/plugins/obstacle_distance/__init__.py`
- Create: `perception/plugins/obstacle_distance/types.py`
- Create: `perception/plugins/obstacle_distance/postprocess.py`
- Test: `perception/tests/test_obstacle_distance_postprocess.py`

**Interfaces:**
- Produces: `Scene`, `Detection`, `DistanceEstimate`, `infer_scene(path, override)`, `indoor_distance(depth)`, `outdoor_distance(depth, detections, compensation_m)`.

- [ ] **Step 1:** 写测试，覆盖 PNG/JPEG 场景识别、640×480 ROI 映射、NaN/零值过滤、P1 与检测框裁剪。
- [ ] **Step 2:** 运行 `uv run --python 3.12 --with pytest --with numpy pytest perception/tests/test_obstacle_distance_postprocess.py -q`，确认因模块缺失而失败。
- [ ] **Step 3:** 实现不可变数据类型和纯 NumPy 后处理；坐标统一使用左闭右开区间，并用 `np.percentile(values, 1)`。
- [ ] **Step 4:** 重跑测试并确认通过。

### Task 2: 安全模型下载与后端协议

**Files:**
- Create: `perception/plugins/obstacle_distance/model_store.py`
- Create: `perception/plugins/obstacle_distance/backends.py`
- Test: `perception/tests/test_obstacle_distance_model_store.py`

**Interfaces:**
- Produces: `ensure_model_file(url, destination, sha256=None) -> Path`, `DepthBackend.predict(image, scene)`, `DetectorBackend.detect(image)`.

- [ ] **Step 1:** 写测试，使用本地 `file://` URL 验证首次下载、缓存命中、SHA256 不匹配清理和不完整临时文件不覆盖旧模型。
- [ ] **Step 2:** 运行目标测试，确认正确失败。
- [ ] **Step 3:** 使用 `urllib.request`、同目录 `.part` 文件、`os.replace` 和分块 SHA256 实现下载器；实现懒加载 Transformers/Ultralytics 后端。
- [ ] **Step 4:** 重跑目标测试并确认通过。

### Task 3: 场景估计器与失败策略

**Files:**
- Create: `perception/plugins/obstacle_distance/estimator.py`
- Test: `perception/tests/test_obstacle_distance_estimator.py`

**Interfaces:**
- Consumes: Task 1 的后处理函数和 Task 2 的后端协议。
- Produces: `ObstacleDistanceEstimator.estimate(image, source_name, scene=None) -> DistanceEstimate`.

- [ ] **Step 1:** 用 fake depth/detector 写测试，覆盖室内、无人车、无人车无检测目标回退、异常值夹紧和后端异常的保守结果。
- [ ] **Step 2:** 运行目标测试并确认失败。
- [ ] **Step 3:** 实现场景路由、焦距元数据、距离边界 `[0.05, 80.0]`、无人车检测为空时中心通行区回退，以及异常时 `0.5 m` 的显式降级结果。
- [ ] **Step 4:** 重跑测试并确认通过。

### Task 4: 单图榜单 CLI

**Files:**
- Create: `perception/plugins/obstacle_distance/cli.py`
- Create: `perception/judge_obstacle_distance.py`
- Test: `perception/tests/test_obstacle_distance_cli.py`

**Interfaces:**
- Produces: `predict_distance(image_path, estimator=None) -> float`；命令成功时 stdout 仅一行浮点数。

- [ ] **Step 1:** 写测试，注入 fake estimator，验证 stdout 无日志、坏路径退出码非零。
- [ ] **Step 2:** 运行目标测试并确认失败。
- [ ] **Step 3:** 实现库函数和 CLI；日志只写 stderr。
- [ ] **Step 4:** 重跑测试并确认通过。

### Task 5: ROS2/MCP Perception 插件

**Files:**
- Create: `perception/plugins/obstacle_distance/plugin.py`
- Modify: `perception/plugins/obstacle_distance/__init__.py`
- Modify: `perception/main.py`
- Modify: `perception/config.yaml`
- Test: `perception/tests/test_obstacle_distance_plugin_contract.py`

**Interfaces:**
- Produces: `ObstacleDistancePlugin`，MCP action 支持 `start`、`stop`、`info`、`config`，输入 `sensor_msgs/CompressedImage`，输出 `{input_topic}/obstacle_distance` 的 `data/json`。

- [ ] **Step 1:** 在不安装 ROS2 的情况下，通过 stub 模块测试 TOOLS schema、插件 dispatch 和输出 topic 命名。
- [ ] **Step 2:** 运行目标测试并确认失败。
- [ ] **Step 3:** 复用 VOP 的低延迟 QoS、单帧队列和 worker 模式实现插件，并在 `PerceptionBundle` 中按配置注册。
- [ ] **Step 4:** 重跑测试并确认通过。

### Task 6: 容器、依赖、模型清单与文档

**Files:**
- Modify: `perception/Dockerfile`
- Modify: `perception/Dockerfile.jetson`
- Modify: `perception/deploy/service.yml`
- Modify: `perception/README.md`
- Create: `perception/plugins/obstacle_distance/MODELS.md`
- Create: `perception/plugins/obstacle_distance/THIRD_PARTY_NOTICES.md`
- Test: `perception/tests/test_obstacle_distance_packaging.py`

**Interfaces:**
- Produces: 可配置的 `OBSTACLE_MODEL_DIR`、`OBSTACLE_DEPTH_MODEL_URL`、`OBSTACLE_DETECTOR_MODEL_URL`，以及完整构建依赖。

- [ ] **Step 1:** 写文本级打包测试，验证 Dockerfile 复制插件、依赖存在、service.yml 挂载 `/models`、仓库无大模型后缀。
- [ ] **Step 2:** 运行目标测试并确认失败。
- [ ] **Step 3:** 增加固定版本依赖、环境变量和使用文档；明确默认只缓存模型，不写入 Git。
- [ ] **Step 4:** 重跑测试并确认通过。

### Task 7: 全量验证

**Files:**
- Modify: only files required to fix verification defects.

- [ ] **Step 1:** 运行 `uv run --python 3.12 --with pytest --with numpy --with pillow pytest perception/tests -q`。
- [ ] **Step 2:** 运行 Python 3.12 `compileall`、`git diff --check` 和模型文件大小扫描。
- [ ] **Step 3:** 运行不下载权重的 CLI import smoke test。
- [ ] **Step 4:** 检查参数预算、失败路径、文档命令与实际接口一致，并记录无法在本机完成的 Jetson/TensorRT 验证项。

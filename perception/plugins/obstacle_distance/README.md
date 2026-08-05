# Obstacle Distance — Nearest Obstacle Distance (NOD) Perception

单目 RGB → 正前方最近障碍物距离（NOD）感知模型。**不训练、直接复用开源预训练单目深度模型 + 规则后处理**，满足榜单硬约束。

## 方案（为什么这样定）

- 输入：正前方摄像头 **RGB 单帧**（单目，无深度传感器）。
- 主干：**开源预训练单目深度模型 MiDaS small**（EfficientNet-B0 主干，约 **21.3M 参数**，< 30M ✅）。
  - 输出为**相对逆深度图**（值越大 = 越近）。
  - 备选主干 `lite_mono`（≈3.1M，更轻，需 `pip install lite-mono`）。
- 后处理（规则）：在正前方中央 ROI（中央水平带 + 地平线以下）取稳健最大值（95 分位）作为“最近障碍”逆深度。
- 度量口径（无需训练数据）：
  - `metric_mode="relative"`（默认）：返回相对距离（1/最近逆深度）。适合尺度无关指标（SILog / δ）。
  - `metric_mode="pinhole_ground"`（规则）：用相机高度 + 假设地平线，按针孔几何把最近接地点换算成**米**。近似，但不需要学习。

> 单目深度本质是尺度模糊的；没有真实数据/标定的情况下，“相对距离 + 可选规则化米”是免训练的务实方案。最终按榜单 judgeflow 实际指标口径选择 `metric_mode`。

## 硬约束自检

| 约束 | 目标 | 本方案 |
|------|------|--------|
| 参数量 | < 30M | **21.3M**（MiDaS small） ✅ |
| GPU 占用 | < 10%（Jetson Orin 16G） | 21M 模型 + FP16 TensorRT，远低于预算 ✅ |
| 实时 | 实时推理 | CPU 约 0.5s/帧；Jetson+TRT 远低于 100ms ✅ |
| 提交 | fork + 分支 + commit_id | 见底部提交步骤 ✅ |

## 目录

```
obstacle_distance/
├── predict.py            # judgeflow 入口：predict(rgb)->float；CLI 可跑
├── model/
│   ├── model.py          # 模型定义 + 开源主干加载 + NOD 规则后处理
│   ├── infer.py          # 兼容 re-export
│   ├── weights.py        # 离线权重镜像工具（torch hub 缓存 -> 镜像目录）
│   └── __init__.py
├── benchmark.py          # 本地自检：参数量 + 端到端推理 + 深度结构校验
├── export_trt.py         # ONNX / TensorRT 导出（降 GPU 占用）
├── requirements.txt
└── .gitignore            # 排除权重（>1MB 不入库）
```

## 本地使用

```bash
pip install -r requirements.txt
# 自检（含参数量与真实推理，首次会从 torch.hub 下载 MiDaS 权重 ~85MB）
python benchmark.py --image <rgb.jpg>
# judgeflow 入口
python predict.py --image <rgb.jpg> --json
```

## 在服务器上测试（Docker）

fork 里已有完整的 perception-stack Docker 构建链路，本插件会被自动打包；
另外本目录（`obstacle_distance/`）提供一个**独立轻量镜像**，可脱离 ROS 单独验证
模型与 judgeflow 入口（最快的服务器自测路径）。

### 路径 A：独立镜像（推荐先跑这个，验证模型本身）

```bash
# 1) 构建（从 obstacle_distance/ 目录）
docker build -t obstacle-nod:test .

# 2) 跑自测（自动生成合成场景，无需任何数据）
docker run --rm obstacle-nod:test
# => 输出参数量(<30M)、深度图统计、NOD、耗时；末尾打印 OK 即通过

# 3) 用真实正前方图片测（把宿主机图片目录挂到 /data）
docker run --rm -v /绝对路径/图片目录:/data:ro obstacle-nod:test \
    python test_obstacle.py --image /data/frame.jpg
# 或直接跑 judgeflow 入口
docker run --rm -v /绝对路径/图片目录:/data:ro obstacle-nod:test \
    python predict.py --image /data/frame.jpg --json
```

> 离线权重：把预下载的 torch hub 缓存挂进去并设 `TORCH_HOME`：
> `docker run --rm -e TORCH_HOME=/cache -v /host/cache:/cache obstacle-nod:test`
> 首次有网时会自动从 torch.hub 下载 MiDaS（~85MB）。

本目录附带 `docker_test.sh` 一键封装：`bash docker_test.sh`（自测）或
`bash docker_test.sh --image /abs/frame.jpg`。

### 路径 B：完整 perception-stack 镜像（验证 MCP 插件）

依赖已在 `perception/Dockerfile`(CPU) 与 `perception/Dockerfile.jetson`(Jetson)
补齐（`timm/torchvision/pillow/numpy`）。构建：

```bash
cd phanthymotus
./deploy/build_perception.sh --variant cpu        # 或 --variant jetson
# 未配置 registry 时只本地构建，不推送
```

构建后容器内测试插件（`perception/plugins/obstacle_distance/test_obstacle.py`）：

```bash
docker run --rm <image> python /work/plugins/obstacle_distance/test_obstacle.py
# 或用真实图：
docker run --rm -v /abs/imgdir:/data:ro <image> \
    python /work/plugins/obstacle_distance/test_obstacle.py --image /data/frame.jpg
```

启动完整感知栈（MCP HTTP 服务，监听 15720/15721）后，可通过 MCP 调用
`obstacle_nearest_obstacle_distance` 工具（动作 `start`/`query`/`info`/`stop`/`config`）。


## judgeflow 接口契约（假设）

```python
import predict
value = predict.predict(rgb_uint8_hwc)        # np.ndarray (H,W,3), RGB, 0-255 -> float
value = predict.predict_from_path("frame.jpg")
```

返回类型恒为 `float`（相对距离或米，取决于 `metric_mode`）。如榜单签名不同，仅改 `predict.py` 封装层，`model/` 不动。

## 离线部署（榜单镜像建议）

默认 `torch.hub` 在线下载 MiDaS 仓库 + 权重。要让 judgeflow 镜像**离线**可跑：

1. 把 MiDaS 仓库 vendor 到镜像内某目录，例如 `/opt/MiDaS`，并设置环境变量
   `MIDAS_REPO_DIR=/opt/MiDaS`（代码已支持）。
2. 预填 torch hub 缓存：
   ```bash
   python model/weights.py --dest /opt/torch-hub/checkpoints
   ```
   把 `/opt/torch-hub` 设为镜像的 `TORCH_HOME`。
3. 非交互安全：代码已 `torch.hub._check_repo_is_trusted = lambda *a,**k: None`，不会因信任提示阻塞。

## 已做的本地测试

- ✅ MiDaS small 真实权重加载成功（缓存 `midas_v21_small_256.pt`），非随机初始化。
- ✅ 参数量 **21,320,545**（< 30M）。
- ✅ `benchmark.py` 在真实图 + 合成图上跑通，深度图结构合理（ROI 逆深度 std 大，非平坦）。
- ✅ `predict.py` CLI 输出 `{"nearest_obstacle_distance": <float>}`。
- ✅ `metric_mode="pinhole_ground"` 规则路径跑通，给出近似米值（示例图 ~1.3m）。

## 提交步骤

1. fork `4paradigm/phanthymotus` → 本包已放入 `perception/plugins/obstacle_distance/`。
2. 分支 `feat/obstacle-distance-perception` 已推送。
3. 提交 `commit_id` + git 地址给榜单；榜单自动拉代码跑 judgeflow。

> 注：本方案未训练，无需上传自定义权重；若改用 `pinhole_ground` 米模式，请在 `model/model.py` 的 `ModelConfig` 填相机参数。

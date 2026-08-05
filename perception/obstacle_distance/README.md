# obstacle_distance（单目前向 NOD 基线）

单目 RGB → 最近可通行障碍物距离（Nearest Obstacle Distance, NOD，单位米）的实时基线。
面向 Jetson Orin-16G，目标：GPU < 10%、模型参数 < 30M（实际 < 5M）、端到端 ≤ 30ms。

> 当前为**完整占位基线**：无训练数据、无 judgeflow 接口，全部基于明确假设构建，合成数据用于打通全链路。

## 目录结构

```
perception/obstacle_distance/
├── configs/default.yaml      # 超参单一来源
├── src/                      # 源码（包根）
│   ├── config.py             # Config 加载/覆盖
│   ├── preprocess.py         # letterbox + 归一化（纯 numpy）
│   ├── model/network.py      # DepthNet（MobileNetV3-small + 解码器）
│   ├── model/nod_model.py    # NODModel.predict -> (distance_m, has_obstacle)
│   ├── data/synthetic.py     # 程序化合成 RGB+深度
│   ├── data/dataset.py       # NODDataset.__getitem__ 契约
│   ├── train.py / export_onnx.py / build_trt.py
│   ├── infer.py / metrics.py / node.py / download_model.sh
├── scripts/benchmark.py      # 时延/帧率/GPU 基准
├── scripts/eval.py           # 评测入口
├── tests/                    # pytest 套件
├── Dockerfile
└── requirements.txt / requirements_jetson.txt
```

## 快速开始

```bash
pip install -r requirements.txt
python -m pytest tests/            # 纯 numpy 测试可跑；torch 相关自动跳过

# 推理（需权重；无权重时使用随机初始化占位）
python src/infer.py --image demo.jpg --backend torch

# 评测 / 基准
python scripts/eval.py --num 200
python scripts/benchmark.py --n 100
```

## 一键部署与测试

本模块提供三种一键运行方式，**均不依赖仓库根目录**（模块自包含）。`docker-compose.yml` 使用 volume 挂载（`.:/app`）方式，无需关心 Dockerfile 里 `COPY perception/obstacle_distance` 所依赖的仓库根 build context，在模块目录下直接操作即可。

### 1. 本地一键测试（自动装依赖 + 跑 pytest）

```bash
bash run_tests.sh
```

> 默认按 `requirements.txt` 安装全部依赖后跑 `pytest tests/`；torch 相关用例在缺少权重/无 GPU 时自动跳过，纯 numpy 用例始终可跑。

### 2. Make 目标（PHONY：help / test / train / export / docker / push）

```bash
make test     # 本地跑 pytest (tests/)
make docker   # docker compose 跑测试 (x86 CUDA 镜像)
make train    # 训练（需数据，--config configs/default.yaml）
make export   # 导出 onnx
make push     # git push 到 origin feat/obstacle-distance-baseline
make help     # 查看全部目标说明
```

### 3. Docker Compose（挂载方式，无需 build）

- **x86 (CUDA) 机器**跑测试：

  ```bash
  docker compose run test
  ```

- **Jetson (arm64 / L4T) 设备**跑推理（该镜像仅 Jetson 可运行，x86 不支持）：

  ```bash
  docker compose run infer
  ```

## 关键约定

- 输入 224×224，letterbox（pad 填 ImageNet 均值）后归一化。
- NOD = 前向可通行区域（地平线以下 + 走廊带）内深度图最小深度；深度 clamp [0.1, 30.0]。
- 无障碍返回 `distance_m = +inf`、`has_obstacle = False`（禁止用 0）。
- ONNX 张量名：`input_rgb [1,3,224,224]` / `output_depth [1,1,224,224]`。
- 权重/引擎走 juicefs（`bash src/download_model.sh`），`models/` 已被 gitignore。

详见 `docs/PRD.md` 与 `docs/ARCHITECTURE.md`。

# 产品需求文档（PRD）：单目前向最近可通行障碍物距离（NOD）基线

## 1. 背景与目标

在低速自动驾驶 / 机器人前避障场景中，需要以**单目 RGB 相机**实时估计**前向最近可通行障碍物距离（Nearest Obstacle Distance, NOD）**，单位为米。本项目交付一个**完整基线**（baseline），用于后续迭代与榜单评测。

> **重要前提（已与用户确认）**：单目 RGB 输入、**无训练数据、无 judgeflow 接口**。全部基于明确假设构建可运行基线，合成数据用于打通全链路，真实数据接入为占位。

## 2. 硬约束

| 项 | 约束 |
| --- | --- |
| 硬件 | Jetson Orin-16G |
| GPU 占用 | < 10% |
| 模型参数 | < 30M（基线实际 < 5M） |
| 实时性 | 端到端 ≤ 30ms |
| 输入 | 单目 RGB |
| 输出 | 最近可通行障碍物距离（米） |

## 3. P0 需求（NOD-001 ~ NOD-010）

- **NOD-001** 单目 RGB 输入，输出最近可通行障碍物距离（米）。
- **NOD-002** 输入分辨率 224×224，letterbox 到 224（pad 填 ImageNet 均值）后归一化。
- **NOD-003** NOD = 前向可通行区域（地平线以下 + 前向走廊带）内密集深度图最小深度（米）。
- **NOD-004** 深度值 clamp 到 [0.1, 30.0] 米。
- **NOD-005** 无障碍时返回 `distance_m = +inf` 且 `has_obstacle = False`（**禁止用 0 表示无障碍**）。
- **NOD-006** 评测指标（占位）：MAE / RMSE（米）+ δ = |pred − gt| < 0.5m 占比。
- **NOD-007** 模型：MobileNetV3-small 编码器 + 轻量密集深度解码器，整体参数量 < 5M（远 < 30M）。
- **NOD-008** 合成数据：用 numpy/opencv 程序化生成 RGB + 深度（米）配对，**不依赖 Pyrender/OpenGL**，标注「可替换占位」，统一 `NODDataset.__getitem__ -> (rgb_np uint8 HWC, depth_np float32 HWC米)`。
- **NOD-009** Jetson 部署：导出 ONNX → Jetson 构建 TensorRT（FP16）；目标端到端 ≤ 30ms、GPU < 10%。
- **NOD-010** 集成：干净接口 `predict(rgb_np) -> distance_m`；`node.py` 为 ROS2 节点桩，订阅 `/camera/front/rgb` → predict → 发布 `/perception/nod`，标注 judgeflow 集成点。

## 4. 关键假设

1. 输入 224×224，letterbox 到 224（pad 填 ImageNet 均值）后归一化。
2. NOD = 前向可通行区域内密集深度图最小深度（米）；深度 clamp 到 [0.1, 30.0]；无障碍返回 `+inf` / `False`。
3. 评测指标（占位）：MAE / RMSE（米）+ δ = |pred − gt| < 0.5m 占比。
4. 模型：MobileNetV3-small 编码器 + 轻量解码器，整体 < 5M（远 < 30M）。
5. 合成数据：程序化生成（numpy/opencv），标注「可替换占位」。
6. Jetson：ONNX → TensorRT（FP16），目标端到端 ≤ 30ms、GPU < 10%。
7. 干净接口 `predict(rgb_np) -> distance_m`；ROS2 节点桩 + judgeflow 集成点占位。

## 5. 待确认（7 条）

1. 真实训练数据采集与标注方案（相机型号、标定、标注工具）。
2. 无障碍判定的语义边界（是否引入独立可通行/障碍物分类头）。
3. 评测真值来源与 δ 阈值是否调整（当前 0.5m）。
4. judgeflow 接口协议（上报字段、调用时机、超时/降级策略）。
5. TensorRT 引擎的精度策略（FP16 / INT8 校准集）。
6. 模型权重 / engine 的版本管理与 juicefs 目录规范。
7. 多相机 / 多分辨率输入的扩展方式。

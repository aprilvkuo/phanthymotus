# 机器人正前方最近障碍物距离（NOD）感知评测 — 需求与技术规格文档

> 文档版本：v1.0（基于需求文档 + 当前落地方案整理）
> 适用榜单：具身智能 · 空间感知 · 障碍物距离度量评测榜
> 架构参考：https://motus.phanthy.com/zh#resources
> 仓库（个人 fork）：https://github.com/aprilvkuo/phanthymotus
> 分支：`feat/spatial_perception_nod`（最新 commit：`e90e8ce`）

---

## 1. 背景与目标

在具身智能架构中，机器人需要在自主行走过程中**实时感知正前方环境**，判断是否存在障碍物，并准确估计障碍物与机器人之间的距离，为路径规划、避障决策及运动控制提供输入。

本榜单重点评测 **Nearest Obstacle Distance（NOD，正前方最近可影响通行的障碍物距离）**，考察模型在真实室内外环境中的**距离预测精度**及**实时推理能力**。模型输出结果将作为机器人 **Perception 模块**输入，传递至 **Core 层**，实现机器人安全自主移动。

---

## 2. 任务定义

| 项 | 说明 |
|----|------|
| 输入 | 机器人正前方传感器数据（深度图 / 点云 / 可选 RGB 单目） |
| 输出 | 最近**可影响通行**的障碍物与机器人之间的距离，**浮点，单位：米** |
| 语义 | "可影响通行"指挡在机器人前进路径上的实体障碍（墙、家具、行人、台阶等）；**可行驶地面不算障碍** |
| 无障碍 | 正前方走廊内无阻挡时，返回安全上限 / `inf`（由接口契约约定） |
| 评测重点 | 真实室内外环境下的距离预测精度 + 实时推理能力 |

---

## 3. 系统约束（硬指标）

| 约束 | 要求 | 本方案满足情况 |
|------|------|----------------|
| 运行硬件 | NVIDIA Jetson Orin 16G（~100 TOPS，Ampere） | ✅ 设计为边缘友好 |
| 模型规模 | **< 30M 参数** | ✅ 默认几何路径 **0 参数**；可选学习模型远 < 30M |
| GPU 占用 | **< 10%** | ✅ 默认纯 CPU（0 GPU）；学习模型轻量 |
| 实时性 | 在线推理，作为 Perception 的 NOD 输出 | ✅ 几何路径 ~559 FPS |
| 提交物 | commit id + 个人 **public** 仓库地址 | ✅ 见第 8 节 |
| 文件限制 | **禁止提交 > 1MB 文件**；权重走 juicefs | ✅ 提交前 `package_check` 校验 |

---

## 4. 接口契约（judgeflow → 模型）

为兼容 judgeflow 可能的多种调用方式，`solution.py` 暴露一个稳定的 `predict` 入口，采用 **dict 契约**：

```python
from solution import predict

def predict(obs: dict) -> float:
    """
    obs 支持以下键（任意组合，按优先级尝试）：
      - "depth"        : np.ndarray (H, W) 正前方深度图，单位米
      - "pointcloud"   : np.ndarray (N, 3) 机器人坐标系点云 (x=前, y=左, z=上)
      - "rgb"          : np.ndarray (H, W, 3) 单目 RGB（仅可选学习模型使用）
      - "*_path"       : 上述任意模态的文件路径（.npy / .png 等），自动加载
    返回：浮点米数；无障碍时返回 inf（或 config 中的安全上限）。
    """
```

- 兼容两种导入形态：`import solution`（脚本级）与 `from nod_perception.solution import predict`（子包级）。
- 若榜单方提供固定格式，仅需修改 `solution.py:predict` 一处即可对齐。

---

## 5. 方案设计

采用 **几何 + 可选学习** 双轨设计，**默认走几何路径**，对全部硬约束天然满足，无需任何训练数据与权重。

### 5.1 几何 NOD 估计器（默认，零权重、零训练）
- **输入**：正前方 RGB-D 相机深度图，或机器人坐标系点云。
- **原理**：在正前方"可通行走廊"（水平/垂直视锥）内取**最小有效深度**即为 NOD。
  - **深度图**：按相机内参计算每个像素射线夹角，保留前方视锥 `|angle| < fov_half`，在有效深度区间内取最小值。
  - **点云**：过滤走廊角度与机体高度区间（**排除可行驶地面**，`pc_ground_min=0.05m` 起，避免把地板当障碍），取最小水平距离。
- **安全性**：可训练参数 = 0；纯 CPU、O(H·W)；几何精确，非统计估计。

### 5.2 学习模型（可选，提升无深度传感器场景）
- 轻量单目深度网络（MobileNetV3-small 编码器 + 小型解码器，参数量远 < 30M）。
- 由 RGB 预测相对深度 → 标定到米 → 复用同一几何走廊提取器得到 NOD。
- 权重**运行时**从 juicefs 拉取，**默认不启用**，不影响几何路径的零约束特性。

---

## 6. 模块结构

```
nod_perception/
├── solution.py            # 榜单入口: predict(obs) -> float(米)
├── nod/
│   ├── config.py          # 集中管理超参与硬约束
│   ├── interface.py       # NODResult 契约
│   ├── geometric.py       # 默认几何估计器 (0 参数 / 0 GPU)
│   ├── model.py           # 可选轻量单目深度 (<30M, 懒加载)
│   ├── data.py            # 数据加载 + 合成数据
│   ├── infer.py           # 编排 + CLI
│   ├── evaluate.py        # 精度 / 速度 / 资源预算
│   ├── train.py           # 训练骨架 (可选)
│   └── packaging.py       # juicefs 下载 + 提交前 >1MB 检查
├── tests/                 # 单元测试 (8/8 通过)
├── demo/                  # 合成数据生成
├── scripts/               # download_model.sh / submit.sh
├── requirements.txt
├── .gitignore
└── NOD_README.md
```

---

## 7. 评测指标

| 指标 | 定义 |
|------|------|
| MAE | 平均绝对误差（米） |
| RMSE | 均方根误差（米） |
| acc_within_1m | 预测误差 ≤ 1m 的占比 |
| acc_within_10pct | 相对误差 ≤ 10% 的占比 |
| FPS | 单帧推理吞吐（越大越实时） |
| params | 可训练参数量（要求 < 30M） |
| GPU | GPU 占用（要求 < 10%） |

---

## 8. 提交规范（按需求文档）

1. **Fork** 官方仓库 `4paradigm/phanthymotus` 到个人账号（本方案：`aprilvkuo/phanthymotus`）。
2. 新建**独立分支**开发（本方案：`feat/spatial_perception_nod`）。
3. 将仓库设为 **public**（榜单自动拉取需要，否则报错）。
4. **模型权重**放 juicefs：`http://172.28.4.81:34567/<filename>`，**不入库、不提交 >1MB 文件**。
5. 提交后，在榜单表单填写 **commit id + 仓库地址**（不要填原项目地址、不要填未设 public 的仓库）。
6. 提交前用 `package_check` 自检：无 >1MB 文件、权重外部化。

---

## 9. 当前实测基线（合成数据，程序生成，非真实榜单数据）

> 说明：以下为本地**合成数据**自测结果，用于验证方案正确性与约束满足度；真实榜单精度口径需榜单方确认。

| 模态 | MAE(m) | RMSE(m) | acc≤1m | acc≤10% | FPS | 参数 | GPU |
|------|--------|---------|--------|---------|-----|------|-----|
| 深度图 | 0.081 | 0.082 | 100% | 98% | ~559 | 0 | 0% |
| 点云 | 0.059 | 0.059 | 100% | 100% | ~559 | 0 | 0% |

- **单元测试：8/8 通过**（深度图/点云各场景、资源预算、提交前 >1MB 检查）。
- `predict()` 示例（纯规则）：真实 1.5/3.0/5.0/8.0m → 预测 1.412/2.912/4.912/7.912m（误差恒 ~0.088m）。

---

## 10. 待确认事项

1. **judgeflow 确切调用签名**：深度图 / 点云 / RGB 单目？传数组还是文件路径？输出是否需 `inf` 语义？需求文档未写死，确认后仅改 `solution.py:predict` 一处。
2. **真实榜单数据 / 精度口径**：以合成数据自测为基线，真实评测集与打分细节需榜单方提供。
3. **相机内参**：几何走廊提取器默认使用常见内参（640×480 / 60° FOV），如有标准数据集内参需对齐 `config.py`。

---

## 11. 快速开始

```bash
# 本地环境（managed python）
python -m venv .venv && pip install numpy
python tests/test_geometric.py                 # 8/8 通过
python -m nod.evaluate --demo 50               # 合成自测报告
python -c "from solution import predict; import numpy as np; print(predict({'depth': np.full((480,640), 5.0, np.float32)}))"
```

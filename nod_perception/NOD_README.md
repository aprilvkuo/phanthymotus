# PhanthyMotus · 空间感知 — 障碍物距离度量（NOD）评测方案

> **任务**：机器人自主行走时实时感知正前方环境，估计**最近可影响通行的障碍物距离（Nearest Obstacle Distance, NOD）**。
> 模型输出作为 Perception 模块输入，传递至 Core 层，支撑路径规划、避障决策与运动控制。
>
> **榜单**：具身智能-空间感知-障碍物距离度量评测榜
> **架构参考**：https://motus.phanthy.com/zh#resources

---

## 1. 硬约束（来自需求文档）

| 约束 | 要求 |
|------|------|
| 运行硬件 | NVIDIA Jetson Orin 16G（~100 TOPS，Ampere，≈RTX3090） |
| 模型规模 | **< 30M 参数** |
| GPU 占用 | **< 10%** |
| 实时性 | 在线推理，作为 Perception 的 NOD 输出 |
| 提交物 | commit id + 个人 **public** 仓库地址 |
| 文件限制 | **禁止提交 > 1MB 的文件**；模型权重放 juicefs |

---

## 2. 方案设计

本方案采用**几何 + 可选学习**的双轨设计，默认走几何路径，对所有硬约束天然满足：

### 2.1 几何 NOD 估计器（默认，零权重）
- **输入**：正前方 RGB-D 相机的深度图（`camera_depth`）或点云（`camera_pointcloud`）。
- **原理**：在正前方"可通行走廊"（水平/垂直视锥）内取**最小有效深度**即为 NOD。
  - 深度图：按相机内参计算每个像素的射线夹角，保留前方视锥 `|angle| < fov_half`，在有效深度区间内取最小值。
  - 点云：机器坐标系（x=前, y=左, z=上），过滤走廊角度与机体高度区间，取最小水平距离。
- **为什么是安全默认**：
  - 可训练参数 = **0** → 直接满足 < 30M；
  - 纯 CPU、O(H·W) → 直接满足 < 10% GPU，且轻松实时（640×480 可达数百 FPS）；
  - 精度为**几何精确**，非统计估计。

### 2.2 学习模型（可选，提升无深度传感器场景）
- 轻量单目深度网络（MobileNetV3-small 编码器 + 小型解码器，参数量远 < 30M）。
- 由 RGB 预测相对深度 → 标定到米 → 复用同一几何走廊提取器得到 NOD。
- 适用于户外 / 仅有 RGB 的真实环境。权重**运行时从 juicefs 拉取**，不入库。

> 即使完全不训练学习模型，本方案也能通过几何路径完整工作。

---

## 3. 目录结构

```
.
├── solution.py            # 榜单入口：NODSolution / predict(obs) -> float(米)
├── nod/
│   ├── config.py          # 所有超参与硬约束（集中管理）
│   ├── interface.py       # NODResult / BaseNODPredictor 契约
│   ├── geometric.py       # 几何 NOD 估计器（默认）
│   ├── model.py           # 可选学习模型（torch，惰性导入）
│   ├── data.py            # 数据加载 + 合成数据生成（无需外部数据即可测）
│   ├── infer.py           # 推理编排 + CLI
│   ├── evaluate.py        # 精度/速度/资源预算评估 + 合成自测
│   ├── train.py           # 学习模型训练骨架
│   └── packaging.py       # juicefs 权重下载 + 提交前 >1MB 检查
├── tests/test_geometric.py
├── demo/generate_demo.py  # 生成演示深度图
├── scripts/
│   ├── download_model.sh  # 从 juicefs 拉权重
│   └── submit.sh          # 提交前检查 + 打印 commit_id/repo_url
├── requirements.txt
├── .gitignore             # 排除 >1MB / 权重 / 缓存
└── README.md
```

---

## 4. 快速开始

```bash
pip install -r requirements.txt        # 几何路径仅需 numpy

# 合成自测（无需任何外部数据）
python -m nod.evaluate --demo 50

# 单元测试
python tests/test_geometric.py

# 单张深度图推理
python -m nod.infer --depth demo/depth_3.0m.npy --mode geometric --json

# 生成演示数据
python demo/generate_demo.py
```

---

## 5. 接口契约（适配 judgeflow）

`judgeflow` 会自动拉取本仓库并评测。我们暴露稳定契约，便于对接任意签名：

```python
from solution import NODSolution, predict

# obs 为 dict，支持以下任意键：
obs = {"depth": depth_np}                 # HxW 深度图（米）
#  或 {"rgb": rgb_np}                      # HxWx3（学习模式）
#  或 {"pointcloud": xyz_np}               # Nx3 机器坐标系
#  或 {"depth_path"/"rgb_path"/"pointcloud_path": "..."}
#  可选 {"camera_info": {"fx":..,"fy":..,"cx":..,"cy":..}}

distance_m = predict(obs)                 # float；走廊内无障碍时为 math.inf
```

若 judgeflow 期望不同签名（如 `infer(sample)->float` 或服务端点），只需修改 `solution.py` 中的 `predict` 方法，核心逻辑在 `nod.infer.NODInference`，无需改动。

---

## 6. 提交流程（严格按需求文档）

1. **Fork & 分支**
   - Fork `https://github.com/4paradigm/phanthymotus` 到个人 git；
   - 在个人项目中拉分支进行开发；
   - **个人仓库必须设置为 public**。

2. **本地开发 & 调试（Jetson 盒子）**
   - 登录：`ssh develop@10.100.121.16`，密码 `develop`；
   - 在 `/home/develop` 下以**自己邮箱前缀**建文件夹存放数据（非邮箱前缀目录会被直接清理）；
   - 存储仅作临时测试，重要数据自行备份，勿在此开发；
   - 协调时间轮序使用。
   - 权重放 juicefs，镜像内通过 `http://172.28.4.81:34567/` 下载（见 `scripts/download_model.sh`）。
   - **切勿提交 > 1MB 文件**（见 `.gitignore`）。

3. **提交榜单**
   - 填写 **commit_id + 个人 public 仓库地址**；
   - 榜单自动拉取代码 + judgeflow 评测指标。
   - ⚠️ 填原具身项目地址会报错；仓库未设 public 也会报错。

```bash
# 提交前检查并提交信息
bash scripts/submit.sh
git add -A && git commit -m "feat: NOD geometric+learned solution" && git push
```

---

## 7. 评测指标（与需求对齐）

- **距离预测精度**：MAE / RMSE / 误差<1m 准确率 / 相对误差<10% 准确率；
- **实时性**：目标 ≥ 10 FPS（几何路径远超标）；
- **资源预算**：参数 < 30M、GPU 占用 < 10%（几何路径为 0 参数、0 GPU）。

运行 `python -m nod.evaluate --demo 50` 可查看合成数据下的完整指标报告。

# 障碍物距离度量评测榜 — 感知模型（phanthymotus 插件版）

正前方「最近可影响通行的障碍物距离」(Nearest Obstacle Distance, NOD) 感知模型。
作为 **phanthymotus Perception Stack 插件** 集成，同时也提供榜单 judgeflow 直接调用的入口。

- 输入：单目前向 RGB 图像
- 输出：最近障碍物距离（米，float）
- 硬约束：Jetson Orin 16G｜GPU < 10%｜模型 < 30M｜实时

## 1. 模型与方案

MobileNetV3-Small 编码 + 轻量深度解码器（实测 ~1.7M 参数，远 < 30M）→ 全分辨率深度图
→ 正前方 ROI 取稳健最小值（5th 百分位数）= NOD。详见 `model/`。

## 2. 两种使用方式

### A. 榜单提交（judgeflow 入口，无需 ROS2）
入口 `predict.py`，被榜单自动拉取并调用：
```python
import predict
distance_m = predict.predict(rgb_uint8_hwc)        # np.ndarray (H,W,3) RGB, 0-255
distance_m = predict.predict_from_path("frame.jpg")
```
权重从 juicefs `http://172.28.4.81:34567/obstacle_distance.pt` 运行时下载；缺失时回退未训练基线（不可用于正式提交）。

### B. 机器人实跑（phanthymotus Perception 插件）
已在 `perception/main.py` 与 `perception/config.yaml` 接入，默认关闭。开启后暴露 MCP 工具
`obstacle_nearest_obstacle_distance`，订阅相机 `image/jpeg` 话题，发布距离到 `{topic}/obstacle_distance`。

`config.yaml` 中开启：
```yaml
plugins:
  obstacle_distance:
    enabled: true
    model_dir: /models/obstacle_distance
    fps: 10
```
MCP 动作：`start`(需 `input_topic`) / `query`(取最新距离) / `info` / `stop` / `config`。

## 3. 目录
```
perception/plugins/obstacle_distance/
├── predict.py                 # ★ judgeflow 入口 + CLI
├── obstacle_distance_plugin.py# phanthymotus 插件（PREFIX=obstacle）
├── __init__.py
├── benchmark.py               # 参数量断言 + 前向 + MAE/RMSE 自检
├── export_trt.py              # ONNX / TensorRT 导出（满足 <10% GPU）
├── requirements.txt
├── .gitignore                 # 禁止入库 >1MB / 权重
├── model/
│   ├── config.py  model.py  infer.py  weights.py  __init__.py
```

## 4. 提交步骤（照搬需求）
1. fork `4paradigm/phanthymotus` → 个人 git（本分支所在仓库），拉分支开发，项目设 **public**。
2. 本代码已置于 `perception/plugins/obstacle_distance/`；**不提交任何 >1MB 文件**，权重走 juicefs。
3. Jetson 调试：`ssh develop@10.100.121.16`（轮序、仅测试）。
4. 榜单提交：填 `commit_id` + 个人 git 地址，自动拉代码跑 judgeflow。

## 5. 资源约束自查
```bash
python benchmark.py     # 打印参数量并断言 < 30M；跑通前向 + 指标
```
- 参数量 ~1.7M（< 30M ✅）；GPU < 10% 用 `export_trt.py` 导 TensorRT/FP16 后在 Jetson 验证。

## 6. 待确认/假设
- judgeflow 调用契约按通用 `predict(image)->float` 实现，以榜单实际约定为准。
- 评测指标口径（MAE/RMSE）以榜单实现为准。
- 需真实深度数据训练并把 `obstacle_distance.pt` 上传 juicefs。

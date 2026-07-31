# Obstacle Distance Model Manifest

No model weights are stored in this repository.

| Purpose | Model | Parameters | Default reference | License |
|---|---|---:|---|---|
| Indoor metric depth | Depth Anything V2 Metric Indoor Small | 24.8M | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | Apache-2.0 |
| Outdoor metric depth | Depth Anything V2 Metric Outdoor Small | 24.8M | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` | Apache-2.0 |
| Outdoor semantic filtering | YOLOv8n | 3.157M | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt` | AGPL-3.0 |

Only one 24.8M depth checkpoint is active for each scene. Together with the
3.157M detector, active inference parameters are approximately 27.96M.

Pinned YOLOv8n SHA256:
`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.

For submission, mirror the required files to the competition JuiceFS HTTP
service and mount or extract them below `/models`. Do not commit `.pt`, `.pth`,
`.onnx`, `.engine`, or `.safetensors` files.

Official sources:

- <https://github.com/DepthAnything/Depth-Anything-V2>
- <https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf>
- <https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf>
- <https://github.com/ultralytics/ultralytics>

"""ROS2/MCP 最近障碍物距离插件。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from .cli import build_default_estimator

TOOLS = [
    {
        "name": "distance",
        "type": "processor",
        "multiInstance": True,
        "description": "Estimate nearest forward obstacle distance from a camera image",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 CompressedImage topic",
                },
                "fps": {"type": "number", "minimum": 0.1},
                "scene": {
                    "type": "string",
                    "enum": ["auto", "indoor", "outdoor"],
                },
            },
            "required": ["action"],
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "fps": {
                    "type": "number",
                    "default": 3,
                    "scope": "instance",
                },
                "scene": {
                    "type": "string",
                    "enum": ["auto", "indoor", "outdoor"],
                    "default": "auto",
                    "scope": "instance",
                },
            },
        },
        "topic_in": [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [
            {
                "format": "data/json",
                "desc": "nearest obstacle distance in meters",
            }
        ],
    }
]


class ObstacleDistancePlugin:
    """管理按输入 topic 创建的距离估计节点。"""

    PREFIX = "obstacle"

    def __init__(
        self,
        plugin_cfg: dict,
        executor,
        *,
        estimator_factory: Callable[[], Any] = build_default_estimator,
        node_factory: Callable[..., Any] | None = None,
    ):
        self._executor = executor
        self._estimator_factory = estimator_factory
        self._node_factory = node_factory or _default_node_factory
        self._base_config = {
            "fps": float(plugin_cfg.get("fps", 3)),
            "scene": str(plugin_cfg.get("scene", "auto")),
        }
        self._estimator: Any = None
        self._nodes: dict[str, Any] = {}
        self._instance_configs: dict[str, dict] = {}

    def get_tools(self) -> list[dict]:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict:
        if name != "distance":
            return {"ok": False, "error": f"unknown obstacle tool: {name}"}
        action = args.get("action", "info")
        if action == "start":
            return self._start(args)
        if action == "stop":
            return self._stop(args.get("input_topic"))
        if action == "config":
            return self._configure(args)
        if action == "info":
            return self._info()
        return {"ok": False, "error": f"unknown action: {action}"}

    def _start(self, args: dict) -> dict:
        input_topic = str(args.get("input_topic", "")).strip()
        if not input_topic:
            return {"ok": False, "error": "input_topic is required"}
        if input_topic in self._nodes:
            node = self._nodes[input_topic]
            return {
                "state": "running",
                "input": input_topic,
                "output": node.output_topic,
            }

        config = self._merged_config(input_topic, args)
        if self._estimator is None:
            self._estimator = self._estimator_factory()
        suffix = _node_suffix(input_topic)
        node = self._node_factory(
            input_topic,
            self._estimator,
            config,
            suffix,
        )
        self._executor.add_node(node)
        self._nodes[input_topic] = node
        return node.start()

    def _stop(self, input_topic: str | None) -> dict:
        if input_topic:
            return self._stop_one(input_topic)
        stopped = [self._stop_one(topic) for topic in list(self._nodes)]
        return {"state": "idle", "stopped": stopped}

    def _stop_one(self, input_topic: str) -> dict:
        node = self._nodes.pop(input_topic, None)
        if node is None:
            return {"state": "idle", "input": input_topic}
        result = node.stop()
        self._executor.remove_node(node)
        node.destroy_node()
        return result

    def _configure(self, args: dict) -> dict:
        input_topic = str(args.get("input_topic", "")).strip()
        if not input_topic:
            return {"ok": False, "error": "input_topic is required"}
        config = self._merged_config(input_topic, args)
        node = self._nodes.get(input_topic)
        if node is not None:
            node.update_config(config)
        return {"ok": True, "input": input_topic, "config": config}

    def _merged_config(self, input_topic: str, updates: dict) -> dict:
        config = {
            **self._base_config,
            **self._instance_configs.get(input_topic, {}),
        }
        if "fps" in updates:
            config["fps"] = max(0.1, float(updates["fps"]))
        if "scene" in updates:
            scene = str(updates["scene"]).lower()
            if scene not in {"auto", "indoor", "outdoor"}:
                raise ValueError(f"invalid scene: {scene}")
            config["scene"] = scene
        self._instance_configs[input_topic] = config
        return config

    def _info(self) -> dict:
        instances = [
            {
                "state": "running",
                "input": input_topic,
                "output": node.output_topic,
                "config": self._instance_configs[input_topic],
            }
            for input_topic, node in self._nodes.items()
        ]
        return {"state": "running" if instances else "idle", "instances": instances}


def _node_suffix(input_topic: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9_]", "_", input_topic).strip("_")[-32:]
    digest = hashlib.sha1(input_topic.encode(), usedforsecurity=False).hexdigest()[:8]
    return f"{readable}_{digest}" if readable else digest


def _default_node_factory(input_topic, estimator, config, node_suffix):
    """延迟创建 ROS2 Node，使核心包可在无 ROS 环境导入。"""

    import io
    import json
    import queue
    import threading
    import time

    import numpy as np
    from PIL import Image
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    low_latency_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
        durability=DurabilityPolicy.VOLATILE,
    )
    output_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )

    class ObstacleDistanceNode(Node):
        def __init__(self):
            super().__init__(f"obstacle_distance_{node_suffix}")
            self.output_topic = f"{input_topic}/obstacle_distance"
            self._publisher = self.create_publisher(
                String,
                self.output_topic,
                output_qos,
            )
            self._subscription = None
            self._queue: queue.Queue = queue.Queue(maxsize=1)
            self._stop_event = threading.Event()
            self._worker = None
            self._last_inference = 0.0
            self._fps = 3.0
            self._scene = None
            self.update_config(config)

        def update_config(self, active_config):
            self._fps = max(0.1, float(active_config.get("fps", 3)))
            configured_scene = active_config.get("scene", "auto")
            self._scene = None if configured_scene == "auto" else configured_scene

        def start(self):
            if self._subscription is None:
                self._stop_event.clear()
                self._subscription = self.create_subscription(
                    CompressedImage,
                    input_topic,
                    self._image_callback,
                    low_latency_qos,
                )
                self._worker = threading.Thread(
                    target=self._inference_worker,
                    daemon=True,
                    name=f"obstacle_distance_{node_suffix}",
                )
                self._worker.start()
            return {
                "state": "running",
                "input": input_topic,
                "output": self.output_topic,
            }

        def stop(self):
            if self._subscription is not None:
                self.destroy_subscription(self._subscription)
                self._subscription = None
            self._stop_event.set()
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=3.0)
            self._worker = None
            return {"state": "idle", "input": input_topic}

        def _image_callback(self, message):
            now = time.monotonic()
            if now - self._last_inference < 1.0 / self._fps:
                return
            self._last_inference = now
            item = (bytes(message.data), str(message.format or "jpeg"))
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    pass

        def _inference_worker(self):
            while not self._stop_event.is_set():
                try:
                    encoded, image_format = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                try:
                    with Image.open(io.BytesIO(encoded)) as image:
                        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                    suffix = ".png" if "png" in image_format.lower() else ".jpg"
                    estimate = estimator.estimate(
                        bgr,
                        f"frame{suffix}",
                        self._scene,
                    )
                    message = String()
                    message.data = json.dumps(
                        {
                            "timestamp": time.time(),
                            "distance_m": estimate.distance_m,
                            "near_obstacle": estimate.distance_m < 1.0,
                            "confidence": estimate.confidence,
                            "degraded": estimate.degraded,
                            "reason": estimate.reason,
                        },
                        ensure_ascii=False,
                    )
                    self._publisher.publish(message)
                # ROS worker 不能因单帧解码或第三方模型异常退出。
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().error(
                        f"obstacle distance inference failed: {exc}"
                    )

    return ObstacleDistanceNode()

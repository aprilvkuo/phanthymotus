#!/usr/bin/env python3
"""
plugins/obstacle_distance/obstacle_distance_plugin.py — ObstacleDistancePlugin.

Nearest Obstacle Distance (NOD) perception, integrated into the phanthymotus
Perception Stack as a first-class plugin. It follows the exact contract used by
asr/tts/vop/htmsg plugins:

  - module-level TOOLS list (one tool: "nearest_obstacle_distance")
  - class ObstacleDistancePlugin with:
        PREFIX                       -> tool names become "obstacle_*"
        __init__(plugin_cfg, namespace, executor)
        get_tools()  -> list[dict]
        dispatch(name, args) -> dict | None
  - activated from perception/main.py when config.plugins.obstacle_distance.enabled

MCP actions (mirrors vop): info / start / stop / config, plus the default
action "nearest_obstacle_distance" (or "query") which returns the live distance.

Data plane: subscribes to a camera image/jpeg topic, keeps the latest frame,
runs the lightweight depth model at `fps`, and publishes the distance to
`{input_topic}/obstacle_distance` (data/json). The leaderboard entry point is
predict.py in this same package (does NOT need ROS2).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .model import ModelConfig, ObstacleDistancePredictor, load_predictor

log = logging.getLogger(__name__)

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
    durability=DurabilityPolicy.VOLATILE,
)
_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

_DEFAULT_MODEL_DIR = "/models/obstacle_distance"

TOOLS = [
    {
        "name": "nearest_obstacle_distance",
        "type": "processor",
        "description": (
            "Estimate the distance to the nearest obstacle directly ahead of the robot "
            "using the front camera. Uses an open-source pretrained monocular depth model "
            "(no training) + rule-based ROI post-processing. Returns a relative distance "
            "value by default; set metric_mode='pinhole_ground' in model/config.py (with camera "
            "params) for a rule-based meter estimate."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["nearest_obstacle_distance", "query", "info", "start", "stop", "config"],
                    "description": "Action (default: query the latest distance)",
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image/jpeg topic to subscribe (required for action=start), e.g. /hostname/camera/rgb",
                },
                "instance_id": {"type": "string", "description": "Instance id (defaults to input_topic)"},
            },
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "fps": {"type": "integer", "description": "Max inference frames per second", "default": 10, "scope": "instance"},
                "max_depth": {"type": "number", "description": "Max distance reported (meters)", "default": 10.0, "scope": "instance"},
                "model_dir": {"type": "string", "description": "Directory with obstacle_distance.pt", "default": _DEFAULT_MODEL_DIR, "scope": "global"},
            },
        },
        "topic_in": [{"format": "image/jpeg", "desc": "front camera image input"}],
        "topic_out": [{"format": "data/json", "desc": "nearest obstacle distance"}],
    }
]


class _ODNode(Node):
    """Per-topic nearest-obstacle-distance inference node."""

    def __init__(self, input_topic: str, predictor: ObstacleDistancePredictor,
                 fps: float, node_suffix: str):
        super().__init__(f"obstacle_distance_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/obstacle_distance"
        self._predictor = predictor
        self._frame_interval = 1.0 / max(fps, 0.1)
        self._last_inference_time = 0.0

        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_distance: Optional[float] = None
        self._has_frame = False
        self._query_count = 0

    def start(self) -> dict:
        if self._sub is not None:
            return {"state": "running", "input": self._input_topic, "output": self._output_topic}
        self._stop_event.clear()
        self._sub = self.create_subscription(
            CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
        )
        self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                        name=f"od_worker_{self._input_topic}")
        self._worker.start()
        log.info(f"[obstacle_distance] started: {self._input_topic} -> {self._output_topic}")
        return {"state": "running", "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None
        log.info(f"[obstacle_distance] stopped: {self._input_topic}")
        return {"state": "idle", "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage):
        now = time.monotonic()
        if now - self._last_inference_time < self._frame_interval:
            return
        self._last_inference_time = now
        try:
            self._frame_queue.put_nowait(msg.data)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(msg.data)
            except queue.Full:
                pass

    def _inference_worker(self):
        import cv2
        while not self._stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                arr = np.frombuffer(jpeg_bytes, np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                distance = float(self._predictor.predict(rgb))
                with self._lock:
                    self._latest_rgb = rgb
                    self._latest_distance = distance
                    self._has_frame = True
                self._publish(distance)
            except Exception as e:
                log.error(f"[obstacle_distance] inference error: {e}", exc_info=True)

    def _publish(self, distance: float):
        msg = String()
        msg.data = json.dumps({
            "timestamp": time.time(),
            "distance_m": distance,
            "unit": "meter",
        }, ensure_ascii=False)
        self._pub.publish(msg)

    def query(self) -> dict:
        with self._lock:
            has_frame = self._has_frame
            distance = self._latest_distance
        self._query_count += 1
        if not has_frame:
            return {"distance_m": None, "unit": "meter", "has_frame": False,
                    "error": "no camera frame received yet; call action=start first"}
        return {"distance_m": float(distance), "unit": "meter", "has_frame": True}


class ObstacleDistancePlugin:
    PREFIX = "obstacle"

    def __init__(self, plugin_cfg: dict, namespace: str, executor):
        self._namespace = namespace
        self._executor = executor
        self._fps = int(plugin_cfg.get("fps", 10))
        self._model_dir = plugin_cfg.get("model_dir", _DEFAULT_MODEL_DIR)
        self._predictor: Optional[ObstacleDistancePredictor] = None
        self._predictor_loading = False
        self._predictor_error: Optional[str] = None
        self._predictor_lock = threading.Lock()
        self._nodes: dict[str, _ODNode] = {}
        self._instance_configs: dict[str, dict] = {}

    def _ensure_predictor(self):
        if self._predictor is not None:
            return
        with self._predictor_lock:
            if self._predictor is not None:
                return
            cfg = ModelConfig()
            wp = os.path.join(self._model_dir, "obstacle_distance_backbone.pt")
            if not os.path.exists(wp):
                wp = os.path.join("/models/obstacle_distance", "obstacle_distance_backbone.pt")
            if not os.path.exists(wp):
                wp = None
            self._predictor = load_predictor(cfg, weights_path=wp)

    def _start_node(self, node_key: str, input_topic: str):
        icfg = self._instance_configs.get(node_key, {})
        fps = int(icfg.get("fps", self._fps))
        suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
        node = _ODNode(input_topic, self._predictor, fps, node_suffix=suffix)
        self._executor.add_node(node)
        self._nodes[node_key] = node
        node.start()
        log.info(f"[obstacle_distance] node started: {input_topic}")

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action in ("info",):
            if self._predictor_loading:
                return {"name": "ObstacleDistance", "state": "loading",
                        "desc": "Loading depth model..."}
            if self._predictor_error:
                return {"name": "ObstacleDistance", "state": "error",
                        "desc": f"Model load failed: {self._predictor_error}"}
            instances = {k: {"input": n._input_topic, "output": n._output_topic,
                             "has_frame": n._has_frame, "query_count": n._query_count}
                         for k, n in self._nodes.items()}
            return {"name": "ObstacleDistance", "state": "running" if instances else "idle",
                    "model_dir": self._model_dir, "instances": instances,
                    "desc": "Nearest obstacle distance estimation (monocular depth)"}

        if action in ("nearest_obstacle_distance", "query"):
            if not self._nodes:
                return {"distance_m": None, "unit": "meter", "has_frame": False,
                        "error": "not started; call action=start with input_topic"}
            node = self._nodes[instance_id] if instance_id in self._nodes else next(iter(self._nodes.values()))
            return node.query()

        if action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required for action=start")
            node_key = instance_id or input_topic
            if node_key not in self._nodes:
                if self._predictor is None:
                    if self._predictor_loading:
                        return {"state": "loading", "message": "Model is still loading, please wait..."}
                    if self._predictor_error:
                        return {"state": "error", "message": f"Model failed to load: {self._predictor_error}"}
                    # Load model in background, then start node.
                    def _bg_start():
                        self._predictor_loading = True
                        self._predictor_error = None
                        try:
                            self._ensure_predictor()
                            self._predictor_loading = False
                            self._start_node(node_key, input_topic)
                        except Exception as e:
                            self._predictor_loading = False
                            self._predictor_error = str(e)
                            log.error(f"[obstacle_distance] model load failed: {e}", exc_info=True)
                    threading.Thread(target=_bg_start, daemon=True, name="od_model_load").start()
                    return {"state": "loading", "input": input_topic,
                            "output": f"{input_topic}/obstacle_distance",
                            "message": "Model loading in background, will start automatically"}
                self._start_node(node_key, input_topic)
            return self._nodes[node_key].start()

        if action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                self._executor.remove_node(node)
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                stopped = []
                for key in list(self._nodes.keys()):
                    n = self._nodes[key]
                    n.stop()
                    self._executor.remove_node(n)
                    del self._nodes[key]
                    stopped.append(key)
                return {"state": "idle", "stopped_instances": stopped}
            return {"state": "idle"}

        if action == "config":
            cfg = {k: v for k, v in args.items()
                   if k not in ("action", "instance_id") and v is not None and v != ""}
            if instance_id:
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    # restart node with new config
                    n = self._nodes[instance_id]
                    input_topic = n._input_topic
                    n.stop()
                    self._executor.remove_node(n)
                    del self._nodes[instance_id]
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            if "fps" in cfg:
                self._fps = int(cfg["fps"])
            if "model_dir" in cfg:
                self._model_dir = cfg["model_dir"]
            return {"status": "configured", "config": cfg}

        return None

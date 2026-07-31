"""ROS2 节点桩(src/node.py)。

订阅 /camera/front/rgb -> predict -> 发布 /perception/nod。
【judgeflow 集成点】：在 cb() 内调用 judgeflow 上报/获取决策(占位)。
rclpy 仅在 Jetson 侧 import；本文件在开发机可安全 import(不触发 ROS)。
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np

# 将 src 加入路径
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from config import Config
    from model.nod_model import NODModel
except ImportError:  # pragma: no cover
    from .config import Config
    from .model.nod_model import NODModel

logger = logging.getLogger(__name__)

TOPIC_RGB = "/camera/front/rgb"
TOPIC_NOD = "/perception/nod"
TOPIC_NOD_HAS = "/perception/nod_has_obstacle"


def _load_msg_type():
    """延迟导入 ROS 消息类型（仅 Jetson 侧存在）。"""
    from sensor_msgs.msg import Image

    return Image


def _ros_image_to_numpy(msg) -> np.ndarray:
    """ROS sensor_msgs/Image -> RGB uint8 HWC。"""
    import cv2
    from cv_bridge import CvBridge

    bridge = CvBridge()
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    if msg.encoding.startswith("bgr") or msg.encoding == "bgr8":
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return cv_img


def main() -> None:
    # Jetson 侧才导入 ROS
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, Bool

    rclpy.init()
    config = Config.default()
    model = NODModel(config)

    Image = _load_msg_type()

    class NODNode(Node):
        def __init__(self) -> None:
            super().__init__("nod_node")
            self.model = model
            self.sub = self.create_subscription(
                Image, TOPIC_RGB, self.cb, 10
            )
            self.pub = self.create_publisher(Float32, TOPIC_NOD, 10)
            self.pub_has = self.create_publisher(Bool, TOPIC_NOD_HAS, 10)

        def cb(self, msg) -> None:
            try:
                rgb = _ros_image_to_numpy(msg)
                distance, has = self.model.predict(rgb)
                # === judgeflow 集成点(占位) ===
                # from judgeflow import report
                # report(distance=distance, has_obstacle=has)
                self.pub.publish(Float32(data=float(distance)))
                self.pub_has.publish(Bool(data=bool(has)))
            except Exception as e:  # pragma: no cover
                logger.error("推理失败: %s", e)

    node = NODNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

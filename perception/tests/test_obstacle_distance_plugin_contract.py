from pathlib import Path

from plugins.obstacle_distance.plugin import TOOLS, ObstacleDistancePlugin


class FakeExecutor:
    def __init__(self):
        self.added = []
        self.removed = []

    def add_node(self, node):
        self.added.append(node)

    def remove_node(self, node):
        self.removed.append(node)


class FakeNode:
    def __init__(self, input_topic: str):
        self.input_topic = input_topic
        self.output_topic = f"{input_topic}/obstacle_distance"
        self.running = False
        self.configs = []

    def start(self):
        self.running = True
        return {
            "state": "running",
            "input": self.input_topic,
            "output": self.output_topic,
        }

    def stop(self):
        self.running = False
        return {"state": "idle", "input": self.input_topic}

    def update_config(self, config):
        self.configs.append(config)

    def destroy_node(self):
        return None


def test_tool_contract_declares_processor_topics() -> None:
    tool = TOOLS[0]

    assert tool["name"] == "distance"
    assert tool["type"] == "processor"
    assert tool["multiInstance"] is True
    assert tool["topic_in"][0]["format"] == "image/jpeg"
    assert tool["topic_out"][0]["format"] == "data/json"


def test_plugin_dispatches_start_config_info_and_stop() -> None:
    executor = FakeExecutor()
    created_nodes: list[FakeNode] = []

    def node_factory(input_topic, estimator, config, node_suffix):
        assert estimator == "estimator"
        assert node_suffix
        node = FakeNode(input_topic)
        created_nodes.append(node)
        return node

    plugin = ObstacleDistancePlugin(
        {"fps": 3, "scene": "outdoor"},
        executor,
        estimator_factory=lambda: "estimator",
        node_factory=node_factory,
    )

    started = plugin.dispatch(
        "distance",
        {"action": "start", "input_topic": "/robot/camera/rgb"},
    )
    configured = plugin.dispatch(
        "distance",
        {
            "action": "config",
            "input_topic": "/robot/camera/rgb",
            "fps": 5,
        },
    )
    info = plugin.dispatch("distance", {"action": "info"})
    stopped = plugin.dispatch(
        "distance",
        {"action": "stop", "input_topic": "/robot/camera/rgb"},
    )

    assert started["output"] == "/robot/camera/rgb/obstacle_distance"
    assert configured["config"]["fps"] == 5
    assert info["instances"][0]["state"] == "running"
    assert stopped["state"] == "idle"
    assert executor.added == created_nodes
    assert executor.removed == created_nodes


def test_perception_bundle_registers_obstacle_plugin() -> None:
    perception_dir = Path(__file__).resolve().parents[1]
    main_source = (perception_dir / "main.py").read_text()
    config_source = (perception_dir / "config.yaml").read_text()

    assert "ObstacleDistancePlugin" in main_source
    assert 'plugins_cfg.get("obstacle"' in main_source
    assert "\n  obstacle:\n" in config_source

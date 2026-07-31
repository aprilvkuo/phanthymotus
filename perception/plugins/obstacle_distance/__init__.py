"""Obstacle Distance perception plugin package.

Imported by perception/main.py when config.plugins.obstacle_distance.enabled is true.
The leaderboard entry point is predict.py in this package (no ROS2 required).
"""
from .obstacle_distance_plugin import ObstacleDistancePlugin, TOOLS

__all__ = ["ObstacleDistancePlugin", "TOOLS"]

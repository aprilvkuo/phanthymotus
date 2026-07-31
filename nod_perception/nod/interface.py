"""Public inference interface for the NOD benchmark.

The benchmark (judgeflow) pulls this project and evaluates the model's
ability to predict the Nearest Obstacle Distance in front of the robot.

We expose a single, stable function-style contract so it can be adapted to
whatever exact signature judgeflow uses. The default and most robust path
is the geometric depth estimator, which needs no weights and trivially
satisfies the <30M-params and <10%-GPU constraints.

Result contract
---------------
NODResult.distance : float (meters). ``inf`` when no obstacle is inside the
                     forward navigable corridor within ``depth_max``.
NODResult.has_obstacle : bool. True when an obstacle is present and near.
NODResult.source : which estimator produced the result (for logging).
NODResult.corridor_ratio : fraction of near pixels inside the corridor.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


@dataclass
class NODResult:
    distance: float                 # meters; inf = no obstacle in corridor
    has_obstacle: bool = False
    source: str = ""
    corridor_ratio: float = 0.0
    extra: Optional[Dict[str, Any]] = None


class BaseNODPredictor(ABC):
    """All NOD estimators implement this contract."""

    @abstractmethod
    def predict(
        self,
        depth: Any = None,
        rgb: Any = None,
        pointcloud: Any = None,
        camera_info: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> NODResult:
        """Predict the nearest obstacle distance in front of the robot.

        Parameters
        ----------
        depth : HxW (or HxWx1) depth image in meters, OR None.
        rgb   : HxWx3 RGB image (only needed by learned mode), OR None.
        pointcloud : Nx3 (or HxWx3) XYZ in robot frame (x=forward,y=left,z=up), OR None.
        camera_info : optional dict overriding fx/fy/cx/cy from config.

        Returns
        -------
        NODResult
        """
        raise NotImplementedError

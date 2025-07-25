import numpy as np
from dataclasses import dataclass


@dataclass
class Obstacle:
    """Dataclass representing a spherical obstacle"""
    center: np.ndarray
    radius: float
    safe_distance: float = 0.0

    def __post_init__(self):
        """Ensure center is a numpy array"""
        if not isinstance(self.center, np.ndarray):
            self.center = np.array(self.center)

    @property
    def danger_threshold(self) -> float:
        """Total threshold distance (radius + safety margin)"""
        return self.radius + self.safe_distance

class PillarObstacle(Obstacle):
    """Dataclass representing a cylindrical pillar obstacle"""

    def __init__(self, center: np.ndarray, radius: float, height: float, safe_distance: float = 0.0):
        super().__init__(center, radius, safe_distance)
        self.height = height

    @property
    def danger_threshold(self) -> float:
        """Total threshold distance for the pillar (radius + safety margin)"""
        return super().danger_threshold
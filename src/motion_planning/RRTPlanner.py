import numpy as np
import mujoco
from typing import Callable, List, Optional, Tuple

class RRTPlanner:
    """Rapidly-exploring Random Tree (RRT) planner for joint space planning"""

    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 step_size: float = 0.3,  # Larger step size for faster planning
                 max_iterations: int = 1000,  # Fewer iterations for faster results
                 goal_threshold: float = 0.15):  # More relaxed goal threshold
        """
        Initialize RRT planner

        Args:
            model: MuJoCo model
            data: MuJoCo data
            step_size: Step size for tree extension
            max_iterations: Maximum planning iterations
            goal_threshold: Distance threshold to consider goal reached
        """
        self.model = model
        self.data = data
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold

        # Use proper joint limits for Kinova Gen3
        self.joint_limits_lower = np.array([-3.14, -2.24, -3.14, -2.57, -3.14, -2.09, -3.14])
        self.joint_limits_upper = np.array([3.14, 2.24, 3.14, 2.57, 3.14, 2.09, 3.14])

        # Tree structure: list of (configuration, parent_index)
        self.tree = []

        # Collision checking function (default: no collision checking)
        self.collision_checker = None

    def set_collision_checker(self, collision_fn: Callable[[np.ndarray], bool]):
        """Set collision checking function"""
        self.collision_checker = collision_fn

    def _sample_random_configuration(self, *, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample a random valid joint configuration"""
        sampler = rng if rng is not None else np.random
        return sampler.uniform(
            self.joint_limits_lower,
            self.joint_limits_upper
        )

    def _find_nearest_node(self, config: np.ndarray) -> int:
        """Find nearest node in tree to given configuration"""
        distances = [np.linalg.norm(node[0] - config) for node in self.tree]
        return int(np.argmin(distances))

    def _extend_towards(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
        """Extend from one configuration towards another by step_size"""
        direction = to_config - from_config
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return to_config

        unit_direction = direction / distance
        return from_config + self.step_size * unit_direction

    def _is_collision_free(self, config: np.ndarray) -> bool:
        """Check if configuration is collision-free"""
        if self.collision_checker is None:
            return True
        return not self.collision_checker(config)

    def _edge_probe_count(self, config1: np.ndarray, config2: np.ndarray) -> int:
        distance = float(np.linalg.norm(config2 - config1))
        probe_step = max(float(self.step_size) * 0.5, 1e-3)
        return max(2, int(np.ceil(distance / probe_step)) + 1)

    def _is_edge_collision_free(self, config1: np.ndarray, config2: np.ndarray) -> bool:
        num_points = self._edge_probe_count(config1, config2)
        return all(self._is_collision_free(config) for config in self._interpolate_path(config1, config2, num_points=num_points))

    def _interpolate_path(self, config1: np.ndarray, config2: np.ndarray,
                          num_points: int = 10) -> List[np.ndarray]:
        """Interpolate between two configurations"""
        return [config1 + t * (config2 - config1)
                for t in np.linspace(0, 1, num_points)]

    def plan(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[np.ndarray], bool]:
        """
        Plan a path from start to goal configuration

        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration

        Returns:
            path: List of joint configurations from start to goal
            success: Whether planning succeeded
        """
        if not self._is_collision_free(start_config) or not self._is_collision_free(goal_config):
            return [], False

        # Initialize tree with start configuration
        self.tree = [(start_config.copy(), -1)]
        sampler = rng if rng is not None else np.random

        for iteration in range(self.max_iterations):
            # Sample random configuration (with bias towards goal)
            if sampler.random() < 0.1:  # 10% bias towards goal
                random_config = goal_config
            else:
                random_config = self._sample_random_configuration(rng=sampler)

            # Find nearest node in tree
            nearest_idx = self._find_nearest_node(random_config)
            nearest_config = self.tree[nearest_idx][0]

            # Extend towards random configuration
            new_config = self._extend_towards(nearest_config, random_config)

            # Require a collision-free edge from nearest to new node.
            if self._is_edge_collision_free(nearest_config, new_config):
                # Add to tree
                self.tree.append((new_config, nearest_idx))

                # Check if we reached the goal
                distance_to_goal = np.linalg.norm(new_config - goal_config)
                if distance_to_goal < self.goal_threshold:
                    # Try to connect directly to goal
                    if self._is_edge_collision_free(new_config, goal_config):
                        self.tree.append((goal_config, len(self.tree) - 1))

                        # Extract path
                        path = self._extract_path(len(self.tree) - 1)
                        return path, True

        # Planning failed
        return [], False

    def _extract_path(self, goal_idx: int) -> List[np.ndarray]:
        """Extract path from tree by backtracking from goal"""
        path = []
        current_idx = goal_idx

        while current_idx != -1:
            path.append(self.tree[current_idx][0])
            current_idx = self.tree[current_idx][1]

        path.reverse()
        return path

    def smooth_path(
        self,
        path: List[np.ndarray],
        max_iterations: int = 100,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> List[np.ndarray]:
        """Smooth path using shortcut smoothing"""
        if len(path) <= 2:
            return path

        smoothed_path = [config.copy() for config in path]
        sampler = rng if rng is not None else np.random

        for _ in range(max_iterations):
            # Select two random points on path
            if rng is not None:
                i = int(sampler.integers(0, len(smoothed_path)))
                j = int(sampler.integers(0, len(smoothed_path)))
            else:
                i = int(np.random.randint(0, len(smoothed_path)))
                j = int(np.random.randint(0, len(smoothed_path)))

            if abs(i - j) <= 1:
                continue

            if i > j:
                i, j = j, i

            # Check if direct connection is collision-free
            interpolated = self._interpolate_path(smoothed_path[i], smoothed_path[j])

            if self._is_edge_collision_free(smoothed_path[i], smoothed_path[j]):
                # Replace path segment with direct connection
                smoothed_path = (smoothed_path[:i + 1] +
                                 interpolated[1:-1] +
                                 smoothed_path[j:])

        return smoothed_path

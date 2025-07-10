import numpy as np
import mujoco
from typing import List, Tuple, Callable, Optional, Dict, Any
from .kinematics import KinematicsSolver
from .utils import Obstacle

class CostFunction:
    """Base class for trajectory optimization cost functions"""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """
        Compute cost for a trajectory

        Args:
            trajectory: (n_waypoints, n_dof) trajectory
            dt: Time step between waypoints

        Returns:
            cost: Scalar cost value
        """
        raise NotImplementedError

    def compute_gradient(self, trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Compute gradient of cost w.r.t. trajectory

        Args:
            trajectory: (n_waypoints, n_dof) trajectory
            dt: Time step between waypoints

        Returns:
            gradient: (n_waypoints, n_dof) gradient
        """
        # Default: numerical gradient with optimized epsilon
        eps = 1e-5  # Larger epsilon for faster, more stable numerical gradient
        gradient = np.zeros_like(trajectory)

        # Compute base cost once
        base_cost = self.compute_cost(trajectory, dt)

        for i in range(trajectory.shape[0]):
            for j in range(trajectory.shape[1]):
                # Forward difference only (faster than central difference)
                trajectory[i, j] += eps
                cost_plus = self.compute_cost(trajectory, dt)
                trajectory[i, j] -= eps  # Restore

                gradient[i, j] = (cost_plus - base_cost) / eps

        return gradient * self.weight


class VelocityCostFunction(CostFunction):
    """Cost function penalizing high joint velocities"""

    def __init__(self, weight: float = 1.0, max_velocity: float = 2.0):
        super().__init__(weight)
        self.max_velocity = max_velocity

    def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """Compute velocity cost"""
        if trajectory.shape[0] < 2:
            return 0.0

        # Compute velocities (finite differences)
        velocities = np.diff(trajectory, axis=0) / dt

        # Quadratic penalty for velocities
        velocity_cost = np.sum(velocities ** 2)

        # Additional penalty for exceeding max velocity
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        excess_velocity = np.maximum(0, velocity_magnitudes - self.max_velocity)
        violation_cost = np.sum(excess_velocity ** 2) * 100  # High penalty

        return self.weight * (velocity_cost + violation_cost)

    def compute_gradient(self, trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Compute analytical gradient for velocity cost"""
        if trajectory.shape[0] < 2:
            return np.zeros_like(trajectory)

        gradient = np.zeros_like(trajectory)

        # Gradient of quadratic velocity cost
        velocities = np.diff(trajectory, axis=0) / dt

        # Interior points (affected by both forward and backward differences)
        for i in range(1, trajectory.shape[0] - 1):
            # Contribution from velocity between waypoints i-1 and i
            gradient[i] += 2 * velocities[i - 1] / dt
            # Contribution from velocity between waypoints i and i+1
            gradient[i] -= 2 * velocities[i] / dt

        # Boundary points
        if trajectory.shape[0] > 1:
            gradient[0] -= 2 * velocities[0] / dt
            gradient[-1] += 2 * velocities[-1] / dt

        return gradient * self.weight


class AccelerationCostFunction(CostFunction):
    """Cost function penalizing high joint accelerations"""

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """Compute acceleration cost"""
        if trajectory.shape[0] < 3:
            return 0.0

        # Compute accelerations (second-order finite differences)
        accelerations = np.diff(trajectory, n=2, axis=0) / (dt ** 2)

        # Quadratic penalty for accelerations
        return self.weight * np.sum(accelerations ** 2)


class SmoothnessCostFunction(CostFunction):
    """Cost function for trajectory smoothness (jerk minimization)"""

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """Compute smoothness cost (jerk squared)"""
        if trajectory.shape[0] < 4:
            return 0.0

        # Compute jerk (third-order finite differences)
        jerk = np.diff(trajectory, n=3, axis=0) / (dt ** 3)

        return self.weight * np.sum(jerk ** 2)


class TrajectoryLengthCostFunction(CostFunction):
    """Cost function for minimizing total trajectory length in joint space"""

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """Compute total trajectory length cost"""
        if trajectory.shape[0] < 2:
            return 0.0

        # Compute distances between consecutive waypoints
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        total_length = np.sum(distances)

        return float(self.weight * total_length)

    def compute_gradient(self, trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Compute analytical gradient for trajectory length cost"""
        if trajectory.shape[0] < 2:
            return np.zeros_like(trajectory)

        gradient = np.zeros_like(trajectory)

        # Compute unit vectors between consecutive waypoints
        for i in range(trajectory.shape[0] - 1):
            diff = trajectory[i + 1] - trajectory[i]
            distance = np.linalg.norm(diff)

            if distance > 1e-8:  # Avoid division by zero
                unit_vector = diff / distance

                # Gradient contribution to point i (negative direction)
                gradient[i] -= unit_vector
                # Gradient contribution to point i+1 (positive direction)
                gradient[i + 1] += unit_vector

        return gradient * self.weight


class ObstacleAvoidanceCostFunction(CostFunction):
    """Cost function for avoiding multiple spherical obstacles in Cartesian space"""

    def __init__(self,
                 kinematics_solver: KinematicsSolver,
                 obstacles: List[Obstacle],
                 weight: float = 1.0):
        """
        Initialize obstacle avoidance cost function

        Args:
            kinematics_solver: KinematicsSolver instance for forward kinematics
            obstacles: List of Obstacle instances to avoid
            weight: Cost function weight
        """
        super().__init__(weight)
        self.kinematics_solver = kinematics_solver
        self.obstacles = obstacles if obstacles else []

    def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """Compute obstacle avoidance cost for all obstacles"""
        if not self.obstacles:
            return 0.0

        total_cost = 0.0

        # Backup current state
        self.kinematics_solver._backup_state()

        try:
            for waypoint in trajectory:
                # Get end-effector position for this waypoint
                ee_position, _ = self.kinematics_solver.forward_kinematics(waypoint)

                # Check against all obstacles
                for obstacle in self.obstacles:
                    # Compute distance to obstacle center
                    distance_to_obstacle = np.linalg.norm(ee_position - obstacle.center)

                    # Always apply cost based on proximity to obstacles
                    if distance_to_obstacle < obstacle.danger_threshold:
                        # Strong quadratic penalty for violations
                        violation = obstacle.danger_threshold - distance_to_obstacle
                        total_cost += violation ** 2
                    else:
                        # Soft attractive force to maintain distance (for better gradients)
                        safe_zone = obstacle.danger_threshold * 1.5  # 50% larger safe zone
                        if distance_to_obstacle < safe_zone:
                            proximity_factor = (safe_zone - distance_to_obstacle) / (
                                        safe_zone - obstacle.danger_threshold)
                            total_cost += 0.01 * proximity_factor ** 2  # Small cost for being near

        finally:
            # Restore original state
            self.kinematics_solver._restore_state()

        return float(self.weight * total_cost)

    def compute_gradient(self, trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Compute simplified analytical gradient for obstacle avoidance cost"""
        if not self.obstacles:
            return np.zeros_like(trajectory)

        gradient = np.zeros_like(trajectory)

        # Backup current state
        self.kinematics_solver._backup_state()

        try:
            for i, waypoint in enumerate(trajectory):
                # Get end-effector position and Jacobian
                ee_position, _ = self.kinematics_solver.forward_kinematics(waypoint)

                # Simplified Jacobian computation - just use finite differences on position
                eps = 1e-4
                jacobian = np.zeros((3, len(waypoint)))

                for j in range(len(waypoint)):
                    waypoint_plus = waypoint.copy()
                    waypoint_plus[j] += eps
                    ee_plus, _ = self.kinematics_solver.forward_kinematics(waypoint_plus)
                    jacobian[:, j] = (ee_plus - ee_position) / eps

                # Compute gradient contribution for each obstacle
                for obstacle in self.obstacles:
                    # Distance to obstacle center
                    distance_vec = ee_position - obstacle.center
                    distance_to_obstacle = np.linalg.norm(distance_vec)

                    # Unit vector pointing away from obstacle center
                    if distance_to_obstacle > 1e-8:
                        unit_vec = distance_vec / distance_to_obstacle
                    else:
                        unit_vec = np.array([1.0, 0.0, 0.0])  # Arbitrary direction

                    # Compute gradient based on distance regime
                    if distance_to_obstacle < obstacle.danger_threshold:
                        # Gradient of squared violation cost
                        violation = obstacle.danger_threshold - distance_to_obstacle
                        cost_gradient = -2 * violation * unit_vec
                    else:
                        # Gradient of proximity cost in safe zone
                        safe_zone = obstacle.danger_threshold * 1.5
                        if distance_to_obstacle < safe_zone:
                            proximity_factor = (safe_zone - distance_to_obstacle) / (
                                        safe_zone - obstacle.danger_threshold)
                            cost_gradient = -2 * 0.01 * proximity_factor * unit_vec / (
                                        safe_zone - obstacle.danger_threshold)
                        else:
                            cost_gradient = np.zeros(3)  # No gradient outside safe zone

                    # Chain rule: gradient w.r.t. joint angles
                    gradient[i] += jacobian.T @ cost_gradient

        finally:
            # Restore original state
            self.kinematics_solver._restore_state()

        return gradient * self.weight

    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle to the list"""
        self.obstacles.append(obstacle)

    def remove_obstacle(self, index: int):
        """Remove an obstacle by index"""
        if 0 <= index < len(self.obstacles):
            del self.obstacles[index]

    def get_obstacle_info(self) -> str:
        """Get formatted string with obstacle information"""
        if not self.obstacles:
            return "No obstacles"

        info_lines = []
        for i, obs in enumerate(self.obstacles):
            info_lines.append(
                f"  Obstacle {i + 1}: center=({obs.center[0]:.2f}, {obs.center[1]:.2f}, {obs.center[2]:.2f}), "
                f"radius={obs.radius:.3f}m, safety={obs.safe_distance:.3f}m"
            )
        return "\n".join(info_lines)


class SafetyImportanceCostFunction(CostFunction):
    """Cost function that penalizes being near obstacles to encourage safer paths"""

    def __init__(self,
                 kinematics_solver: KinematicsSolver,
                 obstacles: List[Obstacle],
                 weight: float = 1.0,
                 safety_radius_multiplier: float = 2.0):
        """
        Initialize safety importance cost function

        Args:
            kinematics_solver: Forward kinematics solver
            obstacles: List of obstacles to avoid
            weight: Cost function weight
            safety_radius_multiplier: How far beyond obstacle radius to consider "unsafe"
                                    Higher values = more cautious behavior
        """
        super().__init__(weight)
        self.kinematics_solver = kinematics_solver
        self.obstacles = obstacles
        self.safety_radius_multiplier = safety_radius_multiplier

    def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """Compute safety importance cost - penalizes proximity to obstacles"""
        if not self.obstacles:
            return 0.0

        total_cost = 0.0

        # Backup current state
        self.kinematics_solver._backup_state()

        try:
            for waypoint in trajectory:
                # Get end-effector position for this waypoint
                ee_position, _ = self.kinematics_solver.forward_kinematics(waypoint)

                # Check against all obstacles
                for obstacle in self.obstacles:
                    # Compute distance to obstacle center
                    distance_to_obstacle = np.linalg.norm(ee_position - obstacle.center)

                    # Define safety zone as larger radius around obstacle
                    safety_zone_radius = obstacle.radius * self.safety_radius_multiplier

                    # Apply cost if within safety zone (encourages going around)
                    if distance_to_obstacle < safety_zone_radius:
                        # Normalized distance (0 = at center, 1 = at safety zone edge)
                        normalized_distance = distance_to_obstacle / safety_zone_radius

                        # Inverse quadratic cost - higher cost for being closer
                        proximity_cost = (1.0 - normalized_distance) ** 2
                        total_cost += proximity_cost

        finally:
            # Restore original state
            self.kinematics_solver._restore_state()

        return float(self.weight * total_cost)

    def compute_gradient(self, trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Compute gradient for safety importance cost"""
        if not self.obstacles:
            return np.zeros_like(trajectory)

        gradient = np.zeros_like(trajectory)

        # Backup current state
        self.kinematics_solver._backup_state()

        try:
            for i, waypoint in enumerate(trajectory):
                # Get end-effector position and simplified Jacobian
                ee_position, _ = self.kinematics_solver.forward_kinematics(waypoint)

                # Simplified Jacobian computation using finite differences
                eps = 1e-4
                jacobian = np.zeros((3, len(waypoint)))

                for j in range(len(waypoint)):
                    waypoint_plus = waypoint.copy()
                    waypoint_plus[j] += eps
                    ee_plus, _ = self.kinematics_solver.forward_kinematics(waypoint_plus)
                    jacobian[:, j] = (ee_plus - ee_position) / eps

                # Compute gradient contribution for each obstacle
                for obstacle in self.obstacles:
                    # Distance to obstacle center
                    distance_vec = ee_position - obstacle.center
                    distance_to_obstacle = np.linalg.norm(distance_vec)

                    # Safety zone radius
                    safety_zone_radius = obstacle.radius * self.safety_radius_multiplier

                    # Only add gradient if within safety zone
                    if distance_to_obstacle < safety_zone_radius:
                        # Handle numerical stability for very small distances
                        if distance_to_obstacle < 1e-6:
                            # Use a small default distance to avoid division by zero
                            distance_to_obstacle = 1e-6
                            distance_vec = np.array([1.0, 0.0, 0.0]) * 1e-6  # Arbitrary direction

                        # Unit vector pointing away from obstacle center
                        unit_vec = distance_vec / distance_to_obstacle

                        # Normalized distance (clamp to avoid numerical issues)
                        normalized_distance = max(float(distance_to_obstacle / safety_zone_radius), 1e-6)

                        # Gradient of (1 - normalized_distance)^2 w.r.t. distance
                        # Since cost decreases as distance increases, gradient should point away from obstacle
                        cost_gradient = -2 * (1.0 - normalized_distance) * unit_vec / safety_zone_radius

                        # Check for numerical issues
                        if np.any(np.isnan(cost_gradient)) or np.any(np.isinf(cost_gradient)):
                            continue  # Skip this obstacle if gradient is invalid

                        # Chain rule: gradient w.r.t. joint angles
                        gradient[i] += jacobian.T @ cost_gradient

        finally:
            # Restore original state
            self.kinematics_solver._restore_state()

        return gradient * self.weight

class FixedZCostFunction(CostFunction):
    def __init__(self, kinematics_solver, target_z=0.529, weight=100.0):
        super().__init__(weight)
        self.kinematics = kinematics_solver
        self.target_z = target_z

    def compute_cost(self, trajectory, dt=0.1):
        total_cost = 0.0
        self.kinematics._backup_state()
        try:
            for q in trajectory:
                ee_pos, _ = self.kinematics.forward_kinematics(q)
                z_error = ee_pos[2] - self.target_z
                total_cost += z_error ** 2
        finally:
            self.kinematics._restore_state()
        return self.weight * total_cost

    def compute_gradient(self, trajectory, dt=0.1):
        grad = np.zeros_like(trajectory)
        self.kinematics._backup_state()
        try:
            for i, q in enumerate(trajectory):
                ee_pos, _ = self.kinematics.forward_kinematics(q)
                z_error = ee_pos[2] - self.target_z

                eps = 1e-4
                n_joints = len(q)
                J_z = np.zeros(n_joints)

                # Finite difference approximation of z-Jacobian
                for j in range(n_joints):
                    q_eps = q.copy()
                    q_eps[j] += eps
                    ee_pos_eps, _ = self.kinematics.forward_kinematics(q_eps)
                    J_z[j] = (ee_pos_eps[2] - ee_pos[2]) / eps

                grad[i] = 2 * z_error * J_z
        finally:
            self.kinematics._restore_state()
        return grad * self.weight

"""
Path planning algorithms for robot motion planning
"""

import numpy as np
import mujoco
from typing import List, Tuple, Callable, Optional, Dict, Any
from scipy.optimize import minimize
from dataclasses import dataclass
from .kinematics import KinematicsSolver


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


class RRTPlanner:
    """Rapidly-exploring Random Tree (RRT) planner for joint space planning"""
    
    def __init__(self, 
                 model: mujoco.MjModel, 
                 data: mujoco.MjData,
                 step_size: float = 0.3,      # Larger step size for faster planning
                 max_iterations: int = 1000,  # Fewer iterations for faster results
                 goal_threshold: float = 0.15): # More relaxed goal threshold
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
    
    def _sample_random_configuration(self) -> np.ndarray:
        """Sample a random valid joint configuration"""
        return np.random.uniform(
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
    
    def _interpolate_path(self, config1: np.ndarray, config2: np.ndarray, 
                         num_points: int = 10) -> List[np.ndarray]:
        """Interpolate between two configurations"""
        return [config1 + t * (config2 - config1) 
                for t in np.linspace(0, 1, num_points)]
    
    def plan(self, start_config: np.ndarray, goal_config: np.ndarray) -> Tuple[List[np.ndarray], bool]:
        """
        Plan a path from start to goal configuration
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            
        Returns:
            path: List of joint configurations from start to goal
            success: Whether planning succeeded
        """
        # Initialize tree with start configuration
        self.tree = [(start_config.copy(), -1)]
        
        for iteration in range(self.max_iterations):
            # Sample random configuration (with bias towards goal)
            if np.random.random() < 0.1:  # 10% bias towards goal
                random_config = goal_config
            else:
                random_config = self._sample_random_configuration()
            
            # Find nearest node in tree
            nearest_idx = self._find_nearest_node(random_config)
            nearest_config = self.tree[nearest_idx][0]
            
            # Extend towards random configuration
            new_config = self._extend_towards(nearest_config, random_config)
            
            # Check if new configuration is valid and collision-free
            if self._is_collision_free(new_config):
                # Add to tree
                self.tree.append((new_config, nearest_idx))
                
                # Check if we reached the goal
                distance_to_goal = np.linalg.norm(new_config - goal_config)
                if distance_to_goal < self.goal_threshold:
                    # Try to connect directly to goal
                    if self._is_collision_free(goal_config):
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
    
    def smooth_path(self, path: List[np.ndarray], max_iterations: int = 100) -> List[np.ndarray]:
        """Smooth path using shortcut smoothing"""
        if len(path) <= 2:
            return path
        
        smoothed_path = [config.copy() for config in path]
        
        for _ in range(max_iterations):
            # Select two random points on path
            i = np.random.randint(0, len(smoothed_path))
            j = np.random.randint(0, len(smoothed_path))
            
            if abs(i - j) <= 1:
                continue
                
            if i > j:
                i, j = j, i
            
            # Check if direct connection is collision-free
            interpolated = self._interpolate_path(smoothed_path[i], smoothed_path[j])
            
            if all(self._is_collision_free(config) for config in interpolated):
                # Replace path segment with direct connection
                smoothed_path = (smoothed_path[:i+1] + 
                               interpolated[1:-1] + 
                               smoothed_path[j:])
        
        return smoothed_path


# Cost Function Interface for TrajOpt
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
        velocity_cost = np.sum(velocities**2)
        
        # Additional penalty for exceeding max velocity
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        excess_velocity = np.maximum(0, velocity_magnitudes - self.max_velocity)
        violation_cost = np.sum(excess_velocity**2) * 100  # High penalty
        
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
            gradient[i] += 2 * velocities[i-1] / dt
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
        accelerations = np.diff(trajectory, n=2, axis=0) / (dt**2)
        
        # Quadratic penalty for accelerations
        return self.weight * np.sum(accelerations**2)


class SmoothnessCostFunction(CostFunction):
    """Cost function for trajectory smoothness (jerk minimization)"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
    
    def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """Compute smoothness cost (jerk squared)"""
        if trajectory.shape[0] < 4:
            return 0.0
        
        # Compute jerk (third-order finite differences)
        jerk = np.diff(trajectory, n=3, axis=0) / (dt**3)
        
        return self.weight * np.sum(jerk**2)


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
            diff = trajectory[i+1] - trajectory[i]
            distance = np.linalg.norm(diff)
            
            if distance > 1e-8:  # Avoid division by zero
                unit_vector = diff / distance
                
                # Gradient contribution to point i (negative direction)
                gradient[i] -= unit_vector
                # Gradient contribution to point i+1 (positive direction)
                gradient[i+1] += unit_vector
        
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
                            proximity_factor = (safe_zone - distance_to_obstacle) / (safe_zone - obstacle.danger_threshold)
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
                            proximity_factor = (safe_zone - distance_to_obstacle) / (safe_zone - obstacle.danger_threshold)
                            cost_gradient = -2 * 0.01 * proximity_factor * unit_vec / (safe_zone - obstacle.danger_threshold)
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
                f"  Obstacle {i+1}: center=({obs.center[0]:.2f}, {obs.center[1]:.2f}, {obs.center[2]:.2f}), "
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


class TrajOptPlanner:
    """Trajectory Optimization planner using cost-based optimization"""
    
    def __init__(self, 
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 n_waypoints: int = 50,  # Reduced from 150 for faster computation
                 dt: float = 0.1):
        """
        Initialize TrajOpt planner
        
        Args:
            model: MuJoCo model
            data: MuJoCo data  
            n_waypoints: Number of waypoints in trajectory
            dt: Time step between waypoints
        """
        self.model = model
        self.data = data
        self.n_waypoints = n_waypoints
        self.dt = dt
        self.n_dof = 7  # Kinova Gen3 has 7 DOF
        
        # Joint limits
        self.joint_limits_lower = np.array([-3.14, -2.24, -3.14, -2.57, -3.14, -2.09, -3.14])
        self.joint_limits_upper = np.array([3.14, 2.24, 3.14, 2.57, 3.14, 2.09, 3.14])
        
        # Cost functions (can be swapped easily)
        self.cost_functions: List[CostFunction] = []
        
        # Constraints
        self.collision_checker = None
        
        # Progress tracking
        self.iteration_count = 0
        self.last_cost = float('inf')
    
    def add_cost_function(self, cost_fn: CostFunction):
        """Add a cost function to the optimization"""
        self.cost_functions.append(cost_fn)
    
    def clear_cost_functions(self):
        """Clear all cost functions"""
        self.cost_functions.clear()
    
    def set_collision_checker(self, collision_fn: Callable[[np.ndarray], bool]):
        """Set collision checking function"""
        self.collision_checker = collision_fn
    
    def _trajectory_to_vector(self, trajectory: np.ndarray) -> np.ndarray:
        """Convert trajectory matrix to flat vector for optimization"""
        return trajectory.flatten()
    
    def _vector_to_trajectory(self, vector: np.ndarray) -> np.ndarray:
        """Convert flat vector back to trajectory matrix"""
        return vector.reshape(self.n_waypoints, self.n_dof)
    
    def _compute_total_cost(self, trajectory_vector: np.ndarray) -> float:
        """Compute total cost for trajectory with progress tracking"""
        trajectory = self._vector_to_trajectory(trajectory_vector)
        
        total_cost = 0.0
        for cost_fn in self.cost_functions:
            total_cost += cost_fn.compute_cost(trajectory, self.dt)
        
        # Progress feedback
        self.iteration_count += 1
        if self.iteration_count % 10 == 0 or total_cost < self.last_cost * 0.9:
            print(f"  Iteration {self.iteration_count}: Cost = {total_cost:.3f}")
            self.last_cost = total_cost
        
        return total_cost
    
    def _compute_total_gradient(self, trajectory_vector: np.ndarray) -> np.ndarray:
        """Compute total gradient for trajectory"""
        trajectory = self._vector_to_trajectory(trajectory_vector)
        
        total_gradient = np.zeros_like(trajectory)
        for cost_fn in self.cost_functions:
            total_gradient += cost_fn.compute_gradient(trajectory, self.dt)
        
        return self._trajectory_to_vector(total_gradient)
    
    def _create_bounds(self, start_config: np.ndarray, goal_config: np.ndarray):
        """Create bounds for optimization variables"""
        bounds = []
        
        for i in range(self.n_waypoints):
            for j in range(self.n_dof):
                if i == 0:  # Start waypoint - fixed
                    bounds.append((start_config[j], start_config[j]))
                elif i == self.n_waypoints - 1:  # Goal waypoint - fixed
                    bounds.append((goal_config[j], goal_config[j]))
                else:  # Interior waypoints - within joint limits
                    bounds.append((self.joint_limits_lower[j], self.joint_limits_upper[j]))
        
        return bounds
    
    def _create_initial_trajectory(self, start_config: np.ndarray, goal_config: np.ndarray) -> np.ndarray:
        """Create initial trajectory guess (linear interpolation)"""
        trajectory = np.zeros((self.n_waypoints, self.n_dof))
        
        for i in range(self.n_waypoints):
            alpha = i / (self.n_waypoints - 1)
            trajectory[i] = (1 - alpha) * start_config + alpha * goal_config
        
        return trajectory
    
    def plan(self, start_config: np.ndarray, goal_config: np.ndarray) -> Tuple[List[np.ndarray], bool]:
        """
        Plan trajectory using trajectory optimization
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            
        Returns:
            trajectory: List of joint configurations
            success: Whether planning succeeded
        """
        if len(self.cost_functions) == 0:
            print("Warning: No cost functions added to TrajOpt planner")
            return [], False
        
        # Reset progress tracking
        self.iteration_count = 0
        self.last_cost = float('inf')
        
        # Create initial trajectory
        initial_trajectory = self._create_initial_trajectory(start_config, goal_config)
        initial_vector = self._trajectory_to_vector(initial_trajectory)
        
        # Create bounds
        bounds = self._create_bounds(start_config, goal_config)
        
        print(f"  Optimizing {self.n_waypoints} waypoints × {self.n_dof} DOF = {len(initial_vector)} variables...")
        initial_cost = self._compute_total_cost(initial_vector)
        print(f"  Initial cost: {initial_cost:.3f}")
        
        # Optimize trajectory with robust settings
        try:
            result = minimize(
                fun=self._compute_total_cost,
                x0=initial_vector,
                method='L-BFGS-B',
                jac=self._compute_total_gradient,
                bounds=bounds,
                options={
                    'maxiter': 500,     # More iterations for complex scenarios
                    'ftol': 1e-6,       # Tighter tolerance for better convergence
                    'gtol': 1e-5,       # Tighter gradient tolerance
                    'maxfun': 1000,     # Limit function evaluations to prevent hanging
                    'disp': False       # Suppress verbose output
                }
            )

            print(f"  Optimization completed in {self.iteration_count} cost evaluations")
            print(f"  Final cost: {result.fun:.3f}")
            print(f"  Status: {result.message}")
            
            if result.success or result.fun < initial_cost * 0.8:  # Accept if significantly improved
                optimized_trajectory = self._vector_to_trajectory(result.x)
                trajectory_list = [waypoint for waypoint in optimized_trajectory]
                return trajectory_list, True
            else:
                print(f"TrajOpt optimization failed: {result.message}")
                # Return initial trajectory as fallback
                initial_trajectory_list = [waypoint for waypoint in initial_trajectory]
                return initial_trajectory_list, False
                
        except Exception as e:
            print(f"TrajOpt planning failed: {e}")
            return [], False


class MotionPlannerFactory:
    """Factory for creating motion planners"""
    
    @staticmethod
    def create_rrt_planner(model: mujoco.MjModel, 
                          data: mujoco.MjData, 
                          **kwargs) -> RRTPlanner:
        """Create an RRT planner"""
        return RRTPlanner(model, data, **kwargs)
    
    @staticmethod
    def create_trajopt_planner(model: mujoco.MjModel,
                              data: mujoco.MjData,
                              **kwargs) -> TrajOptPlanner:
        """Create a TrajOpt planner"""
        return TrajOptPlanner(model, data, **kwargs)
    
    @staticmethod
    def create_collision_checker(model: mujoco.MjModel, 
                               data: mujoco.MjData) -> Callable[[np.ndarray], bool]:
        """Create a collision checking function using MuJoCo"""
        
        # Store original state
        original_qpos = data.qpos.copy()
        original_qvel = data.qvel.copy()
        
        def check_collision(joint_config: np.ndarray) -> bool:
            """Check if joint configuration results in collision"""
            # Set joint configuration
            data.qpos[:7] = joint_config
            data.qvel[:] = 0  # Zero velocities for static check
            
            # Forward dynamics to update contact information
            mujoco.mj_forward(model, data)
            
            # Check for any contacts (collisions)
            has_collision = data.ncon > 0
            
            # Restore original state
            data.qpos[:] = original_qpos
            data.qvel[:] = original_qvel
            mujoco.mj_forward(model, data)
            
            return has_collision
        
        return check_collision 
import numpy as np
import mujoco
from typing import List, Tuple, Callable, Optional, Dict, Any
from .cost_functions import CostFunction
from scipy.optimize import minimize


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
                    'maxiter': 500,  # More iterations for complex scenarios
                    'ftol': 1e-6,  # Tighter tolerance for better convergence
                    'gtol': 1e-5,  # Tighter gradient tolerance
                    'maxfun': 1000,  # Limit function evaluations to prevent hanging
                    'disp': False  # Suppress verbose output
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

import numpy as np
import mujoco
from typing import List, Tuple, Callable, Optional
from .cost_functions import CostFunction, CompositeCostFunction
from .utils import PerformanceTimer
from scipy.optimize import minimize
import time

class UnconstrainedTrajOptPlanner:
    """Trajectory Optimization planner using cost-based optimization (no constraints)"""

    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 n_waypoints: int = 50,  # Reduced from 150 for faster computation
                 dt: float = 0.1,
                 cost_mode: str = 'legacy'):
        """
        Initialize TrajOpt planner

        Args:
            model: MuJoCo model
            data: MuJoCo data
            n_waypoints: Number of waypoints in trajectory
            dt: Time step between waypoints
            cost_mode: 'legacy' (individual cost functions), 'composite' (single composite function)
        """
        self.model = model
        self.data = data
        self.n_waypoints = n_waypoints
        self.dt = dt
        self.n_dof = 7  # Kinova Gen3 has 7 DOF
        self.f_tol = 1e-6
        self.g_tol = 1e-5
        self.max_iter = 500
        self.max_fun = 1000
        
        # Cost mode configuration
        if cost_mode not in ['legacy', 'composite']:
            raise ValueError("cost_mode must be 'legacy' or 'composite'")
        self.cost_mode = cost_mode

        # Joint limits
        self.joint_limits_lower = np.array([-3.14, -2.24, -3.14, -2.57, -3.14, -2.09, -3.14])
        self.joint_limits_upper = np.array([3.14, 2.24, 3.14, 2.57, 3.14, 2.09, 3.14])

        # Cost functions (legacy mode)
        self.cost_functions: List[CostFunction] = []
        
        # Composite cost function (new mode)
        self.composite_cost_function: Optional[CompositeCostFunction] = None

        # Constraints
        self.collision_checker = None

        # Progress tracking
        self.iteration_count = 0
        self.last_cost = float('inf')
        self.last_iteration_time = time.time()
        
        # Performance timing
        self.timer = PerformanceTimer()

    def add_cost_function(self, cost_fn: CostFunction):
        """Add a cost function to the optimization (legacy mode only)"""
        if self.cost_mode != 'legacy':
            raise RuntimeError("add_cost_function() only available in legacy mode. Use set_composite_cost_function() instead.")
        self.cost_functions.append(cost_fn)

    def set_composite_cost_function(self, composite_fn: CompositeCostFunction):
        """Set the composite cost function (composite mode only)"""
        if self.cost_mode != 'composite':
            raise RuntimeError("set_composite_cost_function() only available in composite mode.")
        self.composite_cost_function = composite_fn
        print("  ✅ Composite cost function configured:")
        print(f"     {composite_fn.get_mode_info()}")

    def setup_composite_cost(self, 
                           cost_functions: List[CostFunction], 
                           weights: List[float],
                           formulation: str = 'sum',
                           rho: float = 0.01) -> CompositeCostFunction:
        """
        Convenience method to create and set composite cost function.
        
        Args:
            cost_functions: List of individual cost functions
            weights: Corresponding weights
            formulation: 'sum' or 'max'
            rho: Tie-breaking parameter for max mode
            
        Returns:
            The created CompositeCostFunction
        """
        if self.cost_mode != 'composite':
            raise RuntimeError("setup_composite_cost() only available in composite mode.")
        
        composite_fn = CompositeCostFunction(
            cost_functions=cost_functions,
            weights=weights,
            mode=formulation,
            rho=rho
        )
        self.set_composite_cost_function(composite_fn)
        return composite_fn

    def clear_cost_functions(self):
        """Clear all cost functions"""
        self.cost_functions.clear()
        self.composite_cost_function = None

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

        if self.cost_mode == 'legacy':
            # Legacy mode: sum individual cost functions
            total_cost = 0.0
            for cost_fn in self.cost_functions:
                total_cost += cost_fn.compute_cost(trajectory, self.dt)
        
        elif self.cost_mode == 'composite':
            # Composite mode: use single composite cost function
            if self.composite_cost_function is None:
                raise RuntimeError("No composite cost function set. Use set_composite_cost_function().")
            total_cost = self.composite_cost_function.compute_cost(trajectory, self.dt)
        
        else:
            raise ValueError(f"Invalid cost_mode: {self.cost_mode}")

        # Progress feedback
        self.iteration_count += 1
        average_iteration_time = (time.time() - self.last_iteration_time) / self.iteration_count

        if self.iteration_count % 10 == 0 or total_cost < self.last_cost * 0.9:
            print(f"  Iteration {self.iteration_count}: Cost = {total_cost:.3f}, Avg Time = {average_iteration_time:.3f}s")
        
        # Always update last_cost for convergence detection
        self.last_cost = total_cost

        return total_cost

    def _compute_total_gradient(self, trajectory_vector: np.ndarray) -> np.ndarray:
        """Compute total gradient for trajectory"""
        trajectory = self._vector_to_trajectory(trajectory_vector)

        if self.cost_mode == 'legacy':
            # Legacy mode: sum individual gradients
            total_gradient = np.zeros_like(trajectory)
            for cost_fn in self.cost_functions:
                total_gradient += cost_fn.compute_gradient(trajectory, self.dt)
        
        elif self.cost_mode == 'composite':
            # Composite mode: use single composite gradient
            if self.composite_cost_function is None:
                raise RuntimeError("No composite cost function set. Use set_composite_cost_function().")
            total_gradient = self.composite_cost_function.compute_gradient(trajectory, self.dt)
        
        else:
            raise ValueError(f"Invalid cost_mode: {self.cost_mode}")

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
        # Validate cost functions are set up
        if self.cost_mode == 'legacy' and len(self.cost_functions) == 0:
            print("Warning: No cost functions added to TrajOpt planner")
            return [], False
        elif self.cost_mode == 'composite' and self.composite_cost_function is None:
            print("Warning: No composite cost function set")
            return [], False

        # Reset progress tracking
        self.iteration_count = 0
        self.last_cost = float('inf')

        # Create initial trajectory
        initial_trajectory = self._create_initial_trajectory(start_config, goal_config)
        initial_vector = self._trajectory_to_vector(initial_trajectory)

        # Create bounds
        bounds = self._create_bounds(start_config, goal_config)

        # print(f"  Optimizing {self.n_waypoints} waypoints × {self.n_dof} DOF = {len(initial_vector)} variables...")
        # print(f"  Cost mode: {self.cost_mode.upper()}")
        
        initial_cost = self._compute_total_cost(initial_vector)
        # print(f"  Initial cost: {initial_cost:.3f}")

        # Optimize trajectory with robust settings
        try:
            result = minimize(
                fun=self._compute_total_cost,
                x0=initial_vector,
                method='L-BFGS-B',
                jac=self._compute_total_gradient,
                bounds=bounds,
                options={
                    'maxiter': self.max_iter,  # More iterations for complex scenarios
                    'ftol': self.f_tol,  # Tighter tolerance for better convergence
                    'gtol': self.g_tol,  # Tighter gradient tolerance
                    'maxfun': self.max_fun,  # Limit function evaluations to prevent hanging
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

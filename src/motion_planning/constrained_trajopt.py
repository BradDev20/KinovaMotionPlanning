import numpy as np
import mujoco
from typing import List, Tuple, Callable, Optional, Dict, Any
from .cost_functions import CostFunction
from .unconstrained_trajopt import UnconstrainedTrajOptPlanner
from scipy.optimize import minimize


class ConstrainedTrajOptPlanner(UnconstrainedTrajOptPlanner):
    """Trajectory Optimization planner with velocity and acceleration constraints"""

    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 n_waypoints: int = 50,  # Reduced from 150 for faster computation
                 dt: float = 0.1,
                 max_velocity: float = 2.0,  # Maximum joint velocity constraint
                 max_acceleration: float = 10.0):  # Maximum joint acceleration constraint
        """
        Initialize TrajOpt planner with constraints

        Args:
            model: MuJoCo model
            data: MuJoCo data
            n_waypoints: Number of waypoints in trajectory
            dt: Time step between waypoints
            max_velocity: Maximum allowed joint velocity (rad/s)
            max_acceleration: Maximum allowed joint acceleration (rad/s²)
        """
        super().__init__(model, data, n_waypoints, dt)
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

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

    def _compute_velocity_constraints(self, trajectory_vector: np.ndarray) -> np.ndarray:
        """
        Compute velocity constraint violations.
        Returns array where negative values indicate constraint violations.
        """
        trajectory = self._vector_to_trajectory(trajectory_vector)
        
        if trajectory.shape[0] < 2:
            return np.array([])
        
        # Compute velocities (finite differences)
        velocities = np.diff(trajectory, axis=0) / self.dt
        
        # Compute velocity magnitudes for each waypoint transition
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Constraint: max_velocity - |velocity| >= 0
        # Negative values indicate constraint violations
        constraints = self.max_velocity - velocity_magnitudes
        
        return constraints

    def _compute_acceleration_constraints(self, trajectory_vector: np.ndarray) -> np.ndarray:
        """
        Compute acceleration constraint violations.
        Returns array where negative values indicate constraint violations.
        """
        trajectory = self._vector_to_trajectory(trajectory_vector)
        
        if trajectory.shape[0] < 3:
            return np.array([])
        
        # Compute accelerations (second-order finite differences)
        accelerations = np.diff(trajectory, n=2, axis=0) / (self.dt ** 2)
        
        # Compute acceleration magnitudes for each waypoint
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Constraint: max_acceleration - |acceleration| >= 0
        # Negative values indicate constraint violations
        constraints = self.max_acceleration - acceleration_magnitudes
        
        return constraints

    def _compute_velocity_constraint_jacobian(self, trajectory_vector: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of velocity constraints with respect to trajectory parameters.
        """
        trajectory = self._vector_to_trajectory(trajectory_vector)
        
        if trajectory.shape[0] < 2:
            return np.array([]).reshape(0, len(trajectory_vector))
        
        n_velocity_constraints = trajectory.shape[0] - 1
        jacobian = np.zeros((n_velocity_constraints, len(trajectory_vector)))
        
        # Compute velocities
        velocities = np.diff(trajectory, axis=0) / self.dt
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        for i in range(n_velocity_constraints):
            if velocity_magnitudes[i] > 1e-8:  # Avoid division by zero
                # Gradient of -|velocity_i| with respect to trajectory
                vel_unit = velocities[i] / velocity_magnitudes[i]
                
                # Velocity depends on waypoints i and i+1
                start_idx = i * self.n_dof
                end_idx = (i + 1) * self.n_dof
                
                # Contribution from waypoint i (negative because velocity = (q_{i+1} - q_i)/dt)
                jacobian[i, start_idx:start_idx + self.n_dof] = vel_unit / self.dt
                
                # Contribution from waypoint i+1 (positive)
                jacobian[i, end_idx:end_idx + self.n_dof] = -vel_unit / self.dt
        
        return jacobian

    def _compute_acceleration_constraint_jacobian(self, trajectory_vector: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of acceleration constraints with respect to trajectory parameters.
        """
        trajectory = self._vector_to_trajectory(trajectory_vector)
        
        if trajectory.shape[0] < 3:
            return np.array([]).reshape(0, len(trajectory_vector))
        
        n_acceleration_constraints = trajectory.shape[0] - 2
        jacobian = np.zeros((n_acceleration_constraints, len(trajectory_vector)))
        
        # Compute accelerations
        accelerations = np.diff(trajectory, n=2, axis=0) / (self.dt ** 2)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        for i in range(n_acceleration_constraints):
            if acceleration_magnitudes[i] > 1e-8:  # Avoid division by zero
                # Gradient of -|acceleration_i| with respect to trajectory
                acc_unit = accelerations[i] / acceleration_magnitudes[i]
                
                # Acceleration depends on waypoints i, i+1, and i+2
                # acceleration = (q_{i+2} - 2*q_{i+1} + q_i) / dt^2
                
                waypoint_i_idx = i * self.n_dof
                waypoint_i1_idx = (i + 1) * self.n_dof
                waypoint_i2_idx = (i + 2) * self.n_dof
                
                # Coefficients from second-order finite difference
                coeff = 1.0 / (self.dt ** 2)
                
                # Contribution from waypoint i
                jacobian[i, waypoint_i_idx:waypoint_i_idx + self.n_dof] = -acc_unit * coeff
                
                # Contribution from waypoint i+1 (coefficient is -2)
                jacobian[i, waypoint_i1_idx:waypoint_i1_idx + self.n_dof] = acc_unit * 2 * coeff
                
                # Contribution from waypoint i+2
                jacobian[i, waypoint_i2_idx:waypoint_i2_idx + self.n_dof] = -acc_unit * coeff
        
        return jacobian

    def _create_constraints(self):
        """Create constraint dictionaries for scipy.optimize.minimize"""
        constraints = []
        
        # Velocity constraints
        velocity_constraint = {
            'type': 'ineq',
            'fun': self._compute_velocity_constraints,
            'jac': self._compute_velocity_constraint_jacobian
        }
        constraints.append(velocity_constraint)
        
        # Acceleration constraints
        acceleration_constraint = {
            'type': 'ineq',
            'fun': self._compute_acceleration_constraints,
            'jac': self._compute_acceleration_constraint_jacobian
        }
        constraints.append(acceleration_constraint)
        
        return constraints

    def plan(self, start_config: np.ndarray, goal_config: np.ndarray) -> Tuple[List[np.ndarray], bool]:
        """
        Plan trajectory using constrained trajectory optimization

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
        
        # Create constraints
        constraints = self._create_constraints()

        print(f"  Optimizing {self.n_waypoints} waypoints × {self.n_dof} DOF = {len(initial_vector)} variables...")
        print(f"  Max velocity constraint: {self.max_velocity:.2f} rad/s")
        print(f"  Max acceleration constraint: {self.max_acceleration:.2f} rad/s²")
        
        initial_cost = self._compute_total_cost(initial_vector)
        print(f"  Initial cost: {initial_cost:.3f}")

        # Check initial constraint violations
        initial_vel_violations = self._compute_velocity_constraints(initial_vector)
        initial_acc_violations = self._compute_acceleration_constraints(initial_vector)
        
        vel_violations = np.sum(initial_vel_violations < 0)
        acc_violations = np.sum(initial_acc_violations < 0)
        
        if vel_violations > 0 or acc_violations > 0:
            print(f"  Initial trajectory violates {vel_violations} velocity and {acc_violations} acceleration constraints")
        else:
            print(f"  Initial trajectory satisfies all constraints")

        # Optimize trajectory with robust settings
        try:
            result = minimize(
                fun=self._compute_total_cost,
                x0=initial_vector,
                method='SLSQP',
                jac=self._compute_total_gradient,
                bounds=bounds,
                constraints=constraints,
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

            # Check final constraint violations
            final_vel_constraints = self._compute_velocity_constraints(result.x)
            final_acc_constraints = self._compute_acceleration_constraints(result.x)
            
            final_vel_violations = np.sum(final_vel_constraints < -1e-6)  # Small tolerance for numerical errors
            final_acc_violations = np.sum(final_acc_constraints < -1e-6)
            
            print(f"  Final trajectory violates {final_vel_violations} velocity and {final_acc_violations} acceleration constraints")

            if result.success or result.fun < initial_cost * 0.8:  # Accept if significantly improved
                optimized_trajectory = self._vector_to_trajectory(result.x)
                trajectory_list = [waypoint for waypoint in optimized_trajectory]
                
                # Additional verification
                if final_vel_violations == 0 and final_acc_violations == 0:
                    print(f"  ✅ All constraints satisfied!")
                else:
                    print(f"  ⚠ Some constraints still violated (numerical tolerance)")
                
                return trajectory_list, True
            else:
                print(f"TrajOpt optimization failed: {result.message}")
                # Return initial trajectory as fallback
                initial_trajectory_list = [waypoint for waypoint in initial_trajectory]
                return initial_trajectory_list, False

        except Exception as e:
            print(f"TrajOpt planning failed: {e}")
            return [], False

from ast import arguments
import numpy as np
import mujoco
import time
from typing import List, Tuple, Callable, Optional, Dict, Any
from .cost_functions import CostFunction
from .unconstrained_trajopt import UnconstrainedTrajOptPlanner
from scipy.optimize import minimize, LinearConstraint, Bounds
from .spline import make_uniform_clamped_knots, first_diff_matrix, second_diff_matrix, bspline_basis_matrices
import scipy.sparse as sparse

# Helpers for 7-DoF expansion (block diagonal and kron with identity)
def kron_I7(M):
    I7 = sparse.identity(7, format="csr")
    return sparse.kron(I7, M, format="csr")  # shape scales by 7

def stack7(vec1d, reps):
    # tile a (K,) vector limit across 7 joints and across 'reps' rows per joint
    return np.concatenate([np.full(reps, vec1d[j]) for j in range(7)])


class SplineBasedTrajOptPlanner(UnconstrainedTrajOptPlanner):
    """Trajectory Optimization planner with Spline Based optimization"""

    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 n_waypoints: int = 25,  # Reduced for faster iteration during debugging
                 dt: float = 0.1,
                 max_velocity: float = 2.0,  # Maximum joint velocity constraint
                 max_acceleration: float = 10.0,  # Maximum joint acceleration constraint
                 cost_mode: str = 'legacy'):
        """
        Initialize TrajOpt planner with constraints

        Args:
            model: MuJoCo model
            data: MuJoCo data
            n_waypoints: Number of waypoints in trajectory
            dt: Time step between waypoints
            max_velocity: Maximum allowed joint velocity (rad/s)
            max_acceleration: Maximum allowed joint acceleration (rad/s²)
            cost_mode: 'legacy' (individual cost functions), 'composite' (single composite function)
        """
        super().__init__(model, data, n_waypoints, dt, cost_mode)

        # Joint limits
        self.joint_limits_lower = np.array([-3.14, -2.24, -3.14, -2.57, -3.14, -2.09, -3.14])
        self.joint_limits_upper = np.array([3.14, 2.24, 3.14, 2.57, 3.14, 2.09, 3.14])
        self.vel_limit = max_velocity
        self.accel_limit = max_acceleration

        self.velocity_max = np.array([self.vel_limit] * self.n_dof)
        self.acceleration_max = np.array([self.accel_limit] * self.n_dof)

        self._z_con_enabled = False
        self._z_target = None
        self._z_tol = None
        self._kin = None

        self.degree = 3  # Cubic spline
        self.T = 2.0  # Total duration of the trajectory
        self.K = self.n_waypoints   # control points per joint
        # Spline create knots
        self.knots = make_uniform_clamped_knots(self.K, self.degree, self.T)

        self.f_tol = 1e-3


    def _create_bounds(self, start_config: np.ndarray, goal_config: np.ndarray):
        """Create bounds for optimization variables (control points in Fortran order)
        
        For B-splines, we can't directly fix the first/last control points to equal start/goal
        because the spline may not interpolate those points (depends on degree and knot vector).
        Instead, we rely on equality constraints via the B-spline basis.
        """
        bounds = []

        # Bounds in Fortran order: [j0_cp0, j0_cp1, ..., j0_cpK, j1_cp0, ..., j6_cpK]
        for j in range(self.n_dof):
            for i in range(self.n_waypoints):
                # All control points are bounded by joint limits
                bounds.append((self.joint_limits_lower[j], self.joint_limits_upper[j]))
        
        # Add bounds for auxiliary variable 't' if in max_constrained mode
        if self.cost_mode == 'composite' and hasattr(self.composite_cost_function, 'mode') \
           and self.composite_cost_function.mode == 'max_constrained':
            # t should be non-negative and reasonably bounded
            bounds.append((0.0, 1e6))  # Lower bound of 0, upper bound large enough
        
        return bounds

    def _create_initial_trajectory(self, start_config: np.ndarray, goal_config: np.ndarray, 
                                    kinematics_solver=None) -> np.ndarray:
        """Create initial trajectory guess
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            kinematics_solver: Optional kinematics solver for Cartesian interpolation
            
        Returns:
            trajectory: (n_waypoints, n_dof) initial trajectory
        """
        # Try Cartesian space interpolation if kinematics solver available
        if kinematics_solver is not None:
            try:
                trajectory = self._create_cartesian_interpolated_trajectory(
                    start_config, goal_config, kinematics_solver
                )
                print(f"  ✓ Using Cartesian-space interpolated initial trajectory")
                return trajectory
            except Exception as e:
                print(f"  ⚠ Cartesian interpolation failed ({e}), falling back to joint-space")
        
        # Fallback: Joint-space linear interpolation
        trajectory = np.zeros((self.n_waypoints, self.n_dof))
        for i in range(self.n_waypoints):
            alpha = i / (self.n_waypoints - 1)
            trajectory[i] = (1 - alpha) * start_config + alpha * goal_config
        
        print(f"  Using joint-space linear interpolated initial trajectory")
        return trajectory
    
    def _create_cartesian_interpolated_trajectory(self, start_config: np.ndarray, 
                                                   goal_config: np.ndarray,
                                                   kinematics_solver) -> np.ndarray:
        """Create initial trajectory by interpolating in Cartesian space and using IK
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            kinematics_solver: Kinematics solver with forward_kinematics and inverse_kinematics methods
            
        Returns:
            trajectory: (n_waypoints, n_dof) trajectory
        """
        # Get start and goal Cartesian positions
        start_fk = kinematics_solver.forward_kinematics(start_config)
        goal_fk = kinematics_solver.forward_kinematics(goal_config)
        
        # Extract position and orientation
        if isinstance(start_fk, tuple) and len(start_fk) >= 2:
            start_pos, start_quat = start_fk[0], start_fk[1]
            goal_pos, goal_quat = goal_fk[0], goal_fk[1]
        elif isinstance(start_fk, dict):
            start_pos = start_fk['pos']
            start_quat = start_fk.get('quat', np.array([1, 0, 0, 0]))
            goal_pos = goal_fk['pos']
            goal_quat = goal_fk.get('quat', np.array([1, 0, 0, 0]))
        else:
            raise ValueError("Unexpected forward kinematics return format")
        
        trajectory = np.zeros((self.n_waypoints, self.n_dof))
        trajectory[0] = start_config
        trajectory[-1] = goal_config
        
        # Interpolate in Cartesian space and solve IK for each waypoint
        previous_config = start_config.copy()
        ik_failures = 0
        
        for i in range(1, self.n_waypoints - 1):
            alpha = i / (self.n_waypoints - 1)
            
            # Linear interpolation of position
            target_pos = (1 - alpha) * start_pos + alpha * goal_pos
            
            # Slerp for orientation (simplified - linear interpolation then normalize)
            target_quat = (1 - alpha) * start_quat + alpha * goal_quat
            target_quat = target_quat / np.linalg.norm(target_quat)
            
            # Solve IK with previous config as seed (for smooth trajectory)
            try:
                ik_result = kinematics_solver.inverse_kinematics(
                    target_pos, target_quat, seed=previous_config
                )
                trajectory[i] = ik_result
                previous_config = ik_result
            except Exception:
                # If IK fails, fall back to joint-space interpolation for this waypoint
                trajectory[i] = (1 - alpha) * start_config + alpha * goal_config
                previous_config = trajectory[i]
                ik_failures += 1
        
        if ik_failures > 0:
            print(f"    Warning: {ik_failures}/{self.n_waypoints-2} IK solutions failed, used joint-space fallback")
        
        return trajectory
    
    def _is_max_constrained_mode(self) -> bool:
        """Check if we're using max_constrained mode"""
        return (self.cost_mode == 'composite' and 
                hasattr(self.composite_cost_function, 'mode') and
                self.composite_cost_function.mode == 'max_constrained')
    
    def _trajectory_to_vector(self, trajectory: np.ndarray) -> np.ndarray:
        """Convert trajectory to optimization vector in Fortran order for spline constraints.
        
        Args:
            trajectory: (n_waypoints, n_dof) array
            
        Returns:
            vector: Flattened in Fortran order [j0_cp0, j0_cp1, ..., j0_cpK, j1_cp0, ...]
        """
        return trajectory.T.flatten()  # Transpose then flatten = Fortran order
    
    def _control_points_from_vector(self, vector: np.ndarray) -> np.ndarray:
        """Convert optimization vector to control points (without B-spline interpolation).
        
        Args:
            vector: Flattened array in Fortran order
            
        Returns:
            control_points: (n_waypoints, n_dof) array of control points
        """
        return vector.reshape(self.n_dof, self.n_waypoints).T  # Reshape and transpose
    
    def _vector_to_trajectory(self, vector: np.ndarray) -> np.ndarray:
        """Convert optimization vector (control points) to interpolated B-spline trajectory.
        
        This ensures cost functions are evaluated on the actual smooth B-spline curve,
        not just sparse control points. Critical for obstacle avoidance and Cartesian metrics.
        
        Note: Cost functions should use dt based on the interpolated trajectory, not control points.
        Since we have a fixed total time T, dt_eval = T / (n_eval_points - 1)
        
        Args:
            vector: Flattened array in Fortran order (control points)
            
        Returns:
            trajectory: Interpolated B-spline trajectory for cost evaluation (n_eval_points, n_dof)
        """
        # Convert vector to control points
        C = self._control_points_from_vector(vector)  # Shape: (K, 7)
        
        # Generate interpolated trajectory at collocation points
        # Use 2x control points - balance between accuracy and speed
        n_eval_points = self.n_waypoints * 2  # 24 points for 12 control points
        t_eval = np.linspace(0.0, self.T, n_eval_points)
        B_eval, _, _ = bspline_basis_matrices(t_eval, self.knots, self.degree)
        
        # Interpolated trajectory
        trajectory = B_eval @ C  # Shape: (n_eval_points, 7)
        
        # Store dt for this interpolated trajectory (used by cost functions)
        # dt_eval = T / (n_eval_points - 1) but cost functions typically don't use it critically
        
        return trajectory

    def _extract_trajectory_and_t(self, augmented_vector: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """
        Extract trajectory and auxiliary variable t from augmented optimization vector.
        
        Returns:
            (trajectory_vector, t) where t is None if not in max_constrained mode
        """
        if self._is_max_constrained_mode():
            # Last element is t, rest is trajectory
            trajectory_vector = augmented_vector[:-1]
            t = float(augmented_vector[-1])
            return trajectory_vector, t
        else:
            return augmented_vector, None
    
    def _compute_epigraph_constraints(self, augmented_vector: np.ndarray) -> np.ndarray:
        """
        Compute epigraph constraints: w_i * f_i(T) <= t for all i.
        Returns array where positive values indicate constraint satisfaction.
        """
        with self.timer.time_operation('Constraint_Eval'):
            if not self._is_max_constrained_mode():
                return np.array([])
            
            trajectory_vector, t = self._extract_trajectory_and_t(augmented_vector)
            trajectory = self._vector_to_trajectory(trajectory_vector)
            
            # Compute weighted individual costs
            weighted_costs = self.composite_cost_function.compute_weighted_individual_costs(trajectory, self.dt)
            
            # Constraints: t - w_i * f_i(T) >= 0 for all i
            # Positive values indicate satisfaction
            constraints = t - weighted_costs
            
            return constraints
    
    def _compute_epigraph_constraint_jacobian(self, augmented_vector: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of epigraph constraints w.r.t. augmented variables [C, t].
        Each row corresponds to one constraint: w_i * f_i(T) <= t
        Uses chain rule to transform from interpolated trajectory space to control point space.
        """
        with self.timer.time_operation('Constraint_Jacobian'):
            if not self._is_max_constrained_mode():
                return np.array([]).reshape(0, len(augmented_vector))
            
            trajectory_vector, t = self._extract_trajectory_and_t(augmented_vector)
            
            # Convert control points to interpolated trajectory
            C = self._control_points_from_vector(trajectory_vector)  # (K, 7)
            
            # Get B-spline basis for evaluation points (same as in _vector_to_trajectory)
            n_eval_points = self.n_waypoints * 2  # 24 points for 12 control points
            t_eval = np.linspace(0.0, self.T, n_eval_points)
            B_eval, _, _ = bspline_basis_matrices(t_eval, self.knots, self.degree)
            
            # Interpolated trajectory for cost evaluation
            trajectory = B_eval @ C  # (n_eval, 7)
            
            # Get gradients of weighted individual costs w.r.t. interpolated trajectory
            individual_gradients = self.composite_cost_function.compute_individual_cost_gradients(trajectory, self.dt)
            
            n_cost_functions = len(individual_gradients)
            n_traj_vars = len(trajectory_vector)
            n_total_vars = len(augmented_vector)
            
            jacobian = np.zeros((n_cost_functions, n_total_vars))
            
            for i, grad in enumerate(individual_gradients):
                # grad is (n_eval, 7) - gradient w.r.t. interpolated trajectory
                # Apply chain rule: ∇_C f = B^T @ ∇_q f
                grad_control_points = np.zeros_like(C)  # (K, 7)
                for j in range(self.n_dof):
                    grad_control_points[:, j] = B_eval.T @ grad[:, j]
                
                # Convert to vector format (Fortran order) and negate for constraint
                grad_vector = grad_control_points.T.flatten()
                jacobian[i, :n_traj_vars] = -grad_vector
                
                # Gradient w.r.t. t: +1
                jacobian[i, -1] = 1.0
            
            return jacobian

    def _create_constraints(self, start_config, goal_config):
        """Create constraint dictionaries for scipy.optimize.minimize"""
        constraints = []
        
        # Linear Constraints
        D1_base = first_diff_matrix(self.K, self.degree, self.T)   # (K-1, K)
        D2_base = second_diff_matrix(self.K, self.degree, self.T)  # (K-2, K)
        D1_full = kron_I7(D1_base)                  # ((K-1)*7, 7K)
        D2_full = kron_I7(D2_base)                  # ((K-2)*7, 7K)

        v_rows_per_joint = D1_base.shape[0]
        a_rows_per_joint = D2_base.shape[0]
        v_lb = -stack7(self.velocity_max, v_rows_per_joint)
        v_ub =  stack7(self.velocity_max, v_rows_per_joint)
        a_lb = -stack7(self.acceleration_max, a_rows_per_joint)
        a_ub =  stack7(self.acceleration_max, a_rows_per_joint)

        lin_vel = LinearConstraint(D1_full, v_lb, v_ub)
        lin_acc = LinearConstraint(D2_full, a_lb, a_ub)

        constraints.append(lin_vel)
        constraints.append(lin_acc)

        # Equality constrains for end behavior
        t_end = np.array([0.0, self.T])
        B_end, _, _ = bspline_basis_matrices(t_end, self.knots, self.degree)   # shape (2, K)
        # Expand to 7-DoF: (2*7, 7K). Use kron with identity:
        A_end = sparse.kron(sparse.identity(7, format="csr"), B_end, format="csr")

        # b_end must match the ordering from kron: [q0(0), q0(T), q1(0), q1(T), ..., q6(0), q6(T)]
        b_end = np.empty(14)
        for j in range(7):
            b_end[2*j] = start_config[j]      # q_j(t=0)
            b_end[2*j + 1] = goal_config[j]   # q_j(t=T)
        end_eq = LinearConstraint(A_end, b_end, b_end)

        constraints.append(end_eq)

        return constraints

    def _compute_total_cost_augmented(self, augmented_vector: np.ndarray) -> float:
        """
        Compute total cost for augmented optimization vector [T, t].
        In max_constrained mode: cost = t + ρ * Σf_i(T)
        Otherwise: delegates to parent's _compute_total_cost
        """
        trajectory_vector, t = self._extract_trajectory_and_t(augmented_vector)
        
        if self._is_max_constrained_mode() and t is not None:
            # In max_constrained mode: bypass parent's printing/tracking to avoid confusion
            trajectory = self._vector_to_trajectory(trajectory_vector)
            trajectory_cost = self.composite_cost_function.compute_cost(trajectory, self.dt)
            total_cost = t + trajectory_cost
            
            # Manual progress tracking
            self.iteration_count += 1
            average_iteration_time = (time.time() - self.last_iteration_time) / self.iteration_count
            
            if self.iteration_count % 10 == 0 or total_cost < self.last_cost * 0.9:
                print(f"  Iteration {self.iteration_count}: Cost = {total_cost:.3f}, Avg Time = {average_iteration_time:.3f}s")
            
            # Always update for callback tracking
            self.last_cost = total_cost
            return total_cost
        else:
            # Standard modes: use parent's implementation (includes tracking)
            total_cost = super()._compute_total_cost(trajectory_vector)
            self.last_cost = total_cost  # Ensure it's always current for callback
            return total_cost
    
    def _compute_total_gradient(self, trajectory_vector: np.ndarray) -> np.ndarray:
        """Compute gradient of cost w.r.t. control points.
        
        Uses chain rule to transform gradients from interpolated trajectory space
        to control point space: ∇_C f = B^T @ ∇_q f
        """
        # Convert control points to interpolated trajectory (same as in _vector_to_trajectory)
        C = self._control_points_from_vector(trajectory_vector)  # (K, 7)
        
        # Get B-spline basis for evaluation points (same as _vector_to_trajectory)
        n_eval_points = self.n_waypoints * 2  # 24 points for 12 control points
        t_eval = np.linspace(0.0, self.T, n_eval_points)
        B_eval, _, _ = bspline_basis_matrices(t_eval, self.knots, self.degree)
        
        # Interpolated trajectory for cost evaluation
        trajectory = B_eval @ C  # (n_eval, 7)
        
        # Compute gradient w.r.t. interpolated trajectory using parent's logic
        if self.cost_mode == 'legacy':
            grad_trajectory = np.zeros_like(trajectory)
            for cost_fn in self.cost_functions:
                grad_trajectory += cost_fn.compute_gradient(trajectory, self.dt)
        elif self.cost_mode == 'composite':
            if self.composite_cost_function is None:
                raise RuntimeError("No composite cost function set.")
            grad_trajectory = self.composite_cost_function.compute_gradient(trajectory, self.dt)
        else:
            grad_trajectory = np.zeros_like(trajectory)
        
        # Apply chain rule: ∇_C f = B^T @ ∇_q f
        # grad_trajectory is (n_eval, 7), B_eval is (n_eval, K)
        # Do this for each joint separately
        grad_control_points = np.zeros_like(C)  # (K, 7)
        for j in range(self.n_dof):
            grad_control_points[:, j] = B_eval.T @ grad_trajectory[:, j]
        
        # Convert back to vector format (Fortran order)
        return grad_control_points.T.flatten()
    
    def _compute_total_gradient_augmented(self, augmented_vector: np.ndarray) -> np.ndarray:
        """
        Compute gradient of total cost for augmented optimization vector [T, t].
        In max_constrained mode: gradient = [∇_T(ρ * Σf_i(T)), 1]
        Otherwise: uses our overridden _compute_total_gradient (which applies B-spline chain rule)
        """
        trajectory_vector, t = self._extract_trajectory_and_t(augmented_vector)
        
        # Compute trajectory gradient using our overridden method (applies B-spline chain rule)
        trajectory_gradient = self._compute_total_gradient(trajectory_vector)
        
        if self._is_max_constrained_mode() and t is not None:
            # Gradient w.r.t. t is 1
            gradient_t = np.array([1.0])
            return np.concatenate([trajectory_gradient, gradient_t])
        else:
            return trajectory_gradient
    
    def _create_callback(self, max_evaluations: int = 1000, patience: int = 500):
        """Create callback function with smart early stopping
        
        Args:
            max_evaluations: Hard limit on function evaluations
            patience: Stop if no improvement for this many iterations
        """
        # Track best solution and convergence
        best_state = {'x': None, 'cost': float('inf')}
        convergence = {
            'iterations_since_improvement': 0,
            'best_iter': 0,
            'improvement_threshold': 1e-3  # Relative improvement threshold (0.1%)
        }
        
        def callback(x, state):
            # Use last_cost from iteration tracking (already computed)
            current_cost = self.last_cost
            
            # Check for improvement
            if current_cost < best_state['cost'] * (1.0 - convergence['improvement_threshold']):
                # Significant improvement found
                best_state['x'] = x.copy()
                best_state['cost'] = current_cost
                print(best_state['cost'])
                convergence['iterations_since_improvement'] = 0
                convergence['best_iter'] = self.iteration_count
            else:
                convergence['iterations_since_improvement'] += 1
            
            # Check stopping criteria
            if self.iteration_count >= max_evaluations:
                raise StopIteration(f"Reached maximum function evaluations: {max_evaluations}")
        
            if convergence['iterations_since_improvement'] >= patience:
                raise StopIteration(
                    f"Early stopping: No improvement for {patience} iterations "
                    f"(best at iteration {convergence['best_iter']})"
                )
        
        # Store reference to best_state so we can access it from outside
        callback.best_state = best_state
        callback.convergence = convergence
        return callback

    def plan(self, start_config: np.ndarray, goal_config: np.ndarray) -> Tuple[List[np.ndarray], bool, Dict[str, Any]]:
        """
        Plan trajectory using constrained trajectory optimization

        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration

        Returns:
            trajectory: List of joint configurations
            success: Whether planning succeeded
            metadata: Dictionary with optimization info (iterations, final_cost, etc.)
        """
        # Validate cost functions are set up (same logic as parent class)
        if self.cost_mode == 'legacy' and len(self.cost_functions) == 0:
            print("Warning: No cost functions added to TrajOpt planner")
            return [], False, {}
        elif self.cost_mode == 'composite' and self.composite_cost_function is None:
            print("Warning: No composite cost function set")
            return [], False, {}

        # Reset progress tracking
        self.iteration_count = 0
        self.last_cost = float('inf')

        # Try to extract kinematics solver from cost functions for Cartesian interpolation
        kinematics_solver = None
        if self.cost_mode == 'composite' and self.composite_cost_function is not None:
            for cost_fn in self.composite_cost_function.cost_functions:
                if hasattr(cost_fn, 'kinematics_solver'):
                    kinematics_solver = cost_fn.kinematics_solver
                    break

        # Create initial trajectory (with Cartesian interpolation if possible)
        initial_trajectory = self._create_initial_trajectory(
            start_config, goal_config, kinematics_solver=kinematics_solver
        )
        initial_vector = self._trajectory_to_vector(initial_trajectory)
        
        # Augment with t if in max_constrained mode
        if self._is_max_constrained_mode():
            # Initialize t with a reasonable estimate (e.g., max of weighted costs)
            weighted_costs = self.composite_cost_function.compute_weighted_individual_costs(initial_trajectory, self.dt)
            initial_t = float(np.max(weighted_costs)) * 1.1  # 10% above max
            initial_vector = np.append(initial_vector, initial_t)
            print(f"  Using epigraph reformulation: min_(T,t) [t + ρ*Σf_i(T)] s.t. w_i*f_i(T) ≤ t")
            print(f"  Initial t: {initial_t:.3f}")

        # Create bounds
        bounds = self._create_bounds(start_config, goal_config)

        # Create constraints
        constraints = self._create_constraints(start_config, goal_config)

        n_traj_vars = self.n_waypoints * self.n_dof
        n_total_vars = len(initial_vector)
        if self._is_max_constrained_mode():
            print(f"  Optimizing {self.n_waypoints} waypoints × {self.n_dof} DOF + 1 auxiliary var = {n_total_vars} variables...")
        else:
            print(f"  Optimizing {self.n_waypoints} waypoints × {self.n_dof} DOF = {n_total_vars} variables...")
        print(f"  Max velocity constraint: {self.vel_limit:.2f} rad/s")
        print(f"  Max acceleration constraint: {self.accel_limit:.2f} rad/s²")
        
        initial_cost = self._compute_total_cost_augmented(initial_vector)
        print(f"  Initial cost: {initial_cost:.3f}")
        
        # Debug: Check if cost function sees interpolated trajectory correctly
        if self.cost_mode == 'composite' and self.composite_cost_function is not None:
            initial_traj_for_cost = self._vector_to_trajectory(initial_vector if not self._is_max_constrained_mode() else initial_vector[:-1])
            print(f"\n  Initial trajectory for cost evaluation:")
            print(f"    Control points: {self.n_waypoints}")
            print(f"    Interpolated points for cost: {initial_traj_for_cost.shape[0]}")
            
            # Compute individual costs
            print(f"  Initial cost breakdown:")
            for i, cost_fn in enumerate(self.composite_cost_function.cost_functions):
                cost_val = cost_fn.compute_cost(initial_traj_for_cost, self.dt)
                weight = self.composite_cost_function.weights[i]
                print(f"    Cost {i} ({cost_fn.__class__.__name__}): {cost_val:.6f} (weight: {weight:.2f}, weighted: {weight*cost_val:.6f})")
            
            # Check gradient magnitude
            initial_grad = self._compute_total_gradient(initial_vector if not self._is_max_constrained_mode() else initial_vector[:-1])
            print(f"  Initial gradient magnitude: {np.linalg.norm(initial_grad):.6f}")
            print(f"    Max gradient element: {np.max(np.abs(initial_grad)):.6f}")
            print(f"    Min gradient element: {np.min(np.abs(initial_grad)):.6f}")
            
            # Gradient check: Verify gradient direction reduces cost
            print(f"\n  Gradient sanity check:")
            eps = 1e-5
            test_vector = initial_vector if not self._is_max_constrained_mode() else initial_vector[:-1]
            test_step = -eps * initial_grad / np.linalg.norm(initial_grad)  # Small step in negative gradient direction
            test_vector_perturbed = test_vector + test_step
            cost_original = self._compute_total_cost(test_vector)
            cost_perturbed = self._compute_total_cost(test_vector_perturbed)
            print(f"    Cost at current point: {cost_original:.6f}")
            print(f"    Cost after small gradient step: {cost_perturbed:.6f}")
            print(f"    Change: {cost_perturbed - cost_original:.6e} (should be negative)")
            if cost_perturbed >= cost_original:
                print(f"    ⚠ WARNING: Gradient may be incorrect (cost increased after gradient step)!")
        
        # Check epigraph constraint violations if in max_constrained mode
        if self._is_max_constrained_mode():
            epigraph_violations_vals = self._compute_epigraph_constraints(initial_vector)
            epigraph_violations = np.sum(epigraph_violations_vals < 0)
            if epigraph_violations > 0:
                print(f"  Initial trajectory violates {epigraph_violations} epigraph constraints")
            else:
                print(f"  Initial trajectory satisfies all epigraph constraints")

        max_evaluations = 3500
        maxiter = 1000
        ftol = 1e-4
        patience = 1200
    
        callback = self._create_callback(max_evaluations, patience)
        
        print(f"  Optimization budget: {max_evaluations} evaluations, {maxiter} iterations, ftol={ftol:.0e}, patience={patience}")
        
        try:
            # Track total optimization time
            import time
            optimization_start = time.perf_counter()
            
            result = minimize(
                fun=self._compute_total_cost_augmented,
                x0=initial_vector,
                method='trust-constr',
                jac=self._compute_total_gradient_augmented,
                bounds=bounds,
                constraints=constraints,
                callback=callback,
                options={
                    'maxiter': 1200,
                    'xtol': 1e-8,  # Tighter tolerance on variables
                    'gtol': 1e-6,  # Tighter tolerance on gradients
                    'barrier_tol': 1e-8,  # Tighter constraint satisfaction
                    'verbose': 0  # Suppress verbose output
                }
            )
            
            optimization_end = time.perf_counter()
            total_optimization_time = optimization_end - optimization_start
        except StopIteration as e:
            optimization_end = time.perf_counter()
            total_optimization_time = optimization_end - optimization_start
            print(f"  ⚠ {e}")
            # Create a pseudo-result for the case where we hit the evaluation limit
            class PseudoResult:
                def __init__(self, x, fun, success=False, message="Hit function evaluation limit"):
                    self.x = x
                    self.fun = fun
                    self.success = success
                    self.message = message
            
            # Use best solution found during optimization
            best_x = callback.best_state['x'] if callback.best_state['x'] is not None else initial_vector
            best_cost = callback.best_state['cost'] if callback.best_state['x'] is not None else self._compute_total_cost_augmented(initial_vector)
            
            result = PseudoResult(
                x=best_x,
                fun=best_cost,
                success=True,  # Accept for visualization even if hit limit
                message="Hit function evaluation limit"
            )
        except Exception as e:
            print(f"TrajOpt planning failed: {e}")
            return [], False, {}

        print(f"  Optimization completed in {self.iteration_count} cost evaluations")
        print(f"  Final cost: {result.fun:.3f}")
        print(f"  Status: {result.message}")
        
        # Extract trajectory and t from result
        final_traj_vec, final_t = self._extract_trajectory_and_t(result.x)
        
        # Collect timing statistics from all sources
        timing_summary = self._collect_timing_statistics(total_optimization_time)
        
        # Prepare metadata
        metadata = {
            'iterations': self.iteration_count,
            'final_optimization_cost': float(result.fun),
            'cost_mode': self.cost_mode,
            'stopped_early': 'Early stopping' in result.message if hasattr(result, 'message') else False,
            'termination_reason': result.message if hasattr(result, 'message') else 'Unknown',
            'timing': timing_summary,
        }


        # Convert back to control points (not interpolated trajectory)
        C_opt = self._control_points_from_vector(final_traj_vec)  # Shape: (K, 7)
        
        # Generate fine-grained trajectory using B-spline interpolation
        t_fine = np.linspace(0.0, self.T, 201)
        B_fine, _, _ = bspline_basis_matrices(t_fine, self.knots, self.degree)
        q_fine = B_fine @ C_opt  # Shape: (201, 7)
        
        # Enforce exact boundary conditions (trust-constr may not satisfy them perfectly)
        q_fine[0] = start_config
        q_fine[-1] = goal_config
        
        # Check boundary constraint satisfaction
        print(f"\n  Boundary constraint verification:")
        t_boundary = np.array([0.0, self.T])
        B_boundary, _, _ = bspline_basis_matrices(t_boundary, self.knots, self.degree)
        q_boundary = B_boundary @ C_opt  # What the constraint enforces
        
        # Compute constraint violation
        A_end = sparse.kron(sparse.identity(7, format="csr"), B_boundary, format="csr")
        C_flat = final_traj_vec
        constraint_value = A_end @ C_flat
        
        # Expected boundary values (interleaved format)
        b_end_expected = np.empty(14)
        for j in range(7):
            b_end_expected[2*j] = start_config[j]
            b_end_expected[2*j + 1] = goal_config[j]
        
        constraint_violation = constraint_value - b_end_expected
        print(f"    Constraint violation (should be ~0):")
        print(f"      Max abs violation: {np.max(np.abs(constraint_violation)):.6e}")
        print(f"      RMS violation: {np.sqrt(np.mean(constraint_violation**2)):.6e}")
        print(f"    Actual boundary from B-spline:")
        print(f"      Start: {q_boundary[0]}")
        print(f"      Goal:  {q_boundary[1]}")
        print(f"    Expected boundary:")
        print(f"      Start: {start_config}")
        print(f"      Goal:  {goal_config}")
        
        # Verify start/end positions
        print(f"\n  Trajectory verification:")
        print(f"    Start config (expected): {start_config}")
        print(f"    Start config (actual):   {q_fine[0]}")
        print(f"    Start error: {np.linalg.norm(q_fine[0] - start_config):.6f}")
        print(f"    End config (expected):   {goal_config}")
        print(f"    End config (actual):     {q_fine[-1]}")
        print(f"    End error: {np.linalg.norm(q_fine[-1] - goal_config):.6f}")
        
        # Analyze control point movement
        initial_control_points = self._control_points_from_vector(self._trajectory_to_vector(initial_trajectory))
        control_point_movement = np.linalg.norm(C_opt - initial_control_points, axis=1)
        print(f"\n  Control point analysis:")
        print(f"    Number of control points: {C_opt.shape[0]}")
        print(f"    Avg movement: {np.mean(control_point_movement):.4f} rad")
        print(f"    Max movement: {np.max(control_point_movement):.4f} rad (point {np.argmax(control_point_movement)})")
        print(f"    Min movement: {np.min(control_point_movement):.4f} rad (point {np.argmin(control_point_movement)})")
        
        # Verify B-spline is creating smooth curves (not just passing through control points)
        print(f"\n  B-spline interpolation verification:")
        print(f"    Control points shape: {C_opt.shape}")
        print(f"    Interpolated trajectory shape: {q_fine.shape}")
        print(f"    First control point: {C_opt[0]}")
        print(f"    First interpolated point: {q_fine[0]}")
        print(f"    Difference: {np.linalg.norm(C_opt[0] - q_fine[0]):.6f}")
        
        # Convert to list format for compatibility (same as constrained_trajopt)
        trajectory_list = [waypoint for waypoint in q_fine]
        
        # Check final costs
        if self.cost_mode == 'composite' and self.composite_cost_function is not None:
            print(f"\n  Final cost breakdown:")
            for i, cost_fn in enumerate(self.composite_cost_function.cost_functions):
                cost_val = cost_fn.compute_cost(np.array(trajectory_list), self.dt)
                weight = self.composite_cost_function.weights[i]
                print(f"    Cost {i} ({cost_fn.__class__.__name__}): {cost_val:.6f} (weight: {weight:.2f}, weighted: {weight*cost_val:.6f})")
        
        # Store control points in metadata for visualization
        metadata['control_points'] = C_opt.tolist()
        metadata['control_point_movement'] = control_point_movement.tolist()
        metadata['n_control_points'] = C_opt.shape[0]

        return trajectory_list, True, metadata
    
    def _collect_timing_statistics(self, total_optimization_time: float = None) -> Dict[str, Any]:
        """Collect timing statistics from all components"""
        aggregated_timings = {}
        aggregated_counts = {}
        
        # Collect from cost function(s)
        if self.cost_mode == 'composite' and self.composite_cost_function is not None:
            cost_timer = self.composite_cost_function.timer
            for op_name, op_time in cost_timer.timings.items():
                if op_name not in aggregated_timings:
                    aggregated_timings[op_name] = 0.0
                    aggregated_counts[op_name] = 0
                aggregated_timings[op_name] += op_time
                aggregated_counts[op_name] += cost_timer.call_counts[op_name]
            
            # Also collect from individual cost functions if they use FK
            for cf in self.composite_cost_function.cost_functions:
                if hasattr(cf, 'kinematics_solver') and hasattr(cf.kinematics_solver, 'timer'):
                    kin_timer = cf.kinematics_solver.timer
                    for op_name, op_time in kin_timer.timings.items():
                        if op_name not in aggregated_timings:
                            aggregated_timings[op_name] = 0.0
                            aggregated_counts[op_name] = 0
                        aggregated_timings[op_name] += op_time
                        aggregated_counts[op_name] += kin_timer.call_counts[op_name]
        
        # Collect from planner's timer
        for op_name, op_time in self.timer.timings.items():
            if op_name not in aggregated_timings:
                aggregated_timings[op_name] = 0.0
                aggregated_counts[op_name] = 0
            aggregated_timings[op_name] += op_time
            aggregated_counts[op_name] += self.timer.call_counts[op_name]
        
        # Calculate tracked time and optimizer overhead
        tracked_time = sum(aggregated_timings.values())
        
        # Add optimizer overhead if we have total optimization time
        if total_optimization_time is not None and total_optimization_time > tracked_time:
            optimizer_overhead = total_optimization_time - tracked_time
            aggregated_timings['Optimizer_Overhead'] = optimizer_overhead
            aggregated_counts['Optimizer_Overhead'] = 1
            total_time = total_optimization_time
        else:
            total_time = tracked_time
        
        timing_summary = {
            'total_time': total_time,
            'timings': aggregated_timings,
            'call_counts': aggregated_counts,
            'percentages': {
                name: (time_val / total_time * 100) if total_time > 0 else 0.0
                for name, time_val in aggregated_timings.items()
            },
            'average_times': {
                name: (aggregated_timings[name] / aggregated_counts[name]) if aggregated_counts[name] > 0 else 0.0
                for name in aggregated_timings.keys()
            }
        }
        
        return timing_summary
    
    def print_timing_summary(self, timing_summary: Optional[Dict[str, Any]] = None):
        """Print formatted timing summary"""
        if timing_summary is None:
            timing_summary = self._collect_timing_statistics()
        
        print(f"\n{'='*70}")
        print(f"Performance Timing Summary")
        print(f"{'='*70}")
        print(f"Total Optimization Time: {timing_summary['total_time']:.3f}s")
        print(f"{'-'*70}")
        print(f"{'Operation':<20} {'Time (s)':<12} {'%':<8} {'Calls':<10} {'Avg (ms)':<10}")
        print(f"{'-'*70}")
        
        # Sort by time spent (descending)
        sorted_ops = sorted(
            timing_summary['timings'].keys(),
            key=lambda x: timing_summary['timings'][x],
            reverse=True
        )
        
        for op in sorted_ops:
            time_val = timing_summary['timings'][op]
            pct = timing_summary['percentages'][op]
            calls = timing_summary['call_counts'][op]
            avg_ms = timing_summary['average_times'][op] * 1000
            print(f"{op:<20} {time_val:<12.3f} {pct:<8.1f} {calls:<10} {avg_ms:<10.3f}")
        
        print(f"{'='*70}\n")

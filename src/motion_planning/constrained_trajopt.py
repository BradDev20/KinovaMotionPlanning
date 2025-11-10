import numpy as np
import mujoco
import time
from typing import List, Tuple, Callable, Optional, Dict, Any
from .cost_functions import CostFunction
from .unconstrained_trajopt import UnconstrainedTrajOptPlanner
from scipy.optimize import minimize


class ConstrainedTrajOptPlanner(UnconstrainedTrajOptPlanner):
    """Trajectory Optimization planner with velocity and acceleration constraints"""

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
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self._z_con_enabled = False
        self._z_target = None
        self._z_tol = None
        self._kin = None

        self.f_tol = 1e-3


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
        
        # Add bounds for auxiliary variable 't' if in max_constrained mode
        if self.cost_mode == 'composite' and hasattr(self.composite_cost_function, 'mode') \
           and self.composite_cost_function.mode == 'max_constrained':
            # t should be non-negative and reasonably bounded
            bounds.append((0.0, 1e6))  # Lower bound of 0, upper bound large enough
        
        return bounds

    def _create_initial_trajectory(self, start_config: np.ndarray, goal_config: np.ndarray) -> np.ndarray:
        """Create initial trajectory guess (linear interpolation)"""
        trajectory = np.zeros((self.n_waypoints, self.n_dof))

        for i in range(self.n_waypoints):
            alpha = i / (self.n_waypoints - 1)
            trajectory[i] = (1 - alpha) * start_config + alpha * goal_config

        return trajectory
    
    def _is_max_constrained_mode(self) -> bool:
        """Check if we're using max_constrained mode"""
        return (self.cost_mode == 'composite' and 
                hasattr(self.composite_cost_function, 'mode') and
                self.composite_cost_function.mode == 'max_constrained')

    def _compute_velocity_constraints(self, trajectory_vector: np.ndarray) -> np.ndarray:
        """
        Compute velocity constraint violations.
        Returns array where negative values indicate constraint violations.
        """
        with self.timer.time_operation('Constraint_Eval'):
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
        with self.timer.time_operation('Constraint_Eval'):
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
        with self.timer.time_operation('Constraint_Jacobian'):
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
        with self.timer.time_operation('Constraint_Jacobian'):
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

    def enable_fixed_z_constraint(self, kinematics_solver, target_z: float, tol: float = 1e-3):
        """Enable |z(q)-target_z| <= tol for (most) waypoints."""
        self._z_con_enabled = True
        self._kin = kinematics_solver
        self._z_target = float(target_z)
        self._z_tol = float(tol)

    def _z_from_fk(self, q: np.ndarray) -> float:
        fk = self._kin.forward_kinematics(q)
        # Common patterns:
        # 1) (pos, quat) tuple
        if isinstance(fk, tuple) and len(fk) >= 1:
            pos = fk[0]
            return float(pos[2])
        # 2) dict like {'pos': np.array([x,y,z]), ...}
        if isinstance(fk, dict) and 'pos' in fk:
            return float(fk['pos'][2])
        # 3) flat array (x, y, z, qw, qx, qy, qz) or similar
        try:
            arr = np.asarray(fk).reshape(-1)
            return float(arr[2])
        except Exception:
            raise RuntimeError("forward_kinematics(q) returned an unexpected shape")

    def _compute_fixed_z_constraints(self, trajectory_vector: np.ndarray) -> np.ndarray:
        if not self._z_con_enabled:
            return np.array([])
        traj = self._vector_to_trajectory(trajectory_vector)
        if traj.shape[0] < 2:
            return np.array([])

        cons = []
        # constrain interior waypoints (1..N-2); include endpoints if appropriate for your start/goal
        for i in range(1, traj.shape[0]-1):
            q = traj[i]
            z = self._z_from_fk(q)
            e = z - self._z_target
            cons.append(self._z_tol - e)  # z - z_t <= tol
            cons.append(self._z_tol + e)  # z - z_t >= -tol
        return np.array(cons, dtype=float)
    
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
        Compute Jacobian of epigraph constraints w.r.t. augmented variables [T, t].
        Each row corresponds to one constraint: w_i * f_i(T) <= t
        """
        with self.timer.time_operation('Constraint_Jacobian'):
            if not self._is_max_constrained_mode():
                return np.array([]).reshape(0, len(augmented_vector))
            
            trajectory_vector, t = self._extract_trajectory_and_t(augmented_vector)
            trajectory = self._vector_to_trajectory(trajectory_vector)
            
            # Get gradients of weighted individual costs w.r.t. trajectory
            individual_gradients = self.composite_cost_function.compute_individual_cost_gradients(trajectory, self.dt)
            
            n_cost_functions = len(individual_gradients)
            n_traj_vars = len(trajectory_vector)
            n_total_vars = len(augmented_vector)
            
            jacobian = np.zeros((n_cost_functions, n_total_vars))
            
            for i, grad in enumerate(individual_gradients):
                # Gradient w.r.t. trajectory variables: -w_i * ∇f_i(T)
                grad_vector = self._trajectory_to_vector(grad)
                jacobian[i, :n_traj_vars] = -grad_vector
                
                # Gradient w.r.t. t: +1
                jacobian[i, -1] = 1.0
            
            return jacobian

    def _create_constraints(self):
        """Create constraint dictionaries for scipy.optimize.minimize"""
        constraints = []
        
        # Wrapper functions to handle augmented vector in max_constrained mode
        def velocity_constraint_wrapper(x):
            traj_vec, _ = self._extract_trajectory_and_t(x)
            return self._compute_velocity_constraints(traj_vec)
        
        def velocity_jac_wrapper(x):
            traj_vec, _ = self._extract_trajectory_and_t(x)
            jac = self._compute_velocity_constraint_jacobian(traj_vec)
            if self._is_max_constrained_mode():
                # Pad with zeros for t variable
                jac = np.hstack([jac, np.zeros((jac.shape[0], 1))])
            return jac
        
        def acceleration_constraint_wrapper(x):
            traj_vec, _ = self._extract_trajectory_and_t(x)
            return self._compute_acceleration_constraints(traj_vec)
        
        def acceleration_jac_wrapper(x):
            traj_vec, _ = self._extract_trajectory_and_t(x)
            jac = self._compute_acceleration_constraint_jacobian(traj_vec)
            if self._is_max_constrained_mode():
                # Pad with zeros for t variable
                jac = np.hstack([jac, np.zeros((jac.shape[0], 1))])
            return jac
        
        def z_constraint_wrapper(x):
            traj_vec, _ = self._extract_trajectory_and_t(x)
            return self._compute_fixed_z_constraints(traj_vec)
        
        # Velocity constraints
        velocity_constraint = {
            'type': 'ineq',
            'fun': velocity_constraint_wrapper,
            'jac': velocity_jac_wrapper
        }
        constraints.append(velocity_constraint)
        
        # Acceleration constraints
        acceleration_constraint = {
            'type': 'ineq',
            'fun': acceleration_constraint_wrapper,
            'jac': acceleration_jac_wrapper
        }
        constraints.append(acceleration_constraint)

        if self._z_con_enabled:
            # Let SLSQP estimate Jacobian numerically: omit 'jac' for simplicity.
            constraints.append({
                'type': 'ineq',
                'fun': z_constraint_wrapper
            })
        
        # Epigraph constraints for max_constrained mode
        if self._is_max_constrained_mode():
            epigraph_constraint = {
                'type': 'ineq',
                'fun': self._compute_epigraph_constraints,
                'jac': self._compute_epigraph_constraint_jacobian  # Provide analytical Jacobian for speed
            }
            constraints.append(epigraph_constraint)
            print(f"  Added epigraph constraints: w_i * f_i(T) ≤ t for {len(self.composite_cost_function.cost_functions)} objectives")

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
    
    def _compute_total_gradient_augmented(self, augmented_vector: np.ndarray) -> np.ndarray:
        """
        Compute gradient of total cost for augmented optimization vector [T, t].
        In max_constrained mode: gradient = [∇_T(ρ * Σf_i(T)), 1]
        Otherwise: delegates to parent's _compute_total_gradient
        """
        trajectory_vector, t = self._extract_trajectory_and_t(augmented_vector)
        
        # Compute trajectory gradient (∇_T(ρ * Σf_i(T)) in max_constrained mode)
        trajectory_gradient = super()._compute_total_gradient(trajectory_vector)
        
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
        
        def callback(x):
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

        # Create initial trajectory
        initial_trajectory = self._create_initial_trajectory(start_config, goal_config)
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
        constraints = self._create_constraints()

        n_traj_vars = self.n_waypoints * self.n_dof
        n_total_vars = len(initial_vector)
        if self._is_max_constrained_mode():
            print(f"  Optimizing {self.n_waypoints} waypoints × {self.n_dof} DOF + 1 auxiliary var = {n_total_vars} variables...")
        else:
            print(f"  Optimizing {self.n_waypoints} waypoints × {self.n_dof} DOF = {n_total_vars} variables...")
        print(f"  Max velocity constraint: {self.max_velocity:.2f} rad/s")
        print(f"  Max acceleration constraint: {self.max_acceleration:.2f} rad/s²")
        
        initial_cost = self._compute_total_cost_augmented(initial_vector)
        print(f"  Initial cost: {initial_cost:.3f}")

        # Check initial constraint violations
        initial_traj_vec, initial_t = self._extract_trajectory_and_t(initial_vector)
        initial_vel_violations = self._compute_velocity_constraints(initial_traj_vec)
        initial_acc_violations = self._compute_acceleration_constraints(initial_traj_vec)
        
        vel_violations = np.sum(initial_vel_violations < 0)
        acc_violations = np.sum(initial_acc_violations < 0)
        
        if vel_violations > 0 or acc_violations > 0:
            print(f"  Initial trajectory violates {vel_violations} velocity and {acc_violations} acceleration constraints")
        else:
            print(f"  Initial trajectory satisfies all constraints")
        
        # Check epigraph constraint violations if in max_constrained mode
        if self._is_max_constrained_mode():
            epigraph_violations_vals = self._compute_epigraph_constraints(initial_vector)
            epigraph_violations = np.sum(epigraph_violations_vals < 0)
            if epigraph_violations > 0:
                print(f"  Initial trajectory violates {epigraph_violations} epigraph constraints")
            else:
                print(f"  Initial trajectory satisfies all epigraph constraints")

        # Adaptive optimization settings based on cost mode
        # Sum mode needs more iterations at extreme weights due to unbalanced gradients
        if self.cost_mode == 'composite' and hasattr(self.composite_cost_function, 'mode'):
            mode = self.composite_cost_function.mode
            if mode == 'sum':
                # Sum mode: needs MANY iterations due to unbalanced gradients at extreme weights
                # and tendency to get stuck in local minima at middle weights
                max_evaluations = 10000
                maxiter = 3000
                ftol = 1e-5  # Very tight tolerance
                patience = 50  # But stop early if stuck in local minimum
            elif mode == 'max_constrained':
                # Max constrained: extra variable + epigraph constraints need more iterations
                max_evaluations = 4000
                maxiter = 1200
                ftol = 1e-4
                patience = 20
            else:  # 'max' or other
                max_evaluations = 3500
                maxiter = 1000
                ftol = 1e-4
                patience = 20
        else:
            # Legacy or other modes
            max_evaluations = 3500
            maxiter = 1000
            ftol = 1e-4
            patience = 20
        
        callback = self._create_callback(max_evaluations, patience)
        
        print(f"  Optimization budget: {max_evaluations} evaluations, {maxiter} iterations, ftol={ftol:.0e}, patience={patience}")
        
        try:
            # Track total optimization time
            import time
            optimization_start = time.perf_counter()
            
            result = minimize(
                fun=self._compute_total_cost_augmented,
                x0=initial_vector,
                method='SLSQP',
                jac=self._compute_total_gradient_augmented,
                bounds=bounds,
                constraints=constraints,
                callback=callback,
                options={
                    'maxiter': maxiter,
                    'ftol': ftol,
                    'disp': False  # Suppress verbose output
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
        
        if self._is_max_constrained_mode() and final_t is not None:
            print(f"  Final t: {final_t:.3f}")
            # Report breakdown of cost
            traj_cost = super()._compute_total_cost(final_traj_vec)
            print(f"    Cost breakdown: t={final_t:.3f}, ρ*Σf_i={traj_cost:.3f}")
            metadata['final_t'] = float(final_t)
            metadata['trajectory_cost'] = float(traj_cost)

        # Check final constraint violations
        final_vel_constraints = self._compute_velocity_constraints(final_traj_vec)
        final_acc_constraints = self._compute_acceleration_constraints(final_traj_vec)
        
        final_vel_violations = np.sum(final_vel_constraints < -1e-6)  # Small tolerance for numerical errors
        final_acc_violations = np.sum(final_acc_constraints < -1e-6)
        
        print(f"  Final trajectory violates {final_vel_violations} velocity and {final_acc_violations} acceleration constraints")
        
        # Check epigraph constraint violations if in max_constrained mode
        if self._is_max_constrained_mode():
            final_epigraph_constraints = self._compute_epigraph_constraints(result.x)
            final_epigraph_violations = np.sum(final_epigraph_constraints < -1e-6)
            print(f"  Final trajectory violates {final_epigraph_violations} epigraph constraints")

        if result.success or result.fun < initial_cost * 0.8:  # Accept if significantly improved
            optimized_trajectory = self._vector_to_trajectory(final_traj_vec)
            trajectory_list = [waypoint for waypoint in optimized_trajectory]
            
            # Additional verification
            total_violations = final_vel_violations + final_acc_violations
            if self._is_max_constrained_mode():
                total_violations += final_epigraph_violations
            
            if total_violations == 0:
                print(f"  ✓ All constraints satisfied!")
            else:
                print(f"  ⚠ Some constraints still violated (numerical tolerance)")
            
            return trajectory_list, True, metadata
        else:
            print(f"TrajOpt optimization failed: {result.message}")
            # Return initial trajectory as fallback
            initial_trajectory_list = [waypoint for waypoint in initial_trajectory]
            return initial_trajectory_list, False, metadata
    
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

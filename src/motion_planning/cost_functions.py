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

    def __init__(self, weight: float = 1.0, normalization_bounds: Tuple[float, float] = (0.0, 1.0)):
        # assert 1 >= weight >= 0, "Weight must be between 0 and 1"
        super().__init__(weight)
        self.normalization_bounds = normalization_bounds

    def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """Compute total trajectory length cost"""
        if trajectory.shape[0] < 2:
            return 0.0

        # Compute distances between consecutive waypoints
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        total_length = np.sum(distances)

        # Normalize cost to be between 0 and 1
        normalized_cost = (total_length - self.normalization_bounds[0]) / (self.normalization_bounds[1] - self.normalization_bounds[0])

        return float(self.weight * normalized_cost)

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

        # Apply normalization to gradient (consistent with cost normalization)
        normalization_factor = 1.0 / (self.normalization_bounds[1] - self.normalization_bounds[0])
        return gradient * self.weight * normalization_factor

class ObstacleAvoidanceCostFunction(CostFunction):
    """Cost for avoiding spherical obstacles in Cartesian space, with configurable aggregation."""

    def __init__(self,
                 kinematics_solver: KinematicsSolver,
                 obstacles: List[Obstacle],
                 weight: float = 1.0,
                 normalization_bounds: Tuple[float, float] = (0.0, 1.0),
                 decay_rate: float = 5.0,
                 aggregate: str = "min"):  # "min" | "sum" | "avg"
        """
        Args:
            kinematics_solver: FK provider
            obstacles: list of obstacles
            weight: scalar applied after normalization
            normalization_bounds: (low, high) for post-aggregation normalization
            decay_rate: alpha in exp(-alpha * distance_to_surface)
            aggregate: how to combine waypoint penalties: "min", "sum", or "avg"
        """
        super().__init__(weight)
        assert aggregate in ("min", "sum", "avg"), "aggregate must be 'min' | 'sum' | 'avg'"
        self.kinematics_solver = kinematics_solver
        self.obstacles = obstacles if obstacles else []
        self.normalization_bounds = normalization_bounds
        self.decay_rate = decay_rate
        self.aggregate = aggregate

    # ---------- internals ----------
    def _waypoint_cost_from_surface_dist(self, d_surface: float) -> float:
        """Penalty for a single waypoint given distance to obstacle surface."""
        if d_surface <= 0.0:
            # inside obstacle: very large but smooth penalty
            return 1000.0 * np.exp(-self.decay_rate * d_surface)
        else:
            # outside: decays with clearance (larger clearance -> smaller cost)
            return float(np.exp(-self.decay_rate * d_surface))

    def _closest_surface_distance(self, ee_pos: np.ndarray) -> float:
        """Return min distance-to-surface over all obstacles for a given EE position."""
        # distance to surface = ||ee - center|| - radius
        return min(float(np.linalg.norm(ee_pos - obs.center) - obs.radius) for obs in self.obstacles)

    # ---------- cost ----------
    def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        if not self.obstacles or trajectory.size == 0:
            return 0.0

        self.kinematics_solver._backup_state()
        try:
            wp_costs = []
            for q in trajectory:
                ee, _ = self.kinematics_solver.forward_kinematics(q)
                d_surface = self._closest_surface_distance(ee)
                wp_costs.append(self._waypoint_cost_from_surface_dist(d_surface))
        finally:
            self.kinematics_solver._restore_state()

        if self.aggregate == "min":
            agg_cost = float(min(wp_costs))
        elif self.aggregate == "sum":
            agg_cost = float(np.sum(wp_costs) * dt)  # time-weighted integral
        else:  # "avg"
            agg_cost = float(np.mean(wp_costs))

        # normalize and apply this CF's weight
        lo, hi = self.normalization_bounds
        norm = (agg_cost - lo) / (hi - lo)
        return float(self.weight * norm)

    # ---------- gradient ----------
    def compute_gradient(self, trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
        if not self.obstacles or trajectory.size == 0:
            return np.zeros_like(trajectory)

        n_wp, n_j = trajectory.shape
        grad = np.zeros_like(trajectory)
        eps = 1e-4

        # First pass: cache FK and closest obstacle data
        infos = []
        self.kinematics_solver._backup_state()
        try:
            for q in trajectory:
                ee, _ = self.kinematics_solver.forward_kinematics(q)
                # find closest obstacle and center distance
                dcs = [(np.linalg.norm(ee - obs.center), obs) for obs in self.obstacles]
                dc, obs = min(dcs, key=lambda t: t[0])  # distance to center, closest obstacle
                d_surface = dc - obs.radius
                infos.append((ee, obs, dc, d_surface))
        finally:
            self.kinematics_solver._restore_state()

        # Which waypoints contribute?
        if self.aggregate == "min":
            # Only the bottleneck waypoint(s)
            waypoint_penalties = [self._waypoint_cost_from_surface_dist(d) for *_, d in infos]
            min_val = min(waypoint_penalties)
            idxs = [i for i, v in enumerate(waypoint_penalties) if abs(v - min_val) <= 1e-12]
            wp_weight = 1.0  # no dt/avg factor for min
        elif self.aggregate == "sum":
            idxs = list(range(n_wp))
            wp_weight = dt
        else:  # "avg"
            idxs = list(range(n_wp))
            wp_weight = 1.0 / n_wp

        # Accumulate gradient
        for i in idxs:
            ee, obs, dc, d_surface = infos[i]
            if dc <= 1e-10:
                continue  # degenerate; skip

            # d(cost_wp)/d(d_surface)
            if d_surface <= 0.0:
                dcost_dd = -1000.0 * self.decay_rate * np.exp(-self.decay_rate * d_surface)
            else:
                dcost_dd = -self.decay_rate * np.exp(-self.decay_rate * d_surface)

            # d(d_surface)/d(ee) = (ee - center)/||ee - center|| = unit vector away from center
            unit = (ee - obs.center) / dc
            dcost_dee = dcost_dd * unit  # shape (3,)

            # Jacobian d(ee)/d(q) via finite-diff (keep your analytic FK if you have it)
            q = trajectory[i]
            J = np.zeros((3, n_j))
            self.kinematics_solver._backup_state()
            try:
                for j in range(n_j):
                    q_eps = q.copy(); q_eps[j] += eps
                    ee_eps, _ = self.kinematics_solver.forward_kinematics(q_eps)
                    J[:, j] = (ee_eps - ee) / eps
            finally:
                self.kinematics_solver._restore_state()

            grad[i] += wp_weight * (J.T @ dcost_dee)

        # apply normalization factor and this CF's weight
        lo, hi = self.normalization_bounds
        norm_factor = 1.0 / (hi - lo)
        return grad * self.weight * norm_factor

    # ---------- utils ----------
    def add_obstacle(self, obstacle: Obstacle):
        self.obstacles.append(obstacle)

    def remove_obstacle(self, index: int):
        if 0 <= index < len(self.obstacles):
            del self.obstacles[index]

    def get_obstacle_info(self) -> str:
        if not self.obstacles:
            return "No obstacles"
        lines = []
        for i, obs in enumerate(self.obstacles):
            lines.append(
                f"  Obstacle {i+1}: center=({obs.center[0]:.2f}, {obs.center[1]:.2f}, {obs.center[2]:.2f}), "
                f"radius={obs.radius:.3f}m, safety={obs.safe_distance:.3f}m"
            )
        return "\n".join(lines)


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


class CompositeCostFunction(CostFunction):
    """
    Professional composite cost function that supports both linear sum and weighted maximum formulations.
    
    Formulations:
    - Sum mode: cost = Σ(w_i * f_i(T))
    - Max mode: cost = max_i(w_i * f_i(T)) + ρ * Σ(f_i(T))
    
    The max mode uses tie-breaking to ensure differentiability at switching points.
    """
    
    def __init__(self, 
                 cost_functions: List[CostFunction], 
                 weights: List[float],
                 mode: str = 'sum',
                 rho: float = 0.01,
                 epsilon_tie: float = 1e-10):
        """
        Initialize composite cost function.
        
        Args:
            cost_functions: List of individual cost functions
            weights: Corresponding weights for each cost function
            mode: 'sum' for linear combination, 'max' for weighted maximum with tie-breaking
            rho: Tie-breaking parameter for max mode (should be small, e.g., 0.001-0.1)
            epsilon_tie: Threshold for detecting ties in max mode
        """
        super().__init__(weight=1.0)  # Composite function manages its own weighting
        
        if len(cost_functions) != len(weights):
            raise ValueError("Number of cost functions must match number of weights")
        
        if mode not in ['sum', 'max']:
            raise ValueError("Mode must be 'sum' or 'max'")
            
        if mode == 'max' and not (0.001 <= rho <= 0.1):
            print(f"Warning: rho={rho} is outside recommended range [0.001, 0.1] for max mode")
        
        self.cost_functions = cost_functions
        
        # Normalize weights to sum to 1
        weights_array = np.array(weights)
        weight_sum = np.sum(weights_array)
        if weight_sum <= 0:
            raise ValueError("Sum of weights must be positive")
        self.weights = weights_array / weight_sum
        self.original_weights = weights_array.copy()  # Store original for reference
        
        self.mode = mode
        self.rho = rho
        self.epsilon_tie = epsilon_tie
        
        print(f"  📊 Composite cost function initialized:")
        print(f"     Mode: {mode.upper()}")
        print(f"     Functions: {len(cost_functions)}")
        print(f"     Original weights: {[f'{w:.1f}' for w in self.original_weights]}")
        print(f"     Normalized weights: {[f'{w:.3f}' for w in self.weights]} (sum={np.sum(self.weights):.3f})")
        if mode == 'max':
            print(f"     Tie-breaking parameter ρ: {rho}")
    
    def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """Compute composite cost using specified mode."""
        if not self.cost_functions:
            return 0.0
        
        # Compute all individual costs
        individual_costs = np.array([
            cf.compute_cost(trajectory, dt) for cf in self.cost_functions
        ])
        
        if self.mode == 'sum':
            # Linear weighted sum: cost = Σ(w_i * f_i)
            return float(np.sum(self.weights * individual_costs))
        
        elif self.mode == 'max':
            # Weighted maximum with tie-breaking: cost = max(w_i * f_i) + ρ * Σ(f_i)
            weighted_costs = self.weights * individual_costs
            max_term = np.max(weighted_costs)
            sum_term = self.rho * np.sum(individual_costs)
            return float(max_term + sum_term)
        
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'sum' or 'max'.")
    
    def compute_gradient(self, trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Compute composite gradient using specified mode."""
        if not self.cost_functions:
            return np.zeros_like(trajectory)
        
        # Compute all individual costs and gradients
        individual_costs = np.array([
            cf.compute_cost(trajectory, dt) for cf in self.cost_functions
        ])
        individual_gradients = [
            cf.compute_gradient(trajectory, dt) for cf in self.cost_functions
        ]
        
        if self.mode == 'sum':
            # Linear weighted sum gradient: ∇cost = Σ(w_i * ∇f_i)
            total_gradient = np.zeros_like(trajectory)
            for i, grad in enumerate(individual_gradients):
                total_gradient += self.weights[i] * grad
            return total_gradient
        
        elif self.mode == 'max':
            # Weighted maximum gradient with tie-breaking
            weighted_costs = self.weights * individual_costs
            max_value = np.max(weighted_costs)
            
            # Find indices of functions that achieve the maximum (within epsilon_tie)
            max_indices = np.where(np.abs(weighted_costs - max_value) <= self.epsilon_tie)[0]
            
            # Max term gradient: sum over all tied maximizers
            max_gradient = np.zeros_like(trajectory)
            for idx in max_indices:
                max_gradient += self.weights[idx] * individual_gradients[idx]
            
            # If multiple tied maximizers, average their contributions
            if len(max_indices) > 1:
                max_gradient /= len(max_indices)
            
            # Tie-breaking sum term gradient: ρ * Σ(∇f_i)
            sum_gradient = np.zeros_like(trajectory)
            for grad in individual_gradients:
                sum_gradient += grad
            sum_gradient *= self.rho
            
            return max_gradient + sum_gradient
        
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'sum' or 'max'.")
    
    def get_mode_info(self) -> str:
        """Get formatted information about the current mode and configuration."""
        info = [
            f"Composite Cost Function ({self.mode.upper()} mode)",
            f"  Functions: {len(self.cost_functions)}",
            f"  Original weights: {[f'{w:.1f}' for w in self.original_weights]}",
            f"  Normalized weights: {[f'{w:.3f}' for w in self.weights]} (sum={np.sum(self.weights):.3f})"
        ]
        
        if self.mode == 'max':
            info.extend([
                f"  Tie-breaking ρ: {self.rho}",
                f"  Tie threshold: {self.epsilon_tie}"
            ])
        
        return '\n'.join(info)
    
    def switch_mode(self, new_mode: str, rho: Optional[float] = None):
        """
        Switch between sum and max modes dynamically.
        
        Args:
            new_mode: 'sum' or 'max'
            rho: New tie-breaking parameter (only used if switching to max mode)
        """
        if new_mode not in ['sum', 'max']:
            raise ValueError("Mode must be 'sum' or 'max'")
        
        old_mode = self.mode
        self.mode = new_mode
        
        if new_mode == 'max' and rho is not None:
            self.rho = rho
        
        print(f"  🔄 Switched cost mode: {old_mode.upper()} → {new_mode.upper()}")
        if new_mode == 'max':
            print(f"     Tie-breaking ρ: {self.rho}")


class CostModeFactory:
    """Factory class for creating composite cost functions with standard configurations."""
    
    @staticmethod
    def create_pareto_comparison(cost_functions: List[CostFunction], 
                               weights: List[float],
                               strategy: str = 'safe') -> CompositeCostFunction:
        """
        Create composite cost function optimized for Pareto frontier analysis.
        
        Args:
            cost_functions: List of cost functions to combine
            weights: Weights for each cost function
            strategy: 'safe' (uses sum mode), 'risky' (uses max mode), or 'custom'
        
        Returns:
            Configured CompositeCostFunction
        """
        if strategy == 'safe':
            # Safe strategy: weighted sum promotes balanced solutions
            return CompositeCostFunction(cost_functions, weights, mode='sum')
        
        elif strategy == 'risky':
            # Risky strategy: weighted max focuses on dominant objective
            return CompositeCostFunction(cost_functions, weights, mode='max', rho=0.01)
        
        elif strategy == 'custom':
            # Custom: let user configure manually
            return CompositeCostFunction(cost_functions, weights, mode='sum')
        
        else:
            raise ValueError("Strategy must be 'safe', 'risky', or 'custom'")
    
    @staticmethod
    def create_research_mode(cost_functions: List[CostFunction], 
                           weights: List[float],
                           rho: float = 0.01) -> CompositeCostFunction:
        """Create composite cost function for research with weighted maximum formulation."""
        return CompositeCostFunction(cost_functions, weights, mode='max', rho=rho)



#########################################
# previous ObstacleAvoidanceCostFunction
#########################################

# class ObstacleAvoidanceCostFunction(CostFunction):
#     """Cost function for avoiding multiple spherical obstacles in Cartesian space"""

#     def __init__(self,
#                  kinematics_solver: KinematicsSolver,
#                  obstacles: List[Obstacle],
#                  weight: float = 1.0,
#                  normalization_bounds: Tuple[float, float] = (0.0, 1.0),
#                  decay_rate: float = 5.0):
#         """
#         Initialize obstacle avoidance cost function with exponential decay

#         Args:
#             kinematics_solver: KinematicsSolver instance for forward kinematics
#             obstacles: List of Obstacle instances to avoid
#             weight: Cost function weight
#             decay_rate: Rate of exponential decay (higher = faster decay with distance)
#         """
#         # assert 1 >= weight >= 0, "Weight must be between 0 and 1"
#         super().__init__(weight)
#         self.kinematics_solver = kinematics_solver
#         self.obstacles = obstacles if obstacles else []
#         self.normalization_bounds = normalization_bounds
#         self.decay_rate = decay_rate

#     def compute_cost(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
#         """Compute obstacle avoidance cost based on MINIMUM distance to obstacles across entire trajectory"""
#         if not self.obstacles:
#             return 0.0

#         # Find the minimum distance to obstacles across the entire trajectory
#         min_distance_across_trajectory = float('inf')

#         # Backup current state
#         self.kinematics_solver._backup_state()

#         try:
#             for waypoint in trajectory:
#                 # Get end-effector position for this waypoint
#                 ee_position, _ = self.kinematics_solver.forward_kinematics(waypoint)

#                 # Find minimum distance to any obstacle surface at this waypoint
#                 min_distance_at_waypoint = float('inf')
                
#                 for obstacle in self.obstacles:
#                     # Compute distance to obstacle center
#                     distance_to_center = np.linalg.norm(ee_position - obstacle.center)
#                     # Distance to obstacle surface (negative if inside obstacle)
#                     distance_to_surface = distance_to_center - obstacle.radius
                    
#                     min_distance_at_waypoint = min(min_distance_at_waypoint, float(distance_to_surface))
                
#                 # Track the minimum distance across the entire trajectory
#                 min_distance_across_trajectory = min(min_distance_across_trajectory, min_distance_at_waypoint)

#         finally:
#             # Restore original state
#             self.kinematics_solver._restore_state()

#         # Apply exponential decay to the minimum distance across the trajectory
#         if min_distance_across_trajectory <= 0:
#             # Inside obstacle - very high penalty
#             cost = 1000.0 * np.exp(-self.decay_rate * min_distance_across_trajectory)
#         else:
#             # Outside obstacle - exponential decay with distance
#             # Cost = exp(-α * min_distance) where α is decay_rate
#             cost = np.exp(-self.decay_rate * min_distance_across_trajectory)

#         # Normalize cost to be between 0 and 1
#         normalized_cost = (cost - self.normalization_bounds[0]) / (self.normalization_bounds[1] - self.normalization_bounds[0])
#         return float(self.weight * normalized_cost)

#     def compute_gradient(self, trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
#         """Compute analytical gradient for minimum distance-based obstacle avoidance cost"""
#         if not self.obstacles:
#             return np.zeros_like(trajectory)

#         gradient = np.zeros_like(trajectory)

#         # First pass: find the minimum distance across the entire trajectory
#         min_distance_across_trajectory = float('inf')
#         waypoint_distances = []

#         # Backup current state
#         self.kinematics_solver._backup_state()

#         try:
#             # First pass: compute minimum distance at each waypoint
#             for i, waypoint in enumerate(trajectory):
#                 # Get end-effector position
#                 ee_position, _ = self.kinematics_solver.forward_kinematics(waypoint)

#                 # Find minimum distance to any obstacle surface at this waypoint
#                 min_distance_at_waypoint = float('inf')
#                 closest_obstacle_info = None
                
#                 for obstacle in self.obstacles:
#                     # Distance to obstacle center
#                     distance_vec = ee_position - obstacle.center
#                     distance_to_center = np.linalg.norm(distance_vec)
#                     # Distance to obstacle surface
#                     distance_to_surface = distance_to_center - obstacle.radius
                    
#                     if distance_to_surface < min_distance_at_waypoint:
#                         min_distance_at_waypoint = distance_to_surface
#                         closest_obstacle_info = {
#                             'obstacle': obstacle,
#                             'distance_vec': distance_vec,
#                             'distance_to_center': distance_to_center,
#                             'distance_to_surface': distance_to_surface,
#                             'ee_position': ee_position.copy()
#                         }
                
#                 waypoint_distances.append({
#                     'min_distance': min_distance_at_waypoint,
#                     'closest_obstacle_info': closest_obstacle_info
#                 })
                
#                 # Track global minimum
#                 min_distance_across_trajectory = min(min_distance_across_trajectory, float(min_distance_at_waypoint))

#             # Second pass: compute gradients only for waypoints that achieve the minimum distance
#             tolerance = 1e-6  # Small tolerance for floating-point comparison
            
#             for i, waypoint in enumerate(trajectory):
#                 waypoint_info = waypoint_distances[i]
                
#                 # Only compute gradient if this waypoint achieves (approximately) the minimum distance
#                 if abs(waypoint_info['min_distance'] - min_distance_across_trajectory) <= tolerance:
#                     closest_info = waypoint_info['closest_obstacle_info']
                    
#                     if (closest_info is not None and 
#                         closest_info['distance_to_center'] > 1e-8):
                        
#                         # Compute Jacobian for this waypoint
#                         eps = 1e-4
#                         jacobian = np.zeros((3, len(waypoint)))
#                         ee_position = closest_info['ee_position']

#                         for j in range(len(waypoint)):
#                             waypoint_plus = waypoint.copy()
#                             waypoint_plus[j] += eps
#                             ee_plus, _ = self.kinematics_solver.forward_kinematics(waypoint_plus)
#                             jacobian[:, j] = (ee_plus - ee_position) / eps

#                         # Unit vector pointing away from closest obstacle center
#                         unit_vec = closest_info['distance_vec'] / closest_info['distance_to_center']
                        
#                         # Compute cost gradient based on exponential decay applied to minimum distance
#                         if min_distance_across_trajectory <= 0:
#                             # Inside obstacle - gradient of exponential penalty
#                             # d/dx[1000 * exp(-α * min_distance)] = -1000α * exp(-α * min_distance)
#                             cost_gradient = -1000.0 * self.decay_rate * np.exp(-self.decay_rate * min_distance_across_trajectory) * unit_vec
#                         else:
#                             # Outside obstacle - gradient of exponential decay
#                             # d/dx[exp(-α * min_distance)] = -α * exp(-α * min_distance)
#                             cost_gradient = -self.decay_rate * np.exp(-self.decay_rate * min_distance_across_trajectory) * unit_vec

#                         # Chain rule: gradient w.r.t. joint angles
#                         gradient[i] += jacobian.T @ cost_gradient

#         finally:
#             # Restore original state
#             self.kinematics_solver._restore_state()

#         # Apply normalization to gradient (consistent with cost normalization)
#         normalization_factor = 1.0 / (self.normalization_bounds[1] - self.normalization_bounds[0])
#         return gradient * self.weight * normalization_factor

#     def add_obstacle(self, obstacle: Obstacle):
#         """Add an obstacle to the list"""
#         self.obstacles.append(obstacle)

#     def remove_obstacle(self, index: int):
#         """Remove an obstacle by index"""
#         if 0 <= index < len(self.obstacles):
#             del self.obstacles[index]

#     def get_obstacle_info(self) -> str:
#         """Get formatted string with obstacle information"""
#         if not self.obstacles:
#             return "No obstacles"

#         info_lines = []
#         for i, obs in enumerate(self.obstacles):
#             info_lines.append(
#                 f"  Obstacle {i + 1}: center=({obs.center[0]:.2f}, {obs.center[1]:.2f}, {obs.center[2]:.2f}), "
#                 f"radius={obs.radius:.3f}m, safety={obs.safe_distance:.3f}m"
#             )
#         return "\n".join(info_lines)
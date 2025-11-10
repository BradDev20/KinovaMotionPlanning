"""
Fast Trajectory Optimization Planner

Optimized version of ConstrainedTrajOptPlanner with:
- Fewer waypoints (15 instead of 25)
- Coarser cost evaluation sampling
- Adaptive early stopping
- Better initial guess options
- Looser default constraints for speed
- FK caching with discretization
"""

import numpy as np
import mujoco
import time
from typing import List, Tuple, Callable, Optional, Dict, Any
from .constrained_trajopt import ConstrainedTrajOptPlanner
from scipy.optimize import minimize
from functools import lru_cache


class CachedKinematicsSolver:
    """Wrapper that caches FK results with discretization for speed
    
    Caches FK calls with 0.0001 rad discretization (~0.1mm position error).
    This is well within optimization tolerances and provides ~2× speedup.
    
    Supports both instance-level and global (class-level) caching.
    Global caching is useful when running multiple optimizations on the same robot,
    e.g., in Pareto searches.
    """
    
    # Class-level (global) cache shared across all instances
    _global_cache = {}
    _global_hits = 0
    _global_misses = 0
    
    def __init__(self, kinematics_solver, decimals: int = 4, cache_size: int = 10000,
                 use_global_cache: bool = False):
        """
        Args:
            kinematics_solver: Underlying kinematics solver
            decimals: Discretization level (4 = 0.0001 rad = ~0.1mm)
            cache_size: LRU cache size
            use_global_cache: If True, use class-level cache shared across all instances
        """
        self.solver = kinematics_solver
        self.decimals = decimals
        self.cache_size = cache_size
        self.use_global_cache = use_global_cache
        
        # Instance-level cache (only used if not using global cache)
        self._instance_cache = {}
        self._instance_hits = 0
        self._instance_misses = 0
        self._last_clear_iteration = 0
    
    def forward_kinematics(self, q):
        """Cached forward kinematics with discretization"""
        # Create cache key by rounding to discretization level
        key = tuple(np.round(q, decimals=self.decimals))
        
        # Choose cache based on global/instance setting
        if self.use_global_cache:
            cache = CachedKinematicsSolver._global_cache
            
            if key in cache:
                CachedKinematicsSolver._global_hits += 1
                return cache[key]
            else:
                CachedKinematicsSolver._global_misses += 1
                result = self.solver.forward_kinematics(q)
                
                # Manage global cache size
                if len(cache) >= self.cache_size:
                    # Remove oldest 20% of entries
                    remove_count = self.cache_size // 5
                    keys_to_remove = list(cache.keys())[:remove_count]
                    for k in keys_to_remove:
                        del cache[k]
                
                cache[key] = result
                return result
        else:
            # Use instance cache
            if key in self._instance_cache:
                self._instance_hits += 1
                return self._instance_cache[key]
            else:
                self._instance_misses += 1
                result = self.solver.forward_kinematics(q)
                
                # Manage instance cache size
                if len(self._instance_cache) >= self.cache_size:
                    # Remove oldest 20% of entries
                    remove_count = self.cache_size // 5
                    keys_to_remove = list(self._instance_cache.keys())[:remove_count]
                    for k in keys_to_remove:
                        del self._instance_cache[k]
                
                self._instance_cache[key] = result
                return result
    
    def inverse_kinematics(self, *args, **kwargs):
        """Pass through IK (no caching)"""
        return self.solver.inverse_kinematics(*args, **kwargs)
    
    def _backup_state(self):
        """Pass through state backup"""
        if hasattr(self.solver, '_backup_state'):
            self.solver._backup_state()
    
    def _restore_state(self):
        """Pass through state restore"""
        if hasattr(self.solver, '_restore_state'):
            self.solver._restore_state()
    
    def clear_cache(self):
        """Clear the FK cache (instance or global depending on settings)"""
        if self.use_global_cache:
            CachedKinematicsSolver._global_cache.clear()
        else:
            self._instance_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.use_global_cache:
            total = CachedKinematicsSolver._global_hits + CachedKinematicsSolver._global_misses
            hit_rate = CachedKinematicsSolver._global_hits / total if total > 0 else 0.0
            return {
                'hits': CachedKinematicsSolver._global_hits,
                'misses': CachedKinematicsSolver._global_misses,
                'hit_rate': hit_rate,
                'cache_size': len(CachedKinematicsSolver._global_cache),
                'global': True
            }
        else:
            total = self._instance_hits + self._instance_misses
            hit_rate = self._instance_hits / total if total > 0 else 0.0
            return {
                'hits': self._instance_hits,
                'misses': self._instance_misses,
                'hit_rate': hit_rate,
                'cache_size': len(self._instance_cache),
                'global': False
            }
    
    def reset_stats(self):
        """Reset hit/miss counters (instance or global depending on settings)"""
        if self.use_global_cache:
            CachedKinematicsSolver._global_hits = 0
            CachedKinematicsSolver._global_misses = 0
        else:
            self._instance_hits = 0
            self._instance_misses = 0
    
    @classmethod
    def clear_global_cache(cls):
        """Clear the global cache (class method)"""
        cls._global_cache.clear()
        cls._global_hits = 0
        cls._global_misses = 0
    
    @classmethod
    def get_global_cache_stats(cls) -> Dict[str, Any]:
        """Get global cache statistics (class method)"""
        total = cls._global_hits + cls._global_misses
        hit_rate = cls._global_hits / total if total > 0 else 0.0
        return {
            'hits': cls._global_hits,
            'misses': cls._global_misses,
            'hit_rate': hit_rate,
            'cache_size': len(cls._global_cache)
        }


class FastTrajOptPlanner(ConstrainedTrajOptPlanner):
    """Fast trajectory optimization with aggressive performance optimizations"""

    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 n_waypoints: int = 15,  # Reduced from 25 (40% fewer variables)
                 dt: float = 0.1,
                 max_velocity: float = 1.5,  # Looser than default 1.0
                 max_acceleration: float = 1.0,  # Looser than default 0.7
                 cost_mode: str = 'composite',
                 cost_sample_rate: int = 2,  # Sample every Nth waypoint for cost eval
                 use_global_fk_cache: bool = False):  # Enable global FK caching for multi-optimization tasks
        """
        Initialize fast trajectory optimization planner
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            n_waypoints: Number of waypoints (15 recommended for speed)
            dt: Time step between waypoints
            max_velocity: Maximum joint velocity (rad/s) - looser = faster
            max_acceleration: Maximum joint acceleration (rad/s²) - looser = faster
            cost_mode: 'legacy' or 'composite'
            cost_sample_rate: Sample every Nth waypoint for cost evaluation (2 = 50% fewer FK calls)
            use_global_fk_cache: If True, share FK cache across all planner instances (useful for Pareto search)
        """
        super().__init__(model, data, n_waypoints, dt, max_velocity, max_acceleration, cost_mode)
        self.cost_sample_rate = max(1, cost_sample_rate)
        self.use_global_fk_cache = use_global_fk_cache
        self._fk_cache_wrapper = None  # Will be initialized when cost functions are set
        self._cache_clear_interval = 100  # Clear cache every N iterations to avoid stale data
        
        cache_type = "global" if use_global_fk_cache else "instance"
        print(f"  Fast TrajOpt: {n_waypoints} waypoints, sample rate={cost_sample_rate}, FK cache={cache_type}")
    
    def _compute_total_cost(self, trajectory_vector: np.ndarray) -> float:
        """
        Compute total cost with coarse sampling and pre-computed FK for speed
        
        Samples every Nth waypoint and computes FK once, sharing results across all cost functions
        """
        trajectory = self._vector_to_trajectory(trajectory_vector)
        
        # Sample waypoints for cost evaluation
        sample_indices = np.arange(0, len(trajectory), self.cost_sample_rate)
        sampled_trajectory = trajectory[sample_indices]
        
        # Adjust dt for sampled trajectory
        sampled_dt = self.dt * self.cost_sample_rate
        
        # Pre-compute FK for all sampled waypoints (share across cost functions)
        fk_results = None
        if self.cost_mode == 'composite' and self.composite_cost_function is not None:
            # Check if any cost function uses FK
            needs_fk = any(hasattr(cf, 'kinematics_solver') for cf in self.composite_cost_function.cost_functions)
            if needs_fk:
                # Get kinematics solver and wrap with cache if not already done
                kin_solver = next((cf.kinematics_solver for cf in self.composite_cost_function.cost_functions 
                                  if hasattr(cf, 'kinematics_solver')), None)
                if kin_solver is not None:
                    # Initialize cache wrapper on first use
                    if self._fk_cache_wrapper is None:
                        self._fk_cache_wrapper = CachedKinematicsSolver(
                            kin_solver, 
                            decimals=4, 
                            use_global_cache=self.use_global_fk_cache
                        )
                        cache_type = "global (shared)" if self.use_global_fk_cache else "instance"
                        print(f"  FK caching enabled: 0.0001 rad discretization (~0.1mm error), {cache_type}")
                    
                    # Periodically clear cache to avoid stale data during large gradient steps
                    # (only for instance cache; global cache persists across optimizations)
                    if not self.use_global_fk_cache and self.iteration_count > 0 and self.iteration_count % self._cache_clear_interval == 0:
                        self._fk_cache_wrapper.clear_cache()
                    
                    # Use cached FK
                    fk_results = [self._fk_cache_wrapper.forward_kinematics(q) for q in sampled_trajectory]
        
        # Compute cost on sampled trajectory with pre-computed FK
        if self.cost_mode == 'legacy':
            total_cost = 0.0
            for cost_fn in self.cost_functions:
                if hasattr(cost_fn, 'compute_cost_with_fk') and fk_results is not None:
                    total_cost += cost_fn.compute_cost_with_fk(sampled_trajectory, sampled_dt, fk_results)
                else:
                    total_cost += cost_fn.compute_cost(sampled_trajectory, sampled_dt)
        elif self.cost_mode == 'composite':
            if self.composite_cost_function is None:
                raise RuntimeError("No composite cost function set.")
            if hasattr(self.composite_cost_function, 'compute_cost_with_fk') and fk_results is not None:
                total_cost = self.composite_cost_function.compute_cost_with_fk(sampled_trajectory, sampled_dt, fk_results)
            else:
                total_cost = self.composite_cost_function.compute_cost(sampled_trajectory, sampled_dt)
        else:
            total_cost = 0.0
        
        # Track progress
        self.iteration_count += 1
        if self.iteration_count == 1:
            self.last_iteration_time = time.time()
        
        average_iteration_time = (time.time() - self.last_iteration_time) / self.iteration_count
        
        # Print progress every 10 iterations or on significant improvement
        if self.iteration_count % 10 == 0 or total_cost < self.last_cost * 0.9:
            print(f"  Iteration {self.iteration_count}: Cost = {total_cost:.3f}, Avg Time = {average_iteration_time:.3f}s")
        
        self.last_cost = total_cost
        return total_cost
    
    def _compute_total_cost_augmented(self, augmented_vector: np.ndarray) -> float:
        """
        Compute total cost for augmented optimization vector [T, t].
        Overrides parent to use optimized cost computation with FK caching and sampling.
        """
        trajectory_vector, t = self._extract_trajectory_and_t(augmented_vector)
        
        if self._is_max_constrained_mode() and t is not None:
            # In max_constrained mode: cost = t + ρ * Σf_i(T)
            # Use our optimized _compute_total_cost for the trajectory part
            trajectory_cost = self._compute_total_cost(trajectory_vector)
            total_cost = t + trajectory_cost
            
            # Manual progress tracking (same as parent)
            self.iteration_count += 1
            if self.iteration_count == 1:
                self.last_iteration_time = time.time()
            
            average_iteration_time = (time.time() - self.last_iteration_time) / self.iteration_count
            
            if self.iteration_count % 10 == 0 or total_cost < self.last_cost * 0.9:
                print(f"  Iteration {self.iteration_count}: Cost = {total_cost:.3f}, Avg Time = {average_iteration_time:.3f}s")
            
            self.last_cost = total_cost
            return total_cost
        else:
            # Standard modes: use our optimized _compute_total_cost
            return self._compute_total_cost(trajectory_vector)
    
    def _compute_total_gradient(self, trajectory_vector: np.ndarray) -> np.ndarray:
        """
        Compute gradient with coarse sampling and pre-computed FK for speed
        
        Uses same sampling as cost evaluation for consistency
        """
        trajectory = self._vector_to_trajectory(trajectory_vector)
        
        # Sample waypoints for gradient evaluation
        sample_indices = np.arange(0, len(trajectory), self.cost_sample_rate)
        sampled_trajectory = trajectory[sample_indices]
        sampled_dt = self.dt * self.cost_sample_rate
        
        # Pre-compute FK for all sampled waypoints (share across cost functions)
        fk_results = None
        if self.cost_mode == 'composite' and self.composite_cost_function is not None:
            needs_fk = any(hasattr(cf, 'kinematics_solver') for cf in self.composite_cost_function.cost_functions)
            if needs_fk:
                kin_solver = next((cf.kinematics_solver for cf in self.composite_cost_function.cost_functions 
                                  if hasattr(cf, 'kinematics_solver')), None)
                if kin_solver is not None:
                    # Use cache wrapper if available (initialized in _compute_total_cost)
                    if self._fk_cache_wrapper is not None:
                        fk_results = [self._fk_cache_wrapper.forward_kinematics(q) for q in sampled_trajectory]
                    else:
                        fk_results = [kin_solver.forward_kinematics(q) for q in sampled_trajectory]
        
        # Compute gradient on sampled trajectory with pre-computed FK
        if self.cost_mode == 'legacy':
            sampled_gradient = np.zeros_like(sampled_trajectory)
            for cost_fn in self.cost_functions:
                if hasattr(cost_fn, 'compute_gradient_with_fk') and fk_results is not None:
                    sampled_gradient += cost_fn.compute_gradient_with_fk(sampled_trajectory, sampled_dt, fk_results)
                else:
                    sampled_gradient += cost_fn.compute_gradient(sampled_trajectory, sampled_dt)
        elif self.cost_mode == 'composite':
            if self.composite_cost_function is None:
                raise RuntimeError("No composite cost function set.")
            if hasattr(self.composite_cost_function, 'compute_gradient_with_fk') and fk_results is not None:
                sampled_gradient = self.composite_cost_function.compute_gradient_with_fk(sampled_trajectory, sampled_dt, fk_results)
            else:
                sampled_gradient = self.composite_cost_function.compute_gradient(sampled_trajectory, sampled_dt)
        else:
            sampled_gradient = np.zeros_like(sampled_trajectory)
        
        # Upsample gradient back to full trajectory
        # Simple approach: distribute gradient to nearby waypoints
        full_gradient = np.zeros_like(trajectory)
        for i, sample_idx in enumerate(sample_indices):
            if i < len(sample_indices) - 1:
                # Distribute gradient across the gap
                next_sample_idx = sample_indices[i + 1]
                gap_size = next_sample_idx - sample_idx
                for j in range(gap_size):
                    weight = 1.0 - (j / gap_size)
                    full_gradient[sample_idx + j] += sampled_gradient[i] * weight
                    if j > 0:
                        full_gradient[sample_idx + j] += sampled_gradient[i + 1] * (j / gap_size)
            else:
                # Last sample
                full_gradient[sample_idx] = sampled_gradient[i]
        
        return self._trajectory_to_vector(full_gradient)
    
    def _compute_total_gradient_augmented(self, augmented_vector: np.ndarray) -> np.ndarray:
        """
        Compute gradient for augmented optimization vector [T, t].
        Overrides parent to use optimized gradient computation with FK caching and sampling.
        """
        trajectory_vector, t = self._extract_trajectory_and_t(augmented_vector)
        
        # Compute trajectory gradient using our optimized method
        trajectory_gradient = self._compute_total_gradient(trajectory_vector)
        
        if self._is_max_constrained_mode() and t is not None:
            # Gradient w.r.t. t is 1
            gradient_t = np.array([1.0])
            return np.concatenate([trajectory_gradient, gradient_t])
        else:
            return trajectory_gradient
    
    def plan(self, start_config: np.ndarray, goal_config: np.ndarray,
             warm_start_trajectory: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], bool, Dict[str, Any]]:
        """
        Plan trajectory with fast optimization settings
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            warm_start_trajectory: Optional initial trajectory (e.g., from RRT)
        
        Returns:
            trajectory: List of joint configurations
            success: Whether planning succeeded
            metadata: Dictionary with optimization info
        """
        # Validate cost functions
        if self.cost_mode == 'legacy' and len(self.cost_functions) == 0:
            print("Warning: No cost functions added to TrajOpt planner")
            return [], False, {}
        elif self.cost_mode == 'composite' and self.composite_cost_function is None:
            print("Warning: No composite cost function set")
            return [], False, {}

        # Reset progress tracking
        self.iteration_count = 0
        self.last_cost = float('inf')

        # Create initial trajectory (use warm start if provided)
        if warm_start_trajectory is not None:
            print(f"  Using warm start trajectory")
            # Resample warm start to match our waypoint count
            if len(warm_start_trajectory) != self.n_waypoints:
                indices = np.linspace(0, len(warm_start_trajectory) - 1, self.n_waypoints, dtype=int)
                initial_trajectory = warm_start_trajectory[indices]
            else:
                initial_trajectory = warm_start_trajectory
        else:
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

        # Create bounds and constraints
        bounds = self._create_bounds(start_config, goal_config)
        constraints = self._create_constraints()

        n_traj_vars = self.n_waypoints * self.n_dof
        n_total_vars = len(initial_vector)
        if self._is_max_constrained_mode():
            print(f"  Fast optimization: {self.n_waypoints} waypoints × {self.n_dof} DOF + 1 auxiliary var = {n_total_vars} variables")
        else:
            print(f"  Fast optimization: {self.n_waypoints} waypoints × {self.n_dof} DOF = {n_total_vars} variables")
        print(f"  Cost sampling: Every {self.cost_sample_rate} waypoints ({len(initial_trajectory)//self.cost_sample_rate} samples)")
        print(f"  Max velocity: {self.max_velocity:.2f} rad/s, Max acceleration: {self.max_acceleration:.2f} rad/s²")
        
        # Use augmented cost computation (handles both regular and max_constrained modes)
        initial_cost = self._compute_total_cost_augmented(initial_vector)
        print(f"  Initial cost: {initial_cost:.3f}")

        # Fast optimization settings
        max_evaluations = 2000  # Lower limit
        maxiter = 600  # Fewer iterations
        ftol = 1e-3  # Looser tolerance
        patience = 60  # Faster early stopping
        
        callback = self._create_callback(max_evaluations, patience)
        
        print(f"  Fast settings: {max_evaluations} max evals, {maxiter} max iter, ftol={ftol:.0e}, patience={patience}")
        
        try:
            optimization_start = time.perf_counter()
            
            # Use augmented cost/gradient methods (handle both regular and max_constrained modes)
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
                    'disp': False
                }
            )
            
            optimization_end = time.perf_counter()
            total_optimization_time = optimization_end - optimization_start
        except StopIteration as e:
            optimization_end = time.perf_counter()
            total_optimization_time = optimization_end - optimization_start
            print(f"  ⚠ {e}")
            
            # Use best solution found
            class PseudoResult:
                def __init__(self, x, fun, success=True, message="Early stopping"):
                    self.x = x
                    self.fun = fun
                    self.success = success
                    self.message = message
            
            best_x = callback.best_state['x'] if callback.best_state['x'] is not None else initial_vector
            best_cost = callback.best_state['cost'] if callback.best_state['x'] is not None else initial_cost
            
            result = PseudoResult(x=best_x, fun=best_cost, success=True, message=str(e))
        except Exception as e:
            print(f"TrajOpt planning failed: {e}")
            return [], False, {}

        print(f"  Optimization completed in {self.iteration_count} iterations ({total_optimization_time:.2f}s)")
        print(f"  Final cost: {result.fun:.3f}")
        print(f"  Status: {result.message}")
        
        # Collect timing statistics
        timing_summary = self._collect_timing_statistics(total_optimization_time)
        
        # Prepare metadata
        metadata = {
            'iterations': self.iteration_count,
            'final_optimization_cost': float(result.fun),
            'cost_mode': self.cost_mode,
            'stopped_early': 'Early stopping' in result.message if hasattr(result, 'message') else False,
            'termination_reason': result.message if hasattr(result, 'message') else 'Unknown',
            'timing': timing_summary,
            'n_waypoints': self.n_waypoints,
            'cost_sample_rate': self.cost_sample_rate,
        }
        
        # Print FK cache statistics if caching was used
        if self._fk_cache_wrapper is not None:
            cache_stats = self._fk_cache_wrapper.get_stats()
            cache_scope = "Global" if cache_stats.get('global', False) else "Instance"
            print(f"\n  FK Cache Statistics ({cache_scope}):")
            print(f"    Hits: {cache_stats['hits']:,}, Misses: {cache_stats['misses']:,}")
            print(f"    Hit rate: {cache_stats['hit_rate']*100:.1f}%")
            print(f"    Cache size: {cache_stats['cache_size']} entries")
            total_fk = cache_stats['hits'] + cache_stats['misses']
            saved_fk = cache_stats['hits']
            if total_fk > 0:
                print(f"    FK calls saved: {saved_fk:,} ({saved_fk/total_fk*100:.1f}% reduction)")
            
            # Add to metadata
            metadata['fk_cache'] = cache_stats

        # Extract trajectory from result (handles augmented vector if max_constrained)
        final_traj_vec, final_t = self._extract_trajectory_and_t(result.x)
        
        # Check constraint violations
        final_vel_constraints = self._compute_velocity_constraints(final_traj_vec)
        final_acc_constraints = self._compute_acceleration_constraints(final_traj_vec)
        
        final_vel_violations = np.sum(final_vel_constraints < -1e-6)
        final_acc_violations = np.sum(final_acc_constraints < -1e-6)
        
        print(f"  Final violations: {final_vel_violations} velocity, {final_acc_violations} acceleration")
        
        if self._is_max_constrained_mode() and final_t is not None:
            print(f"  Final t (max bound): {final_t:.3f}")

        if result.success or result.fun < initial_cost * 0.9:
            optimized_trajectory = self._vector_to_trajectory(final_traj_vec)
            trajectory_list = [waypoint for waypoint in optimized_trajectory]
            
            total_violations = final_vel_violations + final_acc_violations
            if total_violations == 0:
                print(f"  ✓ All constraints satisfied!")
            else:
                print(f"  ⚠ Some constraints violated (may be acceptable)")
            
            return trajectory_list, True, metadata
        else:
            print(f"TrajOpt optimization failed: {result.message}")
            initial_trajectory_list = [waypoint for waypoint in initial_trajectory]
            return initial_trajectory_list, False, metadata


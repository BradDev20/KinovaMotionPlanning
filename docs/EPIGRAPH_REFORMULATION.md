# Epigraph Reformulation for Max-Based Cost Functions

## Overview

This document describes the implementation of the epigraph reformulation for trajectory optimization with max-based cost functions. This reformulation moves the `max` operator from the cost function to the constraints, which can improve numerical properties and optimization behavior.

## Mathematical Formulation

### Original Formulation

```
min_T [ max_i(w_i * f_i(T)) + ρ * Σ_i f_i(T) ]
s.t. g(T) ≤ 0
```

where:
- `T` is the trajectory (optimization variable)
- `f_i(T)` are individual cost functions
- `w_i` are normalized weights
- `ρ` is a small regularization parameter
- `g(T)` represents kinodynamic constraints (velocity, acceleration, etc.)

### Epigraph Reformulation

```
min_{T,t} [ t + ρ * Σ_i f_i(T) ]
s.t. w_i * f_i(T) ≤ t  for i = 1, ..., n
s.t. g(T) ≤ 0
```

where:
- `T` is the trajectory (optimization variable)
- `t` is an auxiliary variable representing the upper bound on weighted costs
- The constraints `w_i * f_i(T) ≤ t` ensure that `t ≥ max_i(w_i * f_i(T))`

## Implementation

### CompositeCostFunction Changes

#### New Mode: `'max_constrained'`

Added a third mode to `CompositeCostFunction` in `src/motion_planning/cost_functions.py`:

- **`mode='sum'`**: Linear weighted sum (existing)
- **`mode='max'`**: Weighted maximum with tie-breaking (existing)
- **`mode='max_constrained'`**: Epigraph reformulation (new)

#### Key Methods

1. **`compute_cost(trajectory, dt)`** - In `max_constrained` mode, returns only `ρ * Σf_i(T)`
   - The `t` term is handled by the planner
   
2. **`compute_gradient(trajectory, dt)`** - In `max_constrained` mode, returns `ρ * Σ(∇f_i)`
   - The gradient w.r.t. `t` is 1 (handled by planner)

3. **`compute_weighted_individual_costs(trajectory, dt)`** - New helper method
   - Returns array of `w_i * f_i(T)` for each cost function
   - Used to evaluate the epigraph constraints

4. **`compute_individual_cost_gradients(trajectory, dt)`** - New helper method
   - Returns list of gradients `w_i * ∇f_i(T)`
   - Used to compute Jacobian of epigraph constraints

### ConstrainedTrajOptPlanner Changes

Enhanced `src/motion_planning/constrained_trajopt.py` to handle the augmented optimization vector `[T, t]`:

#### New Methods

1. **`_is_max_constrained_mode()`** - Check if using epigraph reformulation

2. **`_extract_trajectory_and_t(augmented_vector)`** - Extract trajectory and auxiliary variable

3. **`_compute_epigraph_constraints(augmented_vector)`**
   - Computes constraints: `t - w_i * f_i(T) ≥ 0` for all i
   - Returns array where positive values indicate satisfaction

4. **`_compute_epigraph_constraint_jacobian(augmented_vector)`**
   - Computes Jacobian of epigraph constraints w.r.t. `[T, t]`
   - Each row corresponds to one constraint

5. **`_compute_total_cost_augmented(augmented_vector)`**
   - Wrapper for cost computation handling augmented vector
   - Returns `t + ρ * Σf_i(T)` in max_constrained mode

6. **`_compute_total_gradient_augmented(augmented_vector)`**
   - Wrapper for gradient computation handling augmented vector
   - Returns `[∇_T(ρ * Σf_i(T)), 1]` in max_constrained mode

#### Modified Methods

1. **`_create_bounds()`** - Adds bounds for `t` variable: `(0, 1e6)`

2. **`_create_constraints()`** - Adds epigraph constraints when in max_constrained mode
   - All existing constraints (velocity, acceleration, z-constraint) are wrapped to handle augmented vector

3. **`plan()`** - Enhanced to:
   - Initialize `t` with reasonable estimate: 110% of max weighted cost
   - Use augmented cost/gradient functions
   - Report final `t` value and cost breakdown
   - Check epigraph constraint violations

## Usage Example

```python
from src.motion_planning.constrained_trajopt import ConstrainedTrajOptPlanner
from src.motion_planning.cost_functions import CompositeCostFunction

# Create individual cost functions
length_cost = TrajectoryLengthCostFunction(kin, weight=1.0)
obstacle_cost = ObstacleAvoidanceCostFunction(kin, obstacles, weight=1.0)
velocity_cost = VelocityCostFunction(weight=1.0)

# Create composite cost with epigraph reformulation
composite = CompositeCostFunction(
    cost_functions=[length_cost, obstacle_cost, velocity_cost],
    weights=[1.0, 2.0, 0.5],
    mode='max_constrained',  # Use epigraph reformulation
    rho=0.01
)

# Create planner
planner = ConstrainedTrajOptPlanner(
    model, data,
    n_waypoints=20,
    dt=0.1,
    max_velocity=2.0,
    max_acceleration=10.0,
    cost_mode='composite'
)
planner.composite_cost_function = composite

# Plan trajectory
trajectory, success = planner.plan(start_config, goal_config)
```

## Advantages of Epigraph Reformulation

1. **Improved Numerical Properties**: Moving the max operator to constraints can help avoid non-differentiability issues at switching points

2. **Explicit Control**: The auxiliary variable `t` provides an explicit upper bound on all weighted costs

3. **Constraint Satisfaction**: Each objective is explicitly bounded by `t`, making it easier to analyze trade-offs

4. **Better Convergence**: In some cases, the reformulation can lead to better convergence properties for gradient-based optimizers

## Testing

A test script is provided in `src/examples/test_max_constrained.py` that compares:
- Standard `'max'` mode (baseline)
- Epigraph `'max_constrained'` mode (new)

Run the test with:
```bash
python src/examples/test_max_constrained.py
```

## Backward Compatibility

All existing functionality is preserved:
- `'sum'` mode: unchanged
- `'max'` mode: unchanged
- No changes required to existing code unless you want to use the new `'max_constrained'` mode

## References

The epigraph reformulation is a standard technique in convex optimization for handling max operators in objective functions. See:
- Boyd & Vandenberghe, "Convex Optimization", Section 4.1.3
- Bertsekas, "Nonlinear Programming", Section 1.2.3 
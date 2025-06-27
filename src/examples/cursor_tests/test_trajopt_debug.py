#!/usr/bin/env python3
"""
Debug script to diagnose TrajOpt multi-obstacle planning failures
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from motion_planning.kinematics import KinematicsSolver
from motion_planning.planners import (
    Obstacle,
    TrajectoryLengthCostFunction,
    ObstacleAvoidanceCostFunction,
    VelocityCostFunction,
    AccelerationCostFunction
)

def debug_trajopt_costs():
    """Debug TrajOpt cost components for the failing scenario"""
    
    print("🔍 TrajOpt Multi-Obstacle Debug Analysis")
    print("=" * 60)
    
    # Define the same setup as the failing demo
    obstacles = [
        Obstacle(center=np.array([0.5, -0.1, 0.6]), radius=0.08, safe_distance=0.05),
        Obstacle(center=np.array([0.5, -0.35, 0.5]), radius=0.06, safe_distance=0.04),
    ]
    
    # Define configurations
    start_config = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
    goal_config = np.array([0.179, 1.577, -0.436, -1.629, 0.393, 2.080, 0.000])
    
    print(f"Start config: {start_config}")
    print(f"Goal config:  {goal_config}")
    
    # Create a simple linear interpolation trajectory (what TrajOpt starts with)
    n_waypoints = 20  # Smaller for debugging
    trajectory = np.zeros((n_waypoints, 7))
    
    for i in range(n_waypoints):
        alpha = i / (n_waypoints - 1)
        trajectory[i] = (1 - alpha) * start_config + alpha * goal_config
    
    print(f"\nCreated {n_waypoints}-waypoint linear interpolation trajectory")
    
    # Create mock kinematics solver for testing
    class MockKinematics:
        def forward_kinematics(self, config):
            # Simplified FK that maps joint config to approximate EE position
            # This is just for testing - use actual FK in real scenarios
            x = 0.5 + 0.3 * config[0] + 0.2 * config[1]
            y = -0.2 - 0.1 * config[0] + 0.2 * config[2]
            z = 0.8 + 0.3 * config[1] - 0.2 * config[3]
            return np.array([x, y, z]), None
        
        def _backup_state(self):
            pass
        
        def _restore_state(self):
            pass
    
    kinematics = MockKinematics()
    
    # Test each cost function individually
    print(f"\n📊 COST FUNCTION ANALYSIS:")
    print("-" * 40)
    
    # 1. Trajectory Length Cost
    length_cost = TrajectoryLengthCostFunction(weight=2.0)
    length_value = length_cost.compute_cost(trajectory, 0.02)
    print(f"✓ Trajectory Length Cost (weight=2.0): {length_value:.6f}")
    
    # 2. Obstacle Avoidance Cost  
    obstacle_cost = ObstacleAvoidanceCostFunction(
        kinematics_solver=kinematics,
        obstacles=obstacles,
        weight=200.0  # Reduced weight
    )
    obstacle_value = obstacle_cost.compute_cost(trajectory, 0.02)
    print(f"✓ Obstacle Avoidance Cost (weight=200.0): {obstacle_value:.6f}")
    
    # 3. Velocity Cost
    velocity_cost = VelocityCostFunction(weight=0.2)
    velocity_value = velocity_cost.compute_cost(trajectory, 0.02)
    print(f"✓ Velocity Cost (weight=0.2): {velocity_value:.6f}")
    
    # 4. Acceleration Cost
    accel_cost = AccelerationCostFunction(weight=0.05)
    accel_value = accel_cost.compute_cost(trajectory, 0.02)
    print(f"✓ Acceleration Cost (weight=0.05): {accel_value:.6f}")
    
    # Total cost
    total_cost = length_value + obstacle_value + velocity_value + accel_value
    print(f"\n🎯 TOTAL COST: {total_cost:.6f}")
    
    # Check obstacle violations
    print(f"\n🚧 OBSTACLE VIOLATION ANALYSIS:")
    print("-" * 40)
    
    violations = []
    for i, waypoint in enumerate(trajectory):
        ee_pos, _ = kinematics.forward_kinematics(waypoint)
        
        for j, obstacle in enumerate(obstacles):
            distance = np.linalg.norm(ee_pos - obstacle.center)
            distance_from_surface = distance - obstacle.radius
            danger_threshold = obstacle.radius + obstacle.safe_distance
            
            if distance < danger_threshold:
                violation = danger_threshold - distance
                violations.append((i, j, violation, distance_from_surface))
                print(f"⚠ Waypoint {i:2d}, Obstacle {j+1}: violation={violation:.4f}m, "
                      f"surface_dist={distance_from_surface:.4f}m")
    
    if not violations:
        print("✅ No obstacle violations found in initial trajectory")
    else:
        print(f"❌ Found {len(violations)} obstacle violations")
        
        # Analyze most severe violation
        max_violation = max(violations, key=lambda x: x[2])
        print(f"📍 Worst violation: Waypoint {max_violation[0]}, Obstacle {max_violation[1]+1}, "
              f"violation={max_violation[2]:.4f}m")
    
    # Gradient check
    print(f"\n🧮 GRADIENT ANALYSIS:")
    print("-" * 40)
    
    try:
        gradient = obstacle_cost.compute_gradient(trajectory, 0.02)
        gradient_norm = np.linalg.norm(gradient)
        print(f"✓ Obstacle gradient computed: norm={gradient_norm:.6f}")
        
        if gradient_norm < 1e-10:
            print("⚠ Gradient is very small - might cause optimization issues")
        elif gradient_norm > 1e6:
            print("⚠ Gradient is very large - might cause numerical instability")
        else:
            print("✅ Gradient norm appears reasonable")
            
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if obstacle_value > 1000:
        print("• Obstacle cost is very high - consider reducing weight or moving obstacles")
    
    if len(violations) > len(trajectory) * 0.5:
        print("• More than 50% of waypoints violate constraints - infeasible problem")
        print("• Consider: moving obstacles, changing start/goal, or using different planner")
    
    if len(violations) == 0 and obstacle_value > 0:
        print("• No violations but positive cost - check cost function implementation")
    
    print("\n🎯 DEBUG COMPLETE")

if __name__ == "__main__":
    debug_trajopt_costs() 
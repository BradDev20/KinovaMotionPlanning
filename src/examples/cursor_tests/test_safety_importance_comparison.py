#!/usr/bin/env python3
"""
Safety Importance Multi-Objective Comparison Demo

Compares two strategies:
1. RISKY: High trajectory length weight, low safety importance (threads between obstacles)
2. SAFE: Low trajectory length weight, high safety importance (goes around obstacles)
"""

import mujoco
import mujoco.viewer as viewer
import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from motion_planning.kinematics import KinematicsSolver
from motion_planning.planners import (
    TrajOptPlanner,
    TrajectoryLengthCostFunction,
    ObstacleAvoidanceCostFunction,
    SafetyImportanceCostFunction,
    VelocityCostFunction,
    AccelerationCostFunction,
    Obstacle
)

# Create obstacle corridor - robot can thread between or go around
obstacles = [
    Obstacle(center=np.array([0.4, -0.15, 0.7]), radius=0.08, safe_distance=0.05),
    Obstacle(center=np.array([0.4, -0.45, 0.5]), radius=0.08, safe_distance=0.05),
]

def plan_strategy(model, data, kinematics, strategy_name):
    """Plan with different cost weights based on strategy"""
    
    print(f"\n🎯 Planning {strategy_name.upper()} Strategy")
    print("=" * 40)
    
    start_config = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
    target_position = np.array([0.5, -0.3, 0.4])  # Between obstacles
    
    # IK
    goal_config, ik_success = kinematics.inverse_kinematics(target_position, start_config)
    if not ik_success:
        return None, None
    
    # Create planner
    planner = TrajOptPlanner(model, data, n_waypoints=40, dt=0.1)
    
    if strategy_name == "risky":
        # RISKY: Prioritize short paths, minimal safety
        length_weight = 10.0    # HIGH - want short paths
        safety_weight = 0.1     # LOW - don't care about proximity
        obstacle_weight = 100.0 # MODERATE - just avoid collisions
        print("RISKY weights: Length=10.0, Safety=0.1, Obstacle=100.0")
    else:
        # SAFE: Prioritize safety, path length less important  
        length_weight = 1.0     # LOW - path length less important
        safety_weight = 50.0    # HIGH - stay far from obstacles
        obstacle_weight = 200.0 # HIGH - strong collision avoidance
        print("SAFE weights: Length=1.0, Safety=50.0, Obstacle=200.0")
    
    # Add cost functions
    planner.add_cost_function(TrajectoryLengthCostFunction(weight=length_weight))
    planner.add_cost_function(SafetyImportanceCostFunction(
        kinematics_solver=kinematics,
        obstacles=obstacles,
        weight=safety_weight,
        safety_radius_multiplier=4.0  # Large safety zones
    ))
    planner.add_cost_function(ObstacleAvoidanceCostFunction(
        kinematics_solver=kinematics,
        obstacles=obstacles,
        weight=obstacle_weight
    ))
    planner.add_cost_function(VelocityCostFunction(weight=0.1))
    
    # Plan
    trajectory, success = planner.plan(start_config, goal_config)
    
    if success:
        # Analyze
        total_length = sum(np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                          for i in range(len(trajectory)-1))
        
        min_clearances = []
        for waypoint in trajectory:
            ee_pos, _ = kinematics.forward_kinematics(waypoint)
            for obstacle in obstacles:
                dist = np.linalg.norm(ee_pos - obstacle.center) - obstacle.radius
                min_clearances.append(dist)
        
        min_clearance = min(min_clearances)
        avg_clearance = np.mean(min_clearances)
        
        stats = {
            'length': total_length,
            'min_clearance': min_clearance,
            'avg_clearance': avg_clearance
        }
        
        print(f"✅ {strategy_name.upper()} success!")
        print(f"  Length: {total_length:.3f} rad")
        print(f"  Min clearance: {min_clearance:.3f}m")
        print(f"  Avg clearance: {avg_clearance:.3f}m")
        
        return trajectory, stats
    else:
        print(f"❌ {strategy_name.upper()} failed!")
        return None, None

def main():
    print("🎯 Safety Importance Comparison Demo")
    print("Comparing RISKY vs SAFE trajectory strategies")
    
    # Use existing scene
    scene_path = 'robot_models/kinova_gen3/scene.xml'
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    kinematics = KinematicsSolver(scene_path)
    
    # Plan both strategies
    risky_traj, risky_stats = plan_strategy(model, data, kinematics, "risky")
    safe_traj, safe_stats = plan_strategy(model, data, kinematics, "safe")
    
    if risky_traj is None or safe_traj is None:
        print("❌ Planning failed")
        return
    
    # Show comparison
    print(f"\n📊 COMPARISON RESULTS:")
    print("-" * 30)
    print(f"{'Metric':<15} {'RISKY':<10} {'SAFE':<10}")
    print("-" * 30)
    print(f"{'Length (rad)':<15} {risky_stats['length']:<10.3f} {safe_stats['length']:<10.3f}")
    print(f"{'Min Clear (m)':<15} {risky_stats['min_clearance']:<10.3f} {safe_stats['min_clearance']:<10.3f}")
    print(f"{'Avg Clear (m)':<15} {risky_stats['avg_clearance']:<10.3f} {safe_stats['avg_clearance']:<10.3f}")
    print("-" * 30)
    
    # Analysis
    if risky_stats['length'] < safe_stats['length']:
        print("✅ RISKY trajectory is SHORTER")
    else:
        print("⚠️ SAFE trajectory is unexpectedly shorter")
        
    if safe_stats['min_clearance'] > risky_stats['min_clearance']:
        print("✅ SAFE trajectory maintains BETTER clearance")
    else:
        print("⚠️ RISKY trajectory unexpectedly safer")
    
    print(f"\n💡 Multi-objective trade-off demonstrated:")
    print(f"• Different cost weights → different path choices")
    print(f"• High safety weight → robot avoids threading between obstacles")
    print(f"• High length weight → robot prefers shorter, riskier paths")

if __name__ == "__main__":
    main() 
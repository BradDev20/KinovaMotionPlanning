#!/usr/bin/env python3
"""
Simple test script for the new TrajOpt cost functions:
- TrajectoryLengthCostFunction
- ObstacleAvoidanceCostFunction
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from motion_planning.planners import (
    TrajOptPlanner,
    TrajectoryLengthCostFunction, 
    ObstacleAvoidanceCostFunction,
    VelocityCostFunction
)
from motion_planning.kinematics import KinematicsSolver


def test_trajectory_length_cost():
    """Test the trajectory length cost function"""
    print("=== Testing TrajectoryLengthCostFunction ===")
    
    # Create a simple test trajectory
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Start
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Middle
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # End
    ])
    
    # Test cost function
    length_cost = TrajectoryLengthCostFunction(weight=1.0)
    cost = length_cost.compute_cost(trajectory)
    gradient = length_cost.compute_gradient(trajectory)
    
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Length cost: {cost:.4f}")
    print(f"Gradient shape: {gradient.shape}")
    print("✓ TrajectoryLengthCostFunction test passed")
    return True


def test_obstacle_avoidance_cost():
    """Test the obstacle avoidance cost function"""
    print("\n=== Testing ObstacleAvoidanceCostFunction ===")
    
    try:
        # Load kinematics solver
        model_path = os.path.join(os.path.dirname(__file__), '../../..', 
                                 'robot_models', 'kinova_gen3', 'gen3.xml')
        
        if not os.path.exists(model_path):
            print(f"Warning: Robot model not found at {model_path}")
            print("Skipping obstacle avoidance test")
            return False
            
        kinematics = KinematicsSolver(model_path)
        
        # Create test trajectory  
        trajectory = np.array([
            [0.0, 0.0, 0.0, -1.5, 0.0, 1.0, 0.0],
            [0.5, 0.2, 0.3, -1.2, 0.1, 0.8, 0.2],
            [1.0, 0.5, 0.6, -1.0, 0.3, 0.6, 0.4],
        ])
        
        # Get end-effector position for obstacle placement
        ee_pos, _ = kinematics.forward_kinematics(trajectory[1])  # Middle waypoint
        
        # Place obstacle near the trajectory
        obstacle_center = ee_pos + np.array([0.1, 0.1, 0.0])  # Slightly offset
        obstacle_radius = 0.05
        safe_distance = 0.05
        
        # Test cost function
        obstacle_cost = ObstacleAvoidanceCostFunction(
            kinematics_solver=kinematics,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            safe_distance=safe_distance,
            weight=10.0
        )
        
        cost = obstacle_cost.compute_cost(trajectory)
        gradient = obstacle_cost.compute_gradient(trajectory)
        
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"Obstacle center: {obstacle_center}")
        print(f"Obstacle avoidance cost: {cost:.4f}")
        print(f"Gradient shape: {gradient.shape}")
        print("✓ ObstacleAvoidanceCostFunction test passed")
        return True
        
    except Exception as e:
        print(f"Error in obstacle avoidance test: {e}")
        return False


def test_combined_optimization():
    """Test TrajOpt with both new cost functions"""
    print("\n=== Testing Combined TrajOpt Optimization ===")
    
    try:
        import mujoco
        
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), '../../..', 
                                 'robot_models', 'kinova_gen3', 'gen3.xml')
        
        if not os.path.exists(model_path):
            print(f"Warning: Robot model not found at {model_path}")
            print("Skipping combined optimization test")
            return False
            
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        kinematics = KinematicsSolver(model_path)
        
        # Define simple start and goal
        start_config = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.0, 0.0])
        goal_config = np.array([1.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0])
        
        # Create planner
        planner = TrajOptPlanner(model, data, n_waypoints=10)
        
        # Add cost functions
        planner.add_cost_function(TrajectoryLengthCostFunction(weight=1.0))
        planner.add_cost_function(VelocityCostFunction(weight=0.1))
        
        # Add obstacle avoidance if we can get EE positions
        try:
            start_ee, _ = kinematics.forward_kinematics(start_config)
            goal_ee, _ = kinematics.forward_kinematics(goal_config)
            obstacle_center = (start_ee + goal_ee) / 2
            
            planner.add_cost_function(ObstacleAvoidanceCostFunction(
                kinematics_solver=kinematics,
                obstacle_center=obstacle_center,
                obstacle_radius=0.1,
                safe_distance=0.05,
                weight=50.0
            ))
            print("Added obstacle avoidance cost function")
        except:
            print("Could not add obstacle avoidance, continuing without it")
        
        # Plan trajectory
        trajectory, success = planner.plan(start_config, goal_config)
        
        if success:
            print(f"✓ Combined optimization successful!")
            print(f"  Trajectory length: {len(trajectory)} waypoints")
            
            # Compute some metrics
            total_length = 0.0
            for i in range(len(trajectory) - 1):
                total_length += np.linalg.norm(trajectory[i+1] - trajectory[i])
            print(f"  Total joint space length: {total_length:.3f}")
            
        else:
            print("✗ Combined optimization failed")
            
        return success
        
    except Exception as e:
        print(f"Error in combined optimization test: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing new TrajOpt cost functions...\n")
    
    # Run individual tests
    test1 = test_trajectory_length_cost()
    test2 = test_obstacle_avoidance_cost()
    test3 = test_combined_optimization()
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"TrajectoryLengthCostFunction: {'✓' if test1 else '✗'}")
    print(f"ObstacleAvoidanceCostFunction: {'✓' if test2 else '✗'}")
    print(f"Combined Optimization: {'✓' if test3 else '✗'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠ Some tests failed or were skipped")


if __name__ == "__main__":
    main() 
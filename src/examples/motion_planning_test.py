#!/usr/bin/env python3
"""
Motion Planning Test: Validate Components Step by Step

This script tests each component individually to diagnose issues:
1. Forward kinematics
2. Inverse kinematics with reachable goals
3. Simple RRT planning between nearby configurations
4. Integration test

Usage:
    mjpython src/examples/motion_planning_test.py
"""

import numpy as np
import mujoco
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.motion_planning import KinematicsSolver, RRTPlanner, MotionPlanningInterface


def test_forward_kinematics():
    """Test forward kinematics with known configurations"""
    print("=" * 50)
    print("TESTING FORWARD KINEMATICS")
    print("=" * 50)
    
    model = mujoco.MjModel.from_xml_path("robot_models/kinova_gen3/scene.xml")
    data = mujoco.MjData(model)
    ik_solver = KinematicsSolver(model, data)
    
    # Test configurations
    test_configs = [
        ("Home", np.array([0.0, 0.3, 0.0, -1.2, 0.0, 0.8, 0.0])),
        ("Zero", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("Bent", np.array([0.5, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0]))
    ]
    
    for name, config in test_configs:
        pos, rot = ik_solver.forward_kinematics(config)
        print(f"{name:10} -> Position: {pos}")
        print(f"{'':10}    Rotation determinant: {np.linalg.det(rot):.3f}")
    
    return ik_solver


def test_inverse_kinematics(ik_solver):
    """Test inverse kinematics with reachable targets"""
    print("\n" + "=" * 50)
    print("TESTING INVERSE KINEMATICS")
    print("=" * 50)
    
    # Get current pose as a known reachable target
    home_config = np.array([0.0, 0.3, 0.0, -1.2, 0.0, 0.8, 0.0])
    home_pos, home_rot = ik_solver.forward_kinematics(home_config)
    
    print(f"Home position: {home_pos}")
    
    # Test IK with the known reachable pose
    solved_config, success = ik_solver.inverse_kinematics(home_pos, home_rot, 
                                                         initial_guess=np.zeros(7))
    print(f"IK for home pose: {'✅ SUCCESS' if success else '❌ FAILED'}")
    if success:
        print(f"  Original: {[f'{x:.3f}' for x in home_config]}")
        print(f"  Solved:   {[f'{x:.3f}' for x in solved_config]}")
        
        # Verify solution
        verify_pos, verify_rot = ik_solver.forward_kinematics(solved_config)
        pos_error = np.linalg.norm(verify_pos - home_pos)
        print(f"  Position error: {pos_error:.6f} m")
    
    # Test with small position perturbations
    small_targets = [
        ("Small X+", home_pos + np.array([0.05, 0.0, 0.0])),
        ("Small Y+", home_pos + np.array([0.0, 0.05, 0.0])),
        ("Small Z+", home_pos + np.array([0.0, 0.0, 0.05])),
    ]
    
    for name, target_pos in small_targets:
        solved_config, success = ik_solver.inverse_kinematics(target_pos, 
                                                             initial_guess=home_config)
        print(f"IK for {name:8}: {'✅ SUCCESS' if success else '❌ FAILED'}")
        if success:
            final_pos, _ = ik_solver.forward_kinematics(solved_config)
            error = np.linalg.norm(final_pos - target_pos)
            print(f"  Position error: {error:.6f} m")
    
    return home_config, solved_config if success else home_config


def test_rrt_planning(model, data, start_config, goal_config):
    """Test RRT planning between two configurations"""
    print("\n" + "=" * 50)
    print("TESTING RRT PLANNING")
    print("=" * 50)
    
    planner = RRTPlanner(model, data, step_size=0.1, max_iterations=500, goal_threshold=0.1)
    
    print(f"Start config: {[f'{x:.3f}' for x in start_config]}")
    print(f"Goal config:  {[f'{x:.3f}' for x in goal_config]}")
    print(f"Distance:     {np.linalg.norm(goal_config - start_config):.3f}")
    
    # Test planning
    path, success = planner.plan(start_config, goal_config)
    
    print(f"RRT Planning: {'✅ SUCCESS' if success else '❌ FAILED'}")
    if success:
        print(f"  Path length: {len(path)} waypoints")
        print(f"  First waypoint: {[f'{x:.3f}' for x in path[0]]}")
        print(f"  Last waypoint:  {[f'{x:.3f}' for x in path[-1]]}")
        
        # Verify path endpoints
        start_error = np.linalg.norm(path[0] - start_config)
        goal_error = np.linalg.norm(path[-1] - goal_config)
        print(f"  Start error: {start_error:.6f}")
        print(f"  Goal error:  {goal_error:.6f}")
    
    return path if success else []


def test_integration():
    """Test the complete motion planning interface"""
    print("\n" + "=" * 50)
    print("TESTING MOTION PLANNING INTERFACE")
    print("=" * 50)
    
    model = mujoco.MjModel.from_xml_path("robot_models/kinova_gen3/scene.xml")
    data = mujoco.MjData(model)
    
    # Set home configuration
    home_config = np.array([0.0, 0.3, 0.0, -1.2, 0.0, 0.8, 0.0])
    data.qpos[:7] = home_config
    data.ctrl[:7] = home_config
    mujoco.mj_forward(model, data)
    
    mp_interface = MotionPlanningInterface(model, data)
    
    # Test current pose reading
    current_pos, current_rot = mp_interface.get_current_end_effector_pose()
    print(f"Current EE position: {current_pos}")
    
    # Test joint space planning to a nearby configuration
    nearby_config = home_config + np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print(f"Planning to nearby joint configuration...")
    trajectory, success = mp_interface.plan_to_joint_configuration(nearby_config)
    
    print(f"Joint space planning: {'✅ SUCCESS' if success else '❌ FAILED'}")
    if success:
        print(f"  Trajectory length: {len(trajectory)} waypoints")
        
        # Test trajectory execution (without viewer)
        print("Testing trajectory execution...")
        exec_success = mp_interface.execute_trajectory(trajectory, dt=0.0)  # No delay
        print(f"  Execution: {'✅ SUCCESS' if exec_success else '❌ FAILED'}")


def main():
    """Run all tests"""
    print("🔬 Motion Planning Component Tests")
    print("This will test each component to identify issues\n")
    
    try:
        # Test 1: Forward kinematics
        ik_solver = test_forward_kinematics()
        
        # Test 2: Inverse kinematics  
        home_config, target_config = test_inverse_kinematics(ik_solver)
        
        # Test 3: RRT planning
        model = mujoco.MjModel.from_xml_path("robot_models/kinova_gen3/scene.xml")
        data = mujoco.MjData(model)
        path = test_rrt_planning(model, data, home_config, target_config)
        
        # Test 4: Integration
        test_integration()
        
        print("\n" + "=" * 50)
        print("TESTING COMPLETED")
        print("=" * 50)
        print("If tests pass, the motion planning system is working!")
        print("If tests fail, check the specific component that failed.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
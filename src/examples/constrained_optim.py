#!/usr/bin/env python3
"""
TrajOpt Visualization Demo with Multi-Obstacle Avoidance and Length Minimization

This demo creates a scene with multiple virtual obstacles and demonstrates how
the new TrajOpt cost functions work:
1. TrajectoryLengthCostFunction - minimizes trajectory length
2. ObstacleAvoidanceCostFunction - avoids multiple spherical obstacles

Follows the pattern from target_reaching_demo.py for proper visualization.
"""

import mujoco
import mujoco.viewer as viewer
import numpy as np
import time
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from motion_planning.kinematics import KinematicsSolver
from motion_planning.constrained_trajopt import ConstrainedTrajOptPlanner
from motion_planning.utils import Obstacle
from motion_planning.cost_functions import (
    TrajectoryLengthCostFunction,
    SafetyImportanceCostFunction,
    ObstacleAvoidanceCostFunction,
    VelocityCostFunction,
    AccelerationCostFunction
)

# Define multiple obstacles for the demo
obstacles = [
    Obstacle(center=np.array([0.55, -0.1, 0.75]), radius=0.08, safe_distance=0.05),
    # Obstacle(center=np.array([0.6, -0.3, 0.8]), radius=0.2, safe_distance=0.05),
    Obstacle(center=np.array([0.55, -0.45, 0.65]), radius=0.08, safe_distance=0.05),
]


def create_scene_with_virtual_obstacles():
    """Create a scene with visual obstacle markers for multiple obstacles"""
    
    # Read the base scene - use simple scene without gripper to avoid path issues
    base_scene_path = 'robot_models/kinova_gen3/scene.xml'
    
    if not os.path.exists(base_scene_path):
        # If scene.xml doesn't work, try gen3.xml but we'll need to handle gripper paths
        base_scene_path = 'robot_models/kinova_gen3/gen3.xml'
    
    with open(base_scene_path, 'r') as f:
        scene_content = f.read()
    
    # Generate XML for multiple obstacles
    obstacle_xml_parts = []
    
    for i, obstacle in enumerate(obstacles):
        obstacle_xml = f'''
    <!-- Virtual obstacle {i+1} visualization -->
    <body name="obstacle_{i+1}" pos="{obstacle.center[0]:.3f} {obstacle.center[1]:.3f} {obstacle.center[2]:.3f}">
        <geom name="obstacle_{i+1}_geom" type="sphere" size="{obstacle.radius}" 
              rgba="1.0 0.2 0.2 0.6" material="" contype="0" conaffinity="0"/>
        <site name="obstacle_{i+1}_center" pos="0 0 0" size="0.005" rgba="1 0 0 1"/>
    </body>
    
    '''
        obstacle_xml_parts.append(obstacle_xml)
    
    # Add target position marker
    target_xml = '''
    <!-- Target position marker -->
    <body name="target_marker" pos="0.6 -0.3 0.4">
        <geom name="target_geom" type="sphere" size="0.03" 
              rgba="0.0 0.8 0.0 0.8" material="" contype="0" conaffinity="0"/>
        <site name="target_center" pos="0 0 0" size="0.005" rgba="0 1 0 1"/>
    </body>
    
    <!-- Trajectory trace dots (initially hidden at origin) -->
    '''
    
    # Add trajectory trace dots - create enough for long trajectories
    for i in range(200):  # More dots than we'll likely need
        target_xml += f'''
    <body name="trace_dot_{i}" pos="0 0 -10">
        <geom name="trace_dot_{i}_geom" type="sphere" size="0.008" 
              rgba="0.2 0.6 1.0 0.8" material="" contype="0" conaffinity="0"/>
    </body>'''
    
    # Combine all obstacle and target XML
    all_obstacles_xml = ''.join(obstacle_xml_parts) + target_xml
    
    # Insert before the closing worldbody tag
    insert_pos = scene_content.rfind('</worldbody>')
    if insert_pos == -1:
        # Fallback - add before closing mujoco tag
        insert_pos = scene_content.rfind('</mujoco>')
        if insert_pos == -1:
            raise ValueError("Could not find insertion point in XML")
    
    modified_content = (scene_content[:insert_pos] + 
                       all_obstacles_xml + 
                       scene_content[insert_pos:])
    
    # Save to new file
    output_path = 'robot_models/kinova_gen3/trajopt_multi_obstacle_scene.xml'
    with open(output_path, 'w') as f:
        f.write(modified_content)
    
    return output_path


def plan_trajectory_with_multi_obstacle_avoidance(model, data, kinematics):
    """Plan trajectory using TrajOpt with multi-obstacle avoidance"""
    
    # Define start configuration
    start_config = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0])  # Home position
    target_position = np.array([0.6, -0.3, 0.4])  # This matches the green sphere position in XML
    
    # Get start end-effector position for reference
    start_ee_pos, _ = kinematics.forward_kinematics(start_config)    
    goal_config, ik_success = kinematics.inverse_kinematics(
        target_position,
        initial_guess=start_config,
        tolerance=0.001,
        max_iterations=2000
    ) # Use inverse kinematics to find goal configuration
    
    if not ik_success:
        print("IK failed - target may be unreachable")
        return None
    
    print("IK succeeded!")
    print(f"Goal joint configuration: {[f'{x:.3f}' for x in goal_config]}")
    
    # Verify the IK solution
    goal_ee_pos, _ = kinematics.forward_kinematics(goal_config)
    ik_error = np.linalg.norm(goal_ee_pos - target_position)
    
    # Create TrajOpt planner
    planner = ConstrainedTrajOptPlanner(
        model, 
        data, 
        n_waypoints=50, 
        dt=0.1,
        max_velocity=1.0,
        max_acceleration=0.5
    )  # Reasonable waypoints and dt
    
    try:
        strategy_choice = "1"
        if strategy_choice == '1':
            # RISKY Strategy: Prioritize short paths
            print("\nRISKY Strategy Selected: Prioritizing trajectory length")
            length_cost = TrajectoryLengthCostFunction(weight=10.0)  # High weight for short paths
            planner.add_cost_function(length_cost)
            print("  ✓ Trajectory length minimization (weight: 10.0 - HIGH)")
            
            safety_weight = 0.1  # Very low safety importance
            safety_description = "minimal"
        else:
            # SAFE Strategy: Prioritize safety (default)
            print("\nSAFE Strategy Selected: Prioritizing safety importance")
            length_cost = TrajectoryLengthCostFunction(weight=0.01)  # Low weight for path length
            planner.add_cost_function(length_cost)
            print("  ✓ Trajectory length minimization (weight: 1.0 - LOW)")
            
            # Add safety importance cost function
            safety_cost = SafetyImportanceCostFunction(
                kinematics_solver=kinematics,
                obstacles=obstacles,
                weight=300.0,  # High weight for safety
                safety_radius_multiplier=4.0
            )
            planner.add_cost_function(safety_cost)
            print("  ✓ Safety importance (weight: 20.0 - HIGH)")
            
            safety_weight = 300.0
            safety_description = "high"
            
    except (KeyboardInterrupt, EOFError):
        # Default to safe strategy
        print("\n🔵 SAFE Strategy Selected (default)")
        length_cost = TrajectoryLengthCostFunction(weight=1.0)
        planner.add_cost_function(length_cost)
        print("  ✓ Trajectory length minimization (weight: 1.0 - LOW)")
        
        safety_cost = SafetyImportanceCostFunction(
            kinematics_solver=kinematics,
            obstacles=obstacles,
            weight=20.0,
            safety_radius_multiplier=3.0
        )
        planner.add_cost_function(safety_cost)
        print("  ✓ Safety importance (weight: 20.0 - HIGH)")
        
        safety_weight = 20.0
        safety_description = "high"
    
    # 2. Multi-obstacle avoidance (moderate weight for stability)
    obstacle_cost = ObstacleAvoidanceCostFunction(
        kinematics_solver=kinematics,
        obstacles=obstacles,
        weight=200.0  # Reduced weight to prevent numerical issues
    )
    planner.add_cost_function(obstacle_cost)
    
    # Note: Velocity and acceleration are now handled as constraints, not cost functions
    print("  ✓ Velocity constraints (max 2.0 rad/s)")
    print("  ✓ Acceleration constraints (max 10.0 rad/s²)")
    
    # Plan trajectory from start_config to goal_config (found by IK)
    start_time = time.time()
    trajectory, success = planner.plan(start_config, goal_config)
    planning_time = time.time() - start_time
    
    if success:
        print(f"✅ TrajOpt planning successful!")
        print(f"  Planning time: {planning_time:.2f}s")
        print(f"  Trajectory waypoints: {len(trajectory)}")
        
        # Analyze the planned trajectory
        total_length = 0.0
        min_obstacle_distances = [float('inf')] * len(obstacles)
        max_velocity = 0.0
        
        for i in range(len(trajectory) - 1):
            # Joint space distance
            joint_distance = np.linalg.norm(trajectory[i+1] - trajectory[i])
            total_length += joint_distance
            
            # Velocity (approximation)
            velocity = joint_distance / 0.01  # dt = 0.01
            max_velocity = max(max_velocity, float(velocity))
            
            # Check distances to all obstacles
            ee_pos, _ = kinematics.forward_kinematics(trajectory[i])
            for j, obstacle in enumerate(obstacles):
                obstacle_distance = np.linalg.norm(ee_pos - obstacle.center) - obstacle.radius
                min_obstacle_distances[j] = min(min_obstacle_distances[j], float(obstacle_distance))
        
        print(f"  Total joint space length: {total_length:.3f} rad")
        print(f"  Max joint velocity: {max_velocity:.3f} rad/s")
        
        # Check safety for all obstacles
        all_safe = True
        for j, (obstacle, min_dist) in enumerate(zip(obstacles, min_obstacle_distances)):
            safety_status = "✅ Safe" if min_dist >= obstacle.safe_distance else "⚠ Close"
            print(f"  Obstacle {j+1} min distance: {min_dist:.3f}m {safety_status}")
            if min_dist < obstacle.safe_distance:
                all_safe = False
        
        if all_safe:
            print("  ✅ Trajectory maintains safe distance from ALL obstacles")
        else:
            print("  ⚠ Warning: Trajectory too close to one or more obstacles!")
        
        # Verify final position accuracy
        final_ee_pos, _ = kinematics.forward_kinematics(trajectory[-1])
        final_error = np.linalg.norm(final_ee_pos - target_position)
        print(f"  Final positioning error: {final_error*1000:.1f}mm")
        
        return trajectory
    else:
        print("❌ TrajOpt planning failed!")
        return None


def execute_trajectory_in_current_viewer(viewer_handle, model, data, kinematics, trajectory):
    """Execute trajectory in the already open MuJoCo viewer with tracing and replay"""
    # Get target position from the scene (green sphere)
    target_position = np.array([0.6, -0.3, 0.4])  # Matches green sphere position
    
    def clear_trajectory_trace():
        """Hide all trajectory trace dots"""
        for i in range(200):
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"trace_dot_{i}")
                if body_id >= 0:
                    model.body_pos[body_id] = [0, 0, -10]  # Move far below ground
            except:
                pass  # Body doesn't exist
    
    def update_trajectory_trace(ee_positions):
        """Update trajectory trace dots to show EE path"""
        clear_trajectory_trace()
        for i, pos in enumerate(ee_positions[-200:]):  # Show last 200 points
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"trace_dot_{i}")
                if body_id >= 0:
                    model.body_pos[body_id] = pos
            except:
                pass
    
    def execute_single_trajectory():
        """Execute the trajectory once with tracing"""
        for i, obstacle in enumerate(obstacles):
            print(f"    • Obstacle {i+1}: center=({obstacle.center[0]:.2f}, {obstacle.center[1]:.2f}, {obstacle.center[2]:.2f}), "
                  f"radius={obstacle.radius:.3f}m")
        print()
        
        # Clear any existing trace
        clear_trajectory_trace()
        
        # Start at initial position
        data.qpos[:7] = trajectory[0]
        data.ctrl[:7] = trajectory[0]
        mujoco.mj_forward(model, data)
        viewer_handle.sync()
        time.sleep(2)
        
        print("Executing optimized multi-obstacle avoidance trajectory...")
        
        # Track EE positions for tracing
        ee_positions = []
        
        # Execute trajectory with analysis and tracing
        for i, waypoint in enumerate(trajectory):
            # Set robot configuration
            data.qpos[:7] = waypoint
            data.ctrl[:7] = waypoint
            
            # Get current EE position
            current_ee_pos, _ = kinematics.forward_kinematics(waypoint)
            ee_positions.append(current_ee_pos.copy())
            
            # Update trajectory trace every few waypoints for smooth visual
            if i % max(1, len(trajectory)//50) == 0 or i == len(trajectory)-1:
                update_trajectory_trace(ee_positions)
            
            # Step simulation for smooth visualization
            for _ in range(8):  # 8 steps per waypoint
                mujoco.mj_step(model, data)
                viewer_handle.sync()
                time.sleep(0.002)  # Smooth motion
            
            # Real-time analysis
            distance_to_target = np.linalg.norm(current_ee_pos - target_position)
            
            # Calculate distances to all obstacles
            obstacle_distances = []
            for j, obstacle in enumerate(obstacles):
                dist_to_surface = np.linalg.norm(current_ee_pos - obstacle.center) - obstacle.radius
                obstacle_distances.append(dist_to_surface)
            
            # Show progress every few waypoints
            if i % max(1, len(trajectory)//10) == 0 or i == len(trajectory)-1:
                # Find the closest obstacle
                closest_obs_idx = np.argmin(obstacle_distances)
                closest_dist = obstacle_distances[closest_obs_idx]
                safety_status = "✅ Safe" if closest_dist >= obstacles[closest_obs_idx].safe_distance else "⚠ CLOSE"
        
        # Final update of trajectory trace
        update_trajectory_trace(ee_positions)
        
        # Final verification
        final_ee_pos, _ = kinematics.forward_kinematics(trajectory[-1])
        final_target_error = np.linalg.norm(final_ee_pos - target_position)
        
        print(f"  Target reached! Final error: {final_target_error*1000:.1f}mm")
        print(f"  End-effector final position: [{final_ee_pos[0]:.3f}, {final_ee_pos[1]:.3f}, {final_ee_pos[2]:.3f}]")
        print(f"  Target position:           [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        
        # Final distance to all obstacles
        print("  Final distances to all obstacles:")
        for j, obstacle in enumerate(obstacles):
            final_dist = np.linalg.norm(final_ee_pos - obstacle.center) - obstacle.radius
            safety_status = "✅ Safe" if final_dist >= obstacle.safe_distance else "⚠ Close"
            print(f"    Obstacle {j+1}: {final_dist:.3f}m {safety_status}")
        
        if final_target_error < 0.02:  # 2cm tolerance
            print("  ✅ SUCCESS: Target reached within tolerance!")
        else:
            print("  ⚠ Target positioning could be improved")
    
    # Execute trajectory for the first time
    execute_single_trajectory()
    
    
    # Interactive controls
    while True:
        print("=" * 60)
        print("INTERACTIVE CONTROLS:")
        print("  [r] - Replay trajectory")
        print("  [c] - Clear trajectory trace")
        print("  [h] - Show home position")
        print("  [q] - Quit demo")
        print("=" * 60)
        
        try:
            choice = input("Enter your choice: ").strip().lower()
            
            if choice == 'r':
                print("\n🔄 Replaying trajectory...")
                execute_single_trajectory()
                
            elif choice == 'c':
                print("\n🧹 Clearing trajectory trace...")
                clear_trajectory_trace()
                mujoco.mj_forward(model, data)
                viewer_handle.sync()
                print("Trace cleared!")
                
            elif choice == 'h':
                print("\n🏠 Moving to home position...")
                home_config = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
                data.qpos[:7] = home_config
                data.ctrl[:7] = home_config
                mujoco.mj_forward(model, data)
                viewer_handle.sync()
                print("Robot moved to home position!")
                
            elif choice == 'q':
                print("\n👋 Exiting demo...")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
            # Keep simulation running
            for _ in range(10):
                mujoco.mj_step(model, data)
                viewer_handle.sync()
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n👋 Exiting demo...")
            break
        except EOFError:
            print("\n👋 Exiting demo...")
            break


def main():
    """Main demo function"""
    print("🤖 TrajOpt Multi-Obstacle Avoidance Demo")
    print("=" * 60)
    print(f"Demonstrating TrajOpt with {len(obstacles)} obstacles:")
    for i, obs in enumerate(obstacles):
        print(f"  {i+1}. Center: ({obs.center[0]:.2f}, {obs.center[1]:.2f}, {obs.center[2]:.2f}), "
              f"Radius: {obs.radius:.3f}m, Safety: {obs.safe_distance:.3f}m")
    print("=" * 60)
    
    try:
        # Create scene with visual obstacles
        print("Setting up multi-obstacle demo scene...")
        model_path = create_scene_with_virtual_obstacles()
        print(f"✓ Created scene: {model_path}")
        
        # Load model and initialize components
        print("Loading robot model...")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        kinematics = KinematicsSolver(model_path)
        print("✓ Model loaded successfully")
        
        # Launch viewer early to show setup
        print("\n🎯 Launching viewer - you can see the setup before planning begins...")
        print("Red spheres: Obstacles to avoid")
        print("Green sphere: Target position")
        print("Robot will start planning trajectory in a moment...")
        
        with mujoco.viewer.launch_passive(model, data) as viewer_handle:
            # Set robot to home position for visualization
            home_config = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
            data.qpos[:7] = home_config
            data.ctrl[:7] = home_config
            mujoco.mj_forward(model, data)
            viewer_handle.sync()
            
            print("\nViewer launched! Take a moment to inspect the scene...")
            time.sleep(3)
            
            # Plan trajectory with multi-obstacle avoidance
            print("\n" + "="*50)
            print("STARTING TRAJECTORY PLANNING...")
            print("="*50)
            trajectory = plan_trajectory_with_multi_obstacle_avoidance(model, data, kinematics)
            
            if trajectory is not None:
                print("\n" + "="*50)
                print("PLANNING COMPLETE - EXECUTING TRAJECTORY...")
                print("="*50)
                time.sleep(2)
                
                # Execute and visualize trajectory in the same viewer
                execute_trajectory_in_current_viewer(viewer_handle, model, data, kinematics, trajectory)
            else:
                print("Cannot proceed with visualization due to planning failure.")
                print("Press Ctrl+C to exit...")
                try:
                    while True:
                        mujoco.mj_step(model, data)
                        viewer_handle.sync()
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\nExiting demo...")
            
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required files")
        print(f"   {e}")
        print("   Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
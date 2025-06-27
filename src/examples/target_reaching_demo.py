#!/usr/bin/env python3
"""
Enhanced target reaching demonstration with RRT planning and gripper center targeting
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
from motion_planning.planners import RRTPlanner

def create_target_scene_with_offset():
    """Create a target scene with sphere positioned to be reachable by gripper center"""
    
    # Read the base scene
    with open('robot_models/kinova_gen3/gen3_with_gripper_scene.xml', 'r') as f:
        scene_content = f.read()
    
    # Add target sphere positioned for gripper center reachability
    # Use the tested reachable position from our successful demo
    target_pos = "-0.7087273 -0.31248501 0.77091202"
    # target_pos = "-0.508727?3 0.21248501 0.67091202"
    target_sphere = f'''
    <!-- Target sphere for gripper center -->
    <body name="target_sphere" pos="{target_pos}">
        <geom name="target_sphere_geom" type="sphere" size="0.03" 
              rgba="0.0 0.8 0.0 0.3" material="" contype="0" conaffinity="0"/>
    </body>
    '''
    
    # Insert before the closing worldbody tag
    insert_pos = scene_content.rfind('</worldbody>')
    modified_content = (scene_content[:insert_pos] + 
                       target_sphere + 
                       scene_content[insert_pos:])
    
    # Save to new file
    output_path = 'robot_models/kinova_gen3/gripper_center_target_scene.xml'
    with open(output_path, 'w') as f:
        f.write(modified_content)
    
    return output_path

def main():
    print("=== RRT Target Reaching Demo ===")
    print("Using RRT path planning to reach target")
    print()
    
    # Create scene with properly positioned target
    model_path = create_target_scene_with_offset()
    print(f"Created scene: {model_path}")
    
    # Initialize components
    ik_solver = KinematicsSolver(model_path)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Initialize RRT planner
    rrt_planner = RRTPlanner(model, data, step_size=0.01, max_iterations=1000, goal_threshold=0.1)
    
    # Set robot to home position
    home_config = np.array([0.0, 0.5, 0.0, -2.5, 0.0, 0.45, 1.57])
    data.qpos[:7] = home_config
    data.ctrl[:7] = home_config
    mujoco.mj_forward(model, data)
    
    # Get target sphere position
    target_sphere_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target_sphere')
    target_position = data.xpos[target_sphere_id].copy()
    print(f"Target sphere position: {target_position}")
    
    # Show initial distances
    gripper_center_pos, _ = ik_solver.get_end_effector_pose(home_config)
    bracelet_pos = data.xpos[8]  # bracelet_link
    
    print()
    print("Initial distances to target:")
    print(f"Gripper center: {np.linalg.norm(gripper_center_pos - target_position)*100:.1f}cm")
    print(f"Bracelet link: {np.linalg.norm(bracelet_pos - target_position)*100:.1f}cm")
    
    # Step 1: Use inverse kinematics to find target configuration
    print()
    print("Step 1: Finding target configuration with IK...")
    
    target_config, ik_success = ik_solver.inverse_kinematics(
        target_position,
        initial_guess=home_config,
        tolerance=0.01,
        max_iterations=2000
    )
    
    if not ik_success:
        print("❌ IK failed - target may be unreachable")
        return
    
    print("✅ IK succeeded for gripper center targeting!")
    print(f"Target configuration: {[f'{x:.3f}' for x in target_config]}")
    
    # Step 2: Use RRT to plan path from start to target configuration
    print()
    print("Step 2: Planning path with RRT...")
    print(f"RRT parameters:")
    print(f"  - Step size: {rrt_planner.step_size}")
    print(f"  - Max iterations: {rrt_planner.max_iterations}")
    print(f"  - Goal threshold: {rrt_planner.goal_threshold}")
    
    start_time = time.time()
    trajectory, planning_success = rrt_planner.plan(home_config, target_config)
    planning_time = time.time() - start_time
    
    if not planning_success:
        print("❌ RRT planning failed")
        return
    
    print(f"✅ RRT planning succeeded!")
    print(f"  Planning time: {planning_time:.3f}s")
    print(f"  Path length: {len(trajectory)} waypoints")
    
    # Analyze trajectory
    if len(trajectory) >= 2:
        trajectory_array = np.array(trajectory)
        distances = [np.linalg.norm(trajectory_array[i+1] - trajectory_array[i]) 
                    for i in range(len(trajectory_array)-1)]
        total_distance = sum(distances)
        max_step = max(distances) if distances else 0
        avg_step = np.mean(distances) if distances else 0
        
        print(f"  Total joint space distance: {total_distance:.3f} rad")
        print(f"  Max step size: {max_step:.3f} rad")
        print(f"  Avg step size: {avg_step:.3f} rad")
    
    # Step 3: Execute the RRT path
    print()
    print("Step 3: Executing RRT path...")
    
    # Create viewer for demonstration
    with mujoco.viewer.launch_passive(model, data) as viewer_handle:
        print()
        print("🎯 RRT Path Execution:")
        print("Starting at home position...")
        viewer_handle.sync()
        time.sleep(2)
        
        print("Executing RRT path...")
        print("Notice the path found by the RRT algorithm!")
        
        # Execute the RRT trajectory
        for i, waypoint in enumerate(trajectory):
            data.qpos[:7] = waypoint
            data.ctrl[:7] = waypoint
            
            # Step simulation multiple times for smooth visualization
            for _ in range(8):  # 8 steps per waypoint for smooth motion
                mujoco.mj_step(model, data)
                viewer_handle.sync()
                time.sleep(0.02)  # Slower for better visualization
            
            # Show progress
            if i % max(1, len(trajectory)//4) == 0:
                current_gripper_pos, _ = ik_solver.get_end_effector_pose(waypoint)
                distance_to_target = np.linalg.norm(current_gripper_pos - target_position)
                print(f"  Waypoint {i+1}/{len(trajectory)}: Distance to target = {distance_to_target*100:.1f}cm")
        
        # Check final accuracy
        final_gripper_pos, _ = ik_solver.get_end_effector_pose(trajectory[-1])
        final_error = np.linalg.norm(final_gripper_pos - target_position)
        
        print(f"Target reached!")
        print(f"   Final gripper center error: {final_error*100:.1f}cm")
        print()
        print("🎉 RRT path execution complete!")
        print("   Key features of RRT:")
        print("   ✅ Explores configuration space randomly")
        print("   ✅ Can handle complex obstacles")
        print("   ✅ Probabilistically complete")
        print("   ✅ Fast planning for most problems")
        print()
        print("Press Ctrl+C to exit...")
        
        try:
            while True:
                mujoco.mj_step(model, data)
                viewer_handle.sync()
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("Exiting...")

if __name__ == "__main__":
    main() 
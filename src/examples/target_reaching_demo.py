#!/usr/bin/env python3
"""
Enhanced target reaching demonstration with gripper center targeting
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
    print("=== Gripper Center Target Reaching Demo ===")
    print()
    
    # Create scene with properly positioned target
    model_path = create_target_scene_with_offset()
    print(f"Created scene: {model_path}")
    
    # Initialize kinematics solver
    ik_solver = KinematicsSolver(model_path)
    
    # Load model for simulation
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
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
    
    # Attempt to reach target with gripper center
    print()
    print("Attempting to reach target with gripper center...")
    
    target_config, ik_success = ik_solver.inverse_kinematics(
        target_position,
        initial_guess=home_config,
        tolerance=0.01,  # 3cm tolerance
        max_iterations=2000
    )
    
    if ik_success:
        print("✅ IK succeeded for gripper center targeting!")
        
        # Create viewer for demonstration
        with mujoco.viewer.launch_passive(model, data) as viewer_handle:
            print()
            print("🎯 Demonstration sequence:")
            print("Starting at home position...")
            viewer_handle.sync()
            time.sleep(2)
            
            # Move to target configuration gradually
            print("Moving to target position...")
            
            # Interpolate between current and target configuration
            steps = 100
            for i in range(steps + 1):
                alpha = i / steps
                current_config = (1 - alpha) * home_config + alpha * target_config
                
                data.qpos[:7] = current_config
                data.ctrl[:7] = current_config
                mujoco.mj_step(model, data)
                viewer_handle.sync()
                time.sleep(0.02)
            
            # Check final accuracy
            final_gripper_pos, _ = ik_solver.get_end_effector_pose(target_config)
            final_error = np.linalg.norm(final_gripper_pos - target_position)
            
            print(f"Target reached!")
            print(f"   Final gripper center error: {final_error*100:.1f}cm")
            print("Press Ctrl+C to exit...")
            
            try:
                while True:
                    mujoco.mj_step(model, data)
                    viewer_handle.sync()
                    time.sleep(0.01)
            except KeyboardInterrupt:
                print("Exiting...")
                
    else:
        print("❌ IK failed - target may be unreachable")
        
        # Still show the scene for debugging
        with mujoco.viewer.launch_passive(model, data) as viewer_handle:
            print("Opening viewer for debugging...")
            try:
                while True:
                    mujoco.mj_step(model, data)
                    viewer_handle.sync()
                    time.sleep(0.01)
            except KeyboardInterrupt:
                print("Exiting...")

if __name__ == "__main__":
    main() 
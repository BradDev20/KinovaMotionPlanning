#!/usr/bin/env python3
"""
Displays Kinova Gen3 robotic arm with Robotiq gripper
"""
import mujoco
import os
import time
import numpy as np
import sys

# Check to see if the robot model is in this directory.
xml_path = "robot_models/kinova_gen3/scene.xml"
if not os.path.exists(xml_path):
    xml_path = "../../robot_models/kinova_gen3/scene.xml"

class gen3_env(object):
    def __init__(self, headless=False):
        # Load the model and create data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.headless = headless
        
        # Motion control parameters
        self.motion_enabled = True
        self.step_count = 0
        self.increment_per_step = 0.1 * np.pi / 180  # 0.1 degrees in radians
        
        # Set initial robot position
        self.set_initial_position("home")

    def set_initial_position(self, preset="home"):
        """Set the robot to a reasonable initial configuration
        
        Available presets:
        - 'upright': Arm pointing up
        - 'horizontal': Arm extended horizontally
        - 'home': Comfortable starting position
        """
        
        if preset == "upright":
            arm_initial_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif preset == "horizontal":
            arm_initial_positions = [0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0]  # 90 degrees
        elif preset == "home":
            arm_initial_positions = [0.0, 0.5, 0.0, -2.5, 0.0, 0.45, 1.57]
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
        # Set arm joint positions (joints 0-6)
        for i, angle in enumerate(arm_initial_positions):
            self.data.qpos[i] = angle
        
        # IMPORTANT: Set control signals to maintain these positions!
        # This prevents the PD controllers from driving back to zero
        for i, angle in enumerate(arm_initial_positions):
            self.data.ctrl[i] = angle
        
        # Set gripper to open position (joints 7-14)
        gripper_position = 0.0  # 0.0 = open, 0.8 = closed
        self.data.qpos[7] = gripper_position   # right_driver_joint
        self.data.qpos[11] = gripper_position  # left_driver_joint
        self.data.ctrl[7] = gripper_position   # gripper actuator control
        
        # Forward kinematics to update the robot pose
        mujoco.mj_forward(self.model, self.data)
        
        print(f"Robot set to '{preset}' position:")
        print(f"  Arm joints: {[f'{x:.3f}' for x in arm_initial_positions]}")
        print(f"  Control signals: {[f'{x:.3f}' for x in self.data.ctrl[:7]]}")
        print(f"  Gripper: {'closed' if gripper_position > 0.4 else 'open'}")

    def start_viewer(self):
        """Start the interactive viewer (requires mjpython on macOS)"""
        if self.headless:
            print("Running in headless mode - no viewer will be started")
            return
            
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("Viewer started successfully")
        except RuntimeError as e:
            if "mjpython" in str(e):
                print("Interactive viewer requires mjpython on macOS.")
                print("Running in headless mode instead.")
                print("To use the viewer, run with: mjpython gen3.py")
                self.headless = True
            else:
                raise e

    def update_motion(self):
        """Demonstrate incremental joint motion by updating control signals"""
        if not self.motion_enabled:
            return
            
        # Example 1: Slowly rotate joint 1 (base) back and forth
        max_angle = 45 * np.pi / 180  # 45 degrees in radians
        joint_1_target = max_angle * np.sin(self.step_count * 0.01)
        self.data.ctrl[0] = joint_1_target
        
        # Example 2: Slowly move joint 2 (shoulder) up and down
        base_angle_2 = 0.5  # Starting position from "home"
        amplitude_2 = 0.3   # ±0.3 radians (~17 degrees)
        joint_2_target = base_angle_2 + amplitude_2 * np.sin(self.step_count * 0.005)
        self.data.ctrl[1] = joint_2_target
        
        # Example 3: Increment joint 6 (wrist) continuously with bounds
        if self.step_count % 100 == 0:  # Update every 100 steps
            current_target = self.data.ctrl[5]
            new_target = current_target + 10 * self.increment_per_step  # 1 degree increment
            
            # Keep within reasonable bounds (-90 to +90 degrees)
            max_wrist = 90 * np.pi / 180
            if new_target > max_wrist:
                new_target = -max_wrist  # Wrap around
            
            self.data.ctrl[5] = new_target
            
            if self.step_count % 500 == 0:  # Print occasionally
                print(f"Step {self.step_count}: Updated joint targets:")
                print(f"  Joint 1 (base): {joint_1_target:.3f} rad ({joint_1_target*180/np.pi:.1f}°)")
                print(f"  Joint 2 (shoulder): {joint_2_target:.3f} rad ({joint_2_target*180/np.pi:.1f}°)")
                print(f"  Joint 6 (wrist): {new_target:.3f} rad ({new_target*180/np.pi:.1f}°)")

    def step(self):
        # Update control signals for motion demonstration
        self.update_motion()
        
        # Step the simulation
        mujoco.mj_step(self.model, self.data)
        
        # Increment step counter
        self.step_count += 1
        
        # Update the viewer if it exists
        if self.viewer is not None:
            self.viewer.sync()

    def render_offscreen(self, width=640, height=480, camera_name=None):
        """Render the scene offscreen (works in headless mode)"""
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        if camera_name:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            renderer.update_scene(self.data, camera=camera_id)
        else:
            renderer.update_scene(self.data)
        return renderer.render()

    def close(self):
        """Close the viewer"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

if __name__ == '__main__':
    # Check if running under mjpython or if headless mode is requested
    # headless = os.getenv('HEADLESS') is not None or 'mjpython' not in sys.executable
    headless = False

    env = gen3_env(headless=headless)
    env.start_viewer()
    
    t = 0
    print("=" * 60)
    print("JOINT MOTION CONTROL DEMONSTRATION")
    print("=" * 60)
    print("The robot will demonstrate three types of motion:")
    print("1. Joint 1 (base): Sinusoidal rotation ±45°")
    print("2. Joint 2 (shoulder): Sinusoidal motion ±17°")
    print("3. Joint 6 (wrist): Incremental 1° steps every 100 timesteps")
    print("\nPress Ctrl+C to stop the simulation.")
    print("=" * 60)
    
    try:
        while True:
            env.step()
            if not headless:
                time.sleep(0.01)  # Small delay for visualization when using viewer
            t += 1
            
            # Stop after reasonable demonstration time
            if t > 2000:
                break
                
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        env.close()
        print(f"Simulation completed after {t} steps")
        print("Motion demonstration finished!")

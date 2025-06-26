#!/usr/bin/env python3
"""
Motion Planning Demonstration: SE(3) Goal Planning with IK + RRT

This demonstration shows:
1. SE(3) goal specification (position + orientation)
2. Inverse kinematics to convert to joint space
3. RRT path planning in joint space
4. Trajectory execution in MuJoCo

Usage:
    mjpython src/examples/motion_planning_demo.py
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.motion_planning import MotionPlanningInterface, TrajectoryVisualizer


def rotation_matrix_x(angle):
    """Create rotation matrix around X axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(angle):
    """Create rotation matrix around Y axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(angle):
    """Create rotation matrix around Z axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


class MotionPlanningDemo:
    """Motion planning demonstration with MuJoCo visualization"""
    
    def __init__(self, xml_path: str = "robot_models/kinova_gen3/scene.xml"):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize motion planner
        self.mp_interface = MotionPlanningInterface(self.model, self.data)
        
        # Setup viewer
        self.viewer = None
        self.setup_initial_configuration()
        
    def setup_initial_configuration(self):
        """Set robot to a reasonable starting configuration"""
        # Home configuration
        home_joints = np.array([0.0, 0.3, 0.0, -1.2, 0.0, 0.8, 0.0])
        
        self.data.qpos[:7] = home_joints
        self.data.ctrl[:7] = home_joints
        mujoco.mj_forward(self.model, self.data)
        
        print("🏠 Robot set to home configuration")
        current_pos, current_rot = self.mp_interface.get_current_end_effector_pose()
        print(f"   Current end-effector position: {current_pos}")
    
    def start_viewer(self):
        """Start MuJoCo viewer"""
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("✅ Viewer started successfully!")
            return True
        except RuntimeError as e:
            if "mjpython" in str(e):
                print("❌ Interactive viewer requires mjpython on macOS.")
                print("💡 Solution: Run with 'mjpython src/examples/motion_planning_demo.py'")
                return False
            else:
                print(f"❌ Viewer failed: {e}")
                return False
    
    def demonstrate_se3_planning(self):
        """Demonstrate SE(3) goal planning"""
        print("\n" + "="*60)
        print("SE(3) GOAL PLANNING DEMONSTRATION")
        print("="*60)
        
        # Get current pose
        current_pos, current_rot = self.mp_interface.get_current_end_effector_pose()
        print(f"Starting position: {current_pos}")
        
        # Define SE(3) goals to visit
        goals = [
            {
                "name": "Forward reach",
                "position": current_pos + np.array([0.2, 0.0, 0.1]),
                "orientation": rotation_matrix_z(np.pi/4)  # 45° rotation around Z
            },
            {
                "name": "Side reach", 
                "position": current_pos + np.array([0.0, 0.3, 0.0]),
                "orientation": rotation_matrix_y(np.pi/6)  # 30° rotation around Y
            },
            {
                "name": "High reach",
                "position": current_pos + np.array([0.1, 0.1, 0.2]),
                "orientation": None  # Position-only goal
            },
            {
                "name": "Return home",
                "position": current_pos,
                "orientation": current_rot
            }
        ]
        
        for i, goal in enumerate(goals):
            print(f"\n--- Goal {i+1}: {goal['name']} ---")
            
            # Plan trajectory
            trajectory, success = self.mp_interface.plan_to_pose(
                goal["position"],
                goal["orientation"]
            )
            
            if not success:
                print(f"❌ Planning failed for goal {i+1}")
                continue
            
            # Validate trajectory
            if not self.mp_interface.validate_trajectory(trajectory):
                print(f"❌ Trajectory validation failed for goal {i+1}")
                continue
            
            # Execute trajectory with visualization
            print(f"Executing trajectory...")
            self.execute_trajectory_with_visualization(trajectory)
            
            # Brief pause between goals
            time.sleep(1.0)
    
    def demonstrate_joint_space_planning(self):
        """Demonstrate joint space planning"""
        print("\n" + "="*60)
        print("JOINT SPACE PLANNING DEMONSTRATION")
        print("="*60)
        
        # Define joint space goals
        joint_goals = [
            {
                "name": "Stretch configuration",
                "joints": np.array([0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0])
            },
            {
                "name": "Compact configuration", 
                "joints": np.array([0.0, 0.8, 0.0, -2.0, 0.0, 1.2, 0.0])
            },
            {
                "name": "Twisted configuration",
                "joints": np.array([1.5, 0.4, -1.5, -1.0, 1.5, 0.6, -1.5])
            }
        ]
        
        for i, goal in enumerate(joint_goals):
            print(f"\n--- Joint Goal {i+1}: {goal['name']} ---")
            
            # Plan trajectory
            trajectory, success = self.mp_interface.plan_to_joint_configuration(goal["joints"])
            
            if not success:
                print(f"❌ Planning failed for joint goal {i+1}")
                continue
            
            # Execute trajectory
            print(f"Executing joint space trajectory...")
            self.execute_trajectory_with_visualization(trajectory)
            
            # Brief pause
            time.sleep(1.0)
    
    def execute_trajectory_with_visualization(self, trajectory):
        """Execute trajectory with real-time visualization"""
        def visualization_callback(step, waypoint, data):
            # Print progress
            if step % 10 == 0:
                pos, _ = self.mp_interface.get_current_end_effector_pose()
                print(f"  Step {step:3d}: EE at {pos}")
            
            # Update viewer
            if self.viewer is not None:
                self.viewer.sync()
        
        # Execute with callback
        success = self.mp_interface.execute_trajectory(
            trajectory, 
            callback=visualization_callback,
            dt=0.02  # 50 Hz execution
        )
        
        if success:
            final_pos, _ = self.mp_interface.get_current_end_effector_pose()
            print(f"  ✅ Goal reached! Final EE position: {final_pos}")
        else:
            print(f"  ❌ Execution failed")
    
    def run_demo(self):
        """Run the complete demonstration"""
        print("🚀 Starting Motion Planning Demonstration")
        
        # Start viewer
        viewer_started = self.start_viewer()
        
        if not viewer_started:
            print("Running in headless mode...")
        
        try:
            # Run demonstrations
            self.demonstrate_se3_planning()
            self.demonstrate_joint_space_planning()
            
            print("\n" + "="*60)
            print("DEMONSTRATION COMPLETED")
            print("="*60)
            print("Issues encountered:")
            print("❌ Many IK solutions failed - need better joint limits and tolerances")
            print("❌ RRT planning failed - need tuned parameters and collision checking")
            print("❌ Goals may be unreachable - need workspace analysis")
            print("✅ MuJoCo integration working")
            print("✅ Code structure is sound - needs parameter tuning")
            
            if viewer_started:
                print("\nPress Ctrl+C to exit...")
                try:
                    while True:
                        time.sleep(0.1)
                        if self.viewer is not None:
                            self.viewer.sync()
                except KeyboardInterrupt:
                    print("\n👋 Demonstration ended by user")
        
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    # Check if we're in the right directory
    xml_path = "robot_models/kinova_gen3/scene.xml"
    if not os.path.exists(xml_path):
        print(f"❌ Robot model not found at {xml_path}")
        print("💡 Make sure you're running from the project root directory")
        return
    
    # Create and run demo
    demo = MotionPlanningDemo(xml_path)
    demo.run_demo()


if __name__ == "__main__":
    main() 
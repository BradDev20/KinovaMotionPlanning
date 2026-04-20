#!/usr/bin/env python3
"""
Trajectory visualization helpers that are useful for demos and offline analysis.
"""

import time
from typing import Any, Callable, List, Optional

import mujoco
import numpy as np


class TrajectoryVisualizationManager:
    """Manager class for executing and visualizing trajectories in the MuJoCo viewer."""

    def __init__(self, model, data, viewer_handle, kinematics_solver):
        self.model = model
        self.data = data
        self.viewer_handle = viewer_handle
        self.kinematics = kinematics_solver
        self.current_trajectory = None
        self.target_position = None

    def set_target_position(self, target_position: np.ndarray):
        self.target_position = target_position

    def clear_trajectory_trace(self, max_dots: int = 500):
        for index in range(max_dots):
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"trace_dot_{index}")
                if body_id >= 0:
                    self.model.body_pos[body_id] = [0, 0, -10]
            except Exception:
                pass

    def update_trajectory_trace(
        self,
        ee_positions: List[np.ndarray],
        color: Optional[np.ndarray] = None,
        start_dot_offset: int = 0,
    ):
        for index, pos in enumerate(ee_positions[-5000:]):
            dot_id = start_dot_offset + index
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"trace_dot_{dot_id}")
                if body_id >= 0:
                    self.model.body_pos[body_id] = pos
                    if color is not None:
                        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"trace_dot_{dot_id}_geom")
                        if geom_id >= 0:
                            self.model.geom_rgba[geom_id] = color
            except Exception:
                pass

    def execute_trajectory(
        self,
        trajectory: List[np.ndarray],
        steps_per_waypoint: int = 8,
        step_delay: float = 0.002,
        show_trace: bool = True,
        initial_pause: float = 1.0,
        trace_color: Optional[np.ndarray] = None,
        success_override: Optional[bool] = None,
        status_label: Optional[str] = None,
        verbose: bool = True,
    ) -> dict:
        if show_trace:
            self.clear_trajectory_trace()

        self.data.qpos[:7] = trajectory[0]
        self.data.ctrl[:7] = trajectory[0]
        mujoco.mj_forward(self.model, self.data)
        self.viewer_handle.sync()
        time.sleep(initial_pause)

        if verbose:
            print(f"Executing trajectory with {len(trajectory)} waypoints...")

        ee_positions = []
        execution_start = time.time()
        for index, waypoint in enumerate(trajectory):
            self.data.qpos[:7] = waypoint
            self.data.ctrl[:7] = waypoint
            current_ee_pos, _ = self.kinematics.forward_kinematics(waypoint)
            ee_positions.append(current_ee_pos.copy())
            if show_trace and (index % max(1, len(trajectory) // 50) == 0 or index == len(trajectory) - 1):
                self.update_trajectory_trace(ee_positions, trace_color)
            for _ in range(steps_per_waypoint):
                mujoco.mj_step(self.model, self.data)
                self.viewer_handle.sync()
                time.sleep(step_delay)

        if show_trace:
            self.update_trajectory_trace(ee_positions, trace_color)

        execution_time = time.time() - execution_start
        stats = self._analyze_trajectory_execution(
            trajectory,
            ee_positions,
            execution_time,
            success_override=success_override,
            status_label=status_label,
        )
        if verbose:
            self._print_execution_results(stats)
        self.current_trajectory = trajectory
        return stats

    def _analyze_trajectory_execution(
        self,
        trajectory: List[np.ndarray],
        ee_positions: List[np.ndarray],
        execution_time: float,
        success_override: Optional[bool] = None,
        status_label: Optional[str] = None,
    ) -> dict:
        stats = {
            "execution_time": execution_time,
            "num_waypoints": len(trajectory),
            "final_ee_position": ee_positions[-1] if ee_positions else None,
            "target_error": None,
            "success": False,
            "status_label": status_label,
        }
        if self.target_position is not None and ee_positions:
            final_error = np.linalg.norm(ee_positions[-1] - self.target_position)
            stats["target_error"] = final_error
            stats["success"] = final_error < 0.02
        if success_override is not None:
            stats["success"] = bool(success_override)
        return stats

    def _print_execution_results(self, stats: dict):
        if stats["target_error"] is not None:
            print(f"Final error: {stats['target_error'] * 1000:.1f}mm")
            if stats["status_label"]:
                print(stats["status_label"])
            elif stats["success"]:
                print("SUCCESS: Target reached within tolerance!")
            else:
                print("Target positioning could be improved")
        else:
            print("Trajectory execution complete!")

    def move_to_home_position(self, home_config: np.ndarray):
        self.data.qpos[:7] = home_config
        self.data.ctrl[:7] = home_config
        mujoco.mj_forward(self.model, self.data)
        self.viewer_handle.sync()

    def interactive_controls(
        self,
        home_config: np.ndarray,
        custom_commands: Optional[dict[str, tuple[str, Callable[[], Any]]]] = None,
        show_menu: bool = True,
    ) -> bool:
        if show_menu:
            print("=" * 60)
            print("INTERACTIVE CONTROLS:")
            print("  [r] - Replay trajectory")
            print("  [c] - Clear trajectory trace")
            print("  [h] - Show home position")
            if custom_commands:
                for key, (description, _) in custom_commands.items():
                    print(f"  [{key}] - {description}")
            print("  [q] - Quit demo")
            print("=" * 60)

        try:
            choice = input("Enter your choice: ").strip().lower()
            if choice == "r" and self.current_trajectory is not None:
                print("Replaying trajectory...")
                self.execute_trajectory(self.current_trajectory)
            elif choice == "c":
                print("Clearing trajectory trace...")
                self.clear_trajectory_trace()
                mujoco.mj_forward(self.model, self.data)
                self.viewer_handle.sync()
            elif choice == "h":
                print("Moving to home position...")
                self.move_to_home_position(home_config)
            elif choice == "q":
                print("Exiting demo...")
                return True
            elif custom_commands and choice in custom_commands:
                _, callback = custom_commands[choice]
                callback()
            else:
                print("Invalid choice. Please try again.")

            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
                self.viewer_handle.sync()
                time.sleep(0.01)
            return False
        except (KeyboardInterrupt, EOFError):
            print("Exiting demo...")
            return True

    def run_interactive_session(
        self,
        trajectory: List[np.ndarray],
        home_config: np.ndarray,
        target_position: Optional[np.ndarray] = None,
        custom_commands: Optional[dict[str, tuple[str, Callable[[], Any]]]] = None,
    ):
        if target_position is not None:
            self.set_target_position(target_position)
        print("Executing trajectory...")
        self.execute_trajectory(trajectory)
        print("Trajectory execution complete!")
        while True:
            should_quit = self.interactive_controls(home_config, custom_commands)
            if should_quit:
                break


class MultiTrajectoryVisualizer(TrajectoryVisualizationManager):
    def __init__(self, model, data, viewer_handle, kinematics_solver):
        super().__init__(model, data, viewer_handle, kinematics_solver)
        self.trajectories = []
        self.trajectory_colors = []

    def add_trajectory(self, trajectory: List[np.ndarray], color: np.ndarray):
        self.trajectories.append(trajectory)
        self.trajectory_colors.append(color)

    def visualize_all_trajectories(self, verbose: bool = True):
        if verbose:
            print(f"Rendering {len(self.trajectories)} trajectories...")
        dot_offset = 0
        for trajectory, color in zip(self.trajectories, self.trajectory_colors):
            ee_positions = []
            for waypoint in trajectory:
                ee_pos, _ = self.kinematics.forward_kinematics(waypoint)
                ee_positions.append(ee_pos)
            self.update_trajectory_trace(ee_positions, color, dot_offset)
            dot_offset += len(ee_positions)
        mujoco.mj_forward(self.model, self.data)
        self.viewer_handle.sync()
        if verbose:
            print("All trajectories rendered!")


def create_trajectory_visualizer(model, data, viewer_handle, kinematics_solver, multi_trajectory: bool = False):
    if multi_trajectory:
        return MultiTrajectoryVisualizer(model, data, viewer_handle, kinematics_solver)
    return TrajectoryVisualizationManager(model, data, viewer_handle, kinematics_solver)

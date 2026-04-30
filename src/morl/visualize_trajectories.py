from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from .dataset import load_raw_records
from .planning import relative_robot_scene_path
from .schemas import TaskSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay planner and RL trajectories in the same task scene.")
    parser.add_argument("--planner-dirs", nargs="*", default=[], help="Planner dataset directories.")
    parser.add_argument("--evaluation-rollouts", nargs="*", default=[], help="Evaluation rollout pickle files.")
    parser.add_argument("--task-id", nargs="+", required=True, help="One or more task ids to visualize.")
    parser.add_argument("--alpha", type=float, default=None, help="Optional alpha to match per source.")
    parser.add_argument("--max-trajectories", type=int, default=4, help="Maximum trajectories to replay.")
    parser.add_argument("--scene-dir", type=str, default="data/morl/_visualization/scenes", help="Temporary scene output directory.")
    parser.add_argument("--no-viewer", action="store_true", help="Print chosen trajectories without launching the viewer.")
    return parser.parse_args()


def _load_rollout_records(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        records.extend(payload)
    return records


def _pick_by_source(records: list[dict], alpha: float | None, max_trajectories: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        source = str(record.get("source", record.get("planner_mode", "planner")))
        grouped.setdefault(source, []).append(record)

    selected = []
    for source, source_records in sorted(grouped.items()):
        if alpha is None:
            source_records = sorted(source_records, key=lambda item: float(item.get("scalarized_cost", item["length_cost"] + item["obstacle_cost"])))
            selected.append(source_records[0])
        else:
            selected.append(min(source_records, key=lambda item: abs(float(item["alpha"]) - alpha)))
        if len(selected) >= max_trajectories:
            break
    return selected[:max_trajectories]


def _build_overlay_scene(task: TaskSpec, max_waypoints: int, scene_dir: Path) -> str:
    from src.motion_planning.utils import Obstacle
    from src.motion_planning.scene_builder import create_standard_scene

    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_path = scene_dir / f"morl_{task.task_id}_overlay.xml"
    output_filename = relative_robot_scene_path(f"morl_{task.task_id}_overlay.xml")
    obstacles = [
        Obstacle(center=np.asarray(obstacle.center, dtype=np.float64), radius=obstacle.radius, safe_distance=obstacle.safe_distance)
        for obstacle in task.obstacles
    ]
    return create_standard_scene(
        obstacles=obstacles,
        target_position=np.asarray(task.target_position, dtype=np.float64),
        trace_dot_count=max_waypoints * 8,
        output_filename=output_filename,
    )


def _ee_positions(kinematics: KinematicsSolver, trajectory: np.ndarray) -> list[np.ndarray]:
    positions = []
    for waypoint in trajectory:
        ee_position, _ = kinematics.forward_kinematics(waypoint)
        positions.append(ee_position.copy())
    return positions


def _status_label(record: dict) -> str | None:
    if "success" not in record:
        return None
    if bool(record.get("success")):
        return "SUCCESS: Rollout completed without collision or timeout."
    if bool(record.get("collision")):
        return "FAILED: Rollout terminated due to collision."
    if bool(record.get("timeout")):
        return "FAILED: Rollout terminated due to timeout."
    return "FAILED: Rollout ended without satisfying success criteria."


def _selected_records_for_task(all_records: list[dict], task_id: str, alpha: float | None, max_trajectories: int) -> list[dict]:
    matching = [record for record in all_records if record["task_spec"]["task_id"] == task_id]
    if not matching:
        raise RuntimeError(f"No trajectories found for task {task_id}.")
    return _pick_by_source(matching, alpha=alpha, max_trajectories=max_trajectories)


def _replay_selected_task(task_id: str, selected: list[dict], scene_dir: Path, has_more_tasks: bool) -> bool:
    import glfw
    import mujoco
    import mujoco.viewer

    from src.motion_planning.kinematics import KinematicsSolver
    from src.analysis.trajectory_visualizer import TrajectoryVisualizationManager

    task = TaskSpec.from_dict(selected[0]["task_spec"])
    scene_path = _build_overlay_scene(task, max_waypoints=max(record["trajectory"].shape[0] for record in selected), scene_dir=scene_dir)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    kinematics = KinematicsSolver(scene_path)
    viewer_state = {"replay": False, "next_task": False, "quit": False}

    def _key_callback(keycode: int) -> None:
        if keycode in {glfw.KEY_ENTER, glfw.KEY_KP_ENTER}:
            viewer_state["replay"] = True
        elif has_more_tasks and keycode == glfw.KEY_N:
            viewer_state["next_task"] = True
        elif keycode in {glfw.KEY_Q, glfw.KEY_ESCAPE}:
            viewer_state["quit"] = True

    with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as viewer_handle:
        visualizer = TrajectoryVisualizationManager(model, data, viewer_handle, kinematics)
        visualizer.set_target_position(np.asarray(task.target_position, dtype=np.float32))
        visualizer.move_to_home_position(np.asarray(task.start_config, dtype=np.float32))

        trace_stride = max(max(record["trajectory"].shape[0] for record in selected) + 10, 50)
        visualizer.clear_trajectory_trace(max_dots=trace_stride * len(selected))
        for index, record in enumerate(selected):
            positions = _ee_positions(kinematics, np.asarray(record["trajectory"], dtype=np.float32))
            visualizer.update_trajectory_trace(
                positions,
                color=np.asarray(record["color"], dtype=np.float32),
                start_dot_offset=index * trace_stride,
            )
        viewer_handle.sync()
        time.sleep(0.5)

        while True:
            viewer_state["replay"] = False
            for record in selected:
                print(
                    f"Replaying {record['trajectory_id']} "
                    f"(task={task_id}, alpha={record['alpha']:.3f}, length={record['length_cost']:.4f}, obstacle={record['obstacle_cost']:.4f})"
                )
                visualizer.execute_trajectory(
                    [np.asarray(waypoint, dtype=np.float32) for waypoint in record["trajectory"]],
                    show_trace=False,
                    initial_pause=0.2,
                    trace_color=np.asarray(record["color"], dtype=np.float32),
                    success_override=record.get("success"),
                    status_label=_status_label(record),
                    verbose=True,
                )

            if has_more_tasks:
                print("Press Enter in the viewer to replay this task, N for the next task, or Q/Esc to quit.")
            else:
                print("Press Enter in the viewer to replay this task, or Q/Esc to quit.")
            try:
                while True:
                    is_running = getattr(viewer_handle, "is_running", None)
                    if callable(is_running) and not is_running():
                        return False
                    if viewer_state["quit"]:
                        return False
                    if has_more_tasks and viewer_state["next_task"]:
                        viewer_state["next_task"] = False
                        return True
                    if viewer_state["replay"]:
                        break
                    mujoco.mj_step(model, data)
                    viewer_handle.sync()
                    time.sleep(0.01)
            except KeyboardInterrupt:
                return False


def main(argv: list[str] | None = None) -> None:
    import sys

    if argv is not None:
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0], *argv]
    try:
        args = parse_args()
    finally:
        if argv is not None:
            sys.argv = original_argv
    planner_records = []
    for planner_dir in args.planner_dirs:
        planner_records.extend(load_raw_records(planner_dir))
    evaluation_records = _load_rollout_records(args.evaluation_rollouts)
    all_records = planner_records + evaluation_records
    task_ids = [str(task_id) for task_id in args.task_id]
    if args.no_viewer:
        for task_id in task_ids:
            selected = _selected_records_for_task(all_records, task_id, alpha=args.alpha, max_trajectories=args.max_trajectories)
            for record in selected:
                print(
                    f"{record['trajectory_id']} "
                    f"task={task_id} alpha={record['alpha']:.3f} length={record['length_cost']:.4f} obstacle={record['obstacle_cost']:.4f}"
                )
        return

    scene_dir = Path(args.scene_dir)
    for index, task_id in enumerate(task_ids):
        selected = _selected_records_for_task(all_records, task_id, alpha=args.alpha, max_trajectories=args.max_trajectories)
        should_continue = _replay_selected_task(task_id, selected, scene_dir=scene_dir, has_more_tasks=index < len(task_ids) - 1)
        if not should_continue:
            break


if __name__ == "__main__":
    main()

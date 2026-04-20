from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.motion_planning.utils import Obstacle


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class MujocoSceneBuilder:
    """Build MuJoCo scenes with obstacles, targets, and trace markers."""

    def __init__(self, base_scene_path: str | Path | None = None):
        self._repo_root = _repo_root()
        default_base = Path("robot_models/kinova_gen3/scene.xml")
        self.base_scene_path = Path(base_scene_path) if base_scene_path is not None else default_base
        self.xml_elements: list[str] = []

    def add_obstacles(self, obstacles: list["Obstacle"]) -> "MujocoSceneBuilder":
        for index, obstacle in enumerate(obstacles, start=1):
            contact_margin = max(float(obstacle.safe_distance), 1e-4)
            obstacle_xml = f"""
    <!-- Physical obstacle {index} -->
    <body name="obstacle_{index}" pos="{obstacle.center[0]:.3f} {obstacle.center[1]:.3f} {obstacle.center[2]:.3f}">
        <geom name="obstacle_{index}_geom" type="sphere" size="{obstacle.radius}"
              rgba="1.0 0.2 0.2 0.6" material="" contype="1" conaffinity="1" margin="{contact_margin:.5f}"/>
        <site name="obstacle_{index}_center" pos="0 0 0" size="0.005" rgba="1 0 0 1"/>
    </body>
    """
            self.xml_elements.append(obstacle_xml)
        return self

    def add_target(self, position: np.ndarray, size: float = 0.03) -> "MujocoSceneBuilder":
        target_xml = f"""
    <!-- Target position marker -->
    <body name="target_marker" pos="{position[0]} {position[1]} {position[2]}">
        <geom name="target_geom" type="sphere" size="{size}"
              rgba="0.0 0.8 0.0 0.8" material="" contype="0" conaffinity="0"/>
        <site name="target_center" pos="0 0 0" size="0.005" rgba="0 1 0 1"/>
    </body>
    """
        self.xml_elements.append(target_xml)
        return self

    def add_trace_dots(
        self,
        count: int,
        initial_position: np.ndarray | None = None,
        default_color: str | None = None,
    ) -> "MujocoSceneBuilder":
        if initial_position is None:
            initial_position = np.array([0.0, 0.0, -10.0], dtype=np.float64)
        if default_color is None:
            default_color = "0.2 0.6 1.0 0.8"
        self.xml_elements.append("\n    <!-- Trajectory trace dots (initially hidden at origin) -->")
        for index in range(count):
            trace_xml = f"""
    <body name="trace_dot_{index}" pos="{initial_position[0]} {initial_position[1]} {initial_position[2]}">
        <geom name="trace_dot_{index}_geom" type="sphere" size="0.008"
              rgba="{default_color}" material="" contype="0" conaffinity="0"/>
    </body>"""
            self.xml_elements.append(trace_xml)
        return self

    def build_scene(self, output_filename: str) -> str:
        base_scene_path = self._resolve_base_scene_path()
        scene_content = base_scene_path.read_text(encoding="utf-8")
        insert_pos = scene_content.rfind("</worldbody>")
        if insert_pos == -1:
            insert_pos = scene_content.rfind("</mujoco>")
            if insert_pos == -1:
                raise ValueError("Could not find insertion point in XML")
        modified_content = scene_content[:insert_pos] + "".join(self.xml_elements) + scene_content[insert_pos:]
        output_path = self._repo_root / "robot_models" / "kinova_gen3" / output_filename
        output_path.write_text(modified_content, encoding="utf-8")
        return output_path.relative_to(self._repo_root).as_posix()

    def reset(self) -> "MujocoSceneBuilder":
        self.xml_elements = []
        return self

    def _resolve_base_scene_path(self) -> Path:
        candidate = self.base_scene_path
        if not candidate.is_absolute():
            candidate = self._repo_root / candidate
        if candidate.exists():
            return candidate
        fallback = self._repo_root / "robot_models" / "kinova_gen3" / "gen3.xml"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Neither {candidate} nor {fallback} found")


def create_standard_scene(
    obstacles: list["Obstacle"],
    target_position: np.ndarray,
    trace_dot_count: int = 200,
    output_filename: str = "trajopt_scene.xml",
) -> str:
    builder = MujocoSceneBuilder()
    return (
        builder.add_obstacles(obstacles)
        .add_target(target_position)
        .add_trace_dots(trace_dot_count)
        .build_scene(output_filename)
    )


def create_pareto_scene(
    obstacles: list["Obstacle"],
    target_position: np.ndarray,
    max_trajectories: int,
    output_filename: str = "pareto_search_scene.xml",
) -> str:
    max_dots_needed = max_trajectories * 10
    builder = MujocoSceneBuilder()
    return (
        builder.add_obstacles(obstacles)
        .add_target(target_position)
        .add_trace_dots(max_dots_needed, default_color="1.0 1.0 1.0 0.8")
        .build_scene(output_filename)
    )

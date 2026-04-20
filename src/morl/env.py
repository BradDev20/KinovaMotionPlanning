from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from src.motion_planning.cost_functions import MuJoCoRobotObstacleCost
from src.motion_planning.kinematics import KinematicsSolver
from src.motion_planning.planners import MotionPlannerFactory
from src.motion_planning.utils import Obstacle

from .config import EnvConfig
from .planning import build_task_scene
from .schemas import TaskSpec
from .tasks import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER, MAX_OBSTACLES


class KinovaMORLEnv:
    def __init__(self, task: TaskSpec, scene_dir: str | Path, env_config: EnvConfig | None = None):
        self.task = task
        self.scene_dir = Path(scene_dir)
        self.config = env_config or EnvConfig()
        self.scene_path = build_task_scene(task, self.scene_dir)

        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)
        self.kinematics = KinematicsSolver(self.scene_path)
        self.contact_checker = MotionPlannerFactory.create_collision_checker(self.model, self.data)
        self.obstacles = [
            Obstacle(center=obstacle.center_array(), radius=obstacle.radius, safe_distance=obstacle.safe_distance)
            for obstacle in task.obstacles
        ]
        self.obstacle_cost = MuJoCoRobotObstacleCost(
            self.model,
            self.data,
            weight=1.0,
            aggregate="avg",
            collision_penalty=3.0,
        )
        self.max_steps = int(self.config.max_steps or task.horizon)
        self.qpos = np.zeros(7, dtype=np.float32)
        self.qvel = np.zeros(7, dtype=np.float32)
        self.step_count = 0

    @property
    def observation_dim(self) -> int:
        return 7 + 7 + 3 + MAX_OBSTACLES * 5

    @property
    def action_dim(self) -> int:
        return 7

    def _sync_state(self) -> None:
        self.data.qpos[:7] = self.qpos
        self.data.ctrl[:7] = self.qpos
        self.data.qvel[:7] = self.qvel
        mujoco.mj_forward(self.model, self.data)

    def _effective_action_scale(self) -> float:
        physical_limit = float(self.config.max_joint_velocity) * float(self.task.dt)
        return float(min(float(self.config.action_scale), physical_limit))

    def _obstacle_features(self) -> np.ndarray:
        features = np.zeros((MAX_OBSTACLES, 5), dtype=np.float32)
        for index, obstacle in enumerate(self.task.obstacles[:MAX_OBSTACLES]):
            features[index, :3] = np.asarray(obstacle.center, dtype=np.float32)
            features[index, 3] = float(obstacle.radius)
            features[index, 4] = float(obstacle.safe_distance)
        return features.reshape(-1)

    def _current_observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.qpos.astype(np.float32),
                self.qvel.astype(np.float32),
                np.asarray(self.task.target_position, dtype=np.float32),
                self._obstacle_features(),
            ]
        ).astype(np.float32)

    def _min_surface_distance(self, ee_position: np.ndarray) -> float:
        if not self.obstacles:
            return float("inf")
        return float(min(np.linalg.norm(ee_position - obstacle.center) - obstacle.radius for obstacle in self.obstacles))

    def reset(self, start_config: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.qpos = np.asarray(start_config if start_config is not None else self.task.start_config, dtype=np.float32).copy()
        self.qvel = np.zeros_like(self.qpos)
        self.step_count = 0
        self._sync_state()
        return self._current_observation(), {"task_id": self.task.task_id}

    def step(self, action: np.ndarray, clip_action: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if clip_action:
            action_limit = self._effective_action_scale()
            action = np.clip(action, -action_limit, action_limit)

        previous_qpos = self.qpos.copy()
        previous_ee, _ = self.kinematics.forward_kinematics(previous_qpos)

        next_qpos = np.clip(previous_qpos + action, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER).astype(np.float32)
        self.qvel = ((next_qpos - previous_qpos) / float(self.task.dt)).astype(np.float32)
        self.qpos = next_qpos
        self.step_count += 1
        self._sync_state()

        next_ee, _ = self.kinematics.forward_kinematics(next_qpos)
        step_length_cost = float(np.linalg.norm(next_ee - previous_ee))
        collision_result = self.contact_checker.evaluate_current_state()
        min_surface_distance = self._min_surface_distance(next_ee)
        step_safety_cost = float(self.obstacle_cost.compute_configuration_cost(next_qpos))
        final_error = float(np.linalg.norm(next_ee - self.task.target_array()))
        success = final_error <= self.config.success_threshold
        obstacle_collision = any(
            self.contact_checker.is_disallowed_contact(contact, penetration_only=True)
            and contact.classification == "disallowed_obstacle"
            for contact in collision_result.contacts
        )
        self_collision = any(
            self.contact_checker.is_disallowed_contact(contact, penetration_only=True)
            and contact.classification == "disallowed_self"
            for contact in collision_result.contacts
        )
        contact_collision = bool(collision_result.has_collision)
        collision = bool(self.config.terminate_on_contact and contact_collision)
        timeout = self.step_count >= self.max_steps
        done = bool(success or collision or timeout)

        objective_vector = np.asarray([step_length_cost, step_safety_cost], dtype=np.float32)
        reward_vector = -objective_vector.copy()
        if done and not success:
            reward_vector -= float(self.config.terminal_failure_penalty)

        info = {
            "task_id": self.task.task_id,
            "success": success,
            "collision": collision,
            "obstacle_collision": bool(obstacle_collision),
            "self_collision": bool(self_collision),
            "contact_collision": bool(contact_collision),
            "timeout": timeout and not success and not collision,
            "step_length_cost": step_length_cost,
            "step_safety_cost": step_safety_cost,
            "objective_vector": objective_vector.copy(),
            "final_error": final_error,
            "min_surface_distance": min_surface_distance,
            "action_scale": self._effective_action_scale(),
        }
        return self._current_observation(), reward_vector.astype(np.float32), objective_vector.astype(np.float32), done, info

    def rollout(
        self,
        policy: Callable[[np.ndarray, np.ndarray], np.ndarray],
        weights: np.ndarray,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        observation, _ = self.reset()
        observations = [observation.copy()]
        actions = []
        qpos_trajectory = [self.qpos.copy()]
        total_rewards = np.zeros(2, dtype=np.float32)
        total_objectives = np.zeros(2, dtype=np.float32)
        final_info: dict[str, Any] = {"success": False, "collision": False, "timeout": False}

        while True:
            action = np.asarray(policy(observation, weights), dtype=np.float32)
            next_observation, reward_vector, objective_vector, done, info = self.step(action, clip_action=True)
            observations.append(next_observation.copy())
            actions.append(action.copy())
            qpos_trajectory.append(self.qpos.copy())
            total_rewards += reward_vector
            total_objectives += objective_vector
            final_info = info
            observation = next_observation
            if done:
                break

        return {
            "task_spec": self.task.to_dict(),
            "trajectory": np.asarray(qpos_trajectory, dtype=np.float32),
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "reward_total": total_rewards,
            "objective_total": total_objectives,
            "steps": len(actions),
            "success": bool(final_info["success"]),
            "collision": bool(final_info["collision"]),
            "contact_collision": bool(final_info.get("contact_collision", False)),
            "timeout": bool(final_info["timeout"]),
            "final_error": float(final_info["final_error"]),
            "weights": np.asarray(weights, dtype=np.float32),
            "deterministic": bool(deterministic),
        }

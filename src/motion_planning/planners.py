"""
Path planning algorithms for robot motion planning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .RRTPlanner import RRTPlanner
from .unconstrained_trajopt import UnconstrainedTrajOptPlanner


@dataclass(frozen=True)
class CollisionPolicy:
    ignored_visual_geom_prefixes: tuple[str, ...] = ("target_", "trace_dot_")
    ignored_visual_body_prefixes: tuple[str, ...] = ("target_marker", "trace_dot_")
    physical_obstacle_geom_prefixes: tuple[str, ...] = ("obstacle_",)
    physical_obstacle_body_prefixes: tuple[str, ...] = ("obstacle_",)
    allowed_gripper_body_prefixes: tuple[str, ...] = ("gripper_", "left_", "right_")
    allowed_body_pairs: tuple[tuple[str, str], ...] = ()
    allow_adjacent_self_contacts: bool = True
    robot_root_body: str = "base_link"


@dataclass(frozen=True)
class ContactClassification:
    classification: str
    geom1: str
    geom2: str
    body1: str
    body2: str
    distance: float = 0.0
    geom1_id: int = -1
    geom2_id: int = -1
    body1_id: int = -1
    body2_id: int = -1


@dataclass(frozen=True)
class CollisionCheckResult:
    has_collision: bool
    has_disallowed_contact: bool = False
    first_disallowed: ContactClassification | None = None
    first_disallowed_contact: ContactClassification | None = None
    contacts: tuple[ContactClassification, ...] = ()


def default_collision_policy() -> CollisionPolicy:
    return CollisionPolicy()


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return name.startswith(prefixes)


def _normalize_body_pair(body1: str, body2: str) -> tuple[str, str]:
    return (body1, body2) if body1 <= body2 else (body2, body1)


def _is_gripper_body(name: str, policy: CollisionPolicy) -> bool:
    return _matches_prefix(name, policy.allowed_gripper_body_prefixes)


def _is_visual_contact(geom1: str, geom2: str, body1: str, body2: str, policy: CollisionPolicy) -> bool:
    return (
        _matches_prefix(geom1, policy.ignored_visual_geom_prefixes)
        or _matches_prefix(geom2, policy.ignored_visual_geom_prefixes)
        or _matches_prefix(body1, policy.ignored_visual_body_prefixes)
        or _matches_prefix(body2, policy.ignored_visual_body_prefixes)
    )


def _is_obstacle_name(name: str, policy: CollisionPolicy) -> bool:
    return _matches_prefix(name, policy.physical_obstacle_geom_prefixes) or _matches_prefix(name, policy.physical_obstacle_body_prefixes)


def _normalized_allowed_pairs(allowed_pairs: tuple[tuple[str, str], ...]) -> frozenset[tuple[str, str]]:
    return frozenset(_normalize_body_pair(body1, body2) for body1, body2 in allowed_pairs)


def _pair_allowed(body1: str, body2: str, allowed_pairs: frozenset[tuple[str, str]]) -> bool:
    return _normalize_body_pair(body1, body2) in allowed_pairs


def classify_named_contact(
    *,
    geom1: str,
    geom2: str,
    body1: str,
    body2: str,
    robot_body_names: set[str],
    adjacent_body_pairs: set[tuple[str, str]],
    policy: CollisionPolicy,
    allowed_body_pairs: frozenset[tuple[str, str]] | None = None,
    distance: float = 0.0,
    geom1_id: int = -1,
    geom2_id: int = -1,
    body1_id: int = -1,
    body2_id: int = -1,
) -> ContactClassification:
    if _is_visual_contact(geom1, geom2, body1, body2, policy):
        return ContactClassification("ignored_visual", geom1, geom2, body1, body2, distance, geom1_id, geom2_id, body1_id, body2_id)

    is_floor_contact = geom1 == "floor" or geom2 == "floor" or body1 == "floor" or body2 == "floor"
    body1_is_robot = body1 in robot_body_names
    body2_is_robot = body2 in robot_body_names
    geom1_is_obstacle = _is_obstacle_name(geom1, policy)
    geom2_is_obstacle = _is_obstacle_name(geom2, policy)
    body1_is_obstacle = _is_obstacle_name(body1, policy)
    body2_is_obstacle = _is_obstacle_name(body2, policy)
    contact_involves_obstacle = geom1_is_obstacle or geom2_is_obstacle or body1_is_obstacle or body2_is_obstacle
    normalized_allowed_pairs = (
        allowed_body_pairs if allowed_body_pairs is not None else _normalized_allowed_pairs(policy.allowed_body_pairs)
    )

    if is_floor_contact and (body1_is_robot or body2_is_robot):
        return ContactClassification("disallowed_floor", geom1, geom2, body1, body2, distance, geom1_id, geom2_id, body1_id, body2_id)

    if contact_involves_obstacle and (body1_is_robot or body2_is_robot):
        return ContactClassification("disallowed_obstacle", geom1, geom2, body1, body2, distance, geom1_id, geom2_id, body1_id, body2_id)

    if body1_is_robot and body2_is_robot:
        if _pair_allowed(body1, body2, normalized_allowed_pairs):
            return ContactClassification("allowed_adjacent_self", geom1, geom2, body1, body2, distance, geom1_id, geom2_id, body1_id, body2_id)
        if _is_gripper_body(body1, policy) and _is_gripper_body(body2, policy):
            return ContactClassification("allowed_gripper_internal", geom1, geom2, body1, body2, distance, geom1_id, geom2_id, body1_id, body2_id)
        if body1 == body2:
            return ContactClassification("allowed_adjacent_self", geom1, geom2, body1, body2, distance, geom1_id, geom2_id, body1_id, body2_id)
        if policy.allow_adjacent_self_contacts and _normalize_body_pair(body1, body2) in adjacent_body_pairs:
            return ContactClassification("allowed_adjacent_self", geom1, geom2, body1, body2, distance, geom1_id, geom2_id, body1_id, body2_id)
        return ContactClassification("disallowed_self", geom1, geom2, body1, body2, distance, geom1_id, geom2_id, body1_id, body2_id)

    if body1_is_robot or body2_is_robot:
        return ContactClassification("disallowed_world", geom1, geom2, body1, body2, distance, geom1_id, geom2_id, body1_id, body2_id)

    return ContactClassification("ignored_visual", geom1, geom2, body1, body2, distance, geom1_id, geom2_id, body1_id, body2_id)


class ContactCollisionChecker:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy: CollisionPolicy | None = None,
    ):
        self.model = model
        self.data = data
        self.policy = policy or default_collision_policy()
        self.original_qpos = data.qpos.copy()
        self.original_qvel = data.qvel.copy()
        self.body_names = self._body_name_map()
        self.geom_names = self._geom_name_map()
        self.robot_body_names = self._robot_body_names()
        self.adjacent_body_pairs = self._adjacent_body_pairs()
        self.allowed_body_pairs = _normalized_allowed_pairs(self.policy.allowed_body_pairs)
        self.last_result = CollisionCheckResult(has_collision=False)

    def _body_name_map(self) -> dict[int, str]:
        names: dict[int, str] = {}
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            names[body_id] = name or f"body_{body_id}"
        return names

    def _geom_name_map(self) -> dict[int, str]:
        names: dict[int, str] = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            names[geom_id] = name or f"geom_{geom_id}"
        return names

    def _body_ancestors(self, body_id: int) -> list[int]:
        ancestors: list[int] = []
        current = int(body_id)
        visited: set[int] = set()
        while current >= 0 and current not in visited:
            ancestors.append(current)
            visited.add(current)
            parent = int(self.model.body_parentid[current])
            if parent == current:
                break
            current = parent
        return ancestors

    def _robot_body_names(self) -> set[str]:
        root_name = self.policy.robot_root_body
        robot_names: set[str] = set()
        for body_id, body_name in self.body_names.items():
            ancestor_names = {self.body_names[ancestor_id] for ancestor_id in self._body_ancestors(body_id)}
            if root_name in ancestor_names:
                robot_names.add(body_name)
        return robot_names

    def _adjacent_body_pairs(self) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for body_id in range(1, self.model.nbody):
            parent_id = int(self.model.body_parentid[body_id])
            if parent_id < 0 or parent_id == body_id:
                continue
            body_name = self.body_names.get(body_id, f"body_{body_id}")
            parent_name = self.body_names.get(parent_id, f"body_{parent_id}")
            pairs.add(tuple(sorted((body_name, parent_name))))
        return pairs

    def _classify_contact(self, contact: Any) -> ContactClassification:
        return self.classify_contact(contact)

    def classify_contact(self, contact: Any) -> ContactClassification:
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        geom1_name = self.geom_names.get(geom1_id, f"geom_{geom1_id}")
        geom2_name = self.geom_names.get(geom2_id, f"geom_{geom2_id}")
        body1_id = int(self.model.geom_bodyid[geom1_id])
        body2_id = int(self.model.geom_bodyid[geom2_id])
        body1_name = self.body_names.get(body1_id, f"body_{body1_id}")
        body2_name = self.body_names.get(body2_id, f"body_{body2_id}")
        return classify_named_contact(
            geom1=geom1_name,
            geom2=geom2_name,
            body1=body1_name,
            body2=body2_name,
            robot_body_names=self.robot_body_names,
            adjacent_body_pairs=self.adjacent_body_pairs,
            policy=self.policy,
            allowed_body_pairs=self.allowed_body_pairs,
            distance=float(contact.dist),
            geom1_id=geom1_id,
            geom2_id=geom2_id,
            body1_id=body1_id,
            body2_id=body2_id,
        )

    @staticmethod
    def is_disallowed_contact(
        contact: ContactClassification,
        *,
        penetration_only: bool = False,
        distance_tolerance: float = 0.0,
    ) -> bool:
        if not contact.classification.startswith("disallowed_"):
            return False
        if not penetration_only:
            return True
        return float(contact.distance) <= float(distance_tolerance)

    def classify_current_contacts(self) -> tuple[ContactClassification, ...]:
        return tuple(self.classify_contact(self.data.contact[contact_index]) for contact_index in range(int(self.data.ncon)))

    def has_disallowed_collision(
        self,
        *,
        contacts: tuple[ContactClassification, ...] | None = None,
        distance_tolerance: float = 0.0,
    ) -> bool:
        active_contacts = contacts if contacts is not None else self.classify_current_contacts()
        return any(
            self.is_disallowed_contact(contact, penetration_only=True, distance_tolerance=distance_tolerance)
            for contact in active_contacts
        )

    def first_disallowed_contact(
        self,
        *,
        contacts: tuple[ContactClassification, ...] | None = None,
        penetration_only: bool = False,
        distance_tolerance: float = 0.0,
    ) -> ContactClassification | None:
        active_contacts = contacts if contacts is not None else self.classify_current_contacts()
        for contact in active_contacts:
            if self.is_disallowed_contact(
                contact,
                penetration_only=penetration_only,
                distance_tolerance=distance_tolerance,
            ):
                return contact
        return None

    def _restore_state(self) -> None:
        self.data.qpos[:] = self.original_qpos
        self.data.qvel[:] = self.original_qvel
        mujoco.mj_forward(self.model, self.data)

    def evaluate_current_state(self, distance_tolerance: float = 0.0) -> CollisionCheckResult:
        contacts = self.classify_current_contacts()
        first_disallowed_contact = self.first_disallowed_contact(contacts=contacts, penetration_only=False)
        first_disallowed = self.first_disallowed_contact(
            contacts=contacts,
            penetration_only=True,
            distance_tolerance=distance_tolerance,
        )
        result = CollisionCheckResult(
            has_collision=first_disallowed is not None,
            has_disallowed_contact=first_disallowed_contact is not None,
            first_disallowed=first_disallowed,
            first_disallowed_contact=first_disallowed_contact,
            contacts=contacts,
        )
        self.last_result = result
        return result

    def evaluate(self, joint_config: np.ndarray, distance_tolerance: float = 0.0) -> CollisionCheckResult:
        self.data.qpos[:7] = joint_config
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        result = self.evaluate_current_state(distance_tolerance=distance_tolerance)
        self._restore_state()
        return result

    def __call__(self, joint_config: np.ndarray) -> bool:
        return self.evaluate(joint_config).has_collision


class MotionPlannerFactory:
    """Factory for creating motion planners"""

    @staticmethod
    def create_rrt_planner(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        **kwargs,
    ) -> RRTPlanner:
        """Create an RRT planner"""
        return RRTPlanner(model, data, **kwargs)

    @staticmethod
    def create_unconstrained_trajopt_planner(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        **kwargs,
    ) -> UnconstrainedTrajOptPlanner:
        """Create a TrajOpt planner"""
        return UnconstrainedTrajOptPlanner(model, data, **kwargs)

    @staticmethod
    def create_collision_checker(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy: CollisionPolicy | None = None,
    ) -> ContactCollisionChecker:
        """Create a collision checking function using MuJoCo contact classification."""
        return ContactCollisionChecker(model, data, policy=policy)

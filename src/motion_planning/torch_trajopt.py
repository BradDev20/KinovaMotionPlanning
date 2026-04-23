from __future__ import annotations

import re
import time
from typing import Sequence

import numpy as np

from .torch_trajopt_serialization import (
    serialize_dynamics_checkpoint,
    serialize_trajectory_collision_summary,
    serialize_trajectory_dynamics_summary,
)
from .torch_trajopt_types import TorchPlannerJob, TorchPlannerResult

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - runtime guard only
    torch = None
    F = None


_JOINT_STATIC_TRANSFORMS = (
    ((0.0, 0.0, 0.15643), (0.0, 1.0, 0.0, 0.0)),
    ((0.0, 0.005375, -0.12838), (1.0, 1.0, 0.0, 0.0)),
    ((0.0, -0.21038, -0.006375), (1.0, -1.0, 0.0, 0.0)),
    ((0.0, 0.006375, -0.21038), (1.0, 1.0, 0.0, 0.0)),
    ((0.0, -0.20843, -0.006375), (1.0, -1.0, 0.0, 0.0)),
    ((0.0, 0.00017505, -0.10593), (1.0, 1.0, 0.0, 0.0)),
    ((0.0, -0.10593, -0.00017505), (1.0, -1.0, 0.0, 0.0)),
)
_GRIPPER_STATIC_TRANSFORMS = (
    ((0.0, 0.0, -0.061525), (0.0, 1.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0108), (1.0, 0.0, 0.0, -1.0)),
)
_GRIPPER_PINCH_OFFSET = (0.0, 0.0, 0.145)
_JOINT_LIMITS_LOWER = (-3.14, -2.24, -3.14, -2.57, -3.14, -2.09, -3.14)
_JOINT_LIMITS_UPPER = (3.14, 2.24, 3.14, 2.57, 3.14, 2.09, 3.14)


def require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for torch trajectory optimization.")


def _quat_to_rotmat(quat, *, device, dtype):
    q = torch.as_tensor(quat, dtype=dtype, device=device)
    q = q / q.norm().clamp_min(1e-8)
    w, x, y, z = q.unbind()
    return torch.stack(
        (
            torch.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w))),
            torch.stack((2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w))),
            torch.stack((2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y))),
        )
    )


def _rot_z(theta):
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    row_0 = torch.stack((cos_theta, -sin_theta, zeros), dim=-1)
    row_1 = torch.stack((sin_theta, cos_theta, zeros), dim=-1)
    row_2 = torch.stack((zeros, zeros, ones), dim=-1)
    return torch.stack((row_0, row_1, row_2), dim=-2)


def _apply_static_transform(rotation, translation, *, offset, static_rotation):
    translated = translation + torch.matmul(rotation, offset.view(1, 3, 1)).squeeze(-1)
    rotated = torch.matmul(rotation, static_rotation.view(1, 3, 3))
    return rotated, translated


class TorchTrajectoryBatchPlanner:
    def __init__(
        self,
        *,
        device: str,
        n_waypoints: int,
        planner_steps: int,
        max_velocity: float,
        max_acceleration: float,
        warm_start_noise: float,
    ):
        require_torch()
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            resolved_device = torch.device("cpu")
        self.device = resolved_device
        self.n_waypoints = int(n_waypoints)
        self.planner_steps = max(int(planner_steps), 1)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.warm_start_noise = float(max(warm_start_noise, 0.0))
        self.dtype = torch.float32
        self._joint_lower = torch.as_tensor(_JOINT_LIMITS_LOWER, dtype=self.dtype, device=self.device)
        self._joint_upper = torch.as_tensor(_JOINT_LIMITS_UPPER, dtype=self.dtype, device=self.device)
        self._joint_center = (self._joint_lower + self._joint_upper) * 0.5
        self._joint_half_range = (self._joint_upper - self._joint_lower) * 0.5
        self._joint_rotations = tuple(
            _quat_to_rotmat(quat, device=self.device, dtype=self.dtype)
            for _, quat in _JOINT_STATIC_TRANSFORMS
        )
        self._joint_offsets = tuple(
            torch.as_tensor(offset, dtype=self.dtype, device=self.device)
            for offset, _ in _JOINT_STATIC_TRANSFORMS
        )
        self._gripper_rotations = tuple(
            _quat_to_rotmat(quat, device=self.device, dtype=self.dtype)
            for _, quat in _GRIPPER_STATIC_TRANSFORMS
        )
        self._gripper_offsets = tuple(
            torch.as_tensor(offset, dtype=self.dtype, device=self.device)
            for offset, _ in _GRIPPER_STATIC_TRANSFORMS
        )
        self._gripper_pinch_offset = torch.as_tensor(_GRIPPER_PINCH_OFFSET, dtype=self.dtype, device=self.device)
        self._latest_warm_start_metadata: list[dict[str, object]] = []

    def _linear_trajectory(self, start_config, goal_config):
        fractions = torch.linspace(0.0, 1.0, self.n_waypoints, device=self.device, dtype=self.dtype).view(1, self.n_waypoints, 1)
        return (1.0 - fractions) * start_config[:, None, :] + fractions * goal_config[:, None, :]

    def _smooth_warm_start_perturbation(
        self,
        waypoint_count: int,
        joint_count: int,
        rng: np.random.Generator,
        *,
        noise_scale: float,
    ) -> np.ndarray:
        interior_waypoints = max(int(waypoint_count) - 2, 0)
        if interior_waypoints <= 0:
            return np.zeros((0, joint_count), dtype=np.float32)

        samples = np.linspace(0.0, 1.0, interior_waypoints + 2, dtype=np.float32)[1:-1]
        basis_vectors: list[np.ndarray] = []
        for mode in range(1, min(3, interior_waypoints) + 1):
            basis = np.sin(np.pi * float(mode) * samples, dtype=np.float32)
            norm = float(np.linalg.norm(basis))
            if norm > 0.0:
                basis_vectors.append((basis / norm).astype(np.float32, copy=False))
        if not basis_vectors:
            return np.zeros((interior_waypoints, joint_count), dtype=np.float32)

        stacked_basis = np.stack(basis_vectors, axis=0)
        coefficients = rng.normal(
            loc=0.0,
            scale=float(max(noise_scale, 0.0)),
            size=(stacked_basis.shape[0], joint_count),
        ).astype(np.float32)
        return np.tensordot(stacked_basis, coefficients, axes=(0, 0)).astype(np.float32, copy=False)

    def _clip_warm_start_second_differences(
        self,
        trajectory: np.ndarray,
        *,
        dt: float,
        acceleration_cap: float,
        iterations: int = 64,
    ) -> np.ndarray:
        if trajectory.shape[0] < 3:
            return trajectory

        clipped = np.array(trajectory, dtype=np.float32, copy=True)
        second_diff_cap = float(acceleration_cap) * float(dt) * float(dt)
        joint_lower = self._joint_lower.detach().cpu().numpy().astype(np.float32, copy=False)
        joint_upper = self._joint_upper.detach().cpu().numpy().astype(np.float32, copy=False)

        for _ in range(max(int(iterations), 1)):
            any_updated = False
            for waypoint_index in range(1, clipped.shape[0] - 1):
                previous_waypoint = clipped[waypoint_index - 1]
                current_waypoint = clipped[waypoint_index]
                next_waypoint = clipped[waypoint_index + 1]
                second_difference = next_waypoint - 2.0 * current_waypoint + previous_waypoint
                bounded_second_difference = np.clip(second_difference, -second_diff_cap, second_diff_cap)
                if np.any(np.abs(second_difference - bounded_second_difference) > 1e-7):
                    clipped[waypoint_index] = np.clip(
                        0.5 * (previous_waypoint + next_waypoint - bounded_second_difference),
                        joint_lower,
                        joint_upper,
                    )
                    any_updated = True
            if not any_updated:
                break
        return clipped

    def _resample_joint_path(self, path: np.ndarray, waypoint_count: int) -> np.ndarray:
        if path.shape[0] == 0:
            raise ValueError("Cannot resample an empty path.")
        if path.shape[0] == 1:
            return np.repeat(path.astype(np.float32, copy=False), waypoint_count, axis=0)

        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative_lengths = np.concatenate((np.asarray([0.0], dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)))
        total_length = float(cumulative_lengths[-1])
        if total_length <= 1e-8:
            return np.linspace(path[0], path[-1], waypoint_count, dtype=np.float32)

        targets = np.linspace(0.0, total_length, waypoint_count, dtype=np.float32)
        resampled = np.empty((waypoint_count, path.shape[1]), dtype=np.float32)
        segment_index = 0
        for target_index, target in enumerate(targets):
            while (
                segment_index < segment_lengths.shape[0] - 1
                and float(cumulative_lengths[segment_index + 1]) < float(target)
            ):
                segment_index += 1
            start_length = float(cumulative_lengths[segment_index])
            end_length = float(cumulative_lengths[segment_index + 1])
            if end_length <= start_length + 1e-8:
                interpolation = 0.0
            else:
                interpolation = (float(target) - start_length) / (end_length - start_length)
            resampled[target_index] = (
                (1.0 - interpolation) * path[segment_index]
                + interpolation * path[segment_index + 1]
            )

        resampled[0] = path[0]
        resampled[-1] = path[-1]
        return resampled

    @staticmethod
    def _rrt_candidate_is_better(
        *,
        baseline_collision_waypoint_count: int,
        baseline_min_signed_distance: float,
        candidate_collision_waypoint_count: int,
        candidate_min_signed_distance: float,
    ) -> bool:
        if candidate_collision_waypoint_count == 0 and baseline_collision_waypoint_count > 0:
            return True
        return (
            candidate_collision_waypoint_count < baseline_collision_waypoint_count
            and candidate_min_signed_distance > baseline_min_signed_distance
        )

    def _surrogate_collision_checker_for_job(self, job: TorchPlannerJob):
        obstacle_centers = np.asarray(job.obstacle_centers, dtype=np.float32)
        if obstacle_centers.shape[0] == 0:
            return lambda _config: False
        inflated_radii = (
            np.asarray(job.obstacle_radii, dtype=np.float32)
            + np.asarray(job.obstacle_safe_distances, dtype=np.float32)
        )

        def _checker(config: np.ndarray) -> bool:
            config_tensor = torch.as_tensor(config, dtype=self.dtype, device=self.device).view(1, 1, -1)
            with torch.no_grad():
                ee_position = self.forward_kinematics(config_tensor)[0, 0, :].detach().cpu().numpy().astype(np.float32, copy=False)
            distances = np.linalg.norm(obstacle_centers - ee_position[None, :], axis=1) - inflated_radii
            return bool(np.any(distances < 0.0))

        return _checker

    @staticmethod
    def _should_attempt_rrt_warm_start(
        job: TorchPlannerJob,
        *,
        baseline_collision_waypoint_count: int,
        baseline_min_signed_distance: float,
    ) -> bool:
        if str(getattr(job, "task_family", "base")) != "double_corridor":
            return False
        if baseline_collision_waypoint_count >= 10:
            return True
        if baseline_collision_waypoint_count >= 6 and baseline_min_signed_distance <= -0.04:
            return True
        return baseline_collision_waypoint_count >= 3 and baseline_min_signed_distance <= -0.07

    @staticmethod
    def _restart_index_from_request_id(request_id: str) -> int | None:
        match = re.search(r"_r(\d+)$", str(request_id))
        if match is None:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _rrt_warm_start_key(job: TorchPlannerJob) -> tuple[object, ...]:
        return (
            tuple(float(value) for value in job.start_config),
            tuple(float(value) for value in job.goal_config),
            tuple(tuple(float(axis) for axis in center) for center in job.obstacle_centers),
            tuple(float(value) for value in job.obstacle_radii),
            tuple(float(value) for value in job.obstacle_safe_distances),
            int(job.seed),
            str(getattr(job, "task_family", "base")),
            float(job.dt),
        )

    def _apply_rrt_warm_starts(
        self,
        trajectory,
        jobs: Sequence[TorchPlannerJob],
        *,
        existing_metadata: list[dict[str, object]] | None = None,
    ):
        batch_size = trajectory.shape[0]
        if batch_size == 0:
            self._latest_warm_start_metadata = []
            return trajectory

        baseline_summary = self._trajectory_collision_summary(trajectory, jobs)
        trajectory_np = trajectory.detach().cpu().numpy().astype(np.float32, copy=True)
        baseline_collision_waypoint_counts = [
            int(baseline_summary["collision_waypoint_count"][index].detach().cpu().item())
            for index in range(batch_size)
        ]
        baseline_min_signed_distances = [
            float(baseline_summary["min_signed_distance"][index].detach().cpu().item())
            for index in range(batch_size)
        ]
        if existing_metadata is not None and len(existing_metadata) == batch_size:
            metadata = [dict(item) for item in existing_metadata]
        else:
            metadata = [{} for _ in range(batch_size)]

        for index in range(batch_size):
            entry = metadata[index]
            entry.setdefault("warm_start_strategy", "linear")
            entry.setdefault("warm_start_seed_used", False)
            entry.setdefault("warm_start_tag", "")
            entry.setdefault("warm_start_noise_scale", None)
            entry.update(
                {
                    "warm_start_rrt_attempted": False,
                    "warm_start_rrt_success": False,
                    "warm_start_rrt_cache_hit": False,
                    "warm_start_baseline_collision_waypoint_count": baseline_collision_waypoint_counts[index],
                    "warm_start_baseline_min_signed_distance": baseline_min_signed_distances[index],
                    "warm_start_rrt_collision_waypoint_count": None,
                    "warm_start_rrt_min_signed_distance": None,
                }
            )

        eligible_indices: list[int] = []
        grouped_indices: dict[tuple[object, ...], list[int]] = {}
        for index, job in enumerate(jobs):
            baseline_collision_waypoint_count = baseline_collision_waypoint_counts[index]
            baseline_min_signed_distance = baseline_min_signed_distances[index]
            if not self._should_attempt_rrt_warm_start(
                job,
                baseline_collision_waypoint_count=baseline_collision_waypoint_count,
                baseline_min_signed_distance=baseline_min_signed_distance,
            ):
                continue
            restart_index = self._restart_index_from_request_id(job.request_id)
            if restart_index is not None and restart_index != 0:
                continue
            eligible_indices.append(index)
            rrt_key = self._rrt_warm_start_key(job)
            grouped_indices.setdefault(rrt_key, []).append(index)

        representative_by_key: dict[tuple[object, ...], int] = {}
        for rrt_key, indices in grouped_indices.items():
            representative_by_key[rrt_key] = max(
                indices,
                key=lambda candidate_index: (
                    baseline_collision_waypoint_counts[candidate_index],
                    -baseline_min_signed_distances[candidate_index],
                    -candidate_index,
                ),
            )
        if not representative_by_key:
            self._latest_warm_start_metadata = metadata
            return torch.as_tensor(trajectory_np, dtype=self.dtype, device=self.device)
        try:
            from .RRTPlanner import RRTPlanner
        except Exception:
            self._latest_warm_start_metadata = metadata
            return torch.as_tensor(trajectory_np, dtype=self.dtype, device=self.device)

        rrt_cache: dict[tuple[object, ...], dict[str, object]] = {}
        for rrt_key, representative_index in sorted(
            representative_by_key.items(),
            key=lambda item: item[1],
        ):
            job = jobs[representative_index]
            item_metadata = metadata[representative_index]
            item_metadata["warm_start_rrt_attempted"] = True
            rrt_seed = np.random.default_rng(int(job.seed))
            planner = RRTPlanner(
                model=None,
                data=None,
                step_size=0.40,
                max_iterations=120,
                goal_threshold=0.20,
            )
            planner.set_collision_checker(self._surrogate_collision_checker_for_job(job))
            planned_path, planning_success = planner.plan(
                np.asarray(job.start_config, dtype=np.float32),
                np.asarray(job.goal_config, dtype=np.float32),
                rng=rrt_seed,
            )
            cache_entry: dict[str, object] = {
                "success": bool(planning_success),
                "candidate_path": None,
                "candidate_collision_waypoint_count": None,
                "candidate_min_signed_distance": None,
            }
            item_metadata["warm_start_rrt_success"] = bool(planning_success)
            if not planning_success or len(planned_path) < 2:
                cache_entry["success"] = False
                item_metadata["warm_start_rrt_success"] = False
                rrt_cache[rrt_key] = cache_entry
                continue

            smoothed_path = planner.smooth_path(planned_path, max_iterations=16, rng=rrt_seed)
            candidate_path = self._resample_joint_path(np.asarray(smoothed_path, dtype=np.float32), self.n_waypoints)
            candidate_path[0] = trajectory_np[representative_index, 0]
            candidate_path[-1] = trajectory_np[representative_index, -1]
            candidate_path = self._clip_warm_start_second_differences(
                candidate_path,
                dt=float(job.dt),
                acceleration_cap=0.5 * self.max_acceleration,
            )
            candidate_trajectory = torch.as_tensor(candidate_path[None, ...], dtype=self.dtype, device=self.device)
            candidate_summary = self._trajectory_collision_summary(candidate_trajectory, [job])
            candidate_collision_waypoint_count = int(candidate_summary["collision_waypoint_count"][0].detach().cpu().item())
            candidate_min_signed_distance = float(candidate_summary["min_signed_distance"][0].detach().cpu().item())
            cache_entry["candidate_path"] = candidate_path
            cache_entry["candidate_collision_waypoint_count"] = candidate_collision_waypoint_count
            cache_entry["candidate_min_signed_distance"] = candidate_min_signed_distance
            rrt_cache[rrt_key] = cache_entry

        for index in eligible_indices:
            job = jobs[index]
            rrt_key = self._rrt_warm_start_key(job)
            cache_entry = rrt_cache.get(rrt_key)
            if cache_entry is None:
                continue
            representative_index = representative_by_key.get(rrt_key)
            item_metadata = metadata[index]
            if representative_index is not None and representative_index != index:
                item_metadata["warm_start_rrt_cache_hit"] = True
                item_metadata["warm_start_rrt_success"] = bool(cache_entry.get("success", False))
            candidate_path_cached = cache_entry.get("candidate_path")
            if candidate_path_cached is None:
                continue
            candidate_collision_waypoint_count = int(cache_entry.get("candidate_collision_waypoint_count", 0) or 0)
            candidate_min_signed_distance = float(cache_entry.get("candidate_min_signed_distance", 0.0) or 0.0)
            item_metadata["warm_start_rrt_collision_waypoint_count"] = candidate_collision_waypoint_count
            item_metadata["warm_start_rrt_min_signed_distance"] = candidate_min_signed_distance
            if self._rrt_candidate_is_better(
                baseline_collision_waypoint_count=baseline_collision_waypoint_counts[index],
                baseline_min_signed_distance=baseline_min_signed_distances[index],
                candidate_collision_waypoint_count=candidate_collision_waypoint_count,
                candidate_min_signed_distance=candidate_min_signed_distance,
            ):
                trajectory_np[index] = np.asarray(candidate_path_cached, dtype=np.float32)
                item_metadata["warm_start_strategy"] = "rrt"

        self._latest_warm_start_metadata = metadata
        return torch.as_tensor(trajectory_np, dtype=self.dtype, device=self.device)

    def _initialize_latent(
        self,
        start_config,
        goal_config,
        seeds: Sequence[int],
        dts: Sequence[float] | None = None,
        jobs: Sequence[TorchPlannerJob] | None = None,
    ):
        trajectory = self._linear_trajectory(start_config, goal_config)
        batch_size = int(trajectory.shape[0])
        resolved_dts = (
            [float(dt) for dt in dts]
            if dts is not None
            else [1.0 / max(self.n_waypoints - 1, 1)] * len(seeds)
        )

        joint_lower = self._joint_lower.detach().cpu().numpy().astype(np.float32, copy=False)
        joint_upper = self._joint_upper.detach().cpu().numpy().astype(np.float32, copy=False)
        base_np = trajectory.detach().cpu().numpy().astype(np.float32, copy=True)

        metadata: list[dict[str, object]] = [{} for _ in range(batch_size)]
        if jobs is not None and len(jobs) == batch_size:
            for index, job in enumerate(jobs):
                requested_seed = getattr(job, "warm_start_trajectory", None) is not None
                used_seed = False
                tag = str(getattr(job, "warm_start_tag", "") or "")
                if requested_seed:
                    try:
                        warm = np.asarray(job.warm_start_trajectory, dtype=np.float32)
                        if warm.ndim == 2 and warm.shape[0] >= 2 and warm.shape[1] == base_np.shape[2]:
                            candidate = self._resample_joint_path(warm, self.n_waypoints)
                            candidate[0] = base_np[index, 0]
                            candidate[-1] = base_np[index, -1]
                            candidate = np.clip(candidate, joint_lower, joint_upper)
                            candidate = self._clip_warm_start_second_differences(
                                candidate,
                                dt=float(resolved_dts[index]),
                                acceleration_cap=0.5 * self.max_acceleration,
                            )
                            base_np[index] = candidate
                            used_seed = True
                    except Exception as exc:
                        metadata[index]["warm_start_seed_error"] = str(exc)

                # Store seed/noise metadata up-front; RRT stage will extend it.
                metadata[index] = {
                    "warm_start_strategy": "seed" if used_seed else "linear",
                    "warm_start_seed_used": bool(used_seed),
                    "warm_start_seed_requested": bool(requested_seed),
                    "warm_start_tag": tag,
                    "warm_start_noise_scale": None,
                }

        # Apply smooth noise after seed insertion so we generate local variants around the base route.
        if self.n_waypoints > 2:
            interior = base_np[:, 1:-1, :]
            for index, seed in enumerate(seeds):
                noise_scale = self.warm_start_noise
                if jobs is not None and len(jobs) == batch_size:
                    per_job_scale = getattr(jobs[index], "warm_start_noise_scale", None)
                    if per_job_scale is not None:
                        noise_scale = float(per_job_scale)
                if jobs is not None and len(jobs) == batch_size:
                    metadata[index]["warm_start_noise_scale"] = float(noise_scale)
                if noise_scale <= 0.0:
                    continue
                rng = np.random.default_rng(int(seed))
                interior[index] += self._smooth_warm_start_perturbation(
                    self.n_waypoints,
                    interior[index].shape[1],
                    rng,
                    noise_scale=float(noise_scale),
                )
                full_trajectory = np.concatenate((base_np[index, :1, :], interior[index], base_np[index, -1:, :]), axis=0)
                full_trajectory = self._clip_warm_start_second_differences(
                    full_trajectory,
                    dt=float(resolved_dts[index]),
                    acceleration_cap=0.5 * self.max_acceleration,
                )
                interior[index] = full_trajectory[1:-1]

            base_np[:, 1:-1, :] = interior

        trajectory = torch.as_tensor(base_np, dtype=self.dtype, device=self.device)
        if jobs is not None and len(jobs) == trajectory.shape[0]:
            trajectory = self._apply_rrt_warm_starts(trajectory, jobs, existing_metadata=metadata)
        else:
            self._latest_warm_start_metadata = metadata if metadata else []
        normalized = ((trajectory[:, 1:-1, :] - self._joint_center.view(1, 1, -1)) / self._joint_half_range.view(1, 1, -1)).clamp(-0.999, 0.999)
        return torch.atanh(normalized)

    def _decode_trajectory(self, latent, start_config, goal_config):
        interior = self._joint_center.view(1, 1, -1) + self._joint_half_range.view(1, 1, -1) * torch.tanh(latent)
        return torch.cat((start_config[:, None, :], interior, goal_config[:, None, :]), dim=1)

    def forward_kinematics(self, trajectory):
        batch_size, waypoint_count, _ = trajectory.shape
        flat = trajectory.reshape(batch_size * waypoint_count, 7)
        rotation = torch.eye(3, dtype=self.dtype, device=self.device).expand(flat.shape[0], 3, 3).clone()
        translation = torch.zeros(flat.shape[0], 3, dtype=self.dtype, device=self.device)

        for joint_index in range(7):
            rotation, translation = _apply_static_transform(
                rotation,
                translation,
                offset=self._joint_offsets[joint_index],
                static_rotation=self._joint_rotations[joint_index],
            )
            rotation = torch.matmul(rotation, _rot_z(flat[:, joint_index]))

        for offset, static_rotation in zip(self._gripper_offsets, self._gripper_rotations):
            rotation, translation = _apply_static_transform(
                rotation,
                translation,
                offset=offset,
                static_rotation=static_rotation,
            )

        ee_position = translation + torch.matmul(rotation, self._gripper_pinch_offset.view(1, 3, 1)).squeeze(-1)
        return ee_position.reshape(batch_size, waypoint_count, 3)

    def _pack_obstacles(self, jobs):
        max_obstacles = max((len(job.obstacle_radii) for job in jobs), default=0)
        if max_obstacles == 0:
            empty_centers = torch.zeros(len(jobs), 0, 3, dtype=self.dtype, device=self.device)
            empty_radii = torch.zeros(len(jobs), 0, dtype=self.dtype, device=self.device)
            empty_mask = torch.zeros(len(jobs), 0, dtype=torch.bool, device=self.device)
            return empty_centers, empty_radii, empty_mask

        centers = torch.zeros(len(jobs), max_obstacles, 3, dtype=self.dtype, device=self.device)
        radii = torch.zeros(len(jobs), max_obstacles, dtype=self.dtype, device=self.device)
        mask = torch.zeros(len(jobs), max_obstacles, dtype=torch.bool, device=self.device)
        for batch_index, job in enumerate(jobs):
            for obstacle_index, center in enumerate(job.obstacle_centers):
                centers[batch_index, obstacle_index] = torch.as_tensor(center, dtype=self.dtype, device=self.device)
                radii[batch_index, obstacle_index] = float(job.obstacle_radii[obstacle_index]) + float(job.obstacle_safe_distances[obstacle_index])
                mask[batch_index, obstacle_index] = True
        return centers, radii, mask

    def _nearest_obstacle_distances(self, positions, jobs):
        centers, radii, obstacle_mask = self._pack_obstacles(jobs)
        if centers.shape[1] == 0:
            batch_size, waypoint_count, _ = positions.shape
            nearest_distance = torch.full(
                (batch_size, waypoint_count),
                1.0e6,
                dtype=self.dtype,
                device=self.device,
            )
            nearest_obstacle_index = torch.full(
                (batch_size, waypoint_count),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
            return nearest_distance, nearest_obstacle_index, obstacle_mask

        delta = positions[:, :, None, :] - centers[:, None, :, :]
        signed_distance = torch.linalg.vector_norm(delta, dim=-1) - radii[:, None, :]
        signed_distance = torch.where(
            obstacle_mask[:, None, :],
            signed_distance,
            torch.full_like(signed_distance, 1.0e6),
        )
        nearest_distance, nearest_obstacle_index = signed_distance.min(dim=-1)
        return nearest_distance, nearest_obstacle_index, obstacle_mask

    def _objective_terms(self, trajectory, jobs):
        positions = self.forward_kinematics(trajectory)
        steps = positions[:, 1:, :] - positions[:, :-1, :]
        path_length = torch.linalg.vector_norm(steps, dim=-1).sum(dim=-1)
        straight_line = torch.linalg.vector_norm(positions[:, -1, :] - positions[:, 0, :], dim=-1).clamp_min(1e-4)
        length_cost = path_length / straight_line

        nearest_distance, nearest_obstacle_index, obstacle_mask = self._nearest_obstacle_distances(positions, jobs)
        if obstacle_mask.shape[1] == 0:
            obstacle_cost = torch.zeros(length_cost.shape[0], dtype=self.dtype, device=self.device)
        else:
            safety_decay = torch.as_tensor([job.safety_decay_rate for job in jobs], dtype=self.dtype, device=self.device)
            safety_bias = torch.as_tensor([job.safety_bias for job in jobs], dtype=self.dtype, device=self.device)
            collision_penalty = torch.as_tensor([job.safety_collision_penalty for job in jobs], dtype=self.dtype, device=self.device)
            scaled = (0.05 - nearest_distance) * safety_decay[:, None] + safety_bias[:, None]
            waypoint_penalty = F.softplus(scaled).square()
            waypoint_penalty = torch.where(
                nearest_distance < 0.0,
                waypoint_penalty * collision_penalty[:, None],
                waypoint_penalty,
            )
            aggregate_is_max = torch.as_tensor(
                [job.safety_aggregate == "max" for job in jobs],
                dtype=torch.bool,
                device=self.device,
            )
            obstacle_cost_mean = waypoint_penalty.mean(dim=-1)
            obstacle_cost_peak = waypoint_penalty.max(dim=-1).values
            collision_depth = F.relu(-nearest_distance).amax(dim=-1)
            obstacle_cost = obstacle_cost_mean + torch.where(
                aggregate_is_max,
                2.0 * obstacle_cost_peak,
                obstacle_cost_peak,
            ) + 25.0 * collision_depth.square()

        weights = torch.as_tensor(
            [[float(job.alpha), float(1.0 - job.alpha)] for job in jobs],
            dtype=self.dtype,
            device=self.device,
        )
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        objectives = torch.stack((length_cost, obstacle_cost), dim=-1)
        modes_max = torch.as_tensor(
            [job.planner_mode == "max" for job in jobs],
            dtype=torch.bool,
            device=self.device,
        )
        max_term = (objectives * weights).max(dim=-1).values
        rho = torch.as_tensor([job.rho for job in jobs], dtype=self.dtype, device=self.device)
        scalarized_cost = torch.where(
            modes_max,
            max_term + rho * objectives.sum(dim=-1),
            (objectives * weights).sum(dim=-1),
        )

        velocity = torch.diff(trajectory, dim=1) / torch.as_tensor([job.dt for job in jobs], dtype=self.dtype, device=self.device).view(-1, 1, 1)
        acceleration = torch.diff(trajectory, n=2, dim=1) / (
            torch.as_tensor([job.dt for job in jobs], dtype=self.dtype, device=self.device).view(-1, 1, 1) ** 2
        )
        smoothness_penalty = torch.diff(trajectory, n=2, dim=1).square().mean(dim=(1, 2)) if trajectory.shape[1] >= 3 else torch.zeros_like(scalarized_cost)
        velocity_excess = F.relu(torch.abs(velocity) - self.max_velocity) if velocity.shape[1] > 0 else torch.zeros_like(trajectory[:, :0, :])
        acceleration_excess = F.relu(torch.abs(acceleration) - self.max_acceleration) if acceleration.shape[1] > 0 else torch.zeros_like(trajectory[:, :0, :])
        velocity_penalty = velocity_excess.square().mean(dim=(1, 2)) if velocity.shape[1] > 0 else torch.zeros_like(scalarized_cost)
        acceleration_penalty = acceleration_excess.square().mean(dim=(1, 2)) if acceleration.shape[1] > 0 else torch.zeros_like(scalarized_cost)
        acceleration_penalty_peak = acceleration_excess.amax(dim=(1, 2)).square() if acceleration.shape[1] > 0 else torch.zeros_like(scalarized_cost)
        velocity_violation_count = (velocity_excess > 0.0).sum(dim=(1, 2)) if velocity.shape[1] > 0 else torch.zeros_like(scalarized_cost, dtype=torch.int64)
        acceleration_violation_count = (acceleration_excess > 0.0).sum(dim=(1, 2)) if acceleration.shape[1] > 0 else torch.zeros_like(scalarized_cost, dtype=torch.int64)
        max_velocity_excess = velocity_excess.amax(dim=(1, 2)) if velocity.shape[1] > 0 else torch.zeros_like(scalarized_cost)
        max_acceleration_excess = acceleration_excess.amax(dim=(1, 2)) if acceleration.shape[1] > 0 else torch.zeros_like(scalarized_cost)
        total_violation_count = velocity_violation_count + acceleration_violation_count
        total_violation_excess = max_velocity_excess + max_acceleration_excess
        collision_waypoint_count = (nearest_distance < 0.0).sum(dim=1)
        near_collision_waypoint_count = (nearest_distance < 0.02).sum(dim=1)
        min_signed_distance = nearest_distance.amin(dim=1)
        collision_penetration_depth = F.relu(-min_signed_distance)

        total_loss = (
            scalarized_cost
            + 0.05 * smoothness_penalty
            + 20.0 * velocity_penalty
            + 12.0 * acceleration_penalty
            + 40.0 * acceleration_penalty_peak
        )
        return {
            "total_loss": total_loss,
            "scalarized_cost": scalarized_cost,
            "length_cost": length_cost,
            "obstacle_cost": obstacle_cost,
            "velocity_violation_count": velocity_violation_count,
            "acceleration_violation_count": acceleration_violation_count,
            "max_velocity_excess": max_velocity_excess,
            "max_acceleration_excess": max_acceleration_excess,
            "total_violation_count": total_violation_count,
            "total_violation_excess": total_violation_excess,
            "collision_waypoint_count": collision_waypoint_count,
            "near_collision_waypoint_count": near_collision_waypoint_count,
            "min_signed_distance": min_signed_distance,
            "collision_penetration_depth": collision_penetration_depth,
        }

    def _lexicographic_less(self, current_values, best_values):
        better = torch.zeros_like(current_values[0], dtype=torch.bool, device=self.device)
        equal_prefix = torch.ones_like(current_values[0], dtype=torch.bool, device=self.device)
        for current, best in zip(current_values, best_values):
            if current.dtype.is_floating_point:
                less = current < best - 1e-8
                equal = torch.isclose(current, best, rtol=1e-6, atol=1e-8)
            else:
                less = current < best
                equal = current == best
            better = better | (equal_prefix & less)
            equal_prefix = equal_prefix & equal
        return better

    def _select_candidate_update_mask(self, candidate_metrics, best_metrics):
        candidate_feasible = (
            (candidate_metrics["velocity_violation_count"] == 0)
            & (candidate_metrics["acceleration_violation_count"] == 0)
            & (candidate_metrics["collision_waypoint_count"] == 0)
        )
        best_feasible = (
            (best_metrics["velocity_violation_count"] == 0)
            & (best_metrics["acceleration_violation_count"] == 0)
            & (best_metrics["collision_waypoint_count"] == 0)
        )

        candidate_beats_invalid_best = candidate_feasible & (~best_feasible)
        both_feasible = candidate_feasible & best_feasible
        both_invalid = (~candidate_feasible) & (~best_feasible)

        feasible_improvement = both_feasible & self._lexicographic_less(
            (
                candidate_metrics["near_collision_waypoint_count"],
                -candidate_metrics["min_signed_distance"],
                candidate_metrics["total_loss"],
                candidate_metrics["scalarized_cost"],
            ),
            (
                best_metrics["near_collision_waypoint_count"],
                -best_metrics["min_signed_distance"],
                best_metrics["total_loss"],
                best_metrics["scalarized_cost"],
            ),
        )
        invalid_improvement = both_invalid & self._lexicographic_less(
            (
                candidate_metrics["total_violation_count"],
                candidate_metrics["collision_waypoint_count"],
                candidate_metrics["collision_penetration_depth"],
                candidate_metrics["near_collision_waypoint_count"],
                candidate_metrics["total_violation_excess"],
                candidate_metrics["max_acceleration_excess"],
                candidate_metrics["total_loss"],
                candidate_metrics["scalarized_cost"],
            ),
            (
                best_metrics["total_violation_count"],
                best_metrics["collision_waypoint_count"],
                best_metrics["collision_penetration_depth"],
                best_metrics["near_collision_waypoint_count"],
                best_metrics["total_violation_excess"],
                best_metrics["max_acceleration_excess"],
                best_metrics["total_loss"],
                best_metrics["scalarized_cost"],
            ),
        )
        return candidate_beats_invalid_best | feasible_improvement | invalid_improvement

    def _trajectory_collision_summary(self, trajectory, jobs):
        positions = self.forward_kinematics(trajectory)
        nearest_distance, nearest_obstacle_index, obstacle_mask = self._nearest_obstacle_distances(positions, jobs)
        batch_size, waypoint_count = nearest_distance.shape
        zero_float = torch.zeros((batch_size,), dtype=self.dtype, device=self.device)
        zero_int = torch.zeros((batch_size,), dtype=torch.int64, device=self.device)
        batch_indices = torch.arange(batch_size, device=self.device)

        if obstacle_mask.shape[1] == 0:
            large_distance = torch.full((batch_size,), 1.0e6, dtype=self.dtype, device=self.device)
            return {
                "min_signed_distance": large_distance,
                "mean_signed_distance": large_distance,
                "collision_waypoint_count": zero_int,
                "near_collision_waypoint_count": zero_int,
                "collision_penetration_depth": zero_float,
                "worst_collision_waypoint_index": zero_int,
                "worst_collision_obstacle_index": torch.full((batch_size,), -1, dtype=torch.int64, device=self.device),
                "worst_collision_waypoint_fraction": zero_float,
                "region_1_to_4_min_signed_distance": large_distance,
                "region_5_plus_min_signed_distance": large_distance,
                "region_1_to_4_collision_waypoint_count": zero_int,
                "region_5_plus_collision_waypoint_count": zero_int,
                "region_5_plus_minus_1_to_4_min_signed_distance": zero_float,
                "region_5_plus_dominates_collision_peak": zero_int,
            }

        min_signed_distance = nearest_distance.amin(dim=1)
        mean_signed_distance = nearest_distance.mean(dim=1)
        collision_waypoint_count = (nearest_distance < 0.0).sum(dim=1)
        near_collision_waypoint_count = (nearest_distance < 0.02).sum(dim=1)
        collision_penetration_depth = F.relu(-min_signed_distance)
        worst_collision_waypoint_index = nearest_distance.argmin(dim=1)
        worst_collision_waypoint_fraction = (
            worst_collision_waypoint_index.to(dtype=self.dtype) / float(max(waypoint_count - 1, 1))
        )
        worst_collision_obstacle_index = nearest_obstacle_index[batch_indices, worst_collision_waypoint_index]

        def _region_summary(start_idx: int, end_idx: int | None):
            region_distance = nearest_distance[:, start_idx:end_idx]
            region_obstacle_index = nearest_obstacle_index[:, start_idx:end_idx]
            if region_distance.shape[1] == 0:
                return (
                    torch.full((batch_size,), 1.0e6, dtype=self.dtype, device=self.device),
                    zero_int,
                    zero_int,
                    torch.full((batch_size,), -1, dtype=torch.int64, device=self.device),
                )
            region_worst_local_index = region_distance.argmin(dim=1)
            region_worst_waypoint_index = region_worst_local_index + start_idx
            return (
                region_distance.amin(dim=1),
                (region_distance < 0.0).sum(dim=1),
                region_worst_waypoint_index,
                region_obstacle_index[batch_indices, region_worst_local_index],
            )

        (
            region_1_to_4_min_signed_distance,
            region_1_to_4_collision_waypoint_count,
            _region_1_to_4_worst_collision_waypoint_index,
            _region_1_to_4_worst_collision_obstacle_index,
        ) = _region_summary(0, min(4, waypoint_count))
        (
            region_5_plus_min_signed_distance,
            region_5_plus_collision_waypoint_count,
            _region_5_plus_worst_collision_waypoint_index,
            _region_5_plus_worst_collision_obstacle_index,
        ) = _region_summary(4, None)
        region_5_plus_minus_1_to_4_min_signed_distance = (
            region_5_plus_min_signed_distance - region_1_to_4_min_signed_distance
        )
        region_5_plus_dominates_collision_peak = (
            region_5_plus_min_signed_distance < region_1_to_4_min_signed_distance
        ).to(dtype=torch.int64)
        return {
            "min_signed_distance": min_signed_distance,
            "mean_signed_distance": mean_signed_distance,
            "collision_waypoint_count": collision_waypoint_count,
            "near_collision_waypoint_count": near_collision_waypoint_count,
            "collision_penetration_depth": collision_penetration_depth,
            "worst_collision_waypoint_index": worst_collision_waypoint_index,
            "worst_collision_obstacle_index": worst_collision_obstacle_index,
            "worst_collision_waypoint_fraction": worst_collision_waypoint_fraction,
            "region_1_to_4_min_signed_distance": region_1_to_4_min_signed_distance,
            "region_5_plus_min_signed_distance": region_5_plus_min_signed_distance,
            "region_1_to_4_collision_waypoint_count": region_1_to_4_collision_waypoint_count,
            "region_5_plus_collision_waypoint_count": region_5_plus_collision_waypoint_count,
            "region_5_plus_minus_1_to_4_min_signed_distance": region_5_plus_minus_1_to_4_min_signed_distance,
            "region_5_plus_dominates_collision_peak": region_5_plus_dominates_collision_peak,
        }

    def _trajectory_dynamics_summary(self, trajectory, jobs):
        batch_size = trajectory.shape[0]
        zero_float = torch.zeros((batch_size,), dtype=self.dtype, device=self.device)
        zero_int = torch.zeros((batch_size,), dtype=torch.int64, device=self.device)
        zero_profile = torch.zeros((batch_size, trajectory.shape[2]), dtype=self.dtype, device=self.device)

        if trajectory.shape[1] >= 2:
            waypoint_jump = torch.abs(torch.diff(trajectory, dim=1))
            max_adjacent_waypoint_jump = waypoint_jump.amax(dim=(1, 2))
            mean_adjacent_waypoint_jump = waypoint_jump.mean(dim=(1, 2))
        else:
            max_adjacent_waypoint_jump = zero_float
            mean_adjacent_waypoint_jump = zero_float

        if trajectory.shape[1] >= 3:
            dt = torch.as_tensor([job.dt for job in jobs], dtype=self.dtype, device=self.device).view(-1, 1, 1)
            acceleration = torch.abs(torch.diff(trajectory, n=2, dim=1)) / (dt ** 2)
            batch_indices = torch.arange(batch_size, device=self.device)
            max_acceleration_observed = acceleration.amax(dim=(1, 2))
            mean_acceleration_magnitude = acceleration.mean(dim=(1, 2))
            acceleration_violation_count = (acceleration > self.max_acceleration).sum(dim=(1, 2))
            max_acceleration_excess = F.relu(acceleration - self.max_acceleration).amax(dim=(1, 2))
            flat_peak = acceleration.reshape(acceleration.shape[0], -1).argmax(dim=1)
            peak_acceleration_waypoint_index = torch.div(
                flat_peak,
                acceleration.shape[2],
                rounding_mode="floor",
            ) + 1
            peak_acceleration_joint_index = torch.remainder(flat_peak, acceleration.shape[2])
            peak_acceleration_waypoint_joint_profile = acceleration[
                batch_indices,
                peak_acceleration_waypoint_index - 1,
                :,
            ]

            def _region_summary(start_idx: int, end_idx: int | None):
                region = acceleration[:, start_idx:end_idx, :]
                if region.shape[1] == 0:
                    return zero_float, zero_float, zero_int, zero_int, zero_int, zero_profile
                region_flat_peak = region.reshape(region.shape[0], -1).argmax(dim=1)
                region_peak_local_waypoint = torch.div(
                    region_flat_peak,
                    region.shape[2],
                    rounding_mode="floor",
                )
                region_peak_joint_index = torch.remainder(region_flat_peak, region.shape[2])
                region_peak_waypoint_index = region_peak_local_waypoint + start_idx + 1
                return (
                    region.amax(dim=(1, 2)),
                    region.mean(dim=(1, 2)),
                    (region > self.max_acceleration).sum(dim=(1, 2)),
                    region_peak_waypoint_index,
                    region_peak_joint_index,
                    region[batch_indices, region_peak_local_waypoint, :],
                )

            (
                region_1_to_4_max_acceleration_observed,
                region_1_to_4_mean_acceleration_magnitude,
                region_1_to_4_acceleration_violation_count,
                region_1_to_4_peak_acceleration_waypoint_index,
                region_1_to_4_peak_acceleration_joint_index,
                region_1_to_4_peak_waypoint_joint_acceleration_profile,
            ) = _region_summary(
                0,
                min(4, acceleration.shape[1]),
            )
            (
                region_5_plus_max_acceleration_observed,
                region_5_plus_mean_acceleration_magnitude,
                region_5_plus_acceleration_violation_count,
                region_5_plus_peak_acceleration_waypoint_index,
                region_5_plus_peak_acceleration_joint_index,
                region_5_plus_peak_waypoint_joint_acceleration_profile,
            ) = _region_summary(
                4,
                None,
            )
            region_5_plus_minus_1_to_4_max_acceleration_observed = (
                region_5_plus_max_acceleration_observed - region_1_to_4_max_acceleration_observed
            )
            region_5_plus_dominates_acceleration_peak = (
                region_5_plus_max_acceleration_observed > region_1_to_4_max_acceleration_observed
            ).to(dtype=torch.int64)
        else:
            max_acceleration_observed = zero_float
            mean_acceleration_magnitude = zero_float
            acceleration_violation_count = zero_int
            max_acceleration_excess = zero_float
            peak_acceleration_waypoint_index = zero_int
            peak_acceleration_joint_index = zero_int
            peak_acceleration_waypoint_joint_profile = zero_profile
            region_1_to_4_max_acceleration_observed = zero_float
            region_1_to_4_mean_acceleration_magnitude = zero_float
            region_1_to_4_acceleration_violation_count = zero_int
            region_1_to_4_peak_acceleration_waypoint_index = zero_int
            region_1_to_4_peak_acceleration_joint_index = zero_int
            region_1_to_4_peak_waypoint_joint_acceleration_profile = zero_profile
            region_5_plus_max_acceleration_observed = zero_float
            region_5_plus_mean_acceleration_magnitude = zero_float
            region_5_plus_acceleration_violation_count = zero_int
            region_5_plus_peak_acceleration_waypoint_index = zero_int
            region_5_plus_peak_acceleration_joint_index = zero_int
            region_5_plus_peak_waypoint_joint_acceleration_profile = zero_profile
            region_5_plus_minus_1_to_4_max_acceleration_observed = zero_float
            region_5_plus_dominates_acceleration_peak = zero_int

        waypoint_denominator = max(int(trajectory.shape[1]) - 1, 1)
        peak_acceleration_waypoint_fraction = (
            peak_acceleration_waypoint_index.to(dtype=self.dtype) / float(waypoint_denominator)
        )
        max_acceleration_limit = torch.full((batch_size,), float(self.max_acceleration), dtype=self.dtype, device=self.device)
        return {
            "max_acceleration_limit": max_acceleration_limit,
            "max_acceleration_observed": max_acceleration_observed,
            "mean_acceleration_magnitude": mean_acceleration_magnitude,
            "acceleration_violation_count": acceleration_violation_count,
            "max_acceleration_excess": max_acceleration_excess,
            "peak_acceleration_waypoint_index": peak_acceleration_waypoint_index,
            "peak_acceleration_joint_index": peak_acceleration_joint_index,
            "peak_acceleration_waypoint_joint_acceleration_profile": peak_acceleration_waypoint_joint_profile,
            "peak_acceleration_waypoint_fraction": peak_acceleration_waypoint_fraction,
            "max_adjacent_waypoint_jump": max_adjacent_waypoint_jump,
            "mean_adjacent_waypoint_jump": mean_adjacent_waypoint_jump,
            "region_1_to_4_max_acceleration_observed": region_1_to_4_max_acceleration_observed,
            "region_1_to_4_mean_acceleration_magnitude": region_1_to_4_mean_acceleration_magnitude,
            "region_1_to_4_acceleration_violation_count": region_1_to_4_acceleration_violation_count,
            "region_1_to_4_peak_acceleration_waypoint_index": region_1_to_4_peak_acceleration_waypoint_index,
            "region_1_to_4_peak_acceleration_joint_index": region_1_to_4_peak_acceleration_joint_index,
            "region_1_to_4_peak_waypoint_joint_acceleration_profile": (
                region_1_to_4_peak_waypoint_joint_acceleration_profile
            ),
            "region_5_plus_max_acceleration_observed": region_5_plus_max_acceleration_observed,
            "region_5_plus_mean_acceleration_magnitude": region_5_plus_mean_acceleration_magnitude,
            "region_5_plus_acceleration_violation_count": region_5_plus_acceleration_violation_count,
            "region_5_plus_peak_acceleration_waypoint_index": region_5_plus_peak_acceleration_waypoint_index,
            "region_5_plus_peak_acceleration_joint_index": region_5_plus_peak_acceleration_joint_index,
            "region_5_plus_peak_waypoint_joint_acceleration_profile": (
                region_5_plus_peak_waypoint_joint_acceleration_profile
            ),
            "region_5_plus_minus_1_to_4_max_acceleration_observed": (
                region_5_plus_minus_1_to_4_max_acceleration_observed
            ),
            "region_5_plus_dominates_acceleration_peak": region_5_plus_dominates_acceleration_peak,
        }

    def solve_batch(self, jobs: Sequence[TorchPlannerJob]) -> list[TorchPlannerResult]:
        require_torch()
        if not jobs:
            return []

        import dataclasses

        start_time = time.perf_counter()
        start_config = torch.as_tensor([job.start_config for job in jobs], dtype=self.dtype, device=self.device)
        goal_config = torch.as_tensor([job.goal_config for job in jobs], dtype=self.dtype, device=self.device)
        
        updated_jobs = []
        for index, job in enumerate(jobs):
            max_dist = float(torch.max(torch.abs(goal_config[index] - start_config[index])).item())
            min_time = (max_dist * 2.0) / max(0.8 * self.max_velocity, 1e-4)
            min_dt = min_time / max(self.n_waypoints - 1, 1)
            resolved_dt = max(float(job.dt), min_dt)
            updated_jobs.append(dataclasses.replace(job, dt=resolved_dt))
        jobs = updated_jobs

        latent = torch.nn.Parameter(
            self._initialize_latent(
                start_config,
                goal_config,
                [job.seed for job in jobs],
                [job.dt for job in jobs],
                jobs=jobs,
            )
        )
        optimizer = torch.optim.Adam([latent], lr=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.planner_steps)

        best_trajectory = self._decode_trajectory(latent.detach(), start_config, goal_config).detach().clone()
        initial_trajectory = best_trajectory.detach().clone()
        checkpoint_limit = min(int(self.planner_steps), 5)
        dynamics_checkpoints: list[tuple[int, dict[str, object]]] = []
        initial_checkpoint_summary = self._trajectory_dynamics_summary(initial_trajectory, jobs)
        initial_collision_summary = self._trajectory_collision_summary(initial_trajectory, jobs)
        dynamics_checkpoints.append((0, {"dynamics": initial_checkpoint_summary, "collision": initial_collision_summary}))
        initial_metrics = self._objective_terms(initial_trajectory, jobs)
        best_metrics = {key: value.detach().clone() for key, value in initial_metrics.items()}
        best_total_loss = best_metrics["total_loss"].clone()
        best_scalarized = best_metrics["scalarized_cost"].clone()
        best_mean_loss = float("inf")
        steps_without_improvement = 0
        improvement_tolerance = 1e-4
        completed_steps = 0

        for step_index in range(self.planner_steps):
            optimizer.zero_grad(set_to_none=True)
            trajectory = self._decode_trajectory(latent, start_config, goal_config)
            metrics = self._objective_terms(trajectory, jobs)
            mean_loss = metrics["total_loss"].mean()
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_([latent], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            completed_steps = step_index + 1

            with torch.no_grad():
                if completed_steps <= checkpoint_limit:
                    checkpoint_trajectory = self._decode_trajectory(latent.detach(), start_config, goal_config)
                    dynamics_checkpoints.append(
                        (
                            completed_steps,
                            {
                                "dynamics": self._trajectory_dynamics_summary(checkpoint_trajectory, jobs),
                                "collision": self._trajectory_collision_summary(checkpoint_trajectory, jobs),
                            },
                        )
                    )
                improved = self._select_candidate_update_mask(metrics, best_metrics)
                detached_trajectory = trajectory.detach()
                best_trajectory = torch.where(improved[:, None, None], detached_trajectory, best_trajectory)
                for key, value in metrics.items():
                    best_metrics[key] = torch.where(improved, value.detach(), best_metrics[key])
                best_total_loss = best_metrics["total_loss"]
                best_scalarized = best_metrics["scalarized_cost"]
                current_mean_loss = float(metrics["total_loss"].mean().item())
                if current_mean_loss < best_mean_loss * (1.0 - improvement_tolerance):
                    best_mean_loss = current_mean_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
                if steps_without_improvement >= max(20, self.planner_steps // 5):
                    break

        elapsed = time.perf_counter() - start_time
        initial_surrogate_dynamics = initial_checkpoint_summary
        initial_surrogate_collision = initial_collision_summary
        surrogate_dynamics = self._trajectory_dynamics_summary(best_trajectory, jobs)
        surrogate_collision = self._trajectory_collision_summary(best_trajectory, jobs)
        trajectories = best_trajectory.detach().cpu().numpy().astype(np.float64)
        warm_start_metadata = self._latest_warm_start_metadata
        return [
            TorchPlannerResult(
                worker_id=job.worker_id,
                request_id=job.request_id,
                order_index=job.order_index,
                trajectory=trajectories[index],
                dt=float(job.dt),
                iterations=int(completed_steps),
                final_optimization_cost=float(best_total_loss[index].detach().cpu().item()),
                scalarized_surrogate_cost=float(best_scalarized[index].detach().cpu().item()),
                optimizer_steps=int(completed_steps),
                batched_jobs=len(jobs),
                device=str(self.device),
                duration_sec=float(elapsed),
                surrogate_initial_trajectory_dynamics={
                    **serialize_trajectory_dynamics_summary(initial_surrogate_dynamics, index),
                    **serialize_trajectory_collision_summary(initial_surrogate_collision, index),
                    **(warm_start_metadata[index] if index < len(warm_start_metadata) else {}),
                },
                surrogate_trajectory_dynamics={
                    **serialize_trajectory_dynamics_summary(surrogate_dynamics, index),
                    **serialize_trajectory_collision_summary(surrogate_collision, index),
                },
                surrogate_dynamics_checkpoints=tuple(
                    serialize_dynamics_checkpoint(
                        summary["dynamics"],
                        summary["collision"],
                        index,
                        optimizer_iteration=optimizer_iteration,
                    )
                    for optimizer_iteration, summary in dynamics_checkpoints
                ),
                warm_start_metadata=(warm_start_metadata[index] if index < len(warm_start_metadata) else {}),
            )
            for index, job in enumerate(jobs)
        ]

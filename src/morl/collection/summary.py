"""
Helpers for crunching numbers and summarizing how the dataset collection went.
Includes tracking for repairs, surrogate dynamics, and optimization stats.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

@dataclass
class RepairUsageAccumulator:
    """
    Tracks how often we had to "repair" trajectories because they 
    violated dynamics or hit obstacles. 
    """
    repair_used_trajectory_count: int = 0
    slsqp_iteration_total: int = 0
    slsqp_function_evaluation_total: int = 0
    slsqp_gradient_evaluation_total: int = 0
    raw_dynamics_velocity_violation_total: int = 0
    raw_dynamics_acceleration_violation_total: int = 0
    raw_dynamics_max_velocity_excess: float = 0.0
    raw_dynamics_max_acceleration_excess: float = 0.0
    optimizer_duration_total: float = 0.0
    optimizer_duration_values: list[float] = field(default_factory=list)
    rescue_attempted_trajectory_count: int = 0
    rescue_success_trajectory_count: int = 0
    warm_start_rrt_attempted_trajectory_count: int = 0
    warm_start_rrt_replaced_trajectory_count: int = 0

    def ingest(self, record: dict[str, object], repair_reason_counts: dict[str, int]) -> None:
        """Add data from a single record into our running totals."""
        optimization = record.get("optimization", {})
        if not isinstance(optimization, dict):
            return
        duration_raw = optimization.get("duration_sec")
        if duration_raw is not None:
            duration_value = float(duration_raw or 0.0)
            self.optimizer_duration_total += duration_value
            self.optimizer_duration_values.append(duration_value)
        if bool(optimization.get("rescue_attempted")):
            self.rescue_attempted_trajectory_count += 1
        if bool(optimization.get("rescue_success")):
            self.rescue_success_trajectory_count += 1
        warm_start = optimization.get("surrogate_initial_trajectory_dynamics")
        if isinstance(warm_start, dict):
            if bool(warm_start.get("warm_start_rrt_attempted")):
                self.warm_start_rrt_attempted_trajectory_count += 1
            if str(warm_start.get("warm_start_strategy", "")) == "rrt":
                self.warm_start_rrt_replaced_trajectory_count += 1
        if not bool(optimization.get("repair_used")):
            return
        self.repair_used_trajectory_count += 1
        repair = optimization.get("repair", {})
        reason = optimization.get("repair_validation_failure_reason")
        if not isinstance(reason, str) or not reason:
            if isinstance(repair, dict):
                nested_reason = repair.get("validation_failure_reason")
                reason = str(nested_reason) if nested_reason else "unknown"
            else:
                reason = "unknown"
        repair_reason_counts[reason] = repair_reason_counts.get(reason, 0) + 1
        if isinstance(repair, dict):
            self.slsqp_iteration_total += int(repair.get("slsqp_iterations", 0) or 0)
            self.slsqp_function_evaluation_total += int(repair.get("slsqp_function_evaluations", 0) or 0)
            self.slsqp_gradient_evaluation_total += int(repair.get("slsqp_gradient_evaluations", 0) or 0)
        raw_dynamics = optimization.get("raw_dynamics_violation")
        if not isinstance(raw_dynamics, dict) and isinstance(repair, dict):
            nested_raw_dynamics = repair.get("raw_dynamics_violation")
            raw_dynamics = nested_raw_dynamics if isinstance(nested_raw_dynamics, dict) else None
        if isinstance(raw_dynamics, dict):
            self.raw_dynamics_velocity_violation_total += int(raw_dynamics.get("velocity_violation_count", 0) or 0)
            self.raw_dynamics_acceleration_violation_total += int(
                raw_dynamics.get("acceleration_violation_count", 0) or 0
            )
            self.raw_dynamics_max_velocity_excess = max(
                self.raw_dynamics_max_velocity_excess,
                float(raw_dynamics.get("max_velocity_excess", 0.0) or 0.0),
            )
            self.raw_dynamics_max_acceleration_excess = max(
                self.raw_dynamics_max_acceleration_excess,
                float(raw_dynamics.get("max_acceleration_excess", 0.0) or 0.0),
            )

    def finalize(self, raw_trajectory_count: int, repair_reason_counts: dict[str, int]) -> dict[str, float | int]:
        """Calculate averages and final percentages."""
        repair_free_trajectory_count = raw_trajectory_count - self.repair_used_trajectory_count
        repair_rate = (
            float(self.repair_used_trajectory_count) / float(raw_trajectory_count) if raw_trajectory_count > 0 else 0.0
        )
        optimizer_duration_count = len(self.optimizer_duration_values)
        optimizer_duration_mean = (
            float(self.optimizer_duration_total) / float(optimizer_duration_count)
            if optimizer_duration_count > 0
            else 0.0
        )
        if optimizer_duration_count > 0:
            sorted_durations = sorted(self.optimizer_duration_values)
            percentile_index = max(0, math.ceil(0.90 * float(optimizer_duration_count)) - 1)
            optimizer_duration_p90 = float(sorted_durations[min(percentile_index, optimizer_duration_count - 1)])
        else:
            optimizer_duration_p90 = 0.0
        return {
            "repair_used_trajectory_count": self.repair_used_trajectory_count,
            "repair_free_trajectory_count": repair_free_trajectory_count,
            "repair_rate": repair_rate,
            "optimizer_duration_mean_sec": optimizer_duration_mean,
            "optimizer_duration_p90_sec": optimizer_duration_p90,
            "rescue_attempted_trajectory_count": self.rescue_attempted_trajectory_count,
            "rescue_success_trajectory_count": self.rescue_success_trajectory_count,
            "warm_start_rrt_attempted_trajectory_count": self.warm_start_rrt_attempted_trajectory_count,
            "warm_start_rrt_replaced_trajectory_count": self.warm_start_rrt_replaced_trajectory_count,
            "repair_invocation_count_by_validation_failure_reason": repair_reason_counts,
            "repair_slsqp_iteration_total": self.slsqp_iteration_total,
            "repair_slsqp_function_evaluation_total": self.slsqp_function_evaluation_total,
            "repair_slsqp_gradient_evaluation_total": self.slsqp_gradient_evaluation_total,
            "repair_mean_slsqp_iterations": (
                float(self.slsqp_iteration_total) / float(self.repair_used_trajectory_count)
                if self.repair_used_trajectory_count > 0
                else 0.0
            ),
            "repair_mean_slsqp_function_evaluations": (
                float(self.slsqp_function_evaluation_total) / float(self.repair_used_trajectory_count)
                if self.repair_used_trajectory_count > 0
                else 0.0
            ),
            "repair_mean_slsqp_gradient_evaluations": (
                float(self.slsqp_gradient_evaluation_total) / float(self.repair_used_trajectory_count)
                if self.repair_used_trajectory_count > 0
                else 0.0
            ),
            "repair_raw_dynamics_velocity_violation_total": self.raw_dynamics_velocity_violation_total,
            "repair_raw_dynamics_acceleration_violation_total": self.raw_dynamics_acceleration_violation_total,
            "repair_raw_dynamics_max_velocity_excess": self.raw_dynamics_max_velocity_excess,
            "repair_raw_dynamics_max_acceleration_excess": self.raw_dynamics_max_acceleration_excess,
            "repair_mean_raw_dynamics_velocity_violations": (
                float(self.raw_dynamics_velocity_violation_total) / float(self.repair_used_trajectory_count)
                if self.repair_used_trajectory_count > 0
                else 0.0
            ),
            "repair_mean_raw_dynamics_acceleration_violations": (
                float(self.raw_dynamics_acceleration_violation_total) / float(self.repair_used_trajectory_count)
                if self.repair_used_trajectory_count > 0
                else 0.0
            ),
        }

@dataclass
class SurrogateSummaryAccumulator:
    """
    Accumulates stats about how well the surrogate model matches the actual dynamics.
    We track things like max acceleration and collision distances.
    """
    summary_count: int = 0
    max_acceleration_observed_total: float = 0.0
    mean_acceleration_magnitude_total: float = 0.0
    max_acceleration_excess_total: float = 0.0
    peak_acceleration_waypoint_fraction_total: float = 0.0
    max_adjacent_waypoint_jump_total: float = 0.0
    region_1_to_4_max_acceleration_observed_total: float = 0.0
    region_5_plus_max_acceleration_observed_total: float = 0.0
    region_1_to_4_acceleration_violation_count_total: int = 0
    region_5_plus_acceleration_violation_count_total: int = 0
    region_5_plus_minus_1_to_4_max_acceleration_observed_total: float = 0.0
    region_5_plus_dominates_acceleration_peak_count: int = 0
    max_acceleration_observed: float = 0.0
    max_acceleration_excess: float = 0.0
    max_adjacent_waypoint_jump: float = 0.0
    region_1_to_4_max_acceleration_observed: float = 0.0
    region_5_plus_max_acceleration_observed: float = 0.0
    max_acceleration_violation_count: int = 0
    acceleration_limit_exceed_trajectory_count: int = 0
    acceleration_violation_count_total: int = 0
    min_signed_distance_total: float = 0.0
    mean_signed_distance_total: float = 0.0
    collision_waypoint_count_total: int = 0
    near_collision_waypoint_count_total: int = 0
    collision_penetration_depth_total: float = 0.0
    worst_collision_waypoint_fraction_total: float = 0.0
    region_1_to_4_min_signed_distance_total: float = 0.0
    region_5_plus_min_signed_distance_total: float = 0.0
    region_1_to_4_collision_waypoint_count_total: int = 0
    region_5_plus_collision_waypoint_count_total: int = 0
    region_5_plus_minus_1_to_4_min_signed_distance_total: float = 0.0
    region_5_plus_dominates_collision_peak_count: int = 0
    min_signed_distance: float = 1.0e6
    max_collision_waypoint_count: int = 0
    max_near_collision_waypoint_count: int = 0
    max_collision_penetration_depth: float = 0.0
    collision_limit_exceed_trajectory_count: int = 0

    def ingest(self, surrogate: dict[str, object]) -> None:
        """Add surrogate data from one planning run."""
        self.summary_count += 1
        observed = float(surrogate.get("max_acceleration_observed", 0.0) or 0.0)
        mean_magnitude = float(surrogate.get("mean_acceleration_magnitude", 0.0) or 0.0)
        violation_count = int(surrogate.get("acceleration_violation_count", 0) or 0)
        excess = float(surrogate.get("max_acceleration_excess", 0.0) or 0.0)
        peak_fraction = float(surrogate.get("peak_acceleration_waypoint_fraction", 0.0) or 0.0)
        max_jump = float(surrogate.get("max_adjacent_waypoint_jump", 0.0) or 0.0)
        region_1_to_4_observed = float(surrogate.get("region_1_to_4_max_acceleration_observed", 0.0) or 0.0)
        region_5_plus_observed = float(surrogate.get("region_5_plus_max_acceleration_observed", 0.0) or 0.0)
        region_1_to_4_violations = int(surrogate.get("region_1_to_4_acceleration_violation_count", 0) or 0)
        region_5_plus_violations = int(surrogate.get("region_5_plus_acceleration_violation_count", 0) or 0)
        region_delta = float(surrogate.get("region_5_plus_minus_1_to_4_max_acceleration_observed", 0.0) or 0.0)
        region_5_plus_dominates = int(surrogate.get("region_5_plus_dominates_acceleration_peak", 0) or 0)
        min_distance_value = float(surrogate.get("min_signed_distance", 1.0e6) or 1.0e6)
        mean_distance_value = float(surrogate.get("mean_signed_distance", 0.0) or 0.0)
        collision_waypoint_count = int(surrogate.get("collision_waypoint_count", 0) or 0)
        near_collision_waypoint_count = int(surrogate.get("near_collision_waypoint_count", 0) or 0)
        collision_penetration_depth = float(surrogate.get("collision_penetration_depth", 0.0) or 0.0)
        worst_collision_waypoint_fraction = float(surrogate.get("worst_collision_waypoint_fraction", 0.0) or 0.0)
        region_1_to_4_min_distance = float(surrogate.get("region_1_to_4_min_signed_distance", 1.0e6) or 1.0e6)
        region_5_plus_min_distance = float(surrogate.get("region_5_plus_min_signed_distance", 1.0e6) or 1.0e6)
        region_1_to_4_collision_count = int(surrogate.get("region_1_to_4_collision_waypoint_count", 0) or 0)
        region_5_plus_collision_count = int(surrogate.get("region_5_plus_collision_waypoint_count", 0) or 0)
        region_collision_delta = float(surrogate.get("region_5_plus_minus_1_to_4_min_signed_distance", 0.0) or 0.0)
        region_5_plus_dominates_collision = int(surrogate.get("region_5_plus_dominates_collision_peak", 0) or 0)
        self.max_acceleration_observed_total += observed
        self.mean_acceleration_magnitude_total += mean_magnitude
        self.acceleration_violation_count_total += violation_count
        self.max_acceleration_excess_total += excess
        self.peak_acceleration_waypoint_fraction_total += peak_fraction
        self.max_adjacent_waypoint_jump_total += max_jump
        self.region_1_to_4_max_acceleration_observed_total += region_1_to_4_observed
        self.region_5_plus_max_acceleration_observed_total += region_5_plus_observed
        self.region_1_to_4_acceleration_violation_count_total += region_1_to_4_violations
        self.region_5_plus_acceleration_violation_count_total += region_5_plus_violations
        self.region_5_plus_minus_1_to_4_max_acceleration_observed_total += region_delta
        self.region_5_plus_dominates_acceleration_peak_count += region_5_plus_dominates
        self.min_signed_distance_total += min_distance_value
        self.mean_signed_distance_total += mean_distance_value
        self.collision_waypoint_count_total += collision_waypoint_count
        self.near_collision_waypoint_count_total += near_collision_waypoint_count
        self.collision_penetration_depth_total += collision_penetration_depth
        self.worst_collision_waypoint_fraction_total += worst_collision_waypoint_fraction
        self.region_1_to_4_min_signed_distance_total += region_1_to_4_min_distance
        self.region_5_plus_min_signed_distance_total += region_5_plus_min_distance
        self.region_1_to_4_collision_waypoint_count_total += region_1_to_4_collision_count
        self.region_5_plus_collision_waypoint_count_total += region_5_plus_collision_count
        self.region_5_plus_minus_1_to_4_min_signed_distance_total += region_collision_delta
        self.region_5_plus_dominates_collision_peak_count += region_5_plus_dominates_collision
        self.max_acceleration_observed = max(self.max_acceleration_observed, observed)
        self.max_acceleration_excess = max(self.max_acceleration_excess, excess)
        self.max_adjacent_waypoint_jump = max(self.max_adjacent_waypoint_jump, max_jump)
        self.region_1_to_4_max_acceleration_observed = max(
            self.region_1_to_4_max_acceleration_observed,
            region_1_to_4_observed,
        )
        self.region_5_plus_max_acceleration_observed = max(
            self.region_5_plus_max_acceleration_observed,
            region_5_plus_observed,
        )
        self.max_acceleration_violation_count = max(self.max_acceleration_violation_count, violation_count)
        self.min_signed_distance = min(self.min_signed_distance, min_distance_value)
        self.max_collision_waypoint_count = max(self.max_collision_waypoint_count, collision_waypoint_count)
        self.max_near_collision_waypoint_count = max(
            self.max_near_collision_waypoint_count,
            near_collision_waypoint_count,
        )
        self.max_collision_penetration_depth = max(
            self.max_collision_penetration_depth,
            collision_penetration_depth,
        )
        if excess > 0.0:
            self.acceleration_limit_exceed_trajectory_count += 1
        if collision_waypoint_count > 0:
            self.collision_limit_exceed_trajectory_count += 1

    def finalize(self, *, prefix: str) -> dict[str, float | int]:
        """Wrap up the totals into a neat dictionary of averages and maxes."""
        mean_denominator = float(self.summary_count) if self.summary_count > 0 else 1.0
        return {
            f"{prefix}_dynamics_summary_trajectory_count": self.summary_count,
            f"{prefix}_mean_max_acceleration_observed": (
                self.max_acceleration_observed_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_max_acceleration_observed": self.max_acceleration_observed,
            f"{prefix}_mean_mean_acceleration_magnitude": (
                self.mean_acceleration_magnitude_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_acceleration_limit_exceed_trajectory_count": self.acceleration_limit_exceed_trajectory_count,
            f"{prefix}_mean_acceleration_violation_count": (
                float(self.acceleration_violation_count_total) / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_max_acceleration_violation_count": self.max_acceleration_violation_count,
            f"{prefix}_mean_max_acceleration_excess": (
                self.max_acceleration_excess_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_max_acceleration_excess": self.max_acceleration_excess,
            f"{prefix}_mean_peak_acceleration_waypoint_fraction": (
                self.peak_acceleration_waypoint_fraction_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_mean_max_adjacent_waypoint_jump": (
                self.max_adjacent_waypoint_jump_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_max_adjacent_waypoint_jump": self.max_adjacent_waypoint_jump,
            f"{prefix}_mean_region_1_to_4_max_acceleration_observed": (
                self.region_1_to_4_max_acceleration_observed_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_max_region_1_to_4_max_acceleration_observed": self.region_1_to_4_max_acceleration_observed,
            f"{prefix}_mean_region_5_plus_max_acceleration_observed": (
                self.region_5_plus_max_acceleration_observed_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_max_region_5_plus_max_acceleration_observed": self.region_5_plus_max_acceleration_observed,
            f"{prefix}_mean_region_1_to_4_acceleration_violation_count": (
                float(self.region_1_to_4_acceleration_violation_count_total) / mean_denominator
                if self.summary_count > 0
                else 0.0
            ),
            f"{prefix}_mean_region_5_plus_acceleration_violation_count": (
                float(self.region_5_plus_acceleration_violation_count_total) / mean_denominator
                if self.summary_count > 0
                else 0.0
            ),
            f"{prefix}_mean_region_5_plus_minus_1_to_4_max_acceleration_observed": (
                self.region_5_plus_minus_1_to_4_max_acceleration_observed_total / mean_denominator
                if self.summary_count > 0
                else 0.0
            ),
            f"{prefix}_region_5_plus_dominates_acceleration_peak_trajectory_count": (
                self.region_5_plus_dominates_acceleration_peak_count
            ),
            f"{prefix}_mean_min_signed_distance": (
                self.min_signed_distance_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_min_signed_distance": (self.min_signed_distance if self.summary_count > 0 else 0.0),
            f"{prefix}_mean_mean_signed_distance": (
                self.mean_signed_distance_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_collision_limit_exceed_trajectory_count": self.collision_limit_exceed_trajectory_count,
            f"{prefix}_mean_collision_waypoint_count": (
                float(self.collision_waypoint_count_total) / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_max_collision_waypoint_count": self.max_collision_waypoint_count,
            f"{prefix}_mean_near_collision_waypoint_count": (
                float(self.near_collision_waypoint_count_total) / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_max_near_collision_waypoint_count": self.max_near_collision_waypoint_count,
            f"{prefix}_mean_collision_penetration_depth": (
                self.collision_penetration_depth_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_max_collision_penetration_depth": self.max_collision_penetration_depth,
            f"{prefix}_mean_worst_collision_waypoint_fraction": (
                self.worst_collision_waypoint_fraction_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_mean_region_1_to_4_min_signed_distance": (
                self.region_1_to_4_min_signed_distance_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_mean_region_5_plus_min_signed_distance": (
                self.region_5_plus_min_signed_distance_total / mean_denominator if self.summary_count > 0 else 0.0
            ),
            f"{prefix}_mean_region_1_to_4_collision_waypoint_count": (
                float(self.region_1_to_4_collision_waypoint_count_total) / mean_denominator
                if self.summary_count > 0
                else 0.0
            ),
            f"{prefix}_mean_region_5_plus_collision_waypoint_count": (
                float(self.region_5_plus_collision_waypoint_count_total) / mean_denominator
                if self.summary_count > 0
                else 0.0
            ),
            f"{prefix}_mean_region_5_plus_minus_1_to_4_min_signed_distance": (
                self.region_5_plus_minus_1_to_4_min_signed_distance_total / mean_denominator
                if self.summary_count > 0
                else 0.0
            ),
            f"{prefix}_region_5_plus_dominates_collision_peak_trajectory_count": (
                self.region_5_plus_dominates_collision_peak_count
            ),
        }

def _repair_usage_summary(records: list[dict[str, object]]) -> dict[str, float | int]:
    """Helper to get the full repair usage report from a list of records."""
    accumulator = RepairUsageAccumulator()
    repair_reason_counts: dict[str, int] = {}
    for record in records:
        accumulator.ingest(record, repair_reason_counts)
    return accumulator.finalize(len(records), repair_reason_counts)

def _surrogate_trajectory_dynamics_summary_for_key(
    records: list[dict[str, object]],
    *,
    optimization_key: str,
    prefix: str,
) -> dict[str, float | int]:
    """Base logic for summarizing surrogate data for any given key in the optimization dict."""
    accumulator = SurrogateSummaryAccumulator()
    for record in records:
        optimization = record.get("optimization", {})
        if not isinstance(optimization, dict):
            continue
        surrogate = optimization.get(optimization_key)
        if not isinstance(surrogate, dict):
            continue
        accumulator.ingest(surrogate)
    return accumulator.finalize(prefix=prefix)

def _surrogate_trajectory_dynamics_summary(records: list[dict[str, object]]) -> dict[str, float | int]:
    """Get surrogate stats for the final trajectory."""
    return _surrogate_trajectory_dynamics_summary_for_key(
        records,
        optimization_key="surrogate_trajectory_dynamics",
        prefix="surrogate",
    )

def _surrogate_initial_trajectory_dynamics_summary(records: list[dict[str, object]]) -> dict[str, float | int]:
    """Get surrogate stats for the initial trajectory (before optimization)."""
    return _surrogate_trajectory_dynamics_summary_for_key(
        records,
        optimization_key="surrogate_initial_trajectory_dynamics",
        prefix="surrogate_initial",
    )

def _surrogate_dynamics_checkpoint_summary(records: list[dict[str, object]]) -> dict[str, float | int]:
    """Summarize surrogate data from intermediate optimization checkpoints."""
    summaries: dict[str, float | int] = {}
    grouped_records: dict[int, list[dict[str, object]]] = {}
    for record in records:
        optimization = record.get("optimization", {})
        if not isinstance(optimization, dict):
            continue
        checkpoints = optimization.get("surrogate_dynamics_checkpoints")
        if not isinstance(checkpoints, (list, tuple)):
            continue
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            optimizer_iteration_raw = checkpoint.get("optimizer_iteration", -1)
            optimizer_iteration = int(-1 if optimizer_iteration_raw is None else optimizer_iteration_raw)
            if optimizer_iteration < 0:
                continue
            grouped_records.setdefault(optimizer_iteration, []).append(
                {"optimization": {"surrogate_dynamics_checkpoint": checkpoint}}
            )
    for optimizer_iteration, checkpoint_records in sorted(grouped_records.items()):
        summaries.update(
            _surrogate_trajectory_dynamics_summary_for_key(
                checkpoint_records,
                optimization_key="surrogate_dynamics_checkpoint",
                prefix=f"surrogate_checkpoint_iter_{optimizer_iteration}",
            )
        )
    return summaries

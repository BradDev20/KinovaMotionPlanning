from __future__ import annotations


def _cpu_item(summary, key: str, index: int):
    return summary[key][index].detach().cpu().item()


def _cpu_list(summary, key: str, index: int) -> list[float]:
    return [float(value) for value in summary[key][index].detach().cpu().tolist()]


def serialize_trajectory_dynamics_summary(summary, index: int) -> dict[str, object]:
    return {
        "max_acceleration_limit": float(_cpu_item(summary, "max_acceleration_limit", index)),
        "max_acceleration_observed": float(_cpu_item(summary, "max_acceleration_observed", index)),
        "mean_acceleration_magnitude": float(_cpu_item(summary, "mean_acceleration_magnitude", index)),
        "acceleration_violation_count": int(_cpu_item(summary, "acceleration_violation_count", index)),
        "max_acceleration_excess": float(_cpu_item(summary, "max_acceleration_excess", index)),
        "peak_acceleration_waypoint_index": int(_cpu_item(summary, "peak_acceleration_waypoint_index", index)),
        "peak_acceleration_joint_index": int(_cpu_item(summary, "peak_acceleration_joint_index", index)),
        "peak_acceleration_waypoint_joint_acceleration_profile": _cpu_list(
            summary,
            "peak_acceleration_waypoint_joint_acceleration_profile",
            index,
        ),
        "peak_acceleration_waypoint_fraction": float(_cpu_item(summary, "peak_acceleration_waypoint_fraction", index)),
        "max_adjacent_waypoint_jump": float(_cpu_item(summary, "max_adjacent_waypoint_jump", index)),
        "mean_adjacent_waypoint_jump": float(_cpu_item(summary, "mean_adjacent_waypoint_jump", index)),
        "region_1_to_4_max_acceleration_observed": float(
            _cpu_item(summary, "region_1_to_4_max_acceleration_observed", index)
        ),
        "region_1_to_4_mean_acceleration_magnitude": float(
            _cpu_item(summary, "region_1_to_4_mean_acceleration_magnitude", index)
        ),
        "region_1_to_4_acceleration_violation_count": int(
            _cpu_item(summary, "region_1_to_4_acceleration_violation_count", index)
        ),
        "region_1_to_4_peak_acceleration_waypoint_index": int(
            _cpu_item(summary, "region_1_to_4_peak_acceleration_waypoint_index", index)
        ),
        "region_1_to_4_peak_acceleration_joint_index": int(
            _cpu_item(summary, "region_1_to_4_peak_acceleration_joint_index", index)
        ),
        "region_1_to_4_peak_waypoint_joint_acceleration_profile": _cpu_list(
            summary,
            "region_1_to_4_peak_waypoint_joint_acceleration_profile",
            index,
        ),
        "region_5_plus_max_acceleration_observed": float(
            _cpu_item(summary, "region_5_plus_max_acceleration_observed", index)
        ),
        "region_5_plus_mean_acceleration_magnitude": float(
            _cpu_item(summary, "region_5_plus_mean_acceleration_magnitude", index)
        ),
        "region_5_plus_acceleration_violation_count": int(
            _cpu_item(summary, "region_5_plus_acceleration_violation_count", index)
        ),
        "region_5_plus_peak_acceleration_waypoint_index": int(
            _cpu_item(summary, "region_5_plus_peak_acceleration_waypoint_index", index)
        ),
        "region_5_plus_peak_acceleration_joint_index": int(
            _cpu_item(summary, "region_5_plus_peak_acceleration_joint_index", index)
        ),
        "region_5_plus_peak_waypoint_joint_acceleration_profile": _cpu_list(
            summary,
            "region_5_plus_peak_waypoint_joint_acceleration_profile",
            index,
        ),
        "region_5_plus_minus_1_to_4_max_acceleration_observed": float(
            _cpu_item(summary, "region_5_plus_minus_1_to_4_max_acceleration_observed", index)
        ),
        "region_5_plus_dominates_acceleration_peak": int(
            _cpu_item(summary, "region_5_plus_dominates_acceleration_peak", index)
        ),
    }


def serialize_trajectory_collision_summary(summary, index: int) -> dict[str, object]:
    return {
        "min_signed_distance": float(_cpu_item(summary, "min_signed_distance", index)),
        "mean_signed_distance": float(_cpu_item(summary, "mean_signed_distance", index)),
        "collision_waypoint_count": int(_cpu_item(summary, "collision_waypoint_count", index)),
        "near_collision_waypoint_count": int(_cpu_item(summary, "near_collision_waypoint_count", index)),
        "collision_penetration_depth": float(_cpu_item(summary, "collision_penetration_depth", index)),
        "worst_collision_waypoint_index": int(_cpu_item(summary, "worst_collision_waypoint_index", index)),
        "worst_collision_obstacle_index": int(_cpu_item(summary, "worst_collision_obstacle_index", index)),
        "worst_collision_waypoint_fraction": float(_cpu_item(summary, "worst_collision_waypoint_fraction", index)),
        "region_1_to_4_min_signed_distance": float(_cpu_item(summary, "region_1_to_4_min_signed_distance", index)),
        "region_5_plus_min_signed_distance": float(_cpu_item(summary, "region_5_plus_min_signed_distance", index)),
        "region_1_to_4_collision_waypoint_count": int(
            _cpu_item(summary, "region_1_to_4_collision_waypoint_count", index)
        ),
        "region_5_plus_collision_waypoint_count": int(
            _cpu_item(summary, "region_5_plus_collision_waypoint_count", index)
        ),
        "region_5_plus_minus_1_to_4_min_signed_distance": float(
            _cpu_item(summary, "region_5_plus_minus_1_to_4_min_signed_distance", index)
        ),
        "region_5_plus_dominates_collision_peak": int(
            _cpu_item(summary, "region_5_plus_dominates_collision_peak", index)
        ),
    }


def serialize_dynamics_checkpoint(
    dynamics_summary,
    collision_summary,
    index: int,
    *,
    optimizer_iteration: int,
) -> dict[str, object]:
    serialized = serialize_trajectory_dynamics_summary(dynamics_summary, index)
    serialized.update(serialize_trajectory_collision_summary(collision_summary, index))
    serialized["optimizer_iteration"] = int(optimizer_iteration)
    return serialized

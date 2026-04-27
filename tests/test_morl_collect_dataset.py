import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock
import io

import numpy as np

from src.morl.collect_dataset import (
    CollectionTaskDispatch,
    TASK_PROBE_RESTART_COUNT,
    _collect_task_results_parallel,
    _collect_task_sequential,
    _repair_usage_summary,
    _surrogate_dynamics_checkpoint_summary,
    _surrogate_initial_trajectory_dynamics_summary,
    _surrogate_trajectory_dynamics_summary,
    main,
    parse_args,
)
from src.morl.planning import PlannerConfig, finalize_planned_trajectory
from src.motion_planning.torch_trajopt import TorchPlannerResult


class _FakeQueue:
    def __init__(self, *, fail_on_get: bool = False):
        self.items = []
        self.fail_on_get = fail_on_get
        self.closed = False
        self.cancelled = False

    def put(self, item):
        self.items.append(item)

    def get(self, timeout=None):
        if self.fail_on_get:
            raise KeyboardInterrupt()
        if self.items:
            return self.items.pop(0)
        raise AssertionError("Unexpected get() on empty fake queue.")

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        self.cancelled = True


class _FakeProcess:
    def __init__(self, target=None, args=None):
        self.target = target
        self.args = args or ()
        self.started = False
        self.terminated = False
        self.killed = False
        self.join_calls = 0
        self.alive = False

    def start(self):
        self.started = True
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False

    def join(self, timeout=None):
        self.join_calls += 1


class _FakeContext:
    def __init__(self):
        self.queues = []
        self.processes = []

    def Queue(self):
        queue = _FakeQueue(fail_on_get=len(self.queues) == 2)
        self.queues.append(queue)
        return queue

    def Process(self, target=None, args=None):
        process = _FakeProcess(target=target, args=args)
        self.processes.append(process)
        return process


class CollectDatasetInterruptTests(unittest.TestCase):
    def test_parallel_collection_interrupt_terminates_processes(self):
        fake_context = _FakeContext()
        planner_config = PlannerConfig(device="cpu", gpu_batch_size=2, gpu_batch_timeout_ms=10, planner_steps=2)

        with mock.patch("src.morl.collect_dataset.mp.get_context", return_value=fake_context):
            with self.assertRaises(KeyboardInterrupt):
                _collect_task_results_parallel(
                    [object()],
                    alpha_values=[0.0],
                    restart_count=1,
                    scene_dir=Path("test_artifacts") / "interrupt_tmp",
                    planner_mode="sum",
                    planner_config=planner_config,
                    num_workers=2,
                )

        self.assertEqual(len(fake_context.processes), 3)
        self.assertTrue(all(process.started for process in fake_context.processes))
        self.assertTrue(all(process.terminated for process in fake_context.processes))
        self.assertTrue(all(process.join_calls >= 1 for process in fake_context.processes))
        self.assertTrue(all(queue.closed for queue in fake_context.queues))
        self.assertTrue(all(queue.cancelled for queue in fake_context.queues))


class CollectDatasetProbeTests(unittest.TestCase):
    def _dispatch(self, *, restart_count: int = 4, alpha_values: tuple[float, ...] = (0.0, 0.5, 1.0)):
        task = SimpleNamespace(
            task_id="task_0000",
            family="culdesac_escape",
            geometry_regime="nonconvex",
            benchmark_profile="baseline",
            planner_seed=7,
            start_config=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            obstacles=(),
            dt=0.1,
        )
        return CollectionTaskDispatch(
            task=task,
            task_index=0,
            alpha_values=alpha_values,
            restart_count=restart_count,
            mode="max",
            order_offset=0,
        )

    def _context(self, task):
        return SimpleNamespace(
            task=task,
            goal_config=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

    def _planner_result(self, job):
        return TorchPlannerResult(
            worker_id=job.worker_id,
            request_id=job.request_id,
            order_index=job.order_index,
            trajectory=[],
            dt=0.1,
            iterations=1,
            final_optimization_cost=0.0,
            scalarized_surrogate_cost=0.0,
            optimizer_steps=1,
            batched_jobs=1,
            device="cpu",
            duration_sec=0.0,
            surrogate_initial_trajectory_dynamics={
                "max_acceleration_limit": 0.7,
                "max_acceleration_observed": 0.0,
                "mean_acceleration_magnitude": 0.0,
                "acceleration_violation_count": 0,
                "max_acceleration_excess": 0.0,
                "peak_acceleration_waypoint_index": 0,
                "peak_acceleration_waypoint_fraction": 0.0,
                "max_adjacent_waypoint_jump": 0.0,
                "mean_adjacent_waypoint_jump": 0.0,
                "min_signed_distance": 0.1,
                "mean_signed_distance": 0.1,
                "collision_waypoint_count": 0,
                "near_collision_waypoint_count": 0,
                "collision_penetration_depth": 0.0,
                "worst_collision_waypoint_index": 0,
                "worst_collision_obstacle_index": -1,
                "worst_collision_waypoint_fraction": 0.0,
                "region_1_to_4_min_signed_distance": 0.1,
                "region_5_plus_min_signed_distance": 0.1,
                "region_1_to_4_collision_waypoint_count": 0,
                "region_5_plus_collision_waypoint_count": 0,
                "region_5_plus_minus_1_to_4_min_signed_distance": 0.0,
                "region_5_plus_dominates_collision_peak": 0,
            },
            surrogate_trajectory_dynamics={
                "max_acceleration_limit": 0.7,
                "max_acceleration_observed": 0.0,
                "mean_acceleration_magnitude": 0.0,
                "acceleration_violation_count": 0,
                "max_acceleration_excess": 0.0,
                "peak_acceleration_waypoint_index": 0,
                "peak_acceleration_waypoint_fraction": 0.0,
                "max_adjacent_waypoint_jump": 0.0,
                "mean_adjacent_waypoint_jump": 0.0,
                "min_signed_distance": 0.1,
                "mean_signed_distance": 0.1,
                "collision_waypoint_count": 0,
                "near_collision_waypoint_count": 0,
                "collision_penetration_depth": 0.0,
                "worst_collision_waypoint_index": 0,
                "worst_collision_obstacle_index": -1,
                "worst_collision_waypoint_fraction": 0.0,
                "region_1_to_4_min_signed_distance": 0.1,
                "region_5_plus_min_signed_distance": 0.1,
                "region_1_to_4_collision_waypoint_count": 0,
                "region_5_plus_collision_waypoint_count": 0,
                "region_5_plus_minus_1_to_4_min_signed_distance": 0.0,
                "region_5_plus_dominates_collision_peak": 0,
            },
        )

    def test_sequential_collection_skips_remaining_restarts_after_probe_failures(self):
        dispatch = self._dispatch()
        planner_config = PlannerConfig(device="cpu", gpu_batch_size=32, planner_steps=2)
        context = self._context(dispatch.task)

        with mock.patch("src.morl.collect_dataset.prepare_task_planning_context", return_value=context):
            with mock.patch(
                "src.morl.collect_dataset.run_torch_planner_batch",
                side_effect=lambda jobs, planner_config=None: [self._planner_result(job) for job in jobs],
            ) as run_batch:
                with mock.patch(
                    "src.morl.collect_dataset.finalize_planned_trajectory",
                    side_effect=RuntimeError("invalid trajectory"),
                ):
                    results = _collect_task_sequential(
                        dispatch,
                        scene_dir=Path("test_artifacts") / "probe_tmp",
                        planner_config=planner_config,
                    )
        probe_restarts = min(dispatch.restart_count, TASK_PROBE_RESTART_COUNT)
        expected_probe_count = len(dispatch.alpha_values) * probe_restarts
        expected_skipped_count = len(dispatch.alpha_values) * dispatch.restart_count - expected_probe_count
        self.assertEqual(run_batch.call_count, 1)
        self.assertEqual(sum(1 for result in results if result.record is not None), 0)
        self.assertEqual(sum(1 for result in results if bool((result.failure or {}).get("probe_skipped"))), expected_skipped_count)
        self.assertEqual(len(results), 12)

    def test_sequential_collection_continues_after_probe_success(self):
        dispatch = self._dispatch()
        planner_config = PlannerConfig(device="cpu", gpu_batch_size=32, planner_steps=2)
        context = self._context(dispatch.task)

        def finalize_side_effect(context, *, planner_result, alpha, mode, restart_index, planner_config=None):
            if restart_index == 0 and abs(alpha - 0.0) < 1e-6:
                return {
                    "trajectory_id": planner_result.request_id,
                    "task_spec": {"task_id": "task_0000"},
                }
            raise RuntimeError("invalid trajectory")

        with mock.patch("src.morl.collect_dataset.prepare_task_planning_context", return_value=context):
            with mock.patch(
                "src.morl.collect_dataset.run_torch_planner_batch",
                side_effect=lambda jobs, planner_config=None: [self._planner_result(job) for job in jobs],
            ) as run_batch:
                with mock.patch(
                    "src.morl.collect_dataset.finalize_planned_trajectory",
                    side_effect=finalize_side_effect,
                ):
                    results = _collect_task_sequential(
                        dispatch,
                        scene_dir=Path("test_artifacts") / "probe_tmp",
                        planner_config=planner_config,
                    )
        probe_restarts = min(dispatch.restart_count, TASK_PROBE_RESTART_COUNT)
        expected_probe_count = len(dispatch.alpha_values) * probe_restarts
        expected_run_batch_calls = 2 if expected_probe_count < len(dispatch.alpha_values) * dispatch.restart_count else 1
        self.assertEqual(run_batch.call_count, expected_run_batch_calls)
        self.assertEqual(sum(1 for result in results if result.record is not None), 1)
        self.assertEqual(sum(1 for result in results if bool((result.failure or {}).get("probe_skipped"))), 0)
        self.assertEqual(len(results), 12)

    def test_parse_args_rejects_explicit_cuda_when_unavailable(self):
        argv = [
            "collect_dataset.py",
            "--experiment-name",
            "demo",
            "--planner-mode",
            "max",
            "--device",
            "cuda",
        ]
        def fake_resolve(device, *, strict_explicit_cuda=False):
            self.assertEqual(device, "cuda")
            self.assertTrue(strict_explicit_cuda)
            raise RuntimeError("CUDA was requested for dataset collection, but the active PyTorch build has no CUDA device available.")

        with mock.patch("sys.argv", argv):
            with mock.patch("src.morl.collect_dataset.resolve_collection_device", side_effect=fake_resolve):
                with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                    parse_args()

    def test_parse_args_accepts_profile_flag(self):
        argv = [
            "collect_dataset.py",
            "--experiment-name",
            "demo",
            "--planner-mode",
            "max",
            "--profile",
        ]
        with mock.patch("sys.argv", argv):
            with mock.patch("src.morl.collect_dataset.resolve_collection_device", return_value="cpu"):
                args = parse_args()
        self.assertTrue(args.profile)

    def test_main_prints_cumulative_profile_when_enabled(self):
        output = io.StringIO()
        with mock.patch("src.morl.collect_dataset._run_collection", return_value={"ok": True}) as run_collection:
            with mock.patch("sys.stdout", output):
                result = main(
                    [
                        "--experiment-name",
                        "demo",
                        "--planner-mode",
                        "max",
                        "--profile",
                    ]
                )
        self.assertEqual(result, {"ok": True})
        run_collection.assert_called_once()
        self.assertIn("Collection profile (top 15 by total time):", output.getvalue())

    def test_repair_usage_summary_counts_raw_successes(self):
        records = [
            {"optimization": {"repair_used": True}},
            {"optimization": {"repair_used": False}},
            {"optimization": {"repair_used": True}},
        ]

        summary = _repair_usage_summary(records)

        self.assertEqual(summary["repair_used_trajectory_count"], 2)
        self.assertEqual(summary["repair_free_trajectory_count"], 1)
        self.assertAlmostEqual(summary["repair_rate"], 2.0 / 3.0)

    def test_repair_usage_summary_tracks_repair_reasons_and_slsqp_counts(self):
        records = [
            {
                "optimization": {
                    "repair_used": True,
                    "repair_validation_failure_reason": "dynamics",
                    "raw_dynamics_violation": {
                        "velocity_violation_count": 2,
                        "acceleration_violation_count": 1,
                        "max_velocity_excess": 0.3,
                        "max_acceleration_excess": 0.5,
                    },
                    "repair": {
                        "slsqp_iterations": 3,
                        "slsqp_function_evaluations": 11,
                        "slsqp_gradient_evaluations": 3,
                    },
                }
            },
            {
                "optimization": {
                    "repair_used": True,
                    "repair": {
                        "validation_failure_reason": "contacts",
                        "raw_dynamics_violation": {
                            "velocity_violation_count": 4,
                            "acceleration_violation_count": 2,
                            "max_velocity_excess": 0.7,
                            "max_acceleration_excess": 0.25,
                        },
                        "slsqp_iterations": 5,
                        "slsqp_function_evaluations": 17,
                        "slsqp_gradient_evaluations": 5,
                    },
                }
            },
            {"optimization": {"repair_used": False}},
        ]

        summary = _repair_usage_summary(records)

        self.assertEqual(summary["repair_invocation_count_by_validation_failure_reason"], {"dynamics": 1, "contacts": 1})
        self.assertEqual(summary["repair_slsqp_iteration_total"], 8)
        self.assertEqual(summary["repair_slsqp_function_evaluation_total"], 28)
        self.assertEqual(summary["repair_slsqp_gradient_evaluation_total"], 8)
        self.assertAlmostEqual(summary["repair_mean_slsqp_iterations"], 4.0)
        self.assertAlmostEqual(summary["repair_mean_slsqp_function_evaluations"], 14.0)
        self.assertAlmostEqual(summary["repair_mean_slsqp_gradient_evaluations"], 4.0)
        self.assertEqual(summary["repair_raw_dynamics_velocity_violation_total"], 6)
        self.assertEqual(summary["repair_raw_dynamics_acceleration_violation_total"], 3)
        self.assertAlmostEqual(summary["repair_raw_dynamics_max_velocity_excess"], 0.7)
        self.assertAlmostEqual(summary["repair_raw_dynamics_max_acceleration_excess"], 0.5)
        self.assertAlmostEqual(summary["repair_mean_raw_dynamics_velocity_violations"], 3.0)
        self.assertAlmostEqual(summary["repair_mean_raw_dynamics_acceleration_violations"], 1.5)

    def test_repair_usage_summary_handles_zero_records(self):
        summary = _repair_usage_summary([])

        self.assertEqual(summary["repair_used_trajectory_count"], 0)
        self.assertEqual(summary["repair_free_trajectory_count"], 0)
        self.assertEqual(summary["repair_rate"], 0.0)
        self.assertEqual(summary["repair_invocation_count_by_validation_failure_reason"], {})
        self.assertEqual(summary["repair_slsqp_iteration_total"], 0)
        self.assertEqual(summary["repair_slsqp_function_evaluation_total"], 0)
        self.assertEqual(summary["repair_slsqp_gradient_evaluation_total"], 0)
        self.assertEqual(summary["repair_raw_dynamics_velocity_violation_total"], 0)
        self.assertEqual(summary["repair_raw_dynamics_acceleration_violation_total"], 0)
        self.assertEqual(summary["repair_raw_dynamics_max_velocity_excess"], 0.0)
        self.assertEqual(summary["repair_raw_dynamics_max_acceleration_excess"], 0.0)
        self.assertEqual(summary["optimizer_duration_mean_sec"], 0.0)
        self.assertEqual(summary["optimizer_duration_p90_sec"], 0.0)
        self.assertEqual(summary["rescue_attempted_trajectory_count"], 0)
        self.assertEqual(summary["rescue_success_trajectory_count"], 0)
        self.assertEqual(summary["warm_start_rrt_attempted_trajectory_count"], 0)
        self.assertEqual(summary["warm_start_rrt_replaced_trajectory_count"], 0)

    def test_repair_usage_summary_tracks_optimizer_duration_and_strategy_counts(self):
        records = [
            {
                "optimization": {
                    "duration_sec": 1.0,
                    "repair_used": False,
                    "rescue_attempted": True,
                    "rescue_success": True,
                    "surrogate_initial_trajectory_dynamics": {
                        "warm_start_rrt_attempted": True,
                        "warm_start_strategy": "rrt",
                    },
                }
            },
            {
                "optimization": {
                    "duration_sec": 2.0,
                    "repair_used": False,
                    "rescue_attempted": True,
                    "rescue_success": False,
                    "surrogate_initial_trajectory_dynamics": {
                        "warm_start_rrt_attempted": True,
                        "warm_start_strategy": "linear",
                    },
                }
            },
            {"optimization": {"duration_sec": 4.0, "repair_used": False}},
        ]

        summary = _repair_usage_summary(records)
        self.assertAlmostEqual(summary["optimizer_duration_mean_sec"], 7.0 / 3.0)
        self.assertAlmostEqual(summary["optimizer_duration_p90_sec"], 4.0)
        self.assertEqual(summary["rescue_attempted_trajectory_count"], 2)
        self.assertEqual(summary["rescue_success_trajectory_count"], 1)
        self.assertEqual(summary["warm_start_rrt_attempted_trajectory_count"], 2)
        self.assertEqual(summary["warm_start_rrt_replaced_trajectory_count"], 1)

    def test_surrogate_dynamics_summary_aggregates_acceleration_shape(self):
        records = [
            {
                "optimization": {
                    "surrogate_initial_trajectory_dynamics": {
                        "max_acceleration_observed": 0.8,
                        "mean_acceleration_magnitude": 0.3,
                        "acceleration_violation_count": 1,
                        "max_acceleration_excess": 0.1,
                        "peak_acceleration_waypoint_fraction": 0.2,
                        "max_adjacent_waypoint_jump": 0.15,
                        "region_1_to_4_max_acceleration_observed": 0.2,
                        "region_1_to_4_acceleration_violation_count": 1,
                        "region_5_plus_max_acceleration_observed": 0.8,
                        "region_5_plus_acceleration_violation_count": 0,
                        "region_5_plus_minus_1_to_4_max_acceleration_observed": 0.6,
                        "region_5_plus_dominates_acceleration_peak": 1,
                        "min_signed_distance": 0.1,
                        "mean_signed_distance": 0.2,
                        "collision_waypoint_count": 0,
                        "near_collision_waypoint_count": 1,
                        "collision_penetration_depth": 0.0,
                        "worst_collision_waypoint_fraction": 0.2,
                        "region_1_to_4_min_signed_distance": 0.1,
                        "region_5_plus_min_signed_distance": 0.2,
                        "region_1_to_4_collision_waypoint_count": 0,
                        "region_5_plus_collision_waypoint_count": 0,
                        "region_5_plus_minus_1_to_4_min_signed_distance": 0.1,
                        "region_5_plus_dominates_collision_peak": 0,
                    },
                    "surrogate_trajectory_dynamics": {
                        "max_acceleration_observed": 1.5,
                        "mean_acceleration_magnitude": 0.6,
                        "acceleration_violation_count": 2,
                        "max_acceleration_excess": 0.8,
                        "peak_acceleration_waypoint_fraction": 0.5,
                        "max_adjacent_waypoint_jump": 0.4,
                        "region_1_to_4_max_acceleration_observed": 0.4,
                        "region_1_to_4_acceleration_violation_count": 0,
                        "region_5_plus_max_acceleration_observed": 1.5,
                        "region_5_plus_acceleration_violation_count": 2,
                        "region_5_plus_minus_1_to_4_max_acceleration_observed": 1.1,
                        "region_5_plus_dominates_acceleration_peak": 1,
                        "min_signed_distance": -0.05,
                        "mean_signed_distance": 0.05,
                        "collision_waypoint_count": 1,
                        "near_collision_waypoint_count": 2,
                        "collision_penetration_depth": 0.05,
                        "worst_collision_waypoint_fraction": 0.5,
                        "region_1_to_4_min_signed_distance": 0.1,
                        "region_5_plus_min_signed_distance": -0.05,
                        "region_1_to_4_collision_waypoint_count": 0,
                        "region_5_plus_collision_waypoint_count": 1,
                        "region_5_plus_minus_1_to_4_min_signed_distance": -0.15,
                        "region_5_plus_dominates_collision_peak": 1,
                    },
                    "surrogate_dynamics_checkpoints": [
                        {
                            "optimizer_iteration": 0,
                            "max_acceleration_observed": 1.8,
                            "mean_acceleration_magnitude": 0.7,
                            "acceleration_violation_count": 3,
                            "max_acceleration_excess": 1.1,
                            "peak_acceleration_waypoint_fraction": 0.8,
                            "max_adjacent_waypoint_jump": 0.5,
                            "region_1_to_4_max_acceleration_observed": 0.3,
                            "region_1_to_4_acceleration_violation_count": 1,
                            "region_5_plus_max_acceleration_observed": 1.8,
                            "region_5_plus_acceleration_violation_count": 2,
                            "region_5_plus_minus_1_to_4_max_acceleration_observed": 1.5,
                            "region_5_plus_dominates_acceleration_peak": 1,
                            "min_signed_distance": -0.1,
                            "mean_signed_distance": 0.0,
                            "collision_waypoint_count": 1,
                            "near_collision_waypoint_count": 2,
                            "collision_penetration_depth": 0.1,
                            "worst_collision_waypoint_fraction": 0.8,
                            "region_1_to_4_min_signed_distance": 0.2,
                            "region_5_plus_min_signed_distance": -0.1,
                            "region_1_to_4_collision_waypoint_count": 0,
                            "region_5_plus_collision_waypoint_count": 1,
                            "region_5_plus_minus_1_to_4_min_signed_distance": -0.3,
                            "region_5_plus_dominates_collision_peak": 1,
                        }
                    ],
                }
            },
            {
                "optimization": {
                    "surrogate_initial_trajectory_dynamics": {
                        "max_acceleration_observed": 0.3,
                        "mean_acceleration_magnitude": 0.1,
                        "acceleration_violation_count": 0,
                        "max_acceleration_excess": 0.0,
                        "peak_acceleration_waypoint_fraction": 0.4,
                        "max_adjacent_waypoint_jump": 0.05,
                        "region_1_to_4_max_acceleration_observed": 0.3,
                        "region_1_to_4_acceleration_violation_count": 0,
                        "region_5_plus_max_acceleration_observed": 0.0,
                        "region_5_plus_acceleration_violation_count": 0,
                        "region_5_plus_minus_1_to_4_max_acceleration_observed": -0.3,
                        "region_5_plus_dominates_acceleration_peak": 0,
                        "min_signed_distance": 0.3,
                        "mean_signed_distance": 0.35,
                        "collision_waypoint_count": 0,
                        "near_collision_waypoint_count": 0,
                        "collision_penetration_depth": 0.0,
                        "worst_collision_waypoint_fraction": 0.4,
                        "region_1_to_4_min_signed_distance": 0.3,
                        "region_5_plus_min_signed_distance": 0.4,
                        "region_1_to_4_collision_waypoint_count": 0,
                        "region_5_plus_collision_waypoint_count": 0,
                        "region_5_plus_minus_1_to_4_min_signed_distance": 0.1,
                        "region_5_plus_dominates_collision_peak": 0,
                    },
                    "surrogate_trajectory_dynamics": {
                        "max_acceleration_observed": 0.4,
                        "mean_acceleration_magnitude": 0.2,
                        "acceleration_violation_count": 0,
                        "max_acceleration_excess": 0.0,
                        "peak_acceleration_waypoint_fraction": 0.75,
                        "max_adjacent_waypoint_jump": 0.1,
                        "region_1_to_4_max_acceleration_observed": 0.4,
                        "region_1_to_4_acceleration_violation_count": 0,
                        "region_5_plus_max_acceleration_observed": 0.1,
                        "region_5_plus_acceleration_violation_count": 0,
                        "region_5_plus_minus_1_to_4_max_acceleration_observed": -0.3,
                        "region_5_plus_dominates_acceleration_peak": 0,
                        "min_signed_distance": 0.3,
                        "mean_signed_distance": 0.4,
                        "collision_waypoint_count": 0,
                        "near_collision_waypoint_count": 0,
                        "collision_penetration_depth": 0.0,
                        "worst_collision_waypoint_fraction": 0.75,
                        "region_1_to_4_min_signed_distance": 0.3,
                        "region_5_plus_min_signed_distance": 0.4,
                        "region_1_to_4_collision_waypoint_count": 0,
                        "region_5_plus_collision_waypoint_count": 0,
                        "region_5_plus_minus_1_to_4_min_signed_distance": 0.1,
                        "region_5_plus_dominates_collision_peak": 0,
                    },
                    "surrogate_dynamics_checkpoints": [
                        {
                            "optimizer_iteration": 0,
                            "max_acceleration_observed": 0.6,
                            "mean_acceleration_magnitude": 0.25,
                            "acceleration_violation_count": 0,
                            "max_acceleration_excess": 0.0,
                            "peak_acceleration_waypoint_fraction": 0.3,
                            "max_adjacent_waypoint_jump": 0.15,
                            "region_1_to_4_max_acceleration_observed": 0.5,
                            "region_1_to_4_acceleration_violation_count": 0,
                            "region_5_plus_max_acceleration_observed": 0.1,
                            "region_5_plus_acceleration_violation_count": 0,
                            "region_5_plus_minus_1_to_4_max_acceleration_observed": -0.4,
                            "region_5_plus_dominates_acceleration_peak": 0,
                            "min_signed_distance": 0.35,
                            "mean_signed_distance": 0.45,
                            "collision_waypoint_count": 0,
                            "near_collision_waypoint_count": 0,
                            "collision_penetration_depth": 0.0,
                            "worst_collision_waypoint_fraction": 0.3,
                            "region_1_to_4_min_signed_distance": 0.35,
                            "region_5_plus_min_signed_distance": 0.45,
                            "region_1_to_4_collision_waypoint_count": 0,
                            "region_5_plus_collision_waypoint_count": 0,
                            "region_5_plus_minus_1_to_4_min_signed_distance": 0.1,
                            "region_5_plus_dominates_collision_peak": 0,
                        }
                    ],
                }
            },
            {"optimization": {"repair_used": False}},
        ]

        initial_summary = _surrogate_initial_trajectory_dynamics_summary(records)
        summary = _surrogate_trajectory_dynamics_summary(records)
        checkpoint_summary = _surrogate_dynamics_checkpoint_summary(records)

        self.assertEqual(initial_summary["surrogate_initial_dynamics_summary_trajectory_count"], 2)
        self.assertAlmostEqual(initial_summary["surrogate_initial_mean_max_acceleration_observed"], 0.55)
        self.assertAlmostEqual(initial_summary["surrogate_initial_max_acceleration_observed"], 0.8)
        self.assertEqual(initial_summary["surrogate_initial_acceleration_limit_exceed_trajectory_count"], 1)
        self.assertAlmostEqual(initial_summary["surrogate_initial_mean_peak_acceleration_waypoint_fraction"], 0.3)
        self.assertAlmostEqual(initial_summary["surrogate_initial_mean_max_adjacent_waypoint_jump"], 0.1)
        self.assertAlmostEqual(initial_summary["surrogate_initial_mean_region_1_to_4_max_acceleration_observed"], 0.25)
        self.assertAlmostEqual(initial_summary["surrogate_initial_mean_region_5_plus_max_acceleration_observed"], 0.4)
        self.assertAlmostEqual(
            initial_summary["surrogate_initial_mean_region_5_plus_minus_1_to_4_max_acceleration_observed"],
            0.15,
        )
        self.assertEqual(initial_summary["surrogate_initial_region_5_plus_dominates_acceleration_peak_trajectory_count"], 1)
        self.assertEqual(summary["surrogate_dynamics_summary_trajectory_count"], 2)
        self.assertAlmostEqual(summary["surrogate_mean_max_acceleration_observed"], 0.95)
        self.assertAlmostEqual(summary["surrogate_max_acceleration_observed"], 1.5)
        self.assertAlmostEqual(summary["surrogate_mean_mean_acceleration_magnitude"], 0.4)
        self.assertEqual(summary["surrogate_acceleration_limit_exceed_trajectory_count"], 1)
        self.assertAlmostEqual(summary["surrogate_mean_acceleration_violation_count"], 1.0)
        self.assertEqual(summary["surrogate_max_acceleration_violation_count"], 2)
        self.assertAlmostEqual(summary["surrogate_mean_max_acceleration_excess"], 0.4)
        self.assertAlmostEqual(summary["surrogate_max_acceleration_excess"], 0.8)
        self.assertAlmostEqual(summary["surrogate_mean_peak_acceleration_waypoint_fraction"], 0.625)
        self.assertAlmostEqual(summary["surrogate_mean_max_adjacent_waypoint_jump"], 0.25)
        self.assertAlmostEqual(summary["surrogate_max_adjacent_waypoint_jump"], 0.4)
        self.assertAlmostEqual(summary["surrogate_mean_region_1_to_4_max_acceleration_observed"], 0.4)
        self.assertAlmostEqual(summary["surrogate_mean_region_5_plus_max_acceleration_observed"], 0.8)
        self.assertAlmostEqual(summary["surrogate_mean_region_5_plus_acceleration_violation_count"], 1.0)
        self.assertAlmostEqual(summary["surrogate_mean_region_5_plus_minus_1_to_4_max_acceleration_observed"], 0.4)
        self.assertEqual(summary["surrogate_region_5_plus_dominates_acceleration_peak_trajectory_count"], 1)
        self.assertAlmostEqual(summary["surrogate_mean_min_signed_distance"], 0.125)
        self.assertEqual(summary["surrogate_collision_limit_exceed_trajectory_count"], 1)
        self.assertAlmostEqual(summary["surrogate_mean_collision_waypoint_count"], 0.5)
        self.assertAlmostEqual(summary["surrogate_mean_near_collision_waypoint_count"], 1.0)
        self.assertAlmostEqual(summary["surrogate_mean_region_5_plus_min_signed_distance"], 0.175)
        self.assertEqual(summary["surrogate_region_5_plus_dominates_collision_peak_trajectory_count"], 1)
        self.assertEqual(checkpoint_summary["surrogate_checkpoint_iter_0_dynamics_summary_trajectory_count"], 2)
        self.assertAlmostEqual(
            checkpoint_summary["surrogate_checkpoint_iter_0_mean_region_1_to_4_max_acceleration_observed"],
            0.4,
        )
        self.assertAlmostEqual(
            checkpoint_summary["surrogate_checkpoint_iter_0_mean_region_5_plus_max_acceleration_observed"],
            0.95,
        )
        self.assertEqual(
            checkpoint_summary["surrogate_checkpoint_iter_0_region_5_plus_dominates_acceleration_peak_trajectory_count"],
            1,
        )
        self.assertAlmostEqual(checkpoint_summary["surrogate_checkpoint_iter_0_mean_min_signed_distance"], 0.125)
        self.assertEqual(checkpoint_summary["surrogate_checkpoint_iter_0_collision_limit_exceed_trajectory_count"], 1)

    def test_finalize_planned_trajectory_records_repair_validation_reason(self):
        task = SimpleNamespace(
            task_id="task_0000",
            dt=0.1,
            to_dict=lambda: {"task_id": "task_0000"},
        )
        context = SimpleNamespace(
            task=task,
            scene_path="scene.xml",
            goal_config=np.asarray([0.0, 0.0], dtype=np.float64),
            model=object(),
            data=object(),
            length_cost=SimpleNamespace(compute_cost=lambda trajectory, dt: 1.0),
            safety_cost=SimpleNamespace(compute_cost=lambda trajectory, dt: 2.0),
        )
        planner_result = TorchPlannerResult(
            worker_id=0,
            request_id="demo",
            order_index=0,
            trajectory=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
            dt=0.1,
            iterations=1,
            final_optimization_cost=0.0,
            scalarized_surrogate_cost=0.0,
            optimizer_steps=1,
            batched_jobs=1,
            device="cpu",
            duration_sec=0.0,
            surrogate_initial_trajectory_dynamics={
                "max_acceleration_limit": 0.7,
                "max_acceleration_observed": 4.0,
                "mean_acceleration_magnitude": 2.0,
                "acceleration_violation_count": 1,
                "max_acceleration_excess": 3.3,
                "peak_acceleration_waypoint_index": 1,
                "peak_acceleration_waypoint_fraction": 1.0,
                "max_adjacent_waypoint_jump": 0.5,
                "mean_adjacent_waypoint_jump": 0.5,
                "region_1_to_4_max_acceleration_observed": 0.0,
                "region_1_to_4_mean_acceleration_magnitude": 0.0,
                "region_1_to_4_acceleration_violation_count": 0,
                "region_5_plus_max_acceleration_observed": 4.0,
                "region_5_plus_mean_acceleration_magnitude": 2.0,
                "region_5_plus_acceleration_violation_count": 1,
                "region_5_plus_minus_1_to_4_max_acceleration_observed": 4.0,
                "region_5_plus_dominates_acceleration_peak": 1,
                "min_signed_distance": 0.05,
                "mean_signed_distance": 0.1,
                "collision_waypoint_count": 0,
                "near_collision_waypoint_count": 1,
                "collision_penetration_depth": 0.0,
                "worst_collision_waypoint_index": 1,
                "worst_collision_obstacle_index": 0,
                "worst_collision_waypoint_fraction": 1.0,
                "region_1_to_4_min_signed_distance": 0.05,
                "region_5_plus_min_signed_distance": 0.05,
                "region_1_to_4_collision_waypoint_count": 0,
                "region_5_plus_collision_waypoint_count": 0,
                "region_5_plus_minus_1_to_4_min_signed_distance": 0.0,
                "region_5_plus_dominates_collision_peak": 0,
            },
            surrogate_trajectory_dynamics={
                "max_acceleration_limit": 0.7,
                "max_acceleration_observed": 10.0,
                "mean_acceleration_magnitude": 5.0,
                "acceleration_violation_count": 2,
                "max_acceleration_excess": 9.3,
                "peak_acceleration_waypoint_index": 1,
                "peak_acceleration_waypoint_fraction": 1.0,
                "max_adjacent_waypoint_jump": 1.0,
                "mean_adjacent_waypoint_jump": 1.0,
                "region_1_to_4_max_acceleration_observed": 0.0,
                "region_1_to_4_mean_acceleration_magnitude": 0.0,
                "region_1_to_4_acceleration_violation_count": 0,
                "region_5_plus_max_acceleration_observed": 10.0,
                "region_5_plus_mean_acceleration_magnitude": 5.0,
                "region_5_plus_acceleration_violation_count": 2,
                "region_5_plus_minus_1_to_4_max_acceleration_observed": 10.0,
                "region_5_plus_dominates_acceleration_peak": 1,
                "min_signed_distance": -0.02,
                "mean_signed_distance": 0.03,
                "collision_waypoint_count": 1,
                "near_collision_waypoint_count": 2,
                "collision_penetration_depth": 0.02,
                "worst_collision_waypoint_index": 1,
                "worst_collision_obstacle_index": 0,
                "worst_collision_waypoint_fraction": 1.0,
                "region_1_to_4_min_signed_distance": -0.02,
                "region_5_plus_min_signed_distance": -0.02,
                "region_1_to_4_collision_waypoint_count": 1,
                "region_5_plus_collision_waypoint_count": 0,
                "region_5_plus_minus_1_to_4_min_signed_distance": 0.0,
                "region_5_plus_dominates_collision_peak": 0,
            },
            surrogate_dynamics_checkpoints=(
                {
                    "optimizer_iteration": 0,
                    "max_acceleration_observed": 12.0,
                    "mean_acceleration_magnitude": 6.0,
                    "acceleration_violation_count": 3,
                    "max_acceleration_excess": 11.3,
                    "peak_acceleration_waypoint_index": 1,
                    "peak_acceleration_waypoint_fraction": 1.0,
                    "max_adjacent_waypoint_jump": 1.2,
                    "mean_adjacent_waypoint_jump": 1.2,
                    "region_1_to_4_max_acceleration_observed": 0.0,
                    "region_1_to_4_mean_acceleration_magnitude": 0.0,
                    "region_1_to_4_acceleration_violation_count": 0,
                    "region_5_plus_max_acceleration_observed": 12.0,
                    "region_5_plus_mean_acceleration_magnitude": 6.0,
                    "region_5_plus_acceleration_violation_count": 3,
                    "region_5_plus_minus_1_to_4_max_acceleration_observed": 12.0,
                    "region_5_plus_dominates_acceleration_peak": 1,
                    "min_signed_distance": -0.03,
                    "mean_signed_distance": 0.02,
                    "collision_waypoint_count": 1,
                    "near_collision_waypoint_count": 2,
                    "collision_penetration_depth": 0.03,
                    "worst_collision_waypoint_index": 1,
                    "worst_collision_obstacle_index": 0,
                    "worst_collision_waypoint_fraction": 1.0,
                    "region_1_to_4_min_signed_distance": -0.03,
                    "region_5_plus_min_signed_distance": -0.03,
                    "region_1_to_4_collision_waypoint_count": 1,
                    "region_5_plus_collision_waypoint_count": 0,
                    "region_5_plus_minus_1_to_4_min_signed_distance": 0.0,
                    "region_5_plus_dominates_collision_peak": 0,
                },
            ),
        )
        repaired = np.asarray([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)

        with mock.patch("src.morl.planning._validate_trajectory_dynamics", side_effect=[RuntimeError("dyn fail"), None]):
            with mock.patch("src.morl.planning._validate_trajectory_contacts", return_value=None):
                with mock.patch(
                    "src.morl.planning._run_cpu_repair",
                    return_value=(
                        repaired,
                        {
                            "slsqp_iterations": 4,
                            "slsqp_function_evaluations": 13,
                            "slsqp_gradient_evaluations": 4,
                        },
                        True,
                    ),
                ):
                    with mock.patch("src.morl.planning.scalarize_numpy", return_value=np.asarray([3.0], dtype=np.float32)):
                        record = finalize_planned_trajectory(
                            context,
                            planner_result=planner_result,
                            alpha=0.5,
                            mode="max",
                            restart_index=0,
                            planner_config=PlannerConfig(),
                        )

        optimization = record["optimization"]
        self.assertTrue(optimization["repair_used"])
        self.assertEqual(optimization["repair_validation_failure_reason"], "dynamics")
        self.assertEqual(optimization["repair_validation_failure_message"], "dyn fail")
        self.assertEqual(optimization["surrogate_initial_trajectory_dynamics"]["max_acceleration_observed"], 4.0)
        self.assertEqual(optimization["surrogate_trajectory_dynamics"]["max_acceleration_observed"], 10.0)
        self.assertEqual(optimization["surrogate_trajectory_dynamics"]["collision_waypoint_count"], 1)
        self.assertEqual(optimization["surrogate_dynamics_checkpoints"][0]["optimizer_iteration"], 0)
        self.assertEqual(optimization["surrogate_dynamics_checkpoints"][0]["region_5_plus_max_acceleration_observed"], 12.0)
        self.assertEqual(optimization["surrogate_dynamics_checkpoints"][0]["collision_penetration_depth"], 0.03)
        self.assertGreater(optimization["raw_dynamics_violation"]["velocity_violation_count"], 0)
        self.assertGreater(optimization["raw_dynamics_violation"]["max_velocity_excess"], 0.0)
        self.assertEqual(optimization["repair"]["validation_failure_reason"], "dynamics")
        self.assertEqual(optimization["repair"]["slsqp_function_evaluations"], 13)
        self.assertEqual(
            optimization["repair"]["raw_dynamics_violation"]["velocity_violation_count"],
            optimization["raw_dynamics_violation"]["velocity_violation_count"],
        )

    def test_finalize_planned_trajectory_attempts_repair_for_mild_contact_violation(self):
        task = SimpleNamespace(task_id="task_0000", dt=0.1, to_dict=lambda: {"task_id": "task_0000"})
        context = SimpleNamespace(
            task=task,
            scene_path="scene.xml",
            goal_config=np.asarray([0.0, 0.0], dtype=np.float64),
            model=object(),
            data=object(),
            length_cost=SimpleNamespace(compute_cost=lambda trajectory, dt: 1.0),
            safety_cost=SimpleNamespace(compute_cost=lambda trajectory, dt: 2.0),
        )
        repaired = np.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        planner_result = TorchPlannerResult(
            worker_id=0,
            request_id="demo",
            order_index=0,
            trajectory=np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
            dt=0.1,
            iterations=1,
            final_optimization_cost=0.0,
            scalarized_surrogate_cost=0.0,
            optimizer_steps=1,
            batched_jobs=1,
            device="cpu",
            duration_sec=0.0,
            surrogate_initial_trajectory_dynamics={},
            surrogate_trajectory_dynamics={
                "collision_waypoint_count": 1,
                "collision_penetration_depth": 0.005,
                "near_collision_waypoint_count": 1,
                "min_signed_distance": -0.005,
            },
        )

        with mock.patch("src.morl.planning._validate_trajectory_dynamics", return_value=None):
            with mock.patch("src.morl.planning._validate_trajectory_contacts", side_effect=[RuntimeError("contact fail"), None]):
                with mock.patch(
                    "src.morl.planning._run_cpu_repair",
                    return_value=(repaired, {"slsqp_function_evaluations": 1}, True),
                ) as repair_mock:
                    with mock.patch("src.morl.planning.scalarize_numpy", return_value=np.asarray([3.0], dtype=np.float32)):
                        record = finalize_planned_trajectory(
                            context,
                            planner_result=planner_result,
                            alpha=0.5,
                            mode="max",
                            restart_index=0,
                            planner_config=PlannerConfig(),
                        )
        repair_mock.assert_called_once()
        self.assertTrue(record["optimization"]["repair_used"])

if __name__ == "__main__":
    unittest.main()

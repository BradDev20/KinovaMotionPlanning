import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.motion_planning.torch_trajopt import TorchPlannerJob, TorchTrajectoryBatchPlanner
try:
    from src.motion_planning.RRTPlanner import RRTPlanner
except Exception:  # pragma: no cover - optional mujoco dependency
    RRTPlanner = None


@unittest.skipUnless(torch is not None, "PyTorch is required for torch trajectory planner tests.")
class TorchTrajectoryPlannerTests(unittest.TestCase):
    def _old_iid_warm_start(self, planner, job: TorchPlannerJob, *, seed: int) -> np.ndarray:
        trajectory = np.linspace(
            np.asarray(job.start_config, dtype=np.float32),
            np.asarray(job.goal_config, dtype=np.float32),
            planner.n_waypoints,
            dtype=np.float32,
        )
        if planner.n_waypoints > 2 and planner.warm_start_noise > 0.0:
            rng = np.random.default_rng(int(seed))
            trajectory[1:-1] += rng.normal(
                loc=0.0,
                scale=planner.warm_start_noise,
                size=trajectory[1:-1].shape,
            ).astype(np.float32)
        return trajectory

    def _job(
        self,
        *,
        request_id: str,
        alpha: float = 0.5,
        mode: str = "sum",
        aggregate: str = "avg",
        obstacle_centers=(),
        obstacle_radii=(),
        obstacle_safe=(),
        task_family: str = "base",
    ) -> TorchPlannerJob:
        return TorchPlannerJob(
            worker_id=-1,
            request_id=request_id,
            order_index=0,
            start_config=(0.0, -0.5, 0.0, 2.5, 0.0, 1.0, -1.57),
            goal_config=(0.1, -0.45, 0.05, 2.45, 0.05, 0.95, -1.52),
            obstacle_centers=tuple(obstacle_centers),
            obstacle_radii=tuple(obstacle_radii),
            obstacle_safe_distances=tuple(obstacle_safe),
            dt=0.1,
            alpha=alpha,
            planner_mode=mode,
            rho=0.01,
            safety_aggregate=aggregate,
            safety_decay_rate=15.0,
            safety_bias=-0.08,
            safety_collision_penalty=1.0,
            seed=123,
            task_family=task_family,
        )

    def test_batched_planner_preserves_shape_and_endpoints(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=8,
            planner_steps=4,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        jobs = [self._job(request_id="a"), self._job(request_id="b", alpha=0.2, mode="max")]
        results = planner.solve_batch(jobs)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].trajectory.shape, (8, 7))
        np.testing.assert_allclose(results[0].trajectory[0], np.asarray(jobs[0].start_config), atol=1e-6)
        np.testing.assert_allclose(results[0].trajectory[-1], np.asarray(jobs[0].goal_config), atol=1e-6)
        self.assertIn("max_acceleration_observed", results[0].surrogate_initial_trajectory_dynamics)
        self.assertIn("max_acceleration_observed", results[0].surrogate_trajectory_dynamics)
        self.assertIn("peak_acceleration_waypoint_index", results[0].surrogate_trajectory_dynamics)
        self.assertIn("min_signed_distance", results[0].surrogate_trajectory_dynamics)
        self.assertIn("collision_waypoint_count", results[0].surrogate_trajectory_dynamics)
        self.assertEqual([item["optimizer_iteration"] for item in results[0].surrogate_dynamics_checkpoints], [0, 1, 2, 3, 4])
        self.assertIn("region_1_to_4_max_acceleration_observed", results[0].surrogate_dynamics_checkpoints[0])
        self.assertIn("region_5_plus_max_acceleration_observed", results[0].surrogate_dynamics_checkpoints[0])
        self.assertIn("region_1_to_4_min_signed_distance", results[0].surrogate_dynamics_checkpoints[0])
        self.assertEqual(len(results[0].surrogate_dynamics_checkpoints[0]["region_5_plus_peak_waypoint_joint_acceleration_profile"]), 7)

    def test_batched_planner_is_deterministic_without_noise(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=6,
            planner_steps=3,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        job = self._job(request_id="det")
        first = planner.solve_batch([job])[0].trajectory
        second = planner.solve_batch([job])[0].trajectory
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_noisy_warm_start_is_deterministic_per_seed(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=9,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.2,
        )
        first_job = self._job(request_id="seed-a")
        second_job = self._job(request_id="seed-b")
        third_job = TorchPlannerJob(**{**first_job.__dict__, "request_id": "seed-c", "seed": 456})

        start = torch.as_tensor([first_job.start_config, second_job.start_config, third_job.start_config], dtype=planner.dtype)
        goal = torch.as_tensor([first_job.goal_config, second_job.goal_config, third_job.goal_config], dtype=planner.dtype)
        latent = planner._initialize_latent(
            start,
            goal,
            [first_job.seed, second_job.seed, third_job.seed],
            [first_job.dt, second_job.dt, third_job.dt],
        )
        decoded = planner._decode_trajectory(latent, start, goal).detach().cpu().numpy()

        np.testing.assert_allclose(decoded[0], decoded[1], atol=1e-6)
        self.assertFalse(np.allclose(decoded[0], decoded[2], atol=1e-6))

    def test_smooth_warm_start_reduces_initial_acceleration_excess_vs_iid_noise(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=10,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.3,
        )
        job = self._job(request_id="warm")
        start = torch.as_tensor([job.start_config], dtype=planner.dtype)
        goal = torch.as_tensor([job.goal_config], dtype=planner.dtype)

        latent = planner._initialize_latent(start, goal, [job.seed], [job.dt])
        smooth_trajectory = planner._decode_trajectory(latent, start, goal)
        smooth_summary = planner._trajectory_dynamics_summary(smooth_trajectory, [job])

        iid_trajectory = torch.as_tensor(self._old_iid_warm_start(planner, job, seed=job.seed)[None, ...], dtype=planner.dtype)
        iid_summary = planner._trajectory_dynamics_summary(iid_trajectory, [job])

        self.assertLess(
            float(smooth_summary["max_acceleration_excess"][0].item()),
            float(iid_summary["max_acceleration_excess"][0].item()),
        )
        self.assertLessEqual(
            float(smooth_summary["max_acceleration_observed"][0].item()),
            0.5 * planner.max_acceleration + 1e-3,
        )

    def test_obstacle_surrogate_is_higher_for_nearby_obstacle(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=3,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        base_job = self._job(request_id="base")
        trajectory = torch.as_tensor(
            np.linspace(np.asarray(base_job.start_config), np.asarray(base_job.goal_config), 3, dtype=np.float32)[None, ...]
        )
        midpoint = planner.forward_kinematics(trajectory)[0, 1].detach().cpu().numpy()

        near_job = self._job(
            request_id="near",
            obstacle_centers=[tuple(midpoint + np.array([0.01, 0.0, 0.0], dtype=np.float32))],
            obstacle_radii=[0.04],
            obstacle_safe=[0.04],
        )
        far_job = self._job(
            request_id="far",
            obstacle_centers=[tuple(midpoint + np.array([1.0, 1.0, 1.0], dtype=np.float32))],
            obstacle_radii=[0.04],
            obstacle_safe=[0.04],
        )
        stacked = trajectory.repeat(2, 1, 1)
        metrics = planner._objective_terms(stacked, [near_job, far_job])
        self.assertGreater(float(metrics["obstacle_cost"][0].item()), float(metrics["obstacle_cost"][1].item()))
        self.assertLess(float(metrics["min_signed_distance"][0].item()), float(metrics["min_signed_distance"][1].item()))

    def test_collision_summary_detects_worst_waypoint_and_obstacle(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=4,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        job = self._job(
            request_id="collision-summary",
            obstacle_centers=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            obstacle_radii=[0.15, 0.05],
            obstacle_safe=[0.05, 0.02],
        )
        trajectory = torch.as_tensor(
            np.asarray(
                [
                    [
                        job.start_config,
                        job.goal_config,
                        job.goal_config,
                        job.goal_config,
                    ]
                ],
                dtype=np.float32,
            )
        )
        positions = planner.forward_kinematics(trajectory)
        positions[0, 2, :] = torch.as_tensor((0.0, 0.0, 0.0), dtype=planner.dtype)
        with mock.patch.object(planner, "forward_kinematics", return_value=positions):
            summary = planner._trajectory_collision_summary(trajectory, [job])

        self.assertLess(float(summary["min_signed_distance"][0].item()), 0.0)
        self.assertEqual(int(summary["collision_waypoint_count"][0].item()), 1)
        self.assertEqual(int(summary["worst_collision_waypoint_index"][0].item()), 2)
        self.assertEqual(int(summary["worst_collision_obstacle_index"][0].item()), 0)

    def test_obstacle_cost_increases_for_single_colliding_waypoint(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=4,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        job = self._job(
            request_id="collide-cost",
            obstacle_centers=[(0.0, 0.0, 0.0)],
            obstacle_radii=[0.10],
            obstacle_safe=[0.05],
        )
        trajectory = torch.as_tensor(
            np.asarray([[job.start_config] * 4, [job.start_config] * 4], dtype=np.float32)
        )
        safe_positions = torch.full((1, 4, 3), 0.4, dtype=planner.dtype)
        colliding_positions = safe_positions.clone()
        colliding_positions[0, 2, :] = torch.zeros(3, dtype=planner.dtype)
        with mock.patch.object(
            planner,
            "forward_kinematics",
            return_value=torch.cat((safe_positions, colliding_positions), dim=0),
        ):
            metrics = planner._objective_terms(trajectory, [job, job])

        self.assertGreater(float(metrics["obstacle_cost"][1].item()), float(metrics["obstacle_cost"][0].item()))
        self.assertGreater(float(metrics["collision_penetration_depth"][1].item()), 0.0)
        self.assertEqual(int(metrics["collision_waypoint_count"][1].item()), 1)

    def test_max_profile_aggregation_exceeds_average(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=4,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        base_job = self._job(request_id="agg")
        trajectory = torch.as_tensor(
            np.linspace(np.asarray(base_job.start_config), np.asarray(base_job.goal_config), 4, dtype=np.float32)[None, ...]
        )
        hot_spot = planner.forward_kinematics(trajectory)[0, 2].detach().cpu().numpy()
        avg_job = self._job(
            request_id="avg",
            aggregate="avg",
            obstacle_centers=[tuple(hot_spot + np.array([0.005, 0.0, 0.0], dtype=np.float32))],
            obstacle_radii=[0.03],
            obstacle_safe=[0.04],
        )
        max_job = self._job(
            request_id="max",
            aggregate="max",
            obstacle_centers=[tuple(hot_spot + np.array([0.005, 0.0, 0.0], dtype=np.float32))],
            obstacle_radii=[0.03],
            obstacle_safe=[0.04],
        )
        metrics = planner._objective_terms(trajectory.repeat(2, 1, 1), [avg_job, max_job])
        self.assertGreater(float(metrics["obstacle_cost"][1].item()), float(metrics["obstacle_cost"][0].item()))

    def test_dynamics_penalty_increases_for_jerky_trajectory(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=5,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        job = self._job(request_id="dyn")
        smooth = torch.as_tensor(
            np.linspace(np.asarray(job.start_config), np.asarray(job.goal_config), 5, dtype=np.float32)[None, ...]
        )
        jerky = smooth.clone()
        jerky[0, 2, :] += 0.6
        metrics = planner._objective_terms(torch.cat((smooth, jerky), dim=0), [job, job])
        self.assertGreater(float(metrics["total_loss"][1].item()), float(metrics["total_loss"][0].item()))

    def test_dynamics_summary_tracks_jerk_and_peak_location(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=8,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        job = self._job(request_id="dyn-summary")
        smooth = torch.as_tensor(
            np.linspace(np.asarray(job.start_config), np.asarray(job.goal_config), 8, dtype=np.float32)[None, ...]
        )
        jerky = smooth.clone()
        jerky[0, 6, 3] += 0.6
        summary = planner._trajectory_dynamics_summary(torch.cat((smooth, jerky), dim=0), [job, job])

        self.assertGreater(
            float(summary["max_acceleration_observed"][1].item()),
            float(summary["max_acceleration_observed"][0].item()),
        )
        self.assertGreater(
            int(summary["acceleration_violation_count"][1].item()),
            int(summary["acceleration_violation_count"][0].item()),
        )
        self.assertEqual(int(summary["peak_acceleration_waypoint_index"][1].item()), 6)
        self.assertGreater(
            float(summary["max_adjacent_waypoint_jump"][1].item()),
            float(summary["max_adjacent_waypoint_jump"][0].item()),
        )
        self.assertGreater(
            float(summary["region_5_plus_max_acceleration_observed"][1].item()),
            float(summary["region_1_to_4_max_acceleration_observed"][1].item()),
        )
        self.assertEqual(int(summary["region_5_plus_dominates_acceleration_peak"][1].item()), 1)
        self.assertEqual(int(summary["peak_acceleration_joint_index"][1].item()), 3)
        self.assertEqual(int(summary["region_5_plus_peak_acceleration_waypoint_index"][1].item()), 6)
        self.assertEqual(int(summary["region_5_plus_peak_acceleration_joint_index"][1].item()), 3)
        self.assertGreater(
            float(summary["region_5_plus_peak_waypoint_joint_acceleration_profile"][1, 3].item()),
            0.0,
        )
        self.assertAlmostEqual(
            float(summary["region_1_to_4_peak_waypoint_joint_acceleration_profile"][1].amax().item()),
            float(summary["region_1_to_4_max_acceleration_observed"][1].item()),
        )

    def test_feasible_candidate_beats_lower_cost_invalid_candidate(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=6,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        feasible_metrics = {
            "velocity_violation_count": torch.tensor([0], dtype=torch.int64),
            "acceleration_violation_count": torch.tensor([0], dtype=torch.int64),
            "total_violation_count": torch.tensor([0], dtype=torch.int64),
            "total_violation_excess": torch.tensor([0.0], dtype=planner.dtype),
            "max_acceleration_excess": torch.tensor([0.0], dtype=planner.dtype),
            "collision_waypoint_count": torch.tensor([0], dtype=torch.int64),
            "collision_penetration_depth": torch.tensor([0.0], dtype=planner.dtype),
            "near_collision_waypoint_count": torch.tensor([0], dtype=torch.int64),
            "min_signed_distance": torch.tensor([0.5], dtype=planner.dtype),
            "total_loss": torch.tensor([2.0], dtype=planner.dtype),
            "scalarized_cost": torch.tensor([2.0], dtype=planner.dtype),
        }
        invalid_metrics = {
            "velocity_violation_count": torch.tensor([0], dtype=torch.int64),
            "acceleration_violation_count": torch.tensor([1], dtype=torch.int64),
            "total_violation_count": torch.tensor([1], dtype=torch.int64),
            "total_violation_excess": torch.tensor([0.1], dtype=planner.dtype),
            "max_acceleration_excess": torch.tensor([0.1], dtype=planner.dtype),
            "collision_waypoint_count": torch.tensor([1], dtype=torch.int64),
            "collision_penetration_depth": torch.tensor([0.05], dtype=planner.dtype),
            "near_collision_waypoint_count": torch.tensor([1], dtype=torch.int64),
            "min_signed_distance": torch.tensor([-0.05], dtype=planner.dtype),
            "total_loss": torch.tensor([1.0], dtype=planner.dtype),
            "scalarized_cost": torch.tensor([0.5], dtype=planner.dtype),
        }

        update_mask = planner._select_candidate_update_mask(invalid_metrics, feasible_metrics)
        self.assertFalse(bool(update_mask[0].item()))

    def test_less_invalid_candidate_beats_lower_cost_more_invalid_candidate(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=6,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        more_invalid = {
            "velocity_violation_count": torch.tensor([0], dtype=torch.int64),
            "acceleration_violation_count": torch.tensor([3], dtype=torch.int64),
            "total_violation_count": torch.tensor([3], dtype=torch.int64),
            "total_violation_excess": torch.tensor([0.9], dtype=planner.dtype),
            "max_acceleration_excess": torch.tensor([0.6], dtype=planner.dtype),
            "collision_waypoint_count": torch.tensor([3], dtype=torch.int64),
            "collision_penetration_depth": torch.tensor([0.4], dtype=planner.dtype),
            "near_collision_waypoint_count": torch.tensor([4], dtype=torch.int64),
            "min_signed_distance": torch.tensor([-0.4], dtype=planner.dtype),
            "total_loss": torch.tensor([1.0], dtype=planner.dtype),
            "scalarized_cost": torch.tensor([0.2], dtype=planner.dtype),
        }
        less_invalid = {
            "velocity_violation_count": torch.tensor([0], dtype=torch.int64),
            "acceleration_violation_count": torch.tensor([1], dtype=torch.int64),
            "total_violation_count": torch.tensor([1], dtype=torch.int64),
            "total_violation_excess": torch.tensor([0.2], dtype=planner.dtype),
            "max_acceleration_excess": torch.tensor([0.2], dtype=planner.dtype),
            "collision_waypoint_count": torch.tensor([1], dtype=torch.int64),
            "collision_penetration_depth": torch.tensor([0.1], dtype=planner.dtype),
            "near_collision_waypoint_count": torch.tensor([2], dtype=torch.int64),
            "min_signed_distance": torch.tensor([-0.1], dtype=planner.dtype),
            "total_loss": torch.tensor([5.0], dtype=planner.dtype),
            "scalarized_cost": torch.tensor([4.0], dtype=planner.dtype),
        }

        update_mask = planner._select_candidate_update_mask(less_invalid, more_invalid)
        self.assertTrue(bool(update_mask[0].item()))

    @unittest.skipUnless(RRTPlanner is not None, "MuJoCo-backed RRT planner module is required.")
    def test_rrt_planner_is_deterministic_for_fixed_seed(self):
        planner = RRTPlanner(model=None, data=None, step_size=0.3, max_iterations=300, goal_threshold=0.2)
        planner.set_collision_checker(lambda _config: False)
        start = np.asarray((0.0, -0.5, 0.0, 2.5, 0.0, 1.0, -1.57), dtype=np.float32)
        goal = np.asarray((0.2, -0.6, 0.1, 2.2, 0.2, 0.8, -1.2), dtype=np.float32)

        path_a, success_a = planner.plan(start, goal, rng=np.random.default_rng(7))
        path_b, success_b = planner.plan(start, goal, rng=np.random.default_rng(7))
        self.assertEqual(success_a, success_b)
        self.assertEqual(len(path_a), len(path_b))
        np.testing.assert_allclose(np.asarray(path_a), np.asarray(path_b), atol=1e-6)

    @unittest.skipUnless(RRTPlanner is not None, "MuJoCo-backed RRT planner module is required.")
    def test_rrt_rejects_colliding_edge_shortcuts(self):
        planner = RRTPlanner(model=None, data=None, step_size=0.4, max_iterations=350, goal_threshold=0.2)
        planner.joint_limits_lower = np.zeros(7, dtype=np.float32)
        planner.joint_limits_upper = np.ones(7, dtype=np.float32)
        planner.set_collision_checker(lambda config: 0.35 <= float(config[0]) <= 0.65)
        start = np.zeros(7, dtype=np.float32)
        goal = np.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)

        _, success = planner.plan(start, goal, rng=np.random.default_rng(13))
        self.assertFalse(success)

    @unittest.skipUnless(RRTPlanner is not None, "MuJoCo-backed RRT planner module is required.")
    def test_rrt_warm_start_falls_back_to_linear_when_rrt_fails(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=7,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        base_job = self._job(request_id="fallback-base")
        start = torch.as_tensor([base_job.start_config], dtype=planner.dtype, device=planner.device)
        goal = torch.as_tensor([base_job.goal_config], dtype=planner.dtype, device=planner.device)
        linear = planner._linear_trajectory(start, goal)
        ee_positions = planner.forward_kinematics(linear)[0].detach().cpu().numpy()
        colliding_job = self._job(
            request_id="fallback-colliding",
            obstacle_centers=[tuple(ee_positions[planner.n_waypoints // 2])],
            obstacle_radii=[0.20],
            obstacle_safe=[0.00],
            task_family="double_corridor",
        )
        severe_baseline = {
            "collision_waypoint_count": torch.as_tensor([4], dtype=torch.int64, device=planner.device),
            "min_signed_distance": torch.as_tensor([-0.10], dtype=planner.dtype, device=planner.device),
        }
        with mock.patch.object(planner, "_trajectory_collision_summary", return_value=severe_baseline):
            with mock.patch("src.motion_planning.RRTPlanner.RRTPlanner.plan", return_value=([], False)):
                warmed = planner._apply_rrt_warm_starts(linear, [colliding_job])
        np.testing.assert_allclose(
            warmed.detach().cpu().numpy(),
            linear.detach().cpu().numpy(),
            atol=1e-6,
        )
        metadata = planner._latest_warm_start_metadata[0]
        self.assertEqual(metadata["warm_start_strategy"], "linear")
        self.assertTrue(bool(metadata["warm_start_rrt_attempted"]))
        self.assertFalse(bool(metadata["warm_start_rrt_success"]))

    def test_rrt_warm_start_gate_is_family_and_severity_limited(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=7,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        base_job = self._job(request_id="base-family", task_family="base")
        self.assertFalse(
            planner._should_attempt_rrt_warm_start(
                base_job,
                baseline_collision_waypoint_count=6,
                baseline_min_signed_distance=-0.20,
            )
        )
        corridor_job = self._job(request_id="dc-family", task_family="double_corridor")
        self.assertFalse(
            planner._should_attempt_rrt_warm_start(
                corridor_job,
                baseline_collision_waypoint_count=1,
                baseline_min_signed_distance=-0.01,
            )
        )
        self.assertFalse(
            planner._should_attempt_rrt_warm_start(
                corridor_job,
                baseline_collision_waypoint_count=4,
                baseline_min_signed_distance=-0.05,
            )
        )
        self.assertTrue(
            planner._should_attempt_rrt_warm_start(
                corridor_job,
                baseline_collision_waypoint_count=6,
                baseline_min_signed_distance=-0.05,
            )
        )
        self.assertTrue(
            planner._should_attempt_rrt_warm_start(
                corridor_job,
                baseline_collision_waypoint_count=3,
                baseline_min_signed_distance=-0.08,
            )
        )

    @unittest.skipUnless(RRTPlanner is not None, "MuJoCo-backed RRT planner module is required.")
    def test_rrt_warm_start_restricted_to_restart_zero(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=7,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        job_r00 = self._job(request_id="task_max_a0p000_r00", task_family="double_corridor")
        job_r01 = self._job(request_id="task_max_a0p250_r01", task_family="double_corridor")
        start = torch.as_tensor([job_r00.start_config, job_r01.start_config], dtype=planner.dtype, device=planner.device)
        goal = torch.as_tensor([job_r00.goal_config, job_r01.goal_config], dtype=planner.dtype, device=planner.device)
        linear = planner._linear_trajectory(start, goal)
        severe_baseline = {
            "collision_waypoint_count": torch.as_tensor([6, 6], dtype=torch.int64, device=planner.device),
            "min_signed_distance": torch.as_tensor([-0.10, -0.10], dtype=planner.dtype, device=planner.device),
        }
        better_candidate = {
            "collision_waypoint_count": torch.as_tensor([0], dtype=torch.int64, device=planner.device),
            "min_signed_distance": torch.as_tensor([0.05], dtype=planner.dtype, device=planner.device),
        }
        with mock.patch.object(planner, "_trajectory_collision_summary", side_effect=[severe_baseline, better_candidate]):
            with mock.patch("src.motion_planning.RRTPlanner.RRTPlanner") as planner_cls_mock:
                rrt_instance = planner_cls_mock.return_value
                rrt_instance.plan.return_value = ([np.asarray(job_r00.start_config), np.asarray(job_r00.goal_config)], True)
                rrt_instance.smooth_path.return_value = [np.asarray(job_r00.start_config), np.asarray(job_r00.goal_config)]
                planner._apply_rrt_warm_starts(linear, [job_r00, job_r01])
        self.assertEqual(rrt_instance.plan.call_count, 1)
        metadata_r00 = planner._latest_warm_start_metadata[0]
        metadata_r01 = planner._latest_warm_start_metadata[1]
        self.assertTrue(bool(metadata_r00["warm_start_rrt_attempted"]))
        self.assertFalse(bool(metadata_r00["warm_start_rrt_cache_hit"]))
        self.assertFalse(bool(metadata_r01["warm_start_rrt_attempted"]))
        self.assertFalse(bool(metadata_r01["warm_start_rrt_cache_hit"]))

    @unittest.skipUnless(RRTPlanner is not None, "MuJoCo-backed RRT planner module is required.")
    def test_rrt_warm_start_reuses_single_rrt_attempt_across_alphas(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=7,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        jobs = [
            self._job(request_id="task_max_a0p000_r00", alpha=0.0, task_family="double_corridor"),
            self._job(request_id="task_max_a0p250_r00", alpha=0.25, task_family="double_corridor"),
            self._job(request_id="task_max_a0p500_r00", alpha=0.5, task_family="double_corridor"),
        ]
        start = torch.as_tensor([job.start_config for job in jobs], dtype=planner.dtype, device=planner.device)
        goal = torch.as_tensor([job.goal_config for job in jobs], dtype=planner.dtype, device=planner.device)
        linear = planner._linear_trajectory(start, goal)
        severe_baseline = {
            "collision_waypoint_count": torch.as_tensor([6, 6, 6], dtype=torch.int64, device=planner.device),
            "min_signed_distance": torch.as_tensor([-0.10, -0.10, -0.10], dtype=planner.dtype, device=planner.device),
        }
        better_candidate = {
            "collision_waypoint_count": torch.as_tensor([0], dtype=torch.int64, device=planner.device),
            "min_signed_distance": torch.as_tensor([0.05], dtype=planner.dtype, device=planner.device),
        }
        with mock.patch.object(planner, "_trajectory_collision_summary", side_effect=[severe_baseline, better_candidate]) as summary_mock:
            with mock.patch("src.motion_planning.RRTPlanner.RRTPlanner") as planner_cls_mock:
                rrt_instance = planner_cls_mock.return_value
                rrt_instance.plan.return_value = ([np.asarray(jobs[0].start_config), np.asarray(jobs[0].goal_config)], True)
                rrt_instance.smooth_path.return_value = [np.asarray(jobs[0].start_config), np.asarray(jobs[0].goal_config)]
                warmed = planner._apply_rrt_warm_starts(linear, jobs)
        self.assertEqual(rrt_instance.plan.call_count, 1)
        self.assertEqual(summary_mock.call_count, 2)
        metadata = planner._latest_warm_start_metadata
        self.assertTrue(bool(metadata[0]["warm_start_rrt_attempted"]))
        self.assertFalse(bool(metadata[0]["warm_start_rrt_cache_hit"]))
        self.assertFalse(bool(metadata[1]["warm_start_rrt_attempted"]))
        self.assertTrue(bool(metadata[1]["warm_start_rrt_cache_hit"]))
        self.assertFalse(bool(metadata[2]["warm_start_rrt_attempted"]))
        self.assertTrue(bool(metadata[2]["warm_start_rrt_cache_hit"]))
        self.assertEqual(metadata[0]["warm_start_strategy"], "rrt")
        self.assertEqual(metadata[1]["warm_start_strategy"], "rrt")
        self.assertEqual(metadata[2]["warm_start_strategy"], "rrt")
        self.assertFalse(np.allclose(warmed.detach().cpu().numpy(), linear.detach().cpu().numpy(), atol=1e-6))

    @unittest.skipUnless(RRTPlanner is not None, "MuJoCo-backed RRT planner module is required.")
    def test_rrt_warm_start_uses_reduced_budget_and_projection(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=7,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        job = self._job(request_id="rrt-budget", task_family="double_corridor")
        start = torch.as_tensor([job.start_config], dtype=planner.dtype, device=planner.device)
        goal = torch.as_tensor([job.goal_config], dtype=planner.dtype, device=planner.device)
        linear = planner._linear_trajectory(start, goal)
        severe_baseline = {
            "collision_waypoint_count": torch.as_tensor([5], dtype=torch.int64, device=planner.device),
            "min_signed_distance": torch.as_tensor([-0.10], dtype=planner.dtype, device=planner.device),
        }
        better_candidate = {
            "collision_waypoint_count": torch.as_tensor([0], dtype=torch.int64, device=planner.device),
            "min_signed_distance": torch.as_tensor([0.05], dtype=planner.dtype, device=planner.device),
        }
        with mock.patch.object(planner, "_trajectory_collision_summary", side_effect=[severe_baseline, better_candidate]):
            with mock.patch.object(planner, "_clip_warm_start_second_differences", wraps=planner._clip_warm_start_second_differences) as clip_mock:
                with mock.patch("src.motion_planning.RRTPlanner.RRTPlanner") as planner_cls_mock:
                    rrt_instance = planner_cls_mock.return_value
                    rrt_instance.plan.return_value = ([np.asarray(job.start_config), np.asarray(job.goal_config)], True)
                    rrt_instance.smooth_path.return_value = [np.asarray(job.start_config), np.asarray(job.goal_config)]
                    planner._apply_rrt_warm_starts(linear, [job])

        planner_cls_mock.assert_called_once_with(
            model=None,
            data=None,
            step_size=0.40,
            max_iterations=120,
            goal_threshold=0.20,
        )
        rrt_instance.smooth_path.assert_called_once()
        self.assertEqual(rrt_instance.smooth_path.call_args.kwargs["max_iterations"], 16)
        clip_mock.assert_called_once()

    def test_rrt_warm_start_replacement_requires_collision_better_candidate(self):
        planner = TorchTrajectoryBatchPlanner(
            device="cpu",
            n_waypoints=7,
            planner_steps=1,
            max_velocity=1.3,
            max_acceleration=0.7,
            warm_start_noise=0.0,
        )
        self.assertTrue(
            planner._rrt_candidate_is_better(
                baseline_collision_waypoint_count=3,
                baseline_min_signed_distance=-0.10,
                candidate_collision_waypoint_count=0,
                candidate_min_signed_distance=-0.30,
            )
        )
        self.assertTrue(
            planner._rrt_candidate_is_better(
                baseline_collision_waypoint_count=4,
                baseline_min_signed_distance=-0.20,
                candidate_collision_waypoint_count=2,
                candidate_min_signed_distance=-0.05,
            )
        )
        self.assertFalse(
            planner._rrt_candidate_is_better(
                baseline_collision_waypoint_count=4,
                baseline_min_signed_distance=-0.05,
                candidate_collision_waypoint_count=2,
                candidate_min_signed_distance=-0.20,
            )
        )
        self.assertFalse(
            planner._rrt_candidate_is_better(
                baseline_collision_waypoint_count=2,
                baseline_min_signed_distance=-0.20,
                candidate_collision_waypoint_count=2,
                candidate_min_signed_distance=-0.01,
            )
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
    from src.morl.planning import build_task_scene
    from src.morl.tasks import TaskSampler
    from src.motion_planning.planners import (
        CollisionPolicy,
        ContactCollisionChecker,
        ContactClassification,
        MotionPlannerFactory,
        classify_named_contact,
        default_collision_policy,
    )

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    CollisionPolicy = None
    ContactCollisionChecker = None
    ContactClassification = None
    MotionPlannerFactory = None
    classify_named_contact = None
    default_collision_policy = None
    build_task_scene = None
    TaskSampler = None


@unittest.skipUnless(HAS_MUJOCO, "MuJoCo is not installed")
class ContactClassificationTests(unittest.TestCase):
    def setUp(self):
        self.policy = CollisionPolicy()
        self.robot_bodies = {
            "base_link",
            "shoulder_link",
            "bracelet_link",
            "gripper_base",
            "left_driver",
            "right_driver",
        }
        self.adjacent_pairs = {
            tuple(sorted(("base_link", "shoulder_link"))),
            tuple(sorted(("bracelet_link", "gripper_base"))),
        }

    def test_target_visualization_contact_is_ignored(self):
        classification = classify_named_contact(
            geom1="target_geom",
            geom2="bracelet_with_vision_link",
            body1="target_marker",
            body2="bracelet_link",
            robot_body_names=self.robot_bodies,
            adjacent_body_pairs=self.adjacent_pairs,
            policy=self.policy,
        )
        self.assertEqual(classification.classification, "ignored_visual")

    def test_obstacle_contact_is_disallowed(self):
        classification = classify_named_contact(
            geom1="obstacle_1_geom",
            geom2="bracelet_with_vision_link",
            body1="obstacle_1",
            body2="bracelet_link",
            robot_body_names=self.robot_bodies,
            adjacent_body_pairs=self.adjacent_pairs,
            policy=self.policy,
        )
        self.assertEqual(classification.classification, "disallowed_obstacle")

    def test_floor_contact_is_disallowed(self):
        classification = classify_named_contact(
            geom1="floor",
            geom2="bracelet_with_vision_link",
            body1="body_0",
            body2="bracelet_link",
            robot_body_names=self.robot_bodies,
            adjacent_body_pairs=self.adjacent_pairs,
            policy=self.policy,
        )
        self.assertEqual(classification.classification, "disallowed_floor")

    def test_adjacent_self_contact_is_allowed(self):
        classification = classify_named_contact(
            geom1="base_link",
            geom2="shoulder_link",
            body1="base_link",
            body2="shoulder_link",
            robot_body_names=self.robot_bodies,
            adjacent_body_pairs=self.adjacent_pairs,
            policy=self.policy,
        )
        self.assertEqual(classification.classification, "allowed_adjacent_self")

    def test_non_adjacent_self_contact_is_disallowed(self):
        classification = classify_named_contact(
            geom1="base_link",
            geom2="bracelet_with_vision_link",
            body1="base_link",
            body2="bracelet_link",
            robot_body_names=self.robot_bodies,
            adjacent_body_pairs=self.adjacent_pairs,
            policy=self.policy,
        )
        self.assertEqual(classification.classification, "disallowed_self")

    def test_gripper_internal_contact_is_allowed(self):
        classification = classify_named_contact(
            geom1="gripper_driver",
            geom2="gripper_coupler",
            body1="left_driver",
            body2="right_driver",
            robot_body_names=self.robot_bodies,
            adjacent_body_pairs=self.adjacent_pairs,
            policy=self.policy,
        )
        self.assertEqual(classification.classification, "allowed_gripper_internal")

    def test_unsorted_allowed_body_pair_is_normalized(self):
        policy = CollisionPolicy(allowed_body_pairs=(("bracelet_link", "base_link"),))
        classification = classify_named_contact(
            geom1="base_link",
            geom2="bracelet_with_vision_link",
            body1="base_link",
            body2="bracelet_link",
            robot_body_names=self.robot_bodies,
            adjacent_body_pairs=self.adjacent_pairs,
            policy=policy,
        )
        self.assertEqual(classification.classification, "allowed_adjacent_self")

    def test_penetration_only_helper_respects_contact_distance(self):
        proximity_contact = ContactClassification(
            classification="disallowed_obstacle",
            geom1="obstacle_1_geom",
            geom2="bracelet_with_vision_link",
            body1="obstacle_1",
            body2="bracelet_link",
            distance=0.01,
        )
        penetrating_contact = ContactClassification(
            classification="disallowed_obstacle",
            geom1="obstacle_1_geom",
            geom2="bracelet_with_vision_link",
            body1="obstacle_1",
            body2="bracelet_link",
            distance=-0.001,
        )

        self.assertTrue(ContactCollisionChecker.is_disallowed_contact(proximity_contact))
        self.assertFalse(ContactCollisionChecker.is_disallowed_contact(proximity_contact, penetration_only=True))
        self.assertTrue(ContactCollisionChecker.is_disallowed_contact(penetrating_contact, penetration_only=True))


@unittest.skipUnless(HAS_MUJOCO, "MuJoCo is not installed")
class CollisionCheckerIntegrationTests(unittest.TestCase):
    def test_home_like_configuration_is_not_rejected(self):
        task = TaskSampler(seed=17).sample_task(0)
        artifact_dir = Path("test_artifacts") / "collision_checker_tmp"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        scene_path = build_task_scene(task, artifact_dir / "scenes")
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        checker = MotionPlannerFactory.create_collision_checker(model, data, policy=default_collision_policy())

        result = checker.evaluate(np.asarray(task.start_config, dtype=np.float64))

        self.assertFalse(result.has_collision)


if __name__ == "__main__":
    unittest.main()

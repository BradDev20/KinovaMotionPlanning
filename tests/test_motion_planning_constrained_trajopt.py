import unittest

import numpy as np

from src.motion_planning.constrained_trajopt import (
    _acceleration_constraint_jacobian_kernel,
    _acceleration_constraints_kernel,
    _velocity_constraint_jacobian_kernel,
    _velocity_constraints_kernel,
)


class ConstrainedTrajOptKernelTests(unittest.TestCase):
    def test_velocity_constraint_kernel_matches_expected_values(self):
        trajectory = np.asarray(
            [
                [0.0, 0.0],
                [0.2, -0.1],
                [0.3, -0.4],
            ],
            dtype=np.float64,
        )

        constraints = _velocity_constraints_kernel(trajectory, dt=0.1, max_velocity=2.5)

        expected = np.asarray(
            [
                0.5,
                1.5,
                1.5,
                -0.5,
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(constraints, expected)

    def test_acceleration_constraint_kernel_matches_expected_values(self):
        trajectory = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.4, 0.3],
            ],
            dtype=np.float64,
        )

        constraints = _acceleration_constraints_kernel(trajectory, dt=0.1, max_acceleration=25.0)

        expected = np.asarray(
            [
                5.0,
                15.0,
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(constraints, expected)

    def test_velocity_constraint_jacobian_matches_expected_structure(self):
        trajectory = np.asarray(
            [
                [0.0],
                [0.2],
                [0.1],
            ],
            dtype=np.float64,
        )

        jacobian = _velocity_constraint_jacobian_kernel(trajectory, dt=0.1)

        expected = np.asarray(
            [
                [10.0, -10.0, 0.0],
                [0.0, -10.0, 10.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(jacobian, expected)

    def test_acceleration_constraint_jacobian_matches_expected_structure(self):
        trajectory = np.asarray(
            [
                [0.0],
                [0.1],
                [0.4],
            ],
            dtype=np.float64,
        )

        jacobian = _acceleration_constraint_jacobian_kernel(trajectory, dt=0.1)

        expected = np.asarray(
            [
                [-100.0, 200.0, -100.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(jacobian, expected)


if __name__ == "__main__":
    unittest.main()

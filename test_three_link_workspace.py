"""Tests for continuous, workspace-limited inverse kinematics."""

from __future__ import annotations

import unittest

import numpy as np

from compare_standard_dh import (
    TOOL_TRANSFORM,
    compare_forward_kinematics,
    standard_dh_forward_transform,
    standard_dh_transform,
    user_table_dh_forward_transform,
    verify_random_angles,
)
from three_link_workspace import (
    DEFAULT_ANGLE_LIMITS,
    Mechanism,
    clip_angles_deg,
    continuous_angle_deg,
    wrap_angle_deg,
)


def forward_tip(mechanism: Mechanism, angles: tuple[float, float, float]) -> np.ndarray:
    """Return the end-effector position for a joint tuple."""
    return mechanism.end_positions(np.array([angles], dtype=float))[0]


class WrapAngleTests(unittest.TestCase):
    def test_wrap_keeps_values_inside_half_turn(self) -> None:
        self.assertAlmostEqual(wrap_angle_deg(181.0), -179.0)
        self.assertAlmostEqual(wrap_angle_deg(-181.0), 179.0)

    def test_continuous_angle_does_not_jump_across_the_cut(self) -> None:
        unwrapped = continuous_angle_deg(-179.0, 179.0)
        self.assertAlmostEqual(unwrapped, 181.0)
        clipped = clip_angles_deg((unwrapped, 0.0, 0.0), DEFAULT_ANGLE_LIMITS)
        self.assertAlmostEqual(clipped[0], 180.0)


class JacobianTests(unittest.TestCase):
    def test_analytic_jacobian_matches_finite_difference(self) -> None:
        mechanism = Mechanism()
        angles = (30.0, 25.0, -40.0)
        jacobian = mechanism.position_jacobian(angles)
        epsilon_rad = 1e-6
        for index in range(3):
            delta = np.zeros(3)
            delta[index] = np.rad2deg(epsilon_rad)
            plus = forward_tip(mechanism, tuple(np.array(angles) + delta))
            minus = forward_tip(mechanism, tuple(np.array(angles) - delta))
            numeric = (plus - minus) / (2.0 * epsilon_rad)
            np.testing.assert_allclose(jacobian[:, index], numeric, atol=1e-6)


class StandardDhTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mechanism = Mechanism()

    def test_standard_dh_matrix_uses_classic_operation_order(self) -> None:
        theta = 32.0
        d = 7.0
        a = 11.0
        alpha = -48.0
        transform = standard_dh_transform(theta, d, a, alpha)
        rotation_z = np.eye(4)
        rotation_z[:3, :3] = np.array(
            [
                [np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), 0.0],
                [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation_z = np.eye(4)
        translation_z[2, 3] = d
        translation_x = np.eye(4)
        translation_x[0, 3] = a
        rotation_x = np.eye(4)
        rotation_x[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(np.deg2rad(alpha)), -np.sin(np.deg2rad(alpha))],
                [0.0, np.sin(np.deg2rad(alpha)), np.cos(np.deg2rad(alpha))],
            ]
        )
        expected = rotation_z @ translation_z @ translation_x @ rotation_x
        np.testing.assert_allclose(transform, expected, atol=1e-12)

    def test_dh_origin_matches_original_but_axes_need_tool_alignment(self) -> None:
        angles = (30.0, 25.0, -40.0)
        comparison = compare_forward_kinematics(self.mechanism, angles)
        np.testing.assert_allclose(
            comparison.dh_transform[:3, 3],
            comparison.original_transform[:3, 3],
            atol=1e-12,
        )
        self.assertGreater(
            np.linalg.norm(
                comparison.dh_transform[:3, :3]
                - comparison.original_transform[:3, :3]
            ),
            1.0,
        )
        np.testing.assert_allclose(
            comparison.dh_transform @ TOOL_TRANSFORM,
            comparison.original_transform,
            atol=1e-12,
        )

    def test_aligned_standard_dh_matches_original_fk_for_random_angles(self) -> None:
        position_error, rotation_error = verify_random_angles(
            self.mechanism,
            sample_count=1000,
            seed=17,
        )
        self.assertLess(position_error, 1e-9)
        self.assertLess(rotation_error, 1e-9)

    def test_standard_dh_helper_returns_aligned_tool_pose(self) -> None:
        angles = (-120.0, -35.0, 95.0)
        aligned = standard_dh_forward_transform(self.mechanism, angles)
        comparison = compare_forward_kinematics(self.mechanism, angles)
        np.testing.assert_allclose(aligned, comparison.original_transform, atol=1e-12)

    def test_user_supplied_dh_table_returns_three_cumulative_transforms(self) -> None:
        angles = (0.0, 0.0, 0.0)
        transform_01, transform_02, transform_03 = user_table_dh_forward_transform(
            self.mechanism,
            angles,
        )
        self.assertEqual(transform_01.shape, (4, 4))
        self.assertEqual(transform_02.shape, (4, 4))
        self.assertEqual(transform_03.shape, (4, 4))
        np.testing.assert_allclose(
            transform_03,
            transform_01
            @ standard_dh_transform(90.0, 0.0, 20.0, 90.0)
            @ standard_dh_transform(0.0, 0.0, 30.0, 0.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(transform_03[:3, 3], [0.0, 0.0, 50.0], atol=1e-12)


class InverseKinematicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mechanism = Mechanism()

    def test_nearby_cartesian_target_is_reached_without_joint_jump(self) -> None:
        start = (30.0, 20.0, -30.0)
        goal_angles = (40.0, 10.0, -20.0)
        result = self.mechanism.inverse_kinematics(forward_tip(self.mechanism, goal_angles), start)
        self.assertTrue(result.reachable)
        np.testing.assert_allclose(result.position, forward_tip(self.mechanism, goal_angles), atol=1e-3)
        self.assertLess(max(abs(a - b) for a, b in zip(result.angles_deg, start)), 25.0)

    def test_q1_stops_at_180_instead_of_wrapping_to_negative(self) -> None:
        previous = (160.0, 15.0, 20.0)
        history = [previous]
        for command_q1 in np.linspace(162.0, 200.0, 20):
            target = forward_tip(self.mechanism, (float(command_q1), 15.0, 20.0))
            result = self.mechanism.inverse_kinematics(target, previous)
            delta_q1 = result.angles_deg[0] - previous[0]
            self.assertLess(abs(delta_q1), 15.0)
            self.assertGreaterEqual(result.angles_deg[0], -180.0)
            self.assertLessEqual(result.angles_deg[0], 180.0)
            previous = result.angles_deg
            history.append(previous)

        self.assertGreater(history[-1][0], 170.0)
        self.assertAlmostEqual(history[-1][0], 180.0, delta=1.0)
        self.assertFalse(any(angles[0] < 0.0 for angles in history))

    def test_free_joints_keep_moving_after_q1_hits_the_cut(self) -> None:
        previous = (180.0, 8.0, 25.0)
        start_q2, start_q3 = previous[1], previous[2]
        target = forward_tip(self.mechanism, (200.0, 8.0, 25.0))
        result = self.mechanism.inverse_kinematics(target, previous)
        self.assertAlmostEqual(result.angles_deg[0], 180.0, delta=1e-3)
        self.assertTrue(result.saturated[0])
        moved_free = abs(result.angles_deg[1] - start_q2) > 0.05 or abs(result.angles_deg[2] - start_q3) > 0.05
        self.assertTrue(moved_free)
        self.assertFalse(result.reachable)

    def test_q3_at_limit_still_allows_q1_to_track_azimuth(self) -> None:
        start = (20.0, 10.0, 110.0)
        start_tip = forward_tip(self.mechanism, start)
        radius = float(np.hypot(start_tip[0], start_tip[1]))
        start_azimuth = float(np.rad2deg(np.arctan2(start_tip[1], start_tip[0])))
        previous = start
        q1_values = [start[0]]
        for azimuth_deg in (start_azimuth + 15.0, start_azimuth + 30.0, start_azimuth + 45.0):
            target = np.array(
                [
                    radius * np.cos(np.deg2rad(azimuth_deg)),
                    radius * np.sin(np.deg2rad(azimuth_deg)),
                    start_tip[2],
                ]
            )
            result = self.mechanism.inverse_kinematics(target, previous)
            self.assertTrue(result.saturated[2])
            self.assertLess(abs(result.angles_deg[0] - previous[0]), 40.0)
            self.assertGreater(result.angles_deg[0], previous[0])
            q1_values.append(result.angles_deg[0])
            previous = result.angles_deg
        self.assertEqual(q1_values, sorted(q1_values))

    def test_far_target_stays_inside_joint_limits_and_leans_toward_it(self) -> None:
        start = (30.0, 10.0, -20.0)
        target = np.array([200.0, 200.0, 200.0])
        result = self.mechanism.inverse_kinematics(target, start)
        lower = np.array([limits[0] for limits in DEFAULT_ANGLE_LIMITS])
        upper = np.array([limits[1] for limits in DEFAULT_ANGLE_LIMITS])
        np.testing.assert_array_less(lower - 1e-9, np.array(result.angles_deg))
        np.testing.assert_array_less(np.array(result.angles_deg), upper + 1e-9)
        self.assertFalse(result.reachable)
        self.assertTrue(any(result.saturated))
        self.assertGreater(result.position[0], 0.0)
        self.assertGreater(result.position[1], 0.0)
        self.assertLess(abs(wrap_angle_deg(result.angles_deg[0] - 45.0)), 50.0)

    def test_q1_locked_at_limit_still_lets_q2_follow_height(self) -> None:
        previous = (180.0, 8.0, 20.0)
        q2_values = []
        for command_q2 in (5.0, 20.0, 35.0):
            target = forward_tip(self.mechanism, (195.0, command_q2, 20.0))
            result = self.mechanism.inverse_kinematics(target, previous)
            self.assertAlmostEqual(result.angles_deg[0], 180.0, delta=1e-3)
            self.assertTrue(result.saturated[0])
            q2_values.append(result.angles_deg[1])
            previous = result.angles_deg
        self.assertGreater(q2_values[-1], q2_values[0] + 10.0)


    def test_ik_path_records_iterates_and_does_not_clip_the_cartesian_target(self) -> None:
        start = (30.0, 10.0, -20.0)
        target = np.array([200.0, 200.0, 200.0])
        path = self.mechanism.inverse_kinematics_path(target, start)
        self.assertGreater(len(path), 1)
        self.assertFalse(path[-1].reachable)
        np.testing.assert_allclose(target, [200.0, 200.0, 200.0])
        lower = np.array([limits[0] for limits in DEFAULT_ANGLE_LIMITS])
        upper = np.array([limits[1] for limits in DEFAULT_ANGLE_LIMITS])
        for step in path:
            np.testing.assert_array_less(lower - 1e-9, np.array(step.angles_deg))
            np.testing.assert_array_less(np.array(step.angles_deg), upper + 1e-9)
            delta = max(abs(a - b) for a, b in zip(step.angles_deg, path[0].angles_deg))
            self.assertLess(delta, 181.0)


try:
    import mujoco
except ImportError:
    mujoco = None


@unittest.skipUnless(mujoco is not None, "mujoco is not installed")
class MujocoIkTests(unittest.TestCase):
    def setUp(self) -> None:
        from three_link_mujoco import load_mujoco_arm

        self.mechanism = Mechanism()
        self.arm = load_mujoco_arm(self.mechanism)

    def test_mujoco_forward_kinematics_matches_analytical_model(self) -> None:
        samples = (
            (0.0, 0.0, 0.0),
            (30.0, 25.0, -40.0),
            (-90.0, 40.0, 110.0),
            (180.0, -15.0, 20.0),
        )
        for angles in samples:
            analytical = forward_tip(self.mechanism, angles)
            self.arm.set_angles_deg(angles)
            np.testing.assert_allclose(self.arm.tip_position(), analytical, atol=1e-6)

    def test_unconstrained_target_is_written_verbatim_to_mocap(self) -> None:
        outside = np.array([200.0, -150.0, 10.0])
        self.arm.set_target(outside)
        np.testing.assert_allclose(self.arm.target_position(), outside)
        result = self.mechanism.inverse_kinematics(self.arm.target_position(), (30.0, 25.0, -40.0))
        self.assertFalse(result.reachable)
        lower = np.array([limits[0] for limits in DEFAULT_ANGLE_LIMITS])
        upper = np.array([limits[1] for limits in DEFAULT_ANGLE_LIMITS])
        np.testing.assert_array_less(lower - 1e-9, np.array(result.angles_deg))
        np.testing.assert_array_less(np.array(result.angles_deg), upper + 1e-9)

    def test_scripted_demo_keeps_joints_limited_for_out_of_workspace_targets(self) -> None:
        from three_link_mujoco import run_scripted_demo

        history = run_scripted_demo(self.mechanism, self.arm, DEFAULT_ANGLE_LIMITS, hold_frames=0)
        self.assertTrue(any(not frame["reachable"] for frame in history))
        lower = np.array([limits[0] for limits in DEFAULT_ANGLE_LIMITS])
        upper = np.array([limits[1] for limits in DEFAULT_ANGLE_LIMITS])
        for frame in history:
            np.testing.assert_array_less(lower - 1e-9, np.array(frame["angles_deg"]))
            np.testing.assert_array_less(np.array(frame["angles_deg"]), upper + 1e-9)


if __name__ == "__main__":
    unittest.main()

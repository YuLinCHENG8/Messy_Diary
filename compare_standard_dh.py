"""Compare the original three-link FK with an equivalent standard-DH model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from three_link_workspace import Matrix, Mechanism


TOOL_TRANSFORM = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@dataclass(frozen=True)
class ForwardKinematicsComparison:
    """Original and standard-DH transforms evaluated at the same joint angles."""

    original_transform: Matrix
    dh_transform: Matrix
    aligned_dh_transform: Matrix

    @property
    def position_error(self) -> float:
        """Return the DH-versus-original end-position error."""
        difference = self.aligned_dh_transform[:3, 3] - self.original_transform[:3, 3]
        return float(np.linalg.norm(difference))

    @property
    def rotation_error(self) -> float:
        """Return the Frobenius norm of the end-rotation difference."""
        difference = self.aligned_dh_transform[:3, :3] - self.original_transform[:3, :3]
        return float(np.linalg.norm(difference))


def homogeneous_transform(rotation: Matrix, position: np.ndarray) -> Matrix:
    """Combine a 3x3 rotation and a 3-vector into a 4x4 transform."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def standard_dh_transform(
    theta_deg: float,
    d: float,
    a: float,
    alpha_deg: float,
) -> Matrix:
    """Return one classic-DH transform Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    theta = np.deg2rad(theta_deg)
    alpha = np.deg2rad(alpha_deg)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array(
        [
            [
                cos_theta,
                -sin_theta * cos_alpha,
                sin_theta * sin_alpha,
                a * cos_theta,
            ],
            [
                sin_theta,
                cos_theta * cos_alpha,
                -cos_theta * sin_alpha,
                a * sin_theta,
            ],
            [0.0, sin_alpha, cos_alpha, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def original_forward_transform(
    mechanism: Mechanism,
    angles_deg: tuple[float, float, float],
) -> Matrix:
    """Return the original Rz(q1) Ry(q2) Rx(q3) FK as a 4x4 transform."""
    positions, end_rotation = mechanism.forward_kinematics(angles_deg)
    return homogeneous_transform(end_rotation, positions[-1])


def standard_dh_forward_transform(
    mechanism: Mechanism,
    angles_deg: tuple[float, float, float],
    align_tool_frame: bool = True,
) -> Matrix:
    """Return the equivalent standard-DH forward transform.

    The selected classic-DH table is:

        i  theta_i       d_i  a_i  alpha_i
        1  q1             L1    0      -90
        2  q2 - 90         0   L2      -90
        3  q3              0   L3        0

    Its frame {3} has the same origin as the original end frame but different
    axis directions. ``TOOL_TRANSFORM`` rotates frame {3} into the original
    end-frame convention without changing its position.
    """
    q1, q2, q3 = angles_deg
    length_1, length_2, length_3 = mechanism.link_lengths
    transform_01 = standard_dh_transform(q1, length_1, 0.0, -90.0)
    transform_12 = standard_dh_transform(q2 - 90.0, 0.0, length_2, -90.0)
    transform_23 = standard_dh_transform(q3, 0.0, length_3, 0.0)
    transform_03 = transform_01 @ transform_12 @ transform_23
    if align_tool_frame:
        return transform_03 @ TOOL_TRANSFORM
    return transform_03


def user_table_dh_forward_transform(
    mechanism: Mechanism,
    angles_deg: tuple[float, float, float],
) -> tuple[Matrix, Matrix, Matrix]:
    """Return FK transforms using the DH table supplied by the user.

    The table is used exactly as supplied:

        i  theta_i       d_i  a_i  alpha_i
        1  q1 + 90         L1    0      -90
        2  q2 + 90          0   L2      +90
        3  q3               0   L3        0

    The returned values are T_01, T_02, and T_03. This is a classic-DH
    serial chain; the result is not silently aligned with the original FK.
    """
    q1, q2, q3 = angles_deg
    length_1, length_2, length_3 = mechanism.link_lengths
    transform_01 = standard_dh_transform(q1 + 90.0, length_1, 0.0, -90.0)
    transform_12 = standard_dh_transform(q2 + 90.0, 0.0, length_2, 90.0)
    transform_23 = standard_dh_transform(q3, 0.0, length_3, 0.0)
    transform_02 = transform_01 @ transform_12
    transform_03 = transform_02 @ transform_23
    return transform_01, transform_02, transform_03


def compare_forward_kinematics(
    mechanism: Mechanism,
    angles_deg: tuple[float, float, float],
) -> ForwardKinematicsComparison:
    """Evaluate the original and DH models at the same joint angles."""
    original = original_forward_transform(mechanism, angles_deg)
    dh_transform = standard_dh_forward_transform(
        mechanism,
        angles_deg,
        align_tool_frame=False,
    )
    return ForwardKinematicsComparison(
        original_transform=original,
        dh_transform=dh_transform,
        aligned_dh_transform=dh_transform @ TOOL_TRANSFORM,
    )


def verify_random_angles(
    mechanism: Mechanism,
    sample_count: int,
    seed: int,
    tolerance: float = 1e-9,
) -> tuple[float, float]:
    """Check random joint angles and return the largest position/rotation errors."""
    random = np.random.default_rng(seed)
    limits = ((-180.0, 180.0), (-40.0, 40.0), (-110.0, 110.0))
    largest_position_error = 0.0
    largest_rotation_error = 0.0
    for _ in range(sample_count):
        angles = tuple(
            float(random.uniform(lower, upper))
            for lower, upper in limits
        )
        comparison = compare_forward_kinematics(mechanism, angles)
        largest_position_error = max(largest_position_error, comparison.position_error)
        largest_rotation_error = max(largest_rotation_error, comparison.rotation_error)
    if largest_position_error > tolerance or largest_rotation_error > tolerance:
        raise AssertionError(
            "Standard-DH verification failed: "
            f"position error={largest_position_error:.3e}, "
            f"rotation error={largest_rotation_error:.3e}"
        )
    return largest_position_error, largest_rotation_error


def print_matrix(name: str, matrix: Matrix) -> None:
    """Print one matrix with stable numeric formatting."""
    print(f"\n{name}:")
    print(np.array2string(matrix, precision=8, suppress_small=True))


def print_user_table_fk(
    mechanism: Mechanism,
    angles_deg: tuple[float, float, float],
) -> None:
    """Print FK matrices and end position for the user's DH table."""
    transform_01, transform_02, transform_03 = user_table_dh_forward_transform(
        mechanism,
        angles_deg,
    )
    print_matrix("User DH T_01", transform_01)
    print_matrix("User DH T_02", transform_02)
    print_matrix("User DH T_03", transform_03)
    print(f"\nUser DH end position: {transform_03[:3, 3]}")


def parse_arguments() -> argparse.Namespace:
    """Parse comparison options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--angles",
        nargs=3,
        type=float,
        metavar=("Q1", "Q2", "Q3"),
        default=(30.0, 25.0, -40.0),
        help="Joint angles in degrees. Default: 30 25 -40.",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=1000,
        help="Number of random joint-limit samples to verify. Default: 1000.",
    )
    parser.add_argument(
        "--user-table",
        action="store_true",
        help="Print FK using the supplied q1+90, q2+90 DH table.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    """Compare one displayed pose and verify many random poses."""
    arguments = parse_arguments()
    if arguments.random_samples < 0:
        raise ValueError("--random-samples must be nonnegative.")

    mechanism = Mechanism()
    angles = tuple(float(angle) for angle in arguments.angles)
    comparison = compare_forward_kinematics(mechanism, angles)

    print("Standard-DH table:")
    print(" i | theta_i   | d_i | a_i | alpha_i")
    print(" 1 | q1        | L1  | 0   | -90 deg")
    print(" 2 | q2-90 deg | 0   | L2  | -90 deg")
    print(" 3 | q3        | 0   | L3  |   0 deg")
    print(f"\nJoint angles [deg]: {angles}")
    print_matrix("Original FK T_0E", comparison.original_transform)
    print_matrix("Standard-DH T_03 before tool alignment", comparison.dh_transform)
    print_matrix("Fixed tool transform T_3E", TOOL_TRANSFORM)
    print_matrix(
        "Aligned DH T_0E = T_03 @ T_3E",
        comparison.aligned_dh_transform,
    )
    print(f"\nPosition error: {comparison.position_error:.3e}")
    print(f"Rotation error: {comparison.rotation_error:.3e}")

    if arguments.user_table:
        print("\nFK using the supplied DH table:")
        print_user_table_fk(mechanism, angles)

    maximum_position_error, maximum_rotation_error = verify_random_angles(
        mechanism,
        arguments.random_samples,
        arguments.seed,
    )
    print(
        f"\nRandom verification: PASS ({arguments.random_samples} samples)\n"
        f"Maximum position error: {maximum_position_error:.3e}\n"
        f"Maximum rotation error: {maximum_rotation_error:.3e}"
    )


if __name__ == "__main__":
    main()

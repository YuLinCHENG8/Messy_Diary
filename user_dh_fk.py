"""Forward kinematics using the user's standard-DH parameter table."""

from __future__ import annotations

import argparse

import numpy as np


LINK_LENGTHS = (100.0, 20.0, 30.0)


def standard_dh(
    theta_deg: float,
    d: float,
    a: float,
    alpha_deg: float,
) -> np.ndarray:
    """Return A_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)."""
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


def forward_kinematics(
    angles_deg: tuple[float, float, float],
    link_lengths: tuple[float, float, float] = LINK_LENGTHS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate T01, T02, and T03 using the supplied DH table.

    The supplied table is:

        i  theta_i       d_i  a_i  alpha_i
        1  q1 + 90 deg    L1    0      -90 deg
        2  q2 + 90 deg     0    L2      +90 deg
        3  q3              0    L3        0 deg
    """
    q1, q2, q3 = angles_deg
    length_1, length_2, length_3 = link_lengths

    transform_01 = standard_dh(q1 + 90.0, length_1, 0.0, -90.0)
    transform_12 = standard_dh(q2 + 90.0, 0.0, length_2, 90.0)
    transform_23 = standard_dh(q3, 0.0, length_3, 0.0)

    transform_02 = transform_01 @ transform_12
    transform_03 = transform_02 @ transform_23
    return transform_01, transform_02, transform_03


def print_matrix(name: str, matrix: np.ndarray) -> None:
    """Print a homogeneous transform."""
    print(f"\n{name} =")
    print(np.array2string(matrix, precision=8, suppress_small=True))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--angles",
        nargs=3,
        type=float,
        default=(30.0, 25.0, -40.0),
        metavar=("Q1", "Q2", "Q3"),
        help="q1 q2 q3 in degrees. Default: 30 25 -40.",
    )
    return parser.parse_args()


def main() -> None:
    """Calculate and print the user's DH forward kinematics."""
    arguments = parse_arguments()
    angles = tuple(float(angle) for angle in arguments.angles)
    transform_01, transform_02, transform_03 = forward_kinematics(angles)

    print("User DH table:")
    print("i | theta_i  | d_i | a_i | alpha_i")
    print("1 | q1 + 90  | L1  | 0   | -90 deg")
    print("2 | q2 + 90  | 0   | L2  | +90 deg")
    print("3 | q3       | 0   | L3  | 0 deg")
    print(f"\nAngles [deg] = {angles}")
    print_matrix("T01", transform_01)
    print_matrix("T02", transform_02)
    print_matrix("T03", transform_03)
    print(f"\nEnd position = {transform_03[:3, 3]}")
    print(f"End rotation =\n{transform_03[:3, :3]}")


if __name__ == "__main__":
    main()

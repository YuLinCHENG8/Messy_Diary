"""Three-link serial mechanism workspace and pose visualization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


Vector = np.ndarray
Matrix = np.ndarray


@dataclass(frozen=True)
class Mechanism:
    """Link lengths and serial forward-kinematics model."""

    link_lengths: tuple[float, float, float] = (100.0, 20.0, 30.0)

    def forward_kinematics(self, angles_deg: tuple[float, float, float]) -> tuple[list[Vector], Matrix]:
        """Return joint positions and the end-frame rotation matrix.

        The model uses active rotations in the order Rz(q1) Ry(q2) Rx(q3).
        Each link starts along its own local positive z-axis.
        """
        q1, q2, q3 = np.deg2rad(angles_deg)
        l1, l2, l3 = self.link_lengths

        rotation_1 = rotation_z(q1)
        rotation_2 = rotation_y(q2)
        rotation_3 = rotation_x(q3)
        end_rotation = rotation_1 @ rotation_2 @ rotation_3
        local_z = np.array([0.0, 0.0, 1.0])

        base = np.zeros(3)
        joint_1 = base + rotation_1 @ (l1 * local_z)
        joint_2 = joint_1 + rotation_1 @ rotation_2 @ (l2 * local_z)
        end = joint_2 + end_rotation @ (l3 * local_z)

        return [base, joint_1, joint_2, end], end_rotation


def rotation_x(angle_rad: float) -> Matrix:
    """Return an active rotation about the x-axis."""
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ]
    )


def rotation_y(angle_rad: float) -> Matrix:
    """Return an active rotation about the y-axis."""
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
    )


def rotation_z(angle_rad: float) -> Matrix:
    """Return an active rotation about the z-axis."""
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def sample_workspace(
    mechanism: Mechanism,
    angle_values_deg: np.ndarray,
) -> np.ndarray:
    """Sample end points for the Cartesian product of three angle arrays."""
    points = []
    for q1 in angle_values_deg:
        for q2 in angle_values_deg:
            for q3 in angle_values_deg:
                positions, _ = mechanism.forward_kinematics((q1, q2, q3))
                points.append(positions[-1])
    return np.asarray(points)


def set_equal_axes(axis: plt.Axes, points: np.ndarray) -> None:
    """Set equal 3D scale using the supplied points."""
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max((maximum - minimum).max() / 2.0, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def draw_frame(axis: plt.Axes, origin: Vector, rotation: Matrix, scale: float) -> None:
    """Draw the x, y, and z axes of a frame."""
    colors = ("tab:red", "tab:green", "tab:blue")
    labels = ("x", "y", "z")
    for index, (color, label) in enumerate(zip(colors, labels)):
        direction = rotation[:, index] * scale
        axis.quiver(*origin, *direction, color=color, arrow_length_ratio=0.15)
        axis.text(*(origin + direction), label, color=color)


def plot_workspace(
    mechanism: Mechanism,
    workspace_points: np.ndarray,
    pose_angles: list[tuple[float, float, float]],
) -> None:
    """Plot sampled workspace and selected serial-link poses."""
    figure = plt.figure(figsize=(12, 6))
    workspace_axis = figure.add_subplot(121, projection="3d")
    workspace_axis.scatter(
        workspace_points[:, 0],
        workspace_points[:, 1],
        workspace_points[:, 2],
        s=8,
        alpha=0.25,
        label="end-effector workspace",
    )
    workspace_axis.set_title("Sampled 3D workspace")
    workspace_axis.set_xlabel("X")
    workspace_axis.set_ylabel("Y")
    workspace_axis.set_zlabel("Z")
    set_equal_axes(workspace_axis, workspace_points)
    workspace_axis.legend()

    pose_axis = figure.add_subplot(122, projection="3d")
    pose_points = [np.zeros(3)]
    frame_scale = sum(mechanism.link_lengths) * 0.15
    for angles in pose_angles:
        positions, end_rotation = mechanism.forward_kinematics(angles)
        pose_points.extend(positions)
        pose_axis.plot(
            [point[0] for point in positions],
            [point[1] for point in positions],
            [point[2] for point in positions],
            marker="o",
            label=f"q={angles}°",
        )
        draw_frame(pose_axis, positions[-1], end_rotation, frame_scale)

    pose_axis.set_title("Representative poses")
    pose_axis.set_xlabel("X")
    pose_axis.set_ylabel("Y")
    pose_axis.set_zlabel("Z")
    set_equal_axes(pose_axis, np.asarray(pose_points))
    pose_axis.legend(fontsize="small")
    figure.tight_layout()
    plt.show()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        type=float,
        default=30.0,
        help="Sampling step for all three joint angles in degrees.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print the example pose without opening a plot window.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the workspace sampling and visualization."""
    arguments = parse_arguments()
    if arguments.step <= 0.0 or arguments.step > 360.0:
        raise ValueError("--step must be greater than 0 and no greater than 360.")

    mechanism = Mechanism()
    sample_angles = np.arange(-180.0, 180.0, arguments.step)
    workspace_points = sample_workspace(mechanism, sample_angles)
    example_angles = (30.0, 25.0, -40.0)
    positions, end_rotation = mechanism.forward_kinematics(example_angles)

    print(f"Sample count: {len(workspace_points)}")
    print(f"Example angles (q1, q2, q3) [deg]: {example_angles}")
    print(f"End position [x, y, z]:\n{positions[-1]}")
    print(f"End rotation Rz(q1) @ Ry(q2) @ Rx(q3):\n{end_rotation}")

    if not arguments.no_plot:
        pose_angles = [
            (0.0, 0.0, 0.0),
            example_angles,
            (90.0, -45.0, 60.0),
        ]
        plot_workspace(mechanism, workspace_points, pose_angles)


if __name__ == "__main__":
    main()

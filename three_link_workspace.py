"""Three-link serial mechanism workspace and pose visualization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, TextBox


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
    angle_values_deg: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Sample end points for the Cartesian product of three angle ranges."""
    points = []
    for q1 in angle_values_deg[0]:
        for q2 in angle_values_deg[1]:
            for q3 in angle_values_deg[2]:
                positions, _ = mechanism.forward_kinematics((q1, q2, q3))
                points.append(positions[-1])
    return np.asarray(points)


def inclusive_samples(lower: float, upper: float, step: float) -> np.ndarray:
    """Return evenly stepped values and include the upper limit."""
    values = np.arange(lower, upper, step)
    if len(values) == 0 or not np.isclose(values[-1], upper):
        values = np.append(values, upper)
    return values


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
    initial_angles: tuple[float, float, float],
    angle_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> None:
    """Plot workspace and provide sliders/text boxes for one live pose."""
    figure = plt.figure(figsize=(13, 8))
    workspace_axis = figure.add_axes((0.05, 0.20, 0.42, 0.70), projection="3d")
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

    pose_axis = figure.add_axes((0.53, 0.20, 0.42, 0.70), projection="3d")
    frame_scale = sum(mechanism.link_lengths) * 0.15

    slider_axes = [
        figure.add_axes((0.16, 0.12, 0.28, 0.025)),
        figure.add_axes((0.16, 0.08, 0.28, 0.025)),
        figure.add_axes((0.16, 0.04, 0.28, 0.025)),
    ]
    sliders = [
        Slider(
            slider_axes[index],
            f"q{index + 1} (deg)",
            limits[0],
            limits[1],
            valinit=initial_angles[index],
            valstep=1.0,
        )
        for index, limits in enumerate(angle_limits)
    ]

    textbox_axes = [
        figure.add_axes((0.47, 0.12, 0.04, 0.025)),
        figure.add_axes((0.47, 0.08, 0.04, 0.025)),
        figure.add_axes((0.47, 0.04, 0.04, 0.025)),
    ]
    textboxes = [
        TextBox(textbox_axes[index], "", initial=f"{initial_angles[index]:.0f}")
        for index in range(3)
    ]

    def redraw_pose(angles: tuple[float, float, float]) -> None:
        positions, end_rotation = mechanism.forward_kinematics(angles)
        pose_axis.clear()
        pose_axis.plot(
            [point[0] for point in positions],
            [point[1] for point in positions],
            [point[2] for point in positions],
            marker="o",
            linewidth=2.0,
            color="tab:purple",
            label=f"q={tuple(round(angle, 1) for angle in angles)}°",
        )
        draw_frame(pose_axis, positions[-1], end_rotation, frame_scale)
        pose_axis.set_title("Interactive pose")
        pose_axis.set_xlabel("X")
        pose_axis.set_ylabel("Y")
        pose_axis.set_zlabel("Z")
        set_equal_axes(pose_axis, np.asarray(positions))
        pose_axis.legend(fontsize="small")

    def update_from_sliders(_value: float) -> None:
        angles = tuple(slider.val for slider in sliders)
        for textbox, angle in zip(textboxes, angles):
            textbox.set_val(f"{angle:.0f}")
        redraw_pose(angles)
        figure.canvas.draw_idle()

    def update_from_textbox(index: int, text: str) -> None:
        try:
            value = float(text)
        except ValueError:
            textboxes[index].set_val(f"{sliders[index].val:.0f}")
            return
        lower, upper = angle_limits[index]
        sliders[index].set_val(np.clip(value, lower, upper))

    for slider in sliders:
        slider.on_changed(update_from_sliders)
    for index, textbox in enumerate(textboxes):
        textbox.on_submit(lambda text, index=index: update_from_textbox(index, text))

    redraw_pose(initial_angles)
    figure.suptitle("Three-link workspace and interactive joint angles")
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
    angle_limits = ((-180.0, 180.0), (-40.0, 40.0), (-110.0, 110.0))
    sample_angles = tuple(
        inclusive_samples(lower, upper, arguments.step)
        for lower, upper in angle_limits
    )
    workspace_points = sample_workspace(mechanism, sample_angles)
    example_angles = (30.0, 25.0, -40.0)
    positions, end_rotation = mechanism.forward_kinematics(example_angles)

    print(f"Sample count: {len(workspace_points)}")
    print(f"Example angles (q1, q2, q3) [deg]: {example_angles}")
    print(f"End position [x, y, z]:\n{positions[-1]}")
    print(f"End rotation Rz(q1) @ Ry(q2) @ Rx(q3):\n{end_rotation}")

    if not arguments.no_plot:
        plot_workspace(mechanism, workspace_points, example_angles, angle_limits)


if __name__ == "__main__":
    main()

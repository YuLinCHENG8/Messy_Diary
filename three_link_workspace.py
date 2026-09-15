"""Three-link serial mechanism workspace and pose visualization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

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

    def end_positions(self, angles_deg: np.ndarray) -> np.ndarray:
        """Vectorized end-effector positions for many joint-angle samples.

        Args:
            angles_deg: Array of shape (N, 3) with columns [q1, q2, q3] in degrees.

        Returns:
            Array of shape (N, 3) with reachable end-effector positions.
        """
        q1 = np.deg2rad(angles_deg[:, 0])
        q2 = np.deg2rad(angles_deg[:, 1])
        q3 = np.deg2rad(angles_deg[:, 2])
        l1, l2, l3 = self.link_lengths

        cos_q1 = np.cos(q1)
        sin_q1 = np.sin(q1)
        cos_q2 = np.cos(q2)
        sin_q2 = np.sin(q2)
        cos_q3 = np.cos(q3)
        sin_q3 = np.sin(q3)

        # Link 1 after Rz(q1): still along world z.
        p1 = np.column_stack((np.zeros_like(q1), np.zeros_like(q1), np.full_like(q1, l1)))

        # Link 2 after Rz(q1) Ry(q2): local z becomes [sin(q2), 0, cos(q2)], then Rz.
        direction_2 = np.column_stack(
            (
                cos_q1 * sin_q2,
                sin_q1 * sin_q2,
                cos_q2,
            )
        )
        p2 = p1 + l2 * direction_2

        # Link 3 after Rz(q1) Ry(q2) Rx(q3): local z mapped by the full rotation.
        # R @ [0,0,1] equals the third column of R = Rz Ry Rx.
        direction_3_x = cos_q1 * sin_q2 * cos_q3 + sin_q1 * sin_q3
        direction_3_y = sin_q1 * sin_q2 * cos_q3 - cos_q1 * sin_q3
        direction_3_z = cos_q2 * cos_q3
        direction_3 = np.column_stack((direction_3_x, direction_3_y, direction_3_z))
        return p2 + l3 * direction_3


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


def inclusive_samples(lower: float, upper: float, step: float) -> np.ndarray:
    """Return evenly stepped values and include the upper limit."""
    values = np.arange(lower, upper, step)
    if len(values) == 0 or not np.isclose(values[-1], upper):
        values = np.append(values, upper)
    return values


def default_joint_steps(base_step: float) -> tuple[float, float, float]:
    """Choose denser steps for the smaller joint ranges."""
    return (
        base_step,
        max(base_step / 6.0, 1.0),
        max(base_step / 3.0, 2.0),
    )


def sample_reachable_workspace(
    mechanism: Mechanism,
    angle_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    steps_deg: tuple[float, float, float],
) -> np.ndarray:
    """Sample reachable end-effector positions inside joint limits."""
    angle_grids = [
        inclusive_samples(limits[0], limits[1], step)
        for limits, step in zip(angle_limits, steps_deg)
    ]
    q1_grid, q2_grid, q3_grid = np.meshgrid(*angle_grids, indexing="ij")
    angles = np.column_stack(
        (
            q1_grid.ravel(),
            q2_grid.ravel(),
            q3_grid.ravel(),
        )
    )
    return mechanism.end_positions(angles)


def set_equal_axes(axis: plt.Axes, points: np.ndarray) -> None:
    """Set equal numeric limits and equal on-screen scale for all axes."""
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max((maximum - minimum).max() / 2.0, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


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
    """Plot reachable end space and provide sliders/text boxes for one live pose."""
    figure = plt.figure(figsize=(13, 8))
    workspace_axis = figure.add_axes((0.05, 0.20, 0.42, 0.70), projection="3d")
    workspace_axis.mouse_init(rotate_btn=1, zoom_btn=3)
    scatter = workspace_axis.scatter(
        workspace_points[:, 0],
        workspace_points[:, 1],
        workspace_points[:, 2],
        c=workspace_points[:, 2],
        cmap="viridis",
        s=4,
        alpha=0.35,
        linewidths=0.0,
    )
    figure.colorbar(scatter, ax=workspace_axis, shrink=0.65, label="end z")
    tip_plot = workspace_axis.plot([], [], [], "o", color="tab:red", markersize=8, label="current tip")[0]
    workspace_axis.set_title("Reachable workspace (drag to rotate)")
    workspace_axis.set_xlabel("X")
    workspace_axis.set_ylabel("Y")
    workspace_axis.set_zlabel("Z")
    set_equal_axes(workspace_axis, workspace_points)
    workspace_axis.legend(loc="upper left")

    pose_axis = figure.add_axes((0.53, 0.20, 0.42, 0.70), projection="3d")
    pose_axis.mouse_init(rotate_btn=1, zoom_btn=3)
    frame_scale = sum(mechanism.link_lengths) * 0.15
    pose_bounds = np.vstack((workspace_points, np.zeros((1, 3))))

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
        tip = positions[-1]
        tip_plot.set_data([tip[0]], [tip[1]])
        tip_plot.set_3d_properties([tip[2]])

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
        draw_frame(pose_axis, tip, end_rotation, frame_scale)
        pose_axis.set_title("Interactive pose (drag to rotate)")
        pose_axis.set_xlabel("X")
        pose_axis.set_ylabel("Y")
        pose_axis.set_zlabel("Z")
        set_equal_axes(pose_axis, pose_bounds)
        pose_axis.legend(fontsize="small")

    def update_from_sliders(_value: float) -> None:
        angles = tuple(float(slider.val) for slider in sliders)
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
        sliders[index].set_val(float(np.clip(value, lower, upper)))

    for slider in sliders:
        slider.on_changed(update_from_sliders)
    for index, textbox in enumerate(textboxes):
        textbox.on_submit(lambda text, index=index: update_from_textbox(index, text))

    redraw_pose(initial_angles)
    figure.suptitle("Reachable tip workspace within joint limits")
    plt.show()


def export_workspace_html(
    mechanism: Mechanism,
    workspace_points: np.ndarray,
    angles: tuple[float, float, float],
    output_path: Path,
) -> None:
    """Export a rotatable and zoomable 3D workspace view as HTML."""
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError(
            "HTML export requires Plotly. Install it with: pip install plotly"
        ) from error

    positions, end_rotation = mechanism.forward_kinematics(angles)
    end_point = positions[-1]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=workspace_points[:, 0],
            y=workspace_points[:, 1],
            z=workspace_points[:, 2],
            mode="markers",
            marker={
                "size": 2,
                "color": workspace_points[:, 2],
                "colorscale": "Viridis",
                "opacity": 0.35,
                "colorbar": {"title": "end z"},
            },
            name="reachable workspace",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[point[0] for point in positions],
            y=[point[1] for point in positions],
            z=[point[2] for point in positions],
            mode="lines+markers",
            line={"color": "purple", "width": 8},
            marker={"size": 5},
            name="current pose",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[end_point[0]],
            y=[end_point[1]],
            z=[end_point[2]],
            mode="markers",
            marker={"color": "red", "size": 7},
            name="current tip",
        )
    )
    figure.update_layout(
        title=(
            "Reachable end-effector workspace"
            f" | q=({angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f})°"
        ),
        scene={
            "xaxis": {"title": "X", "scaleanchor": "y"},
            "yaxis": {"title": "Y", "scaleanchor": "x"},
            "zaxis": {"title": "Z"},
            "aspectmode": "cube",
        },
        width=1100,
        height=750,
    )
    figure.write_html(output_path, include_plotlyjs=True)
    print(f"Interactive HTML exported to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        type=float,
        default=10.0,
        help="Base sampling step for q1 in degrees. q2/q3 use denser steps automatically.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print the example pose without opening a plot window.",
    )
    parser.add_argument(
        "--export-html",
        type=Path,
        help="Export the fixed workspace and current pose to an interactive HTML file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the workspace sampling and visualization."""
    arguments = parse_arguments()
    if arguments.step <= 0.0 or arguments.step > 360.0:
        raise ValueError("--step must be greater than 0 and no greater than 360.")

    mechanism = Mechanism()
    angle_limits = ((-180.0, 180.0), (-40.0, 40.0), (-110.0, 110.0))
    steps_deg = default_joint_steps(arguments.step)
    workspace_points = sample_reachable_workspace(mechanism, angle_limits, steps_deg)
    example_angles = (30.0, 25.0, -40.0)
    positions, end_rotation = mechanism.forward_kinematics(example_angles)

    horizontal_radius = np.linalg.norm(workspace_points[:, :2], axis=1)
    print(f"Sample count: {len(workspace_points)}")
    print(f"Joint steps [deg]: q1={steps_deg[0]}, q2={steps_deg[1]}, q3={steps_deg[2]}")
    print(f"Reachable XYZ min: {workspace_points.min(axis=0)}")
    print(f"Reachable XYZ max: {workspace_points.max(axis=0)}")
    print(f"Reachable XY radius range: [{horizontal_radius.min():.2f}, {horizontal_radius.max():.2f}]")
    print(f"Example angles (q1, q2, q3) [deg]: {example_angles}")
    print(f"End position [x, y, z]:\n{positions[-1]}")
    print(f"End rotation Rz(q1) @ Ry(q2) @ Rx(q3):\n{end_rotation}")

    if arguments.export_html is not None:
        export_workspace_html(
            mechanism,
            workspace_points,
            example_angles,
            arguments.export_html,
        )

    if not arguments.no_plot:
        plot_workspace(mechanism, workspace_points, example_angles, angle_limits)


if __name__ == "__main__":
    main()

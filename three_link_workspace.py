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
JointLimits = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

DEFAULT_ANGLE_LIMITS: JointLimits = ((-180.0, 180.0), (-40.0, 40.0), (-110.0, 110.0))
REACHABLE_POSITION_TOLERANCE = 1e-3
AT_LIMIT_TOLERANCE_DEG = 1e-4


@dataclass(frozen=True)
class InverseKinematicsResult:
    """Continuous inverse-kinematics solution measured from the previous pose."""

    angles_deg: tuple[float, float, float]
    position: Vector
    position_error: float
    reachable: bool
    saturated: tuple[bool, bool, bool]
    iterations: int


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

    def position_jacobian(self, angles_deg: tuple[float, float, float]) -> Matrix:
        """Return the 3x3 position Jacobian with respect to joint angles in radians."""
        q1, q2, q3 = np.deg2rad(angles_deg)
        _l1, l2, l3 = self.link_lengths
        cos_q1 = np.cos(q1)
        sin_q1 = np.sin(q1)
        cos_q2 = np.cos(q2)
        sin_q2 = np.sin(q2)
        cos_q3 = np.cos(q3)
        sin_q3 = np.sin(q3)
        reach_12 = l2 + l3 * cos_q3
        column_1 = np.array(
            [
                -sin_q1 * sin_q2 * reach_12 + cos_q1 * l3 * sin_q3,
                cos_q1 * sin_q2 * reach_12 + sin_q1 * l3 * sin_q3,
                0.0,
            ]
        )
        column_2 = np.array(
            [
                cos_q1 * cos_q2 * reach_12,
                sin_q1 * cos_q2 * reach_12,
                -sin_q2 * reach_12,
            ]
        )
        column_3 = np.array(
            [
                -cos_q1 * sin_q2 * l3 * sin_q3 + sin_q1 * l3 * cos_q3,
                -sin_q1 * sin_q2 * l3 * sin_q3 - cos_q1 * l3 * cos_q3,
                -cos_q2 * l3 * sin_q3,
            ]
        )
        return np.column_stack((column_1, column_2, column_3))

    def inverse_kinematics(
        self,
        target: Vector,
        previous_angles_deg: tuple[float, float, float],
        angle_limits: JointLimits = DEFAULT_ANGLE_LIMITS,
        position_tolerance: float = REACHABLE_POSITION_TOLERANCE,
        max_iterations: int = 80,
        max_step_deg: float = 20.0,
        damping: float = 1e-3,
    ) -> InverseKinematicsResult:
        """Track a Cartesian target without leaving the joint-limit workspace.

        The solver starts from ``previous_angles_deg`` and never wraps a joint
        across a branch cut such as ±180°. Joints that are already at a limit
        and would move further out are locked; the remaining free joints keep
        reducing the Cartesian error through a reduced damped least-squares step.
        """
        target = np.asarray(target, dtype=float)
        angles = np.array(previous_angles_deg, dtype=float)
        lower, upper = limit_arrays(angle_limits)
        position = self.end_positions(angles.reshape(1, 3))[0]
        error_norm = float(np.linalg.norm(target - position))
        if error_norm <= position_tolerance:
            return self._ik_result(angles, position, error_norm, angle_limits, 0)

        for used_iterations in range(1, max_iterations + 1):
            if error_norm <= position_tolerance:
                return self._ik_result(angles, position, error_norm, angle_limits, used_iterations)

            error = target - position
            current_angles = tuple(float(angle) for angle in angles)
            jacobian_deg = self.position_jacobian(current_angles) * np.pi / 180.0
            delta_deg = damped_least_squares_with_locks(
                jacobian_deg,
                error,
                angles,
                lower,
                upper,
                damping,
            )
            step_scale = float(np.max(np.abs(delta_deg)))
            if step_scale > max_step_deg:
                delta_deg *= max_step_deg / step_scale
            if float(np.linalg.norm(delta_deg)) < 1e-10:
                return self._ik_result(angles, position, error_norm, angle_limits, used_iterations)

            next_angles = np.clip(angles + delta_deg, lower, upper)
            next_position = self.end_positions(next_angles.reshape(1, 3))[0]
            next_error = float(np.linalg.norm(target - next_position))
            shrink_count = 0
            while next_error >= error_norm - 1e-12 and shrink_count < 4:
                delta_deg *= 0.5
                if float(np.linalg.norm(delta_deg)) < 1e-10:
                    break
                next_angles = np.clip(angles + delta_deg, lower, upper)
                next_position = self.end_positions(next_angles.reshape(1, 3))[0]
                next_error = float(np.linalg.norm(target - next_position))
                shrink_count += 1
            if next_error >= error_norm - 1e-12:
                if next_error < error_norm:
                    return self._ik_result(
                        next_angles, next_position, next_error, angle_limits, used_iterations
                    )
                return self._ik_result(angles, position, error_norm, angle_limits, used_iterations)
            angles = next_angles
            position = next_position
            error_norm = next_error

        return self._ik_result(angles, position, error_norm, angle_limits, used_iterations)

    def _ik_result(
        self,
        angles: np.ndarray,
        position: Vector,
        error_norm: float,
        angle_limits: JointLimits,
        iterations: int,
    ) -> InverseKinematicsResult:
        """Pack solver output and joint-limit flags."""
        angles_deg = tuple(float(angle) for angle in angles)
        return InverseKinematicsResult(
            angles_deg=angles_deg,
            position=position,
            position_error=error_norm,
            reachable=error_norm <= REACHABLE_POSITION_TOLERANCE,
            saturated=joints_at_limits(angles_deg, angle_limits),
            iterations=iterations,
        )


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


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def continuous_angle_deg(raw_deg: float, previous_deg: float) -> float:
    """Return the angle nearest to ``previous_deg`` that represents ``raw_deg``."""
    return previous_deg + wrap_angle_deg(raw_deg - previous_deg)


def limit_arrays(angle_limits: JointLimits) -> tuple[np.ndarray, np.ndarray]:
    """Split joint limits into lower and upper arrays."""
    lower = np.array([limits[0] for limits in angle_limits], dtype=float)
    upper = np.array([limits[1] for limits in angle_limits], dtype=float)
    return lower, upper


def joints_at_limits(
    angles_deg: tuple[float, float, float],
    angle_limits: JointLimits,
    tolerance_deg: float = AT_LIMIT_TOLERANCE_DEG,
) -> tuple[bool, bool, bool]:
    """Return whether each joint is at a lower or upper bound."""
    saturated = []
    for angle, limits in zip(angles_deg, angle_limits):
        at_lower = angle <= limits[0] + tolerance_deg
        at_upper = angle >= limits[1] - tolerance_deg
        saturated.append(bool(at_lower or at_upper))
    return (saturated[0], saturated[1], saturated[2])


def clip_angles_deg(
    angles_deg: tuple[float, float, float],
    angle_limits: JointLimits,
) -> tuple[float, float, float]:
    """Clamp each joint to its allowed interval without wrapping."""
    lower, upper = limit_arrays(angle_limits)
    clipped = np.clip(np.array(angles_deg, dtype=float), lower, upper)
    return tuple(float(angle) for angle in clipped)


def damped_least_squares_with_locks(
    jacobian_deg: Matrix,
    error: Vector,
    angles_deg: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Solve for a joint step, locking joints that would leave their limits.

    Locked joints stay still. The remaining columns of the Jacobian are used to
    keep reducing the Cartesian error, so unsaturated joints can still move
    along the workspace boundary.
    """
    free = np.ones(3, dtype=bool)
    delta = np.zeros(3, dtype=float)
    for _ in range(3):
        if not np.any(free):
            return np.zeros(3, dtype=float)
        jacobian_free = jacobian_deg[:, free]
        damped = jacobian_free @ jacobian_free.T + (damping**2) * np.eye(3)
        delta_free = jacobian_free.T @ np.linalg.solve(damped, error)
        delta[:] = 0.0
        delta[free] = delta_free
        locked_this_pass = False
        for index in range(3):
            if not free[index]:
                continue
            at_lower = angles_deg[index] <= lower[index] + AT_LIMIT_TOLERANCE_DEG
            at_upper = angles_deg[index] >= upper[index] - AT_LIMIT_TOLERANCE_DEG
            pushing_out = (at_lower and delta[index] < 0.0) or (at_upper and delta[index] > 0.0)
            if pushing_out:
                free[index] = False
                locked_this_pass = True
        if not locked_this_pass:
            break
    return delta


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
    angle_limits: JointLimits,
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


def ik_status_text(result: InverseKinematicsResult) -> str:
    """Build a short status line for the interactive plot."""
    locked = [f"q{index + 1}" for index, saturated in enumerate(result.saturated) if saturated]
    if result.reachable:
        if locked:
            return f"IK reachable | at limit: {', '.join(locked)}"
        return "IK reachable | joints free"
    if locked:
        free = [f"q{index + 1}" for index, saturated in enumerate(result.saturated) if not saturated]
        free_text = ", ".join(free) if free else "none"
        return (
            f"Outside workspace | locked: {', '.join(locked)}"
            f" | still moving: {free_text}"
            f" | err={result.position_error:.2f}"
        )
    return f"Outside workspace | err={result.position_error:.2f}"


def plot_workspace(
    mechanism: Mechanism,
    workspace_points: np.ndarray,
    initial_angles: tuple[float, float, float],
    angle_limits: JointLimits,
) -> None:
    """Plot reachable end space and provide joint plus Cartesian IK controls."""
    initial_positions, _ = mechanism.forward_kinematics(initial_angles)
    initial_tip = initial_positions[-1]
    xyz_minimum = workspace_points.min(axis=0) - 25.0
    xyz_maximum = workspace_points.max(axis=0) + 25.0
    xyz_labels = ("X", "Y", "Z")

    figure = plt.figure(figsize=(13, 8.4))
    workspace_axis = figure.add_axes((0.05, 0.26, 0.42, 0.64), projection="3d")
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
    target_plot = workspace_axis.plot(
        [], [], [], "x", color="black", markersize=8, markeredgewidth=2, label="IK target"
    )[0]
    workspace_axis.set_title("Reachable workspace (drag to rotate)")
    workspace_axis.set_xlabel("X")
    workspace_axis.set_ylabel("Y")
    workspace_axis.set_zlabel("Z")
    set_equal_axes(workspace_axis, workspace_points)
    workspace_axis.legend(loc="upper left")

    pose_axis = figure.add_axes((0.53, 0.26, 0.42, 0.64), projection="3d")
    pose_axis.mouse_init(rotate_btn=1, zoom_btn=3)
    frame_scale = sum(mechanism.link_lengths) * 0.15
    pose_bounds = np.vstack((workspace_points, np.zeros((1, 3))))
    status_axis = figure.add_axes((0.16, 0.18, 0.72, 0.04))
    status_axis.axis("off")
    status_text = status_axis.text(0.0, 0.5, "", fontsize=10, va="center")

    joint_slider_axes = [
        figure.add_axes((0.14, 0.12, 0.28, 0.025)),
        figure.add_axes((0.14, 0.08, 0.28, 0.025)),
        figure.add_axes((0.14, 0.04, 0.28, 0.025)),
    ]
    joint_sliders = [
        Slider(
            joint_slider_axes[index],
            f"q{index + 1} (deg)",
            limits[0],
            limits[1],
            valinit=initial_angles[index],
            valstep=0.1,
        )
        for index, limits in enumerate(angle_limits)
    ]
    joint_textbox_axes = [
        figure.add_axes((0.44, 0.12, 0.05, 0.025)),
        figure.add_axes((0.44, 0.08, 0.05, 0.025)),
        figure.add_axes((0.44, 0.04, 0.05, 0.025)),
    ]
    joint_textboxes = [
        TextBox(joint_textbox_axes[index], "", initial=f"{initial_angles[index]:.1f}")
        for index in range(3)
    ]

    xyz_slider_axes = [
        figure.add_axes((0.62, 0.12, 0.26, 0.025)),
        figure.add_axes((0.62, 0.08, 0.26, 0.025)),
        figure.add_axes((0.62, 0.04, 0.26, 0.025)),
    ]
    xyz_sliders = [
        Slider(
            xyz_slider_axes[index],
            f"{xyz_labels[index]}",
            float(xyz_minimum[index]),
            float(xyz_maximum[index]),
            valinit=float(initial_tip[index]),
            valstep=0.5,
        )
        for index in range(3)
    ]
    xyz_textbox_axes = [
        figure.add_axes((0.90, 0.12, 0.06, 0.025)),
        figure.add_axes((0.90, 0.08, 0.06, 0.025)),
        figure.add_axes((0.90, 0.04, 0.06, 0.025)),
    ]
    xyz_textboxes = [
        TextBox(xyz_textbox_axes[index], "", initial=f"{initial_tip[index]:.1f}")
        for index in range(3)
    ]

    ui_state = {"busy": False, "target": initial_tip.copy()}

    def current_joint_angles() -> tuple[float, float, float]:
        return tuple(float(slider.val) for slider in joint_sliders)

    def current_target() -> Vector:
        return np.array([float(slider.val) for slider in xyz_sliders], dtype=float)

    def set_slider_value(slider: Slider, value: float) -> None:
        slider.set_val(value)

    def redraw_pose(
        angles: tuple[float, float, float],
        target: Vector,
        result: InverseKinematicsResult | None = None,
    ) -> None:
        positions, end_rotation = mechanism.forward_kinematics(angles)
        tip = positions[-1]
        tip_color = "tab:red" if result is None or result.reachable else "tab:orange"
        tip_plot.set_data([tip[0]], [tip[1]])
        tip_plot.set_3d_properties([tip[2]])
        tip_plot.set_color(tip_color)
        target_plot.set_data([target[0]], [target[1]])
        target_plot.set_3d_properties([target[2]])

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
        pose_axis.plot(
            [target[0]],
            [target[1]],
            [target[2]],
            "x",
            color="black",
            markersize=8,
            markeredgewidth=2,
            label="IK target",
        )
        draw_frame(pose_axis, tip, end_rotation, frame_scale)
        pose_axis.set_title("Interactive pose (drag to rotate)")
        pose_axis.set_xlabel("X")
        pose_axis.set_ylabel("Y")
        pose_axis.set_zlabel("Z")
        set_equal_axes(pose_axis, pose_bounds)
        pose_axis.legend(fontsize="small")
        if result is None:
            status_text.set_text("Joint-space pose | tip is the IK target")
        else:
            status_text.set_text(ik_status_text(result))

    def update_from_joint_sliders(_value: float) -> None:
        if ui_state["busy"]:
            return
        ui_state["busy"] = True
        try:
            angles = current_joint_angles()
            for textbox, angle in zip(joint_textboxes, angles):
                textbox.set_val(f"{angle:.1f}")
            positions, _ = mechanism.forward_kinematics(angles)
            tip = positions[-1]
            ui_state["target"] = tip.copy()
            for slider, textbox, value in zip(xyz_sliders, xyz_textboxes, tip):
                set_slider_value(slider, float(value))
                textbox.set_val(f"{value:.1f}")
            redraw_pose(angles, tip)
            figure.canvas.draw_idle()
        finally:
            ui_state["busy"] = False

    def update_from_xyz_sliders(_value: float) -> None:
        if ui_state["busy"]:
            return
        ui_state["busy"] = True
        try:
            target = current_target()
            ui_state["target"] = target
            for textbox, value in zip(xyz_textboxes, target):
                textbox.set_val(f"{value:.1f}")
            result = mechanism.inverse_kinematics(target, current_joint_angles(), angle_limits)
            for slider, textbox, angle in zip(joint_sliders, joint_textboxes, result.angles_deg):
                set_slider_value(slider, float(angle))
                textbox.set_val(f"{angle:.1f}")
            redraw_pose(result.angles_deg, target, result)
            figure.canvas.draw_idle()
        finally:
            ui_state["busy"] = False

    def update_from_joint_textbox(index: int, text: str) -> None:
        try:
            value = float(text)
        except ValueError:
            joint_textboxes[index].set_val(f"{joint_sliders[index].val:.1f}")
            return
        lower, upper = angle_limits[index]
        joint_sliders[index].set_val(float(np.clip(value, lower, upper)))

    def update_from_xyz_textbox(index: int, text: str) -> None:
        try:
            value = float(text)
        except ValueError:
            xyz_textboxes[index].set_val(f"{xyz_sliders[index].val:.1f}")
            return
        clipped = float(np.clip(value, xyz_minimum[index], xyz_maximum[index]))
        xyz_sliders[index].set_val(clipped)

    for slider in joint_sliders:
        slider.on_changed(update_from_joint_sliders)
    for slider in xyz_sliders:
        slider.on_changed(update_from_xyz_sliders)
    for index, textbox in enumerate(joint_textboxes):
        textbox.on_submit(lambda text, index=index: update_from_joint_textbox(index, text))
    for index, textbox in enumerate(xyz_textboxes):
        textbox.on_submit(lambda text, index=index: update_from_xyz_textbox(index, text))

    redraw_pose(initial_angles, initial_tip)
    figure.suptitle("Workspace-limited continuous IK: joints stay in limits, no ±180° wrap")
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
            "xaxis": {"title": "X"},
            "yaxis": {"title": "Y"},
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
    angle_limits = DEFAULT_ANGLE_LIMITS
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

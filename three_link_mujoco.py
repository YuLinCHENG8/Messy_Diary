"""MuJoCo visualization of three-link inverse-kinematics iterates."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from three_link_workspace import (
    DEFAULT_ANGLE_LIMITS,
    InverseKinematicsResult,
    JointLimits,
    Mechanism,
    Vector,
)

try:
    import mujoco
except ImportError as error:  # pragma: no cover - optional runtime dependency
    mujoco = None
    _MUJOCO_IMPORT_ERROR = error
else:
    _MUJOCO_IMPORT_ERROR = None


EXAMPLE_ANGLES_DEG = (30.0, 25.0, -40.0)
TARGET_STEP = 4.0


@dataclass
class MujocoArm:
    """MuJoCo model that matches the analytical three-link FK chain."""

    model: "mujoco.MjModel"
    data: "mujoco.MjData"
    tip_site_id: int
    target_body_id: int
    mocap_id: int

    def set_angles_deg(self, angles_deg: tuple[float, float, float]) -> None:
        """Write joint angles and update kinematics without stepping physics."""
        self.data.qpos[:] = np.deg2rad(angles_deg)
        mujoco.mj_forward(self.model, self.data)

    def tip_position(self) -> np.ndarray:
        """Return the end-effector site position in world coordinates."""
        return self.data.site_xpos[self.tip_site_id].copy()

    def set_target(self, target: Vector) -> None:
        """Place the unconstrained IK target. The value is not workspace-clipped."""
        self.data.mocap_pos[self.mocap_id] = np.asarray(target, dtype=float)

    def target_position(self) -> np.ndarray:
        """Return the current unconstrained target position."""
        return self.data.mocap_pos[self.mocap_id].copy()


def require_mujoco() -> None:
    """Raise a clear error if the MuJoCo Python bindings are missing."""
    if mujoco is None:
        raise RuntimeError(
            "MuJoCo is required for this visualization. Install it with: pip install mujoco"
        ) from _MUJOCO_IMPORT_ERROR


def build_mujoco_xml(
    mechanism: Mechanism,
    angle_limits: JointLimits = DEFAULT_ANGLE_LIMITS,
    initial_angles_deg: tuple[float, float, float] = EXAMPLE_ANGLES_DEG,
) -> str:
    """Build a MuJoCo XML string whose zero pose matches the analytical FK."""
    length_1, length_2, length_3 = mechanism.link_lengths
    radius_1 = max(length_1 * 0.035, 2.5)
    radius_2 = max(length_2 * 0.12, 2.2)
    radius_3 = max(length_3 * 0.08, 2.0)
    q1_min, q1_max = angle_limits[0]
    q2_min, q2_max = angle_limits[1]
    q3_min, q3_max = angle_limits[2]
    positions, _ = mechanism.forward_kinematics(initial_angles_deg)
    tip = positions[-1]
    extent = float(sum(mechanism.link_lengths))
    return f"""
<mujoco model="three_link_arm">
  <compiler angle="degree" autolimits="true"/>
  <option gravity="0 0 0" timestep="0.01" integrator="RK4"/>
  <statistic center="0 0 {length_1}" extent="{extent}"/>
  <visual>
    <global azimuth="140" elevation="-20" offwidth="960" offheight="720"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.25 0.25 0.25"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.15 0.2" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 {extent * 2}" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="{extent} {extent} 1" material="grid"
          contype="0" conaffinity="0" rgba="0.2 0.25 0.3 1"/>
    <body name="link1" pos="0 0 0">
      <joint name="q1" type="hinge" axis="0 0 1" range="{q1_min} {q1_max}" limited="true"
             damping="0" armature="0"/>
      <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 {length_1}" size="{radius_1}"
            rgba="0.45 0.55 0.85 1" contype="0" conaffinity="0"/>
      <body name="link2" pos="0 0 {length_1}">
        <joint name="q2" type="hinge" axis="0 1 0" range="{q2_min} {q2_max}" limited="true"
               damping="0" armature="0"/>
        <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0 {length_2}" size="{radius_2}"
              rgba="0.55 0.75 0.45 1" contype="0" conaffinity="0"/>
        <body name="link3" pos="0 0 {length_2}">
          <joint name="q3" type="hinge" axis="1 0 0" range="{q3_min} {q3_max}" limited="true"
                 damping="0" armature="0"/>
          <geom name="link3_geom" type="capsule" fromto="0 0 0 0 0 {length_3}" size="{radius_3}"
                rgba="0.85 0.6 0.3 1" contype="0" conaffinity="0"/>
          <site name="tip" pos="0 0 {length_3}" size="{radius_3 * 0.7}" rgba="0.15 0.85 0.4 1"/>
        </body>
      </body>
    </body>
    <body name="ik_target" mocap="true" pos="{tip[0]} {tip[1]} {tip[2]}">
      <geom name="ik_target_geom" type="sphere" size="{radius_3 * 1.1}"
            rgba="0.95 0.2 0.2 0.85" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def load_mujoco_arm(
    mechanism: Mechanism,
    angle_limits: JointLimits = DEFAULT_ANGLE_LIMITS,
    initial_angles_deg: tuple[float, float, float] = EXAMPLE_ANGLES_DEG,
) -> MujocoArm:
    """Create a MuJoCo arm whose kinematics match the analytical model."""
    require_mujoco()
    model = mujoco.MjModel.from_xml_string(build_mujoco_xml(mechanism, angle_limits, initial_angles_deg))
    data = mujoco.MjData(model)
    arm = MujocoArm(
        model=model,
        data=data,
        tip_site_id=model.site("tip").id,
        target_body_id=model.body("ik_target").id,
        mocap_id=int(np.asarray(model.body("ik_target").mocapid).reshape(-1)[0]),
    )
    arm.set_angles_deg(initial_angles_deg)
    return arm


def apply_one_ik_iteration(
    mechanism: Mechanism,
    arm: MujocoArm,
    angles_deg: tuple[float, float, float],
    angle_limits: JointLimits = DEFAULT_ANGLE_LIMITS,
) -> tuple[tuple[float, float, float], InverseKinematicsResult]:
    """Advance one DLS step toward the current unconstrained MuJoCo target."""
    result = mechanism.inverse_kinematics(
        arm.target_position(),
        angles_deg,
        angle_limits=angle_limits,
        max_iterations=1,
    )
    arm.set_angles_deg(result.angles_deg)
    return result.angles_deg, result


def demo_targets(mechanism: Mechanism) -> list[tuple[str, np.ndarray]]:
    """Return scripted targets, including points outside the reachable workspace."""
    inside, _ = mechanism.forward_kinematics(EXAMPLE_ANGLES_DEG)
    folded, _ = mechanism.forward_kinematics((120.0, -20.0, 80.0))
    near_cut, _ = mechanism.forward_kinematics((175.0, 15.0, 20.0))
    return [
        ("reachable example pose", inside[-1].copy()),
        ("reachable folded pose", folded[-1].copy()),
        ("near q1 = 180 deg", near_cut[-1].copy()),
        ("outside workspace +X+Y", np.array([80.0, 90.0, 170.0])),
        ("outside workspace far", np.array([200.0, 200.0, 200.0])),
        ("outside workspace below", np.array([0.0, 0.0, 40.0])),
        ("back to reachable example", inside[-1].copy()),
    ]


def format_status(angles_deg: tuple[float, float, float], result: object) -> str:
    """Build a one-line IK status string."""
    locked = [f"q{index + 1}" for index, saturated in enumerate(result.saturated) if saturated]
    locked_text = ", ".join(locked) if locked else "none"
    state = "reachable" if result.reachable else "OUT OF WORKSPACE"
    return (
        f"{state} | q=({angles_deg[0]:6.1f}, {angles_deg[1]:6.1f}, {angles_deg[2]:6.1f}) deg"
        f" | err={result.position_error:7.2f} | locked: {locked_text}"
    )


def run_scripted_demo(
    mechanism: Mechanism,
    arm: MujocoArm,
    angle_limits: JointLimits,
    hold_frames: int = 45,
    settle_iterations: int = 80,
) -> list[dict[str, object]]:
    """Drive scripted unconstrained targets and collect IK execution traces."""
    angles = EXAMPLE_ANGLES_DEG
    arm.set_angles_deg(angles)
    history: list[dict[str, object]] = []
    for label, target in demo_targets(mechanism):
        arm.set_target(target)
        path = mechanism.inverse_kinematics_path(
            target,
            angles,
            angle_limits=angle_limits,
            max_iterations=settle_iterations,
        )
        for step in path:
            arm.set_angles_deg(step.angles_deg)
            history.append(
                {
                    "label": label,
                    "target": target.copy(),
                    "angles_deg": step.angles_deg,
                    "position": step.position.copy(),
                    "position_error": step.position_error,
                    "reachable": step.reachable,
                    "saturated": step.saturated,
                    "iteration": step.iterations,
                }
            )
        angles = path[-1].angles_deg
        for _ in range(hold_frames):
            history.append(history[-1])
    return history


def _key_delta(keycode: int) -> np.ndarray | None:
    """Map a GLFW/ASCII keycode to an unconstrained target offset."""
    mapping = {
        65: np.array([-TARGET_STEP, 0.0, 0.0]),  # A
        68: np.array([TARGET_STEP, 0.0, 0.0]),  # D
        87: np.array([0.0, TARGET_STEP, 0.0]),  # W
        83: np.array([0.0, -TARGET_STEP, 0.0]),  # S
        69: np.array([0.0, 0.0, TARGET_STEP]),  # E
        81: np.array([0.0, 0.0, -TARGET_STEP]),  # Q
    }
    return mapping.get(keycode)


def run_interactive_viewer(
    mechanism: Mechanism,
    angle_limits: JointLimits = DEFAULT_ANGLE_LIMITS,
    initial_angles_deg: tuple[float, float, float] = EXAMPLE_ANGLES_DEG,
) -> None:
    """Open MuJoCo and stream one IK iteration per frame toward a free target."""
    require_mujoco()
    from mujoco import viewer

    arm = load_mujoco_arm(mechanism, angle_limits, initial_angles_deg)
    angles = initial_angles_deg
    presets = demo_targets(mechanism)
    command = {"delta": np.zeros(3), "preset_index": None, "reset": False}

    def key_callback(keycode: int) -> None:
        delta = _key_delta(keycode)
        if delta is not None:
            command["delta"] = command["delta"] + delta
            return
        if keycode in (49, 50, 51, 52, 53, 54, 55):
            command["preset_index"] = keycode - 49
            return
        if keycode == 82:  # R
            command["reset"] = True

    print("MuJoCo IK viewer")
    print("  Target has no workspace limits. Joints stay inside angle limits.")
    print("  A/D: X   W/S: Y   Q/E: Z")
    print("  1-7: scripted targets (3-6 are outside the reachable workspace)")
    print("  R: reset to the example pose")
    with viewer.launch_passive(arm.model, arm.data, key_callback=key_callback) as handle:
        while handle.is_running():
            frame_start = time.time()
            if command["reset"]:
                angles = initial_angles_deg
                arm.set_angles_deg(angles)
                positions, _ = mechanism.forward_kinematics(angles)
                arm.set_target(positions[-1])
                command["reset"] = False
            if command["preset_index"] is not None:
                index = command["preset_index"]
                if 0 <= index < len(presets):
                    label, target = presets[index]
                    arm.set_target(target)
                    print(f"Target -> {label}: {np.round(target, 2)}")
                command["preset_index"] = None
            if np.any(command["delta"]):
                arm.set_target(arm.target_position() + command["delta"])
                command["delta"] = np.zeros(3)
            angles, result = apply_one_ik_iteration(mechanism, arm, angles, angle_limits)
            print(format_status(angles, result), end="\r", flush=True)
            handle.sync()
            elapsed = time.time() - frame_start
            time.sleep(max(0.0, 1.0 / 60.0 - elapsed))
    print()


def record_demo_frames(
    mechanism: Mechanism,
    output_path: str,
    angle_limits: JointLimits = DEFAULT_ANGLE_LIMITS,
    width: int = 960,
    height: int = 720,
) -> None:
    """Render a scripted unconstrained-target IK demo to an image file sequence or video."""
    require_mujoco()
    arm = load_mujoco_arm(mechanism, angle_limits, EXAMPLE_ANGLES_DEG)
    history = run_scripted_demo(mechanism, arm, angle_limits, hold_frames=8)
    renderer = mujoco.Renderer(arm.model, height=height, width=width)
    frames: list[np.ndarray] = []
    sampled = history[::2]
    try:
        for frame in sampled:
            arm.set_target(frame["target"])
            arm.set_angles_deg(frame["angles_deg"])
            renderer.update_scene(arm.data)
            frames.append(renderer.render().copy())
    except Exception as error:
        raise RuntimeError(
            "Offscreen recording needs a working MuJoCo GL backend. "
            "On a desktop run without MUJOCO_GL; on a server try MUJOCO_GL=egl."
        ) from error
    output = _save_frames(frames, output_path)
    print(f"Wrote IK execution recording to {output} ({len(frames)} frames)")


def _save_frames(frames: list[np.ndarray], output_path: str) -> str:
    """Save rendered RGB frames as mp4, gif, or a directory of PNG files."""
    from pathlib import Path

    from PIL import Image

    path_lower = output_path.lower()
    images = [Image.fromarray(frame) for frame in frames]
    destination = Path(output_path)
    if path_lower.endswith(".gif"):
        images[0].save(destination, save_all=True, append_images=images[1:], duration=33, loop=0)
        return str(destination)
    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None
    if imageio is not None and path_lower.endswith(".mp4"):
        imageio.mimsave(output_path, frames, fps=30)
        return output_path
    if destination.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        images[-1].save(destination)
        return str(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        image.save(destination / f"ik_{index:04d}.png")
    return str(destination)


def parse_arguments() -> argparse.Namespace:
    """Parse MuJoCo IK viewer options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run scripted reachable and out-of-workspace targets without a window.",
    )
    parser.add_argument(
        "--record",
        type=str,
        help="Render the scripted demo to mp4/gif/png, or to a PNG directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Launch the MuJoCo IK viewer or a headless execution demo."""
    arguments = parse_arguments()
    require_mujoco()
    mechanism = Mechanism()
    if arguments.record:
        record_demo_frames(mechanism, arguments.record)
        return
    if arguments.demo:
        arm = load_mujoco_arm(mechanism)
        history = run_scripted_demo(mechanism, arm, DEFAULT_ANGLE_LIMITS, hold_frames=0)
        last_label = None
        for frame in history:
            if frame["label"] != last_label:
                print(f"\n== {frame['label']}  target={np.round(frame['target'], 2)} ==")
                last_label = frame["label"]
            fake_result = type("Status", (), frame)()
            print(
                f"  iter {frame['iteration']:3d}  {format_status(frame['angles_deg'], fake_result)}"
            )
        print(f"\nRecorded {len(history)} IK execution frames.")
        return
    run_interactive_viewer(mechanism)


if __name__ == "__main__":
    main()

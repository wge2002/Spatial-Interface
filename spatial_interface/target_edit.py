"""Geometry for the opt-in target-pose editor; no simulator or model dependencies."""

import math
import os

MAX_POSITION_DELTA_M = 0.1
MAX_ROTATION_DEG = 90.0


def control_interface(environ=None):
    """Which control surface is exposed.

    'legacy'   the frozen 18 tools.
    'compact'  legacy + edit_target.
    'geometry' compact + the point-cloud reference/proxy tools. Experimental;
               it must never change what 'legacy' or 'compact' expose.
    'fast_geometry'
               a REPLACEMENT surface, not an addition: only fg_look, fg_bind,
               fg_check, fg_run, fg_state and end_episode. The other three modes
               are cumulative, this one is not — the whole point is that no
               per-nudge tool is reachable, so a model cannot fall back to
               driving the target gripper by hand and re-acquire the waypoint
               cost the fast surface exists to avoid. Experimental; it must never
               change what the other three expose.
    'coarse_fine_policy'
               also a REPLACEMENT surface: only cf_look, cf_policy, cf_state and
               end_episode. Unlike 'fast_geometry' it carries no named skill — the
               model authors the whole policy as a short program of poses, gripper
               changes and dwells, and gets a fresh wrist-depth observation after
               each one. Experimental; it must never change what the others expose.
    """
    mode = (os.environ if environ is None else environ).get("VIA_CONTROL_INTERFACE", "legacy")
    if mode not in ("legacy", "compact", "geometry", "fast_geometry",
                    "coarse_fine_policy", "direct_geometry"):
        raise ValueError("VIA_CONTROL_INTERFACE must be 'legacy', 'compact', "
                         "'geometry', 'fast_geometry', 'coarse_fine_policy' or 'direct_geometry'")
    return mode


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite numbers")
    try:
        valid = all(type(x) in (int, float) and math.isfinite(x) for x in value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must contain three finite numbers (not booleans)")
    return [float(x) for x in value]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _unit(v, name):
    length = math.sqrt(_dot(v, v))
    if abs(length - 1) > 0.001:
        raise ValueError(f"{name} must be a unit vector (tolerance 0.001)")
    return [x / length for x in v]


def validate_edit(arguments):
    """Validate the entire declaration before interacting with the browser."""
    fields = {"frame", "position", "delta_position", "approach", "opening"}
    if not isinstance(arguments, dict) or set(arguments) - fields:
        raise ValueError("Only frame, position, delta_position, approach and opening are allowed")
    if arguments.get("frame") != "robot":
        raise ValueError("frame must be 'robot'")
    if "position" in arguments and "delta_position" in arguments:
        raise ValueError("Specify position or delta_position, not both")
    if ("approach" in arguments) != ("opening" in arguments):
        raise ValueError("Supply approach and opening together")
    if not set(arguments) - {"frame"}:
        raise ValueError("Supply a position, delta_position or orientation")
    edit = {k: _vector(v, k) for k, v in arguments.items() if k != "frame"}
    if "delta_position" in edit and any(abs(x) > MAX_POSITION_DELTA_M
                                       for x in edit["delta_position"]):
        raise ValueError(f"Each position delta must be within +/-{MAX_POSITION_DELTA_M} m")
    if "approach" in edit:
        a = _unit(edit["approach"], "approach")
        o = _unit(edit["opening"], "opening")
        dot = _dot(a, o)
        if abs(dot) > 0.001:
            raise ValueError("approach and opening must be orthogonal (tolerance 0.001)")
        # Correct only accepted floating-point/rounding error, retaining approach.
        o = [x - dot * y for x, y in zip(o, a)]
        edit["approach"], edit["opening"] = a, _unit(o, "opening")
    return edit


def prepare_edit(edit, pose, ui_z_offset):
    """Resolve a validated declaration against a target snapshot, with step limits."""
    def xyz(value):
        return [value[k] for k in ("x", "y", "z")]

    current = xyz(pose["robot_position"])
    position = edit.get("position", [x + d for x, d in
                                     zip(current, edit.get("delta_position", [0, 0, 0]))])
    if any(abs(x - y) > MAX_POSITION_DELTA_M + 1e-10 for x, y in zip(position, current)):
        raise ValueError(f"Each position delta must be within +/-{MAX_POSITION_DELTA_M} m")
    old_a, old_o = xyz(pose["robot_approach"]), xyz(pose["robot_opening"])
    a, o = edit.get("approach", old_a), edit.get("opening", old_o)
    trace = _dot(a, old_a) + _dot(o, old_o) + _dot(_cross(a, o), _cross(old_a, old_o))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))
    if angle > MAX_ROTATION_DEG + 1e-7:
        raise ValueError(f"Orientation change must be at most {MAX_ROTATION_DEG:g} degrees")

    def to_ui(v):
        return [v[0], v[2], -v[1]]

    # Local -Y is the approach axis, local +Z is the opening axis.
    y, z = to_ui([-x for x in a]), to_ui(o)
    return {
        "expected_position": xyz(pose["ui_position"]),
        "expected_approach": old_a,
        "expected_opening": old_o,
        "expected_gripper_open": pose["gripper_open"],
        "position": [position[0] * 10, (position[2] - ui_z_offset) * 10, -position[1] * 10],
        "basis": [_cross(y, z), y, z],
        "robot_position": position,
        "robot_approach": a,
        "robot_opening": o,
    }


# Compare-and-set in one browser turn. No keypress, record button, websocket send
# or gripper mesh swap occurs here. Rejection happens before either pose mutation.
APPLY_EDIT_JS = """edit => {
    if (typeof selectedObject === 'undefined' || !selectedObject)
        return {ok: false, reason: 'Target unavailable'};
    const obj = selectedObject;
    const close = (a, b) => a.every((x, i) => Math.abs(x - b[i]) < 1e-7);
    const dir = v => { v.applyQuaternion(obj.quaternion); return [v.x, -v.z, v.y]; };
    const idx = controlPoints.indexOf(obj);
    const url = idx >= 0 ? meshUrls[idx] : null;
    const open = url == null ? null : !url.includes('_closed');
    if (!close(obj.position.toArray(), edit.expected_position) ||
        !close(dir(new THREE.Vector3(0, -1, 0)), edit.expected_approach) ||
        !close(dir(new THREE.Vector3(0, 0, 1)), edit.expected_opening) ||
        open !== edit.expected_gripper_open)
        return {ok: false, reason: 'Target changed during editing; observe and retry'};
    if (!isTranslationMode || transformControl)
        return {ok: false, reason: 'Target editing mode changed; observe and retry'};
    const basis = edit.basis.map(v => new THREE.Vector3().fromArray(v));
    const q = new THREE.Quaternion().setFromRotationMatrix(
        new THREE.Matrix4().makeBasis(...basis)).normalize();
    obj.position.fromArray(edit.position);
    obj.quaternion.copy(q);
    updateCurvePath();
    return {ok: true};
}"""

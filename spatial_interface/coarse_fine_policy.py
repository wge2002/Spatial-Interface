"""Pure geometry and validation for VIA_CONTROL_INTERFACE=coarse_fine_policy.

The model authors the whole program: which region is a proxy, which frame a pose
is expressed in, the continuous angle about the proxy's up axis, the offset, when
the gripper opens or closes, and how long to dwell. This module supplies only
arithmetic — frame resolution, axis construction, rotation splitting, fail-closed
validation of the entire program before any motion — and deliberately supplies no
menu of named skills, no candidate ranking, no recovery and no retry.

No simulator, browser or MCP dependency, so every rule here is testable directly.
``spatial_interface/coarse_fine_tools.py`` is the browser and execution edge.
"""

from __future__ import annotations

import math

try:
    from . import fast_geometry as fg
except ImportError:  # direct import
    import fast_geometry as fg


class Rejected(Exception):
    """A program the executor refuses to run. Raised before any motion."""


MAX_STEPS = 12          # steps in one cf_policy program
MAX_DWELL_STEPS = 400   # sim steps one dwell may request
MIN_DWELL_STEPS = 1
MAX_OFFSET_M = 0.5      # a single offset longer than this is a typo, not a plan
DEFAULT_TOLERANCE_M = 0.015
MAX_TOLERANCE_M = 0.10
MIN_TOLERANCE_M = 0.002
# A pose step commands an orientation too, so arrival has to be judged on both.
# 10 deg is loose enough for the controller's own settling and tight enough that a
# gripper facing a materially different way is not called arrived.
DEFAULT_ORIENTATION_TOLERANCE_DEG = 10.0
MAX_ORIENTATION_TOLERANCE_DEG = 45.0
MIN_ORIENTATION_TOLERANCE_DEG = 1.0
DEFAULT_OBSERVE_RADIUS_M = 0.08
MAX_OBSERVE_RADIUS_M = 0.30
DEFAULT_WAYPOINTS = 8
MAX_WAYPOINTS = 24
DEFAULT_SECONDS = 240.0
MAX_SECONDS = 540.0

# One target edit is capped at 90 deg by target_edit.MAX_ROTATION_DEG. The step
# here is smaller on purpose — see `orientation_steps`, where the margin is the
# fix for the defect the fast splitter has.
MAX_STEP_ROTATION_DEG = 60.0

# Below this the relative rotation is identity to any precision the controller can
# act on, and the recovered axis would be float noise. See `rotation_between`.
NEAR_IDENTITY_RAD = 1e-7


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Rejected(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise Rejected(f"{name} must be a finite number")
    return value


def _vec3(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise Rejected(f"{name} must be three finite numbers")
    return [_finite(v, f"{name}[{i}]") for i, v in enumerate(value)]


def _unit(v):
    """Normalize, or None. Deliberately does NOT raise.

    The geometry helpers below are called during execution against MEASURED axes,
    where an unusable vector must become a refusal carrying its evidence — not an
    exception from inside an arithmetic helper. Validation of model-supplied input
    raises `Rejected`; that happens in the validate_* functions, before any motion.
    """
    if v is None:
        return None
    try:
        vals = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    if len(vals) != 3 or not all(math.isfinite(x) for x in vals):
        return None
    length = math.sqrt(sum(x * x for x in vals))
    if length < 1e-9:
        return None
    return [x / length for x in vals]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(v):
    return math.sqrt(_dot(v, v))


def _matvec(R, v):
    return [_dot(row, v) for row in R]


def basis_from_axes(approach, opening):
    """Right-handed orthonormal (approach, opening, third), or None.

    Same construction ``fast_geometry`` uses, re-implemented here so this module
    does not depend on that one's private helper: normalize, remove the residual
    non-orthogonality by projection rather than assuming it away, complete with
    the cross product.
    """
    a, o = _unit(approach), _unit(opening)
    if a is None or o is None:
        return None
    dot = _dot(a, o)
    if abs(abs(dot) - 1.0) < 1e-6:
        return None  # parallel pair carries no frame
    o = [x - dot * y for x, y in zip(o, a)]
    if _norm(o) < 1e-9:
        return None
    o = _unit(o)
    if o is None:
        return None
    return {"approach": a, "opening": o, "third": _cross(a, o)}


def axes_from_angles(up, azimuth_deg, tilt_deg):
    """Gripper axes from a continuous angle pair about a proxy's up axis.

    ``tilt_deg`` 0 means approaching straight along -up (down onto a horizontal
    object); larger tilt leans the approach away from -up by that many degrees.
    ``azimuth_deg`` turns the whole thing about up, measured from a deterministic
    reference: world +x projected off up, or +y where up is nearly +/-x. Opening is
    the tangential direction, which is perpendicular to the approach for every
    (azimuth, tilt) rather than only for tilt 0.

    Continuous by construction — no snapping to a candidate list. Returns None if
    ``up`` is unusable.
    """
    u = _unit(up)
    if u is None:
        return None
    reference = [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0]
    e1 = [x - _dot(reference, u) * y for x, y in zip(reference, u)]
    if _norm(e1) < 1e-9:
        return None
    e1 = _unit(e1)
    if e1 is None:
        return None
    e2 = _cross(u, e1)
    az, tilt = math.radians(azimuth_deg), math.radians(tilt_deg)
    radial = [math.cos(az) * e1[i] + math.sin(az) * e2[i] for i in range(3)]
    approach = [-math.cos(tilt) * u[i] + math.sin(tilt) * radial[i] for i in range(3)]
    opening = _cross(u, radial)
    return basis_from_axes(approach, opening)


def rotation_between(current, target):
    """(axis, angle_deg) of the rotation carrying ``current`` axes onto ``target``.

    Returns None if either basis is unusable. angle_deg is in [0, 180].
    """
    now, want = basis_from_axes(*current), basis_from_axes(*target)
    if now is None or want is None:
        return None
    keys = ("approach", "opening", "third")
    # R = B_target^T @ B_current, with basis rows (approach, opening, third):
    # it carries each current axis onto the matching target axis.
    R = [[sum(want[k][i] * now[k][j] for k in keys) for j in range(3)]
         for i in range(3)]
    trace = R[0][0] + R[1][1] + R[2][2]
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
    # acos is ill-conditioned near identity: a 1e-16 error in the trace becomes
    # ~1e-8 rad of angle, and the antisymmetric part it would be divided by is of
    # the same order, so the recovered axis is noise. Below this the rotation is
    # identity to any precision the robot can act on, and the axis is arbitrary.
    if angle < NEAR_IDENTITY_RAD:
        return [0.0, 0.0, 1.0], 0.0
    if math.pi - angle < 1e-6:
        # A half turn: (R - I) is singular in the axis direction, so take the axis
        # from the largest diagonal of (R + I) instead of the antisymmetric part,
        # which vanishes here.
        best = max(range(3), key=lambda i: R[i][i])
        column = [R[i][best] + (1.0 if i == best else 0.0) for i in range(3)]
        axis = _unit(column)
        return None if axis is None else (axis, 180.0)
    sin = math.sin(angle)
    axis = _unit([(R[2][1] - R[1][2]) / (2 * sin), (R[0][2] - R[2][0]) / (2 * sin),
                  (R[1][0] - R[0][1]) / (2 * sin)])
    return None if axis is None else (axis, math.degrees(angle))


def rotate_about(axis, angle_deg, vector):
    """Rodrigues rotation of one vector about a unit axis."""
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    dot = _dot(axis, vector)
    cross = _cross(axis, vector)
    return [vector[i] * c + cross[i] * s + axis[i] * dot * (1 - c) for i in range(3)]


def orientation_steps(current, target, *, max_deg=MAX_STEP_ROTATION_DEG):
    """Virtual orientation edits from ``current`` axes to ``target`` axes.

    Fixes the defect in ``fast_geometry.orientation_steps``, which interpolates
    approach and opening independently and re-orthogonalizes afterwards. That
    re-orthogonalization moves the intermediate pose off the interpolation, so the
    angle actually measured between consecutive emitted steps is NOT the requested
    fraction of the total: for a turn near 180 deg the two interpolated vectors
    nearly cancel, the projection swings the recovered basis, and a step can exceed
    ``target_edit.MAX_ROTATION_DEG`` and be refused mid-program.

    Here the relative rotation is taken once as an axis and an angle, and each step
    is an exact rotation of the ORIGINAL basis by ``k/count`` of that angle. Every
    consecutive pair is then exactly ``angle/count`` apart by construction — no
    projection is applied afterwards, so nothing can move the step off its share.
    ``max_deg`` keeps a margin under the editor's 90 deg cap.

    Returns [] when the axes already match and None when either basis is unusable.
    """
    resolved = rotation_between(current, target)
    if resolved is None:
        return None
    axis, angle = resolved
    if angle <= 1e-6:
        return []
    now = basis_from_axes(*current)
    want = basis_from_axes(*target)
    count = max(1, int(math.ceil(angle / max_deg)))
    out = []
    for k in range(1, count + 1):
        if k == count:
            # End exactly on the requested axes rather than on a rotation of the
            # measured ones, which would differ by the measurement's own
            # non-orthogonality.
            out.append({"approach": list(want["approach"]),
                        "opening": list(want["opening"])})
            break
        f = angle * k / count
        out.append({"approach": rotate_about(axis, f, now["approach"]),
                    "opening": rotate_about(axis, f, now["opening"])})
    return out


# ── program validation ───────────────────────────────────────────────────────
#
# Fail-closed and complete: the WHOLE program is validated before the first
# waypoint, so a program whose fourth step is malformed does not move the robot
# three times first. Nothing here is defaulted silently except where a default is
# documented in the tool schema.

FRAME_ROBOT = "robot"
FRAME_GRIPPER = "gripper"
PROXY_PREFIX = "proxy:"


def validate_frame(value):
    """'robot', 'gripper' or 'proxy:<name>'. Returns (kind, name)."""
    if not isinstance(value, str) or not value:
        raise Rejected("frame must be 'robot', 'gripper' or 'proxy:<name>'")
    if value == FRAME_ROBOT:
        return FRAME_ROBOT, None
    if value == FRAME_GRIPPER:
        return FRAME_GRIPPER, None
    if value.startswith(PROXY_PREFIX):
        name = value[len(PROXY_PREFIX):]
        if not name:
            raise Rejected("proxy frame needs a name: 'proxy:<name>'")
        return PROXY_PREFIX, name
    raise Rejected(f"unknown frame {value!r}; use 'robot', 'gripper' or 'proxy:<name>'")


def validate_pose_step(spec):
    """One pose step. Position XOR offset; axes XOR angles; both optional.

    `offset` means "from where the gripper is now" in the robot and gripper frames,
    and "from the proxy's centre" in a proxy frame. `position` is always absolute
    within the named frame.
    """
    if not isinstance(spec, dict):
        raise Rejected("a pose step must be an object")
    unknown = set(spec) - {"frame", "position", "offset", "approach", "opening",
                           "azimuth_deg", "tilt_deg", "tolerance_m",
                           "orientation_tolerance_deg"}
    if unknown:
        raise Rejected(f"unknown pose fields: {sorted(unknown)}")
    kind, proxy = validate_frame(spec.get("frame", FRAME_ROBOT))
    has_position, has_offset = "position" in spec, "offset" in spec
    if has_position and has_offset:
        raise Rejected("a pose step takes position OR offset, not both")
    if not has_position and not has_offset:
        raise Rejected("a pose step needs position or offset")
    if has_position and kind == FRAME_GRIPPER:
        raise Rejected("frame 'gripper' has no absolute position; use offset")
    out = {"kind": "pose", "frame": kind, "proxy": proxy}
    if has_position:
        out["position"] = _vec3(spec["position"], "position")
    else:
        offset = _vec3(spec["offset"], "offset")
        if _norm(offset) > MAX_OFFSET_M:
            raise Rejected(f"offset length {_norm(offset):.3f} m exceeds the "
                           f"{MAX_OFFSET_M:g} m limit for one step")
        out["offset"] = offset
    has_axes = "approach" in spec or "opening" in spec
    has_angles = "azimuth_deg" in spec or "tilt_deg" in spec
    if has_axes and has_angles:
        raise Rejected("give explicit approach/opening OR azimuth_deg/tilt_deg, "
                       "not both")
    if has_axes:
        if "approach" not in spec or "opening" not in spec:
            raise Rejected("approach and opening must be given together")
        basis = basis_from_axes(_vec3(spec["approach"], "approach"),
                               _vec3(spec["opening"], "opening"))
        if basis is None:
            raise Rejected("approach and opening must be a usable non-parallel pair")
        out["approach"] = basis["approach"]
        out["opening"] = basis["opening"]
        out["orientation_source"] = "explicit_axes"
    elif has_angles:
        if kind != PROXY_PREFIX:
            raise Rejected("azimuth_deg/tilt_deg are measured about a proxy's up "
                           "axis, so the step's frame must be 'proxy:<name>'")
        azimuth = _finite(spec.get("azimuth_deg", 0.0), "azimuth_deg")
        tilt = _finite(spec.get("tilt_deg", 0.0), "tilt_deg")
        if not -180.0 <= azimuth <= 360.0:
            raise Rejected("azimuth_deg must be within [-180, 360]")
        if not -180.0 <= tilt <= 180.0:
            raise Rejected("tilt_deg must be within [-180, 180]")
        out["azimuth_deg"] = azimuth
        out["tilt_deg"] = tilt
        out["orientation_source"] = "angles_about_proxy_up"
    else:
        out["orientation_source"] = "unchanged"
    tolerance = _finite(spec.get("tolerance_m", DEFAULT_TOLERANCE_M), "tolerance_m")
    if not MIN_TOLERANCE_M <= tolerance <= MAX_TOLERANCE_M:
        raise Rejected(f"tolerance_m must be within "
                       f"[{MIN_TOLERANCE_M:g}, {MAX_TOLERANCE_M:g}]")
    out["tolerance_m"] = tolerance
    angle_tolerance = _finite(
        spec.get("orientation_tolerance_deg", DEFAULT_ORIENTATION_TOLERANCE_DEG),
        "orientation_tolerance_deg")
    if not (MIN_ORIENTATION_TOLERANCE_DEG <= angle_tolerance
            <= MAX_ORIENTATION_TOLERANCE_DEG):
        raise Rejected(f"orientation_tolerance_deg must be within "
                       f"[{MIN_ORIENTATION_TOLERANCE_DEG:g}, "
                       f"{MAX_ORIENTATION_TOLERANCE_DEG:g}]")
    out["orientation_tolerance_deg"] = angle_tolerance
    return out


def validate_step(spec):
    if not isinstance(spec, dict) or len(spec) != 1:
        raise Rejected("each step is an object with exactly one of "
                       "pose / gripper / dwell / observe")
    (key, value), = spec.items()
    if key == "pose":
        return validate_pose_step(value)
    if key == "gripper":
        if value not in ("open", "close"):
            raise Rejected("gripper must be 'open' or 'close'")
        return {"kind": "gripper", "action": value}
    if key == "dwell":
        if not isinstance(value, dict) or set(value) - {"sim_steps"}:
            raise Rejected("dwell takes {'sim_steps': int}")
        steps = value.get("sim_steps")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise Rejected("dwell.sim_steps must be an integer")
        if not MIN_DWELL_STEPS <= steps <= MAX_DWELL_STEPS:
            raise Rejected(f"dwell.sim_steps must be within "
                           f"[{MIN_DWELL_STEPS}, {MAX_DWELL_STEPS}]")
        return {"kind": "dwell", "sim_steps": steps}
    if key == "observe":
        if not isinstance(value, dict) or set(value) - {"at", "radius_m"}:
            raise Rejected("observe takes {'at': 'gripper', 'radius_m': number}")
        at = value.get("at", FRAME_GRIPPER)
        if at != FRAME_GRIPPER:
            raise Rejected("observe.at must be 'gripper'")
        radius = _finite(value.get("radius_m", DEFAULT_OBSERVE_RADIUS_M), "radius_m")
        if not 0.005 <= radius <= MAX_OBSERVE_RADIUS_M:
            raise Rejected(f"observe.radius_m must be within "
                           f"[0.005, {MAX_OBSERVE_RADIUS_M:g}]")
        return {"kind": "observe", "at": at, "radius_m": radius}
    raise Rejected(f"unknown step {key!r}; use pose / gripper / dwell / observe")


def validate_bind(spec):
    """The optional inline bind. Region shape is checked by fast_geometry itself."""
    if not isinstance(spec, dict):
        raise Rejected("bind must be an object")
    unknown = set(spec) - {"surface", "name", "region", "shape"}
    if unknown:
        raise Rejected(f"unknown bind fields: {sorted(unknown)}")
    request = fg.validate_bind({
        "surface": spec.get("surface", "canvas"),
        "objects": [{"name": spec.get("name", ""),
                     "region": spec.get("region"),
                     "shape": spec.get("shape", "blob")}],
    })
    return request


def validate_budget(spec):
    if spec is None:
        spec = {}
    if not isinstance(spec, dict) or set(spec) - {"waypoints", "seconds"}:
        raise Rejected("budget takes {'waypoints': int, 'seconds': number}")
    waypoints = spec.get("waypoints", DEFAULT_WAYPOINTS)
    if isinstance(waypoints, bool) or not isinstance(waypoints, int):
        raise Rejected("budget.waypoints must be an integer")
    if not 1 <= waypoints <= MAX_WAYPOINTS:
        raise Rejected(f"budget.waypoints must be within [1, {MAX_WAYPOINTS}]")
    seconds = _finite(spec.get("seconds", DEFAULT_SECONDS), "budget.seconds")
    if not 10.0 <= seconds <= MAX_SECONDS:
        raise Rejected(f"budget.seconds must be within [10, {MAX_SECONDS:g}]")
    return {"waypoints": waypoints, "seconds": seconds}


def validate_program(arguments):
    """The whole cf_policy call. Raises Rejected before anything is executed."""
    if not isinstance(arguments, dict):
        raise Rejected("arguments must be an object")
    unknown = set(arguments) - {"bind", "steps", "budget"}
    if unknown:
        raise Rejected(f"unknown fields: {sorted(unknown)}")
    steps = arguments.get("steps")
    if not isinstance(steps, list) or not steps:
        raise Rejected("steps must be a non-empty array")
    if len(steps) > MAX_STEPS:
        raise Rejected(f"a program takes at most {MAX_STEPS} steps")
    program = {
        "bind": validate_bind(arguments["bind"]) if arguments.get("bind") else None,
        "steps": [validate_step(s) for s in steps],
        "budget": validate_budget(arguments.get("budget")),
    }
    physical = sum(1 for s in program["steps"]
                   if s["kind"] in ("pose", "gripper", "dwell"))
    if physical > program["budget"]["waypoints"]:
        raise Rejected(f"the program has {physical} physical steps but the budget "
                       f"allows {program['budget']['waypoints']} waypoints; nothing "
                       f"was executed")
    program["physical_steps"] = physical
    return program


def step_angles_deg(current, steps):
    """Measured angle of each emitted step from its predecessor.

    The same trace formula ``target_edit.prepare_edit`` applies, so a test can
    assert directly that no emitted step would be refused.
    """
    angles = []
    at = current
    for step in steps:
        resolved = rotation_between(at, (step["approach"], step["opening"]))
        angles.append(None if resolved is None else resolved[1])
        at = (step["approach"], step["opening"])
    return angles


# ── frame resolution ─────────────────────────────────────────────────────────

def resolve_pose(step, *, measured, proxy=None):
    """The absolute robot-frame target this pose step asks for.

    ``measured`` is ``geometry_ref.measured_end_effector`` for the frame the step
    is being solved against — the robot's own telemetry, never the virtual target.
    ``proxy`` is the bound card when the step names one.

    Returns {'status': 'ok', 'position', 'approach', 'opening', ...} or
    {'status': 'refused', 'reason': ...}. Refuses rather than substituting a
    default whenever the evidence a step names is missing: that is the whole
    fail-closed contract, and a silently substituted frame origin would move the
    robot to a pose the model did not ask for.
    """
    if not measured or measured.get("status") != "ok":
        return {"status": "refused", "reason": "no_measured_end_effector",
                "detail": (measured or {}).get("reason")}
    at = [measured["fingertip_position"][k] for k in "xyz"]
    now_a = measured.get("approach")
    now_o = measured.get("opening")
    current_axes = ([now_a[k] for k in "xyz"] if now_a else None,
                    [now_o[k] for k in "xyz"] if now_o else None)

    origin = up = None
    if step["frame"] == PROXY_PREFIX:
        if not proxy:
            return {"status": "refused", "reason": "proxy_not_bound",
                    "detail": step["proxy"]}
        if not proxy.get("valid"):
            return {"status": "refused", "reason": "proxy_not_valid",
                    "detail": proxy.get("reasons") or proxy.get("invalid_reason")}
        center = proxy.get("center")
        if not center:
            return {"status": "refused", "reason": "proxy_has_no_center"}
        origin = [center[k] for k in "xyz"] if isinstance(center, dict) else list(center)
        up_raw = proxy.get("up")
        if up_raw:
            up = [up_raw[k] for k in "xyz"] if isinstance(up_raw, dict) else list(up_raw)
    elif step["frame"] == FRAME_GRIPPER:
        origin = at
    else:
        origin = [0.0, 0.0, 0.0]

    if "position" in step:
        position = [origin[i] + step["position"][i] for i in range(3)]
    elif step["frame"] == FRAME_ROBOT:
        # A robot-frame offset is a WORLD-AXES displacement from where the gripper
        # is now. Measuring it from the origin instead would make it a second
        # spelling of `position` and leave no way at all to say "5 cm down from
        # here" without first reading a pose and doing the addition by hand.
        position = [at[i] + step["offset"][i] for i in range(3)]
    elif step["frame"] == FRAME_GRIPPER:
        # A gripper-frame offset is expressed in the gripper's OWN axes, which is
        # what makes "back off 4 cm along the approach" one number rather than a
        # world-frame vector the model has to recompute for every orientation.
        basis = basis_from_axes(*current_axes) if all(current_axes) else None
        if basis is None:
            return {"status": "refused", "reason": "no_measured_gripper_axes",
                    "detail": "a gripper-frame offset needs the measured axes"}
        d = step["offset"]
        position = [at[i] + d[0] * basis["approach"][i] + d[1] * basis["opening"][i]
                    + d[2] * basis["third"][i] for i in range(3)]
    else:
        position = [origin[i] + step["offset"][i] for i in range(3)]

    source = step["orientation_source"]
    if source == "explicit_axes":
        approach, opening = step["approach"], step["opening"]
    elif source == "angles_about_proxy_up":
        if up is None:
            return {"status": "refused", "reason": "proxy_has_no_up_axis",
                    "detail": ("azimuth/tilt are measured about the proxy's up "
                               "axis, which this card does not carry")}
        basis = axes_from_angles(up, step["azimuth_deg"], step["tilt_deg"])
        if basis is None:
            return {"status": "refused", "reason": "angles_unusable",
                    "detail": "the proxy's up axis is not a usable direction"}
        approach, opening = basis["approach"], basis["opening"]
    else:
        if not all(current_axes):
            return {"status": "refused", "reason": "no_measured_gripper_axes",
                    "detail": "the step leaves orientation unchanged but the "
                              "current axes are not readable"}
        basis = basis_from_axes(*current_axes)
        if basis is None:
            return {"status": "refused", "reason": "no_measured_gripper_axes"}
        approach, opening = basis["approach"], basis["opening"]

    return {"status": "ok",
            "position": [round(v, 6) for v in position],
            "approach": [round(v, 9) for v in approach],
            "opening": [round(v, 9) for v in opening],
            "frame": step["frame"] + (f":{step['proxy']}" if step["proxy"] else ""),
            "origin": [round(v, 6) for v in origin],
            "orientation_source": source,
            "displacement_m": round(_norm([position[i] - at[i] for i in range(3)]), 5),
            "tolerance_m": step["tolerance_m"],
            "orientation_tolerance_deg": step["orientation_tolerance_deg"]}


def pose_error(target, measured):
    """Position and orientation error of the measured pose against the target.

    ``within_tolerance`` is True only when BOTH the position and the orientation
    were verified and both met their tolerance. It is None — not True — when
    orientation could not be measured at all, and the caller must treat that as
    "arrival is unknown" rather than as arrival.

    A pose command carries an orientation as well as a position, so a reading that
    checked only the position could report a fingertip exactly on target while the
    gripper faced the wrong way — the executor would then run the next step, which
    may be a close, on a pose that never arrived. Whichever part failed is named in
    ``failed``, so an abort message can say which one.
    """
    if not measured or measured.get("status") != "ok":
        return {"status": "unknown", "reason": "no_measured_end_effector",
                "within_tolerance": None}
    at = [measured["fingertip_position"][k] for k in "xyz"]
    delta = [at[i] - target["position"][i] for i in range(3)]
    tolerance = target.get("tolerance_m", DEFAULT_TOLERANCE_M)
    angle_tolerance = target.get("orientation_tolerance_deg",
                                 DEFAULT_ORIENTATION_TOLERANCE_DEG)
    out = {"status": "ok",
           "position_error_m": round(_norm(delta), 5),
           "position_delta_m": [round(v, 5) for v in delta],
           "tolerance_m": tolerance,
           "orientation_tolerance_deg": angle_tolerance,
           "position_within_tolerance": _norm(delta) <= tolerance,
           "orientation_within_tolerance": None,
           "within_tolerance": None,
           "failed": []}
    now_a, now_o = measured.get("approach"), measured.get("opening")
    resolved = None
    if now_a and now_o:
        resolved = rotation_between(
            ([now_a[k] for k in "xyz"], [now_o[k] for k in "xyz"]),
            (target["approach"], target["opening"]))
    if resolved is None:
        out["orientation_error_deg"] = None
        out["orientation_reason"] = ("the measured gripper axes are not a usable "
                                     "pair, so the orientation could not be checked")
    else:
        out["orientation_error_deg"] = round(resolved[1], 3)
        out["orientation_within_tolerance"] = resolved[1] <= angle_tolerance
    if not out["position_within_tolerance"]:
        out["failed"].append("position")
    if out["orientation_within_tolerance"] is False:
        out["failed"].append("orientation")
    if out["orientation_within_tolerance"] is None:
        out["within_tolerance"] = None  # unknown: unverifiable, never arrival
        out["reason"] = "orientation_unverifiable"
    else:
        out["within_tolerance"] = not out["failed"]
    return out

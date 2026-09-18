"""Pure projection from a logged dg tool payload to the model-facing receipt.

Nothing here senses, actuates, retries, classifies a stop cause or judges a
grasp. Every function is a pure function of a payload the caller already owns:
the source payload is never mutated, so the raw trace on disk stays complete and
this module can be replayed offline against historical logs.

Three things the receipt deliberately does NOT do:

* It never infers *why* a command stopped. It reports the measured endpoints, the
  residuals against the committed target and the tolerances that were in force.
  Collision, contact, mechanical saturation and grasp are not observable through
  this channel and are not guessed.
* It never upgrades an uncertain backend result to a completed one, and never
  presents a hypothetical measured start for a command that was not submitted.
* It never substitutes a requested goal for a measurement. A missing measurement
  is reported unknown, not as a zero and not as an older reading.

Arrival numbers come from `geometry_workspace`'s shared helpers, which are the
same code the Executor gate runs, so a displayed residual cannot disagree with
the decision that stopped the program. Displayed values are rounded; the gate and
the raw log keep full precision.

The projection is asymmetric on purpose: a command that completed says so in one
short line, while the command that failed, the command whose result is uncertain
and the final executed command carry their full targets, endpoints and residuals.
Every command's full diagnostics stay available through `dg_state` and in the raw
log; this module only chooses what the routine reply spends characters on.
"""
from __future__ import annotations

import math

try:
    from . import geometry_workspace as gw
except ImportError:  # pragma: no cover - direct-script import path
    import geometry_workspace as gw

POSITION_DECIMALS = 5
ANGLE_DECIMALS = 2

# Model-facing name for the robot's configured control point. The raw logs and
# the internal compile path keep the legacy `fingertip_position` key; this is a
# rename of the public copy only, and it carries no extra offset.
EEF_KEY = "eef_position"
LEGACY_EEF_KEY = "fingertip_position"

# A measured pose is only used as a diagnostic anchor when the reading says it is
# an actual robot measurement of the exact kind the gate consumes. Anything else
# -- an unknown reading, a rounded display copy, an unfamiliar source -- is
# reported unknown rather than silently trusted.
MEASUREMENT_SOURCES = ("robot_proprioception",)

# Reported instead of a zero when the measurement that would anchor a start or
# end simply does not exist. A missing measurement is not a measurement of zero.
UNKNOWN_START = "unknown_no_verified_start_measurement"
UNKNOWN_END = "unknown_no_verified_end_measurement"
UNKNOWN_WIDTH = "unknown"

# Command statuses. `not_submitted` is the ONLY one that means nothing was sent.
SUBMITTED_STATUSES = ("submitted", "completed", "stopped", "uncertain")
NOT_SUBMITTED = "not_submitted"
UNCERTAIN = "uncertain"
NO_ARRIVAL_CHECK = "submitted_no_recorded_arrival_check"

OBSERVATION_LEGEND = ("vectors [x,y,z] robot metres; measured_eef is the robot's "
                      "measured control point, no extra offset; local_depth "
                      "summarises returns near its own sample point -- robot, object "
                      "and background alike -- and identifies nothing; jaw width and "
                      "commanded state do not prove a grasp")

# Adds the command-diagnostic half. Emitted once per receipt instead of repeating
# the same caveats on every command entry.
COMMAND_LEGEND = (OBSERVATION_LEGEND +
                  "; before/after are measured residuals to the committed absolute "
                  "target, judged against that command's own tolerances; a start is "
                  "the measured pre-program state or the previous actual "
                  "acknowledgement, never a requested goal; completed commands are "
                  "summarised and dg_state has every command in full; no contact, "
                  "collision or grasp is inferred")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _round(value, decimals):
    return round(float(value), decimals) if _finite(value) else None


def _xyz(value, decimals=POSITION_DECIMALS):
    """Accept the dict or list form used across the payload; round for display.

    Emitted as a plain `[x, y, z]` list: the same three numbers as the `{"x":..}`
    form the raw payload uses, at roughly half the characters. The axis order is
    stated once in the receipt's own legend rather than repeated per vector.
    """
    if isinstance(value, dict):
        value = [value.get(k) for k in "xyz"]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    out = [_round(v, decimals) for v in value]
    return None if any(v is None for v in out) else out


def _translation(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) != 4:
        return None
    try:
        return [float(pose[i][3]) for i in range(3)]
    except (TypeError, ValueError, IndexError):
        return None


def _axes(pose):
    """Columns 2 and 1 are approach (+Z) and opening (+Y), matching direct_control."""
    try:
        return ([float(pose[i][2]) for i in range(3)],
                [float(pose[i][1]) for i in range(3)])
    except (TypeError, ValueError, IndexError):
        return None, None


def _valid_pose(pose):
    try:
        gw.rigid(pose)
    except Exception:
        return False
    return True


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def pose_from_measured_view(view):
    """Rebuild the 4x4 of a measured EEF view from its position and axes.

    `direct_control.measured_view` publishes the proprioceptive pose as position
    plus the approach/opening columns. Recovering the matrix lets the pre-program
    state act as the measured START of the first command. Returns None -- not a
    guess -- when the reading is absent, is not an actual robot measurement, does
    not say it succeeded, or is not a proper rigid transform.
    """
    if not isinstance(view, dict):
        return None
    if view.get("status") != "ok" or view.get("source") not in MEASUREMENT_SOURCES:
        return None
    position = view.get(EEF_KEY, view.get(LEGACY_EEF_KEY))
    if isinstance(position, dict):
        position = [position.get(k) for k in "xyz"]
    approach, opening = view.get("approach"), view.get("opening")
    if isinstance(approach, dict):
        approach = [approach.get(k) for k in "xyz"]
    if isinstance(opening, dict):
        opening = [opening.get(k) for k in "xyz"]
    for vector in (position, approach, opening):
        if not isinstance(vector, (list, tuple)) or len(vector) != 3 or not all(
                _finite(v) for v in vector):
            return None
    a = list(map(float, approach))
    o = list(map(float, opening))
    norm = math.sqrt(sum(x * x for x in a))
    if norm <= 0:
        return None
    a = [x / norm for x in a]
    dot = sum(x * y for x, y in zip(a, o))
    o = [y - dot * x for x, y in zip(a, o)]
    norm = math.sqrt(sum(x * x for x in o))
    if norm <= 0:
        return None
    o = [x / norm for x in o]
    n = [o[1] * a[2] - o[2] * a[1], o[2] * a[0] - o[0] * a[2], o[0] * a[1] - o[1] * a[0]]
    pose = [[n[i], o[i], a[i], float(position[i])] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
    return pose if _valid_pose(pose) else None


def measured_width(view):
    """The measured jaw width of a verified reading, else None."""
    if not isinstance(view, dict) or view.get("status") != "ok":
        return None
    width = view.get("gripper_width_m")
    return float(width) if _finite(width) else None


def command_kind(resolved):
    """pose / gripper / hold, read from the committed command's own fields."""
    if not isinstance(resolved, dict):
        return None
    if "pose" in resolved:
        return "pose"
    if "state" in resolved:
        return "gripper"
    if "seconds" in resolved:
        return "hold"
    return None


def arrival(measured, target, position_tolerance_m, orientation_tolerance_deg):
    """Residuals plus the gate's own verdict; None when either pose is unknown.

    The comparison is done at full precision on the unrounded values, exactly as
    the Executor gate does it, so a displayed figure that looks borderline never
    contradicts the verdict. The two `within` booleans are present only when a
    tolerance was actually in force.
    """
    if measured is None or target is None:
        return None
    distance, angle = gw.arrival_error(measured, target)
    out = {"position_error_m": _round(distance, POSITION_DECIMALS),
           "orientation_error_deg": _round(angle, ANGLE_DECIMALS)}
    if _finite(position_tolerance_m) and _finite(orientation_tolerance_deg):
        out.update(position_within=bool(distance <= position_tolerance_m),
                   orientation_within=bool(angle <= orientation_tolerance_deg))
    return out


def _endpoint(record):
    """The actual pose/width this command ended at, or None when unverified.

    An acknowledgement only counts when the record says the command reached a
    recorded outcome. An `uncertain` record -- a backend exception or an
    acknowledgement the boundary refused -- has no verified endpoint even if
    something is attached to it.
    """
    if record.get("status") not in ("completed", "stopped"):
        return None, None
    ack = record.get("acknowledgement")
    if not isinstance(ack, dict) or ack.get("status") not in ("completed", "stopped"):
        return None, None
    pose = ack.get("measured_pose")
    width = ack.get("gripper_width_m")
    return (pose if _valid_pose(pose) else None), (float(width) if _finite(width) else None)


def _transport_index(payload):
    """Transport records keyed by command id; the backend's own submission log."""
    out = {}
    for order, entry in enumerate(_list(payload.get("executed"))):
        if isinstance(entry, dict) and entry.get("command_id") is not None:
            out[entry["command_id"]] = (order, entry)
    return out


def _status_of(record, transport):
    """The honest status of one command.

    A command the transport actually submitted is never reported as not run: when
    the execution result has no record for it -- the exception path, where the
    program never reached its arrival checks -- it is `uncertain`, because the
    outer gate never inspected an acknowledgement. Backend-level completion does
    not become an outer completion.
    """
    status = record.get("status")
    submitted = transport.get("submitted") if transport else None
    if status in SUBMITTED_STATUSES:
        return status, None
    if submitted:
        return UNCERTAIN, NO_ARRIVAL_CHECK
    if transport is not None:
        return NOT_SUBMITTED, "transport_attempted_not_submitted"
    return status or NOT_SUBMITTED, None


def command_diagnostics(payload):
    """One measured record per committed command, in committed order.

    START is the actual robot state the command began from: the pre-program
    measurement for the first command, the previous command's actual
    acknowledgement for every later one -- including gripper and hold commands,
    which also acknowledge a pose. A requested target is NEVER used as a measured
    start. When a needed measurement is missing or unverified the field is
    reported unknown, and both the pose chain and the jaw-width chain break there
    rather than carrying a stale reading forward.

    A command that was never submitted carries its identity, kind and status and
    nothing else: it has no measured start, because it never started.
    """
    payload = _dict(payload)
    commit = _dict(payload.get("commit"))
    resolved = _list(commit.get("resolved"))
    requested = _list(commit.get("requested"))
    records = _list(_dict(payload.get("execution")).get("records"))
    transports = _transport_index(payload)

    if not resolved:
        return _transport_only_diagnostics(payload)

    before = _dict(payload.get("observation_before")).get("measured_end_effector")
    cursor_pose = pose_from_measured_view(before)
    cursor_source = "measured_pre_program" if cursor_pose else UNKNOWN_START
    cursor_width = measured_width(before) if cursor_pose else None

    out = []
    for index, command in enumerate(resolved):
        record = _dict(records[index]) if index < len(records) else {}
        command_id = record.get("command_id") or f"{commit.get('id')}/{index}"
        transport = transports.get(command_id, (None, None))[1]
        kind = command_kind(command)
        status, note = _status_of(record, transport)
        end_pose, end_width = _endpoint(record)

        entry = {"index": index, "command_id": command_id, "kind": kind, "status": status}
        if record.get("error"):
            entry["stop_reason"] = record["error"]
        elif note:
            entry["stop_reason"] = note
        reply = _dict(transport).get("reply") if transport else None
        steps = _dict(reply).get("sim_steps_used")
        if steps is not None:
            # Actual physics advanced by THIS command, straight from the reply.
            entry["sim_steps_used"] = steps
        if transport is not None and not transport.get("submitted"):
            entry["submitted"] = False

        if status == NOT_SUBMITTED:
            # No start, no end, no residual: nothing about this command was measured.
            out.append(entry)
            continue

        if cursor_pose is None:
            entry["measured_start"] = UNKNOWN_START
        else:
            entry["start_" + EEF_KEY] = _xyz(_translation(cursor_pose))
            # Named only when it is NOT the ordinary previous-ack chain, so the
            # one case that matters -- the first command's pre-program state --
            # stays visible without repeating a constant on every later command.
            if cursor_source != "measured_previous_ack":
                entry["start_source"] = cursor_source
        if end_pose is None:
            entry["measured_end"] = UNKNOWN_END
        else:
            entry["end_" + EEF_KEY] = _xyz(_translation(end_pose))

        if kind == "pose":
            target = command.get("pose")
            target = target if _valid_pose(target) else None
            position_tolerance = command.get("position_tolerance_m")
            orientation_tolerance = command.get("orientation_tolerance_deg")
            entry["tolerance_m"] = position_tolerance
            entry["orientation_tolerance_deg"] = orientation_tolerance
            if target is not None:
                entry["target_position"] = _xyz(_translation(target))
                approach, opening = _axes(target)
                entry["target_approach"] = _xyz(approach)
                entry["target_opening"] = _xyz(opening)
            else:
                entry["target"] = "unknown_committed_target_unavailable"
            # The committed target is an absolute robot-frame pose. A proxy-relative
            # request matrix is a different quantity and is not shown as the target.
            request = _dict(requested[index]) if index < len(requested) else {}
            if request.get("reference"):
                entry["target_resolved_from_proxy_request"] = True
            entry["before"] = arrival(cursor_pose, target, position_tolerance,
                                      orientation_tolerance) or UNKNOWN_START
            entry["after"] = arrival(end_pose, target, position_tolerance,
                                     orientation_tolerance) or UNKNOWN_END
            if end_pose is not None and target is not None:
                entry["target_minus_actual_m"] = _xyz(
                    [t - m for t, m in zip(_translation(target), _translation(end_pose))])
        elif kind == "gripper":
            # Commanded jaw state and the measured widths around it. Width is not
            # evidence of a grasp; the legend says so once for the whole receipt.
            entry["gripper_command"] = command.get("state")
            entry["gripper_width_before_m"] = (_round(cursor_width, POSITION_DECIMALS)
                                               if cursor_width is not None else UNKNOWN_WIDTH)
            entry["gripper_width_after_m"] = (_round(end_width, POSITION_DECIMALS)
                                              if end_width is not None else UNKNOWN_WIDTH)
        elif kind == "hold":
            entry["hold_requested_s"] = command.get("seconds")
            ack = _dict(record.get("acknowledgement"))
            elapsed = ack.get("elapsed_s") if end_pose is not None else None
            entry["hold_elapsed_s"] = elapsed if _finite(elapsed) else UNKNOWN_WIDTH

        out.append(entry)
        if end_pose is None:
            # A submitted command with no verified endpoint breaks the measured
            # chain: later starts must not silently reuse an older measurement,
            # and neither must the jaw width.
            cursor_pose, cursor_source, cursor_width = None, UNKNOWN_START, None
        else:
            cursor_pose, cursor_source = end_pose, "measured_previous_ack"
            cursor_width = end_width
    return out


def _transport_only_diagnostics(payload):
    """Commands the transport attempted when no committed program is available.

    The exception path can leave a payload with submitted transport records and
    no execution result. Those commands were attempted, so they are shown with
    their submission flags and an unknown outcome -- never as commands that did
    not run, and never with an invented measured start.
    """
    out = []
    for index, entry in enumerate(_list(payload.get("executed"))):
        entry = _dict(entry)
        submitted = bool(entry.get("submitted"))
        record = {"index": index, "command_id": entry.get("command_id"),
                  "kind": _dict(entry.get("request")).get("kind"),
                  "status": UNCERTAIN if submitted else NOT_SUBMITTED,
                  "submitted": submitted}
        if submitted:
            record["stop_reason"] = NO_ARRIVAL_CHECK
            record["measured_end"] = UNKNOWN_END
        steps = _dict(entry.get("reply")).get("sim_steps_used")
        if steps is not None:
            record["sim_steps_used"] = steps
        out.append(record)
    return out


def not_executed_tail(diagnostics):
    """Index and count of the commands that never ran, so a stop is unambiguous."""
    for entry in diagnostics:
        if entry.get("status") == NOT_SUBMITTED:
            index = entry["index"]
            return {"from_index": index, "count": len(diagnostics) - index,
                    "note": "these commands did not run; do not assume their effect"}
    return None


def _summary(entry):
    """The short form of a command that completed without incident."""
    out = {k: entry[k] for k in ("index", "kind", "status", "sim_steps_used") if k in entry}
    for key in ("gripper_command", "gripper_width_before_m", "gripper_width_after_m",
                "hold_requested_s", "hold_elapsed_s"):
        if key in entry:
            out[key] = entry[key]
    return out


def project_commands(diagnostics):
    """Asymmetric view: routine commands short, decisive commands complete.

    Full detail is kept for every command that failed, stopped or is uncertain,
    and for the last executed command and pose, because those carry the
    residuals a next decision is made from. Commands that completed on the way
    there are summarised, and commands that never ran say only that. `dg_state`
    still has all of them in full.
    """
    executed = [e["index"] for e in diagnostics
                if e.get("status") in SUBMITTED_STATUSES]
    last_executed = executed[-1] if executed else None
    poses = [e["index"] for e in diagnostics
             if e.get("kind") == "pose" and e.get("status") in SUBMITTED_STATUSES]
    decisive = {last_executed, poses[-1] if poses else None}
    out = []
    for entry in diagnostics:
        if entry.get("status") == NOT_SUBMITTED:
            out.append({k: entry[k] for k in ("index", "kind", "status", "submitted")
                        if k in entry})
        elif entry.get("status") == "completed" and entry["index"] not in decisive:
            out.append(_summary(entry))
        else:
            out.append(entry)
    return out


# ── observation and proxy projection ─────────────────────────────────────────

def _cloud_agrees(frame, cloud_frame):
    """True when the cloud id is the observation id's own epoch and cloud counter.

    The two ids are written `e<epoch>-s<seq>-c<cloud>` and `e<epoch>-c<cloud>`, so
    agreement is a string fact, not an inference: when it holds the cloud id adds
    nothing the frame id does not already say, and when it fails the receipt shows
    the cloud id verbatim so the mismatch is visible.
    """
    if not isinstance(frame, str) or not isinstance(cloud_frame, str):
        return cloud_frame is None
    head, _, cloud = frame.rpartition("-c")
    epoch = head.rpartition("-s")[0]
    return bool(epoch) and cloud_frame == f"{epoch}-c{cloud}"


def observation_receipt(observation):
    """The single latest measured observation, compacted for display.

    `what` is kept first and verbatim: it is the marker a client uses to identify
    a whole observation object, and dropping it silently disables the client's
    own observation de-duplication. Frame pairing, the reasons a reading is
    unusable, depth-unknown reasons and the remaining simulation budget are all
    retained: they tell the model whether its evidence can be trusted. Local
    depth keeps its summary statistics and drops only the raw sample list, which
    stays recoverable through dg_state.
    """
    if not isinstance(observation, dict):
        return None
    out = {}
    if "what" in observation:
        out["what"] = observation["what"]
    for key in ("frame", "paired", "depth_paired_with_frame",
                "sim_steps_used", "sim_steps_budget"):
        if key in observation:
            out[key] = observation[key]
    if _finite(observation.get("sim_steps_used")) and _finite(observation.get("sim_steps_budget")):
        out["sim_steps_remaining"] = observation["sim_steps_budget"] - observation["sim_steps_used"]
    if not _cloud_agrees(observation.get("frame"), observation.get("cloud_frame")):
        # The cloud id repeats the observation id's own epoch and cloud counter,
        # so it is only worth the characters when the two actually disagree.
        out["cloud_frame"] = observation.get("cloud_frame")
    for key in ("pairing", "note"):
        # Why this frame is not usable as geometry. Never dropped.
        if observation.get(key) is not None:
            out[key] = observation[key]

    eef = observation.get("measured_end_effector")
    if isinstance(eef, dict):
        measured = {"status": eef.get("status")}
        for key in ("source", "frame"):
            if key in eef:
                measured[key] = eef[key]
        if eef.get("reason") is not None:
            measured["reason"] = eef["reason"]
        if eef.get("status") != "ok" or eef.get("source") not in MEASUREMENT_SOURCES:
            measured["status"] = "unknown"
            measured.setdefault("reason", "unverified_proprioception_reading")
            eef = {}  # A stale number attached to an unknown reading is not evidence.
        position = eef.get(EEF_KEY, eef.get(LEGACY_EEF_KEY))
        measured[EEF_KEY] = _xyz(position)
        # Measured axes are decision-relevant for approach direction; keep them.
        measured["approach"] = _xyz(eef.get("approach"))
        measured["opening"] = _xyz(eef.get("opening"))
        width = eef.get("gripper_width_m")
        measured["gripper_width_m"] = (_round(width, POSITION_DECIMALS)
                                       if _finite(width) else UNKNOWN_WIDTH)
        measured["gripper_state_class"] = eef.get("gripper_state_class")
        out["measured_eef"] = measured

    depth = observation.get("local_depth")
    if isinstance(depth, dict):
        local = {k: depth[k] for k in ("status", "reason", "radius_m", "support_points",
                                       "residual_rms_m", "nearest_return_m", "source")
                 if k in depth}
        if depth.get("centroid") is not None:
            local["centroid"] = _xyz(depth["centroid"])
        for k in ("z_min", "z_max"):
            if _finite(depth.get(k)):
                local[k] = _round(depth[k], POSITION_DECIMALS)
        if depth.get("at") is not None:
            # The centre these returns were sampled around. It comes from the
            # frame's own telemetry, so it need not equal measured_eef to the
            # last digit; both are kept rather than reconciled.
            local["at"] = _xyz(depth["at"])
        if depth.get("samples_xyz"):
            local["samples"] = "omitted from this summary"
        out["local_depth"] = local

    wrist = observation.get("wrist_depth")
    if isinstance(wrist, dict):
        out["wrist_depth"] = {k: wrist[k] for k in ("status", "reason", "depth_frame", "returns_valid",
                                                    "returns_total", "camera")
                              if k in wrist}
    return out


def proxy_receipt(proxy):
    """A geometry fit: validity, centre, extent, quality and why it was refused.

    A fit is a shape hypothesis over selected returns. It is not an object
    identity, and the label is the model's own. Diagnostic refusal reasons and
    every fit-quality number are always kept -- they are what lets a rejected
    bind be repaired rather than blindly repeated.
    """
    if not isinstance(proxy, dict):
        return None
    out = {"name": proxy.get("name"), "shape": proxy.get("shape"),
           "valid": proxy.get("valid"), "source_frame": proxy.get("from_frame")}
    if proxy.get("source"):
        out["source"] = proxy["source"]
    if proxy.get("center") is not None:
        out["center"] = _xyz(proxy["center"])
    if proxy.get("base_center") is not None:
        out["base_center"] = _xyz(proxy["base_center"])
    for key in ("top_z", "radius_m", "height_m"):
        if _finite(proxy.get(key)):
            out[key] = _round(proxy[key], POSITION_DECIMALS)
    if proxy.get("extent_m") is not None:
        out["extent_m"] = _xyz(proxy["extent_m"])
    if proxy.get("up") is not None:
        out["up"] = _xyz(proxy["up"])
    if proxy.get("up_source"):
        out["up_source"] = proxy["up_source"]
    fit = proxy.get("fit")
    if isinstance(fit, dict):
        out["fit_quality"] = dict(fit)
    if proxy.get("reasons"):
        out["reasons"] = proxy["reasons"]
    if proxy.get("message"):
        out["message"] = proxy["message"]
    out["identity_note"] = "your label on a shape fit; not a verified object identity"
    return out


def feedback_receipt(feedback):
    """The feedback envelope, with its provenance fields intact.

    Variant, spatial status, the frame the image was drawn on, whether the drawn
    selection is historical, and per-reference source / fit validity / identity
    provenance all survive: they are how the model tells a sensor measurement
    from its own hypothesis. Feedback errors stay visible, because a missing
    image is not a silent success.
    """
    if not isinstance(feedback, dict):
        return None
    out = {"variant": feedback.get("variant")}
    if feedback.get("identity") is not None:
        out["identity"] = feedback["identity"]
    spatial = feedback.get("spatial")
    if isinstance(spatial, dict):
        entry = {k: spatial[k] for k in ("status", "image_frame", "selection_source_frame",
                                         "selection_historical")
                 if k in spatial}
        if spatial.get("selection_surface") is not None:
            entry["selection_surface"] = spatial["selection_surface"]
        regions = _list(spatial.get("regions"))
        if regions:
            entry["regions"] = [{"name": _dict(r).get("name"), "returns": _dict(r).get("returns")}
                                for r in regions]
        references = _list(spatial.get("references"))
        if references:
            entry["references"] = [
                {k: _dict(r).get(k) for k in ("name", "source", "source_frame",
                                              "fit_valid", "historical", "identity_verified")}
                for r in references]
        out["spatial"] = entry
    temporal = feedback.get("temporal")
    if isinstance(temporal, dict):
        out["temporal"] = {k: temporal[k] for k in
                           ("status", "reason", "before_frame", "after_frame",
                            "sim_steps_delta", "measured_eef_delta_m",
                            "measured_jaw_width_before_m", "measured_jaw_width_after_m",
                            "commanded_gripper_before", "commanded_gripper_after")
                           if k in temporal}
    if feedback.get("errors"):
        out["errors"] = feedback["errors"]
    return out


def policy_receipt(payload):
    """Project a dg_policy payload into the compact model-facing receipt.

    Pure: `payload` is read, never written. Everything dropped here is either
    recomputable from what is kept or retrievable through dg_state.
    """
    if not isinstance(payload, dict):
        return payload
    out = {"status": payload.get("status")}
    for key in ("program_id", "reason", "cost_s", "timing_id"):
        if payload.get(key) is not None:
            out[key] = payload[key]

    diagnostics = command_diagnostics(payload)
    # Kept even when empty: an empty list is the explicit statement that this
    # call submitted no command at all, which a missing key would only imply.
    out["commands"] = project_commands(diagnostics)
    tail = not_executed_tail(diagnostics)
    if tail:
        out["not_executed"] = tail

    bind = payload.get("bind")
    if isinstance(bind, dict):
        entry = {"status": bind.get("status")}
        if bind.get("reason"):
            entry["reason"] = bind["reason"]
        for key in ("frame", "frame_after", "message"):
            if key in bind:
                entry[key] = bind[key]
        proxy = proxy_receipt(bind.get("proxy"))
        if proxy:
            entry["proxy"] = proxy
        out["bind"] = entry

    check = payload.get("geometry_check")
    if isinstance(check, dict):
        support = check.get("support")
        reference = _dict(check.get("reference"))
        entry = {}
        if isinstance(reference.get("ref"), dict):
            entry["reference"] = reference["ref"]
        if isinstance(support, dict):
            entry["support"] = {k: support[k] for k in
                                ("total_points", "support_points", "radius_m",
                                 "nearest_distance_m", "support_distance_rms_m", "scope")
                                if k in support}
        entry["note"] = "point support near the proxy origin; not identity, contact or grasp"
        out["geometry_check"] = entry

    observation = observation_receipt(
        payload.get("observation") or payload.get("observation_before"))
    if observation:
        out["observation"] = observation

    feedback = feedback_receipt(payload.get("feedback"))
    if feedback:
        out["feedback"] = feedback

    # Exactly one legend per receipt. The command form is a superset of the
    # observation form, so only the wider one is emitted when commands are shown.
    if out.get("commands"):
        out["legend"] = COMMAND_LEGEND
    elif out.get("observation") or out.get("bind"):
        out["legend"] = OBSERVATION_LEGEND
    return out


def look_receipt(payload):
    """Project a dg_look payload: one measured observation and its feedback."""
    if not isinstance(payload, dict):
        return payload
    out = {"status": payload.get("status")}
    if payload.get("timing_id") is not None:
        out["timing_id"] = payload["timing_id"]
    observation = observation_receipt(payload.get("observation"))
    if observation:
        out["observation"] = observation
    feedback = feedback_receipt(payload.get("feedback"))
    if feedback:
        out["feedback"] = feedback
    if observation:
        out["legend"] = OBSERVATION_LEGEND
    return out


def receipt(payload):
    """Dispatch on the payload's own shape; dg_state payloads pass through."""
    if not isinstance(payload, dict):
        return payload
    if any(k in payload for k in ("commit", "execution", "executed", "compiled_targets",
                                  "observation_before", "program_id")):
        return policy_receipt(payload)
    return look_receipt(payload)


# ── stored program history and bounded state queries ─────────────────────────
#
# dg_state reads what earlier calls already produced. It performs no sensing and
# no motion, it never re-measures a stored reading, and an invalid query is
# refused without touching the robot. Every page it returns is bounded, so no
# query can dump an episode.

DEFAULT_PROGRAM_PAGE = 8
MAX_PROGRAM_PAGE = 20
MAX_COMMAND_PAGE = 12
DEFAULT_PROXY_PAGE = 8
MAX_PROXY_PAGE = 20

STATE_QUERY_FIELDS = ("programs", "program_id", "commands", "proxies",
                      "include_depth_samples")


class StateQueryError(ValueError):
    """An invalid dg_state query. Refused, with no sensing and no actuation."""


def _page(spec, field, default, maximum):
    """Validate one {offset, limit} page specification."""
    spec = spec if spec is not None else {}
    if not isinstance(spec, dict):
        raise StateQueryError(f"{field}_must_be_an_object")
    unknown = set(spec) - {"offset", "limit"}
    if unknown:
        raise StateQueryError(f"{field}_unknown_fields:" + ",".join(sorted(unknown)))
    offset, limit = spec.get("offset", 0), spec.get("limit", default)
    for name, value in (("offset", offset), ("limit", limit)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise StateQueryError(f"{field}_{name}_must_be_an_integer")
    if offset < 0:
        raise StateQueryError(f"{field}_offset_must_not_be_negative")
    if not 1 <= limit <= maximum:
        raise StateQueryError(f"{field}_limit_must_be_between_1_and_{maximum}")
    return offset, limit


def validate_state_query(arguments):
    """Normalize a dg_state query or refuse it. Never partially applied."""
    arguments = {} if arguments is None else arguments
    if not isinstance(arguments, dict):
        raise StateQueryError("query_must_be_an_object")
    unknown = set(arguments) - set(STATE_QUERY_FIELDS)
    if unknown:
        raise StateQueryError("unknown_query_fields:" + ",".join(sorted(unknown)))
    program_id = arguments.get("program_id")
    if program_id is not None and (not isinstance(program_id, str) or not program_id):
        raise StateQueryError("program_id_must_be_a_non_empty_string")
    samples = arguments.get("include_depth_samples", False)
    if not isinstance(samples, bool):
        raise StateQueryError("include_depth_samples_must_be_a_boolean")
    for field in ("commands", "include_depth_samples"):
        if program_id is None and field in arguments:
            raise StateQueryError(field + "_requires_program_id")
    programs = _page(arguments.get("programs"), "programs",
                     DEFAULT_PROGRAM_PAGE, MAX_PROGRAM_PAGE)
    commands = _page(arguments.get("commands"), "commands",
                     MAX_COMMAND_PAGE, MAX_COMMAND_PAGE)
    proxies = _page(arguments.get("proxies"), "proxies",
                    DEFAULT_PROXY_PAGE, MAX_PROXY_PAGE)
    return {"program_id": program_id, "include_depth_samples": samples,
            "programs": programs, "commands": commands, "proxies": proxies}


def _depth_samples(observation):
    """The raw local-depth sample list of one stored observation, if it had one."""
    depth = _dict(_dict(observation).get("local_depth"))
    samples = depth.get("samples_xyz")
    if not samples:
        return None
    return {"frame": _dict(observation).get("frame"), "at": depth.get("at"),
            "radius_m": depth.get("radius_m"), "samples_xyz": [list(p) for p in samples],
            "note": "stored returns from that observation: robot, object and background alike"}


def program_record(payload, call_index):
    """A private, self-contained copy of one dg_policy call, for dg_state.

    Rejected and stopped programs are kept too: a refused fit or a stop is the
    history a next decision is made from. The record owns its data -- it copies
    what it needs out of the payload -- so later calls cannot mutate it and it
    cannot mutate the raw payload.
    """
    payload = _dict(payload)
    diagnostics = command_diagnostics(payload)
    before, after = payload.get("observation_before"), payload.get("observation")
    record = {"call_index": call_index,
              "program_id": payload.get("program_id"),
              "status": payload.get("status"),
              "commands": diagnostics,
              "observation_frame_before": _dict(before).get("frame"),
              "observation_frame_after": _dict(after).get("frame")}
    for key in ("reason", "cost_s", "timing_id"):
        if payload.get(key) is not None:
            record[key] = payload[key]
    bind = payload.get("bind")
    if isinstance(bind, dict):
        record["bind"] = {key: bind[key] for key in
                          ("status", "reason", "frame", "frame_after", "message") if key in bind}
        record["bind"]["proxy"] = proxy_receipt(bind.get("proxy"))
    check = payload.get("geometry_check")
    if isinstance(check, dict):
        reference = _dict(check.get("reference")).get("ref")
        record["geometry_check"] = {
            "reference": dict(reference) if isinstance(reference, dict) else None}
    feedback = feedback_receipt(payload.get("feedback"))
    if feedback:
        record["feedback"] = feedback
    samples = {label: _depth_samples(view)
               for label, view in (("before", before), ("after", after))}
    samples = {k: v for k, v in samples.items() if v}
    if samples:
        record["depth_samples"] = samples
    return record


def program_summary(record):
    """One line per stored program: what it was, how it ended, how far it got."""
    commands = _list(record.get("commands"))
    executed = [c for c in commands if c.get("status") in SUBMITTED_STATUSES]
    stopped = [c["index"] for c in commands
               if c.get("status") in ("stopped", UNCERTAIN)]
    out = {"call_index": record.get("call_index"), "program_id": record.get("program_id"),
           "status": record.get("status")}
    if record.get("reason") is not None:
        out["reason"] = record["reason"]
    if commands:
        out["command_count"] = len(commands)
        out["executed_count"] = len(executed)
    if stopped:
        out["stopped_at_index"] = stopped[0]
    if record.get("bind") is not None:
        out["bind_status"] = _dict(record["bind"]).get("status")
    out["observation_frame_after"] = record.get("observation_frame_after")
    return out


def program_detail(record, query):
    """Full diagnostics for one stored program, one bounded command page at a time."""
    commands = _list(record.get("commands"))
    offset, limit = query["commands"]
    page = commands[offset:offset + limit]
    out = {k: record[k] for k in ("call_index", "program_id", "status", "reason", "cost_s",
                                  "observation_frame_before", "observation_frame_after")
           if k in record}
    out["command_count"] = len(commands)
    out["commands_offset"] = offset
    out["commands"] = page
    out["has_more_commands"] = offset + len(page) < len(commands)
    for key in ("bind", "geometry_check", "feedback"):
        if record.get(key) is not None:
            out[key] = record[key]
    if query["include_depth_samples"]:
        out["depth_samples"] = record.get("depth_samples") or {
            "status": "unknown", "reason": "this program stored no local depth samples"}
    out["note"] = ("stored records of an earlier call; nothing here was re-measured "
                   "and no command was re-run")
    return out


def proxy_state_entry(name, *, revision, bound_frame, current, card):
    """One stored geometric hypothesis, with its binding frame and staleness.

    A proxy is a fit or an explicit hypothesis from the observation it was bound
    on. It does not follow an object. `current` says only whether that same
    observation is still the current one -- never whether the object is still
    there.
    """
    out = {"name": name, "revision": revision, "bound_frame": bound_frame,
           "current": bool(current)}
    card = _dict(card)
    if card.get("center") is not None:
        out["center"] = _xyz(card["center"])
    if card.get("up") is not None:
        out["up"] = _xyz(card["up"])
    for key in ("shape", "valid", "source", "up_source"):
        if card.get(key) is not None:
            out[key] = card[key]
    if not current:
        out["note"] = "bound on an earlier observation; historical, not tracked"
    return out


def state_receipt(*, policy_calls, records, proxies, last_observation, query):
    """The dg_state reply: bounded program history, proxies and the stored observation.

    `last_observation` is the stored copy of the most recent measured observation,
    projected exactly as a policy receipt projects it, and marked as stored rather
    than fresh. Proxies say which observation they were bound on and whether that
    binding is still current, so a stale hypothesis cannot pass as a measurement.
    """
    offset, limit = query["programs"]
    ordered = list(reversed(records))  # most recent first
    out = {"status": "ok", "policy_calls": policy_calls, "programs_total": len(records)}
    if query["program_id"] is not None:
        selected = [r for r in records if r.get("program_id") == query["program_id"]]
        if not selected:
            return {"status": "rejected", "reason": "unknown_program_id",
                    "programs_total": len(records),
                    "known_program_ids": [r.get("program_id") for r in ordered[:MAX_PROGRAM_PAGE]],
                    "note": "no sensing or motion happened; nothing was changed"}
        out["program"] = program_detail(selected[-1], query)
    else:
        page = ordered[offset:offset + limit]
        out["programs_offset"] = offset
        out["programs"] = [program_summary(r) for r in page]
        out["has_more_programs"] = offset + len(page) < len(ordered)
        out["programs_note"] = ("most recent first; ask with program_id for one "
                                "program's full command diagnostics")
    proxy_offset, proxy_limit = query["proxies"]
    items = list(proxies)
    page = items[proxy_offset:proxy_offset + proxy_limit]
    out["proxies_total"] = len(items)
    out["proxies_offset"] = proxy_offset
    out["proxies"] = page
    out["has_more_proxies"] = proxy_offset + len(page) < len(items)
    observation = observation_receipt(last_observation)
    if observation:
        out["last_observation"] = observation
        out["last_observation_note"] = ("stored copy of the most recent observation; "
                                        "dg_look takes a new one")
    out["legend"] = COMMAND_LEGEND
    return out

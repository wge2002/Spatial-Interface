"""Sparse local point-cloud references and a coarse geometric proxy.

Model-free and browser-agnostic: every function takes plain numbers, so the whole
file is unit-testable without a simulator, a model or Chromium. The JS constants
are evaluated by the MCP server through ``page.evaluate``; keeping them here
rather than in ``spatial_interface/UI/template_index.html`` leaves the UI that the legacy and
compact interfaces render byte-identical.

What this module deliberately does NOT do:

* it never reads MuJoCo object state, segmentation ids or the success flag — the
  only inputs are the RGB-D point cloud already in the browser, the camera
  calibration, the robot's own telemetry, and a model-chosen (u, v);
* it never claims a verified object identity (a model-supplied string is stored
  as ``model_label``), collision freedom, contact dynamics, or a grasp;
* it never emits motion. The proxy resolves a candidate pose with the same
  limits as the target editor and reports geometry only.
"""

from __future__ import annotations

import math

UI_PER_M = 10.0  # 1 UI unit = 0.1 m (see mcp_server.ui_to_robot)

# Local-support gates. A reference below MIN_SUPPORT points is reported unknown
# rather than as a small object: the cloud is downsampled to --num_point, so a
# thin return is evidence of occlusion or sampling, not of geometry.
MIN_SUPPORT = 40
DEFAULT_RADIUS_M = 0.05
MIN_RADIUS_M = 0.01
MAX_RADIUS_M = 0.12
SAMPLE_CAP = 3000

# A principal axis is only reported when the neighbourhood is actually elongated;
# an isotropic blob has no direction the data supports.
AXIS_RATIO_MIN = 1.6
AXIS_VARIANCE_MIN_M2 = 1e-6  # (1 mm)^2

# Refresh gates. Beyond these the match is reported unknown instead of tracked.
MAX_TRACK_SHIFT_M = 0.05
EXTENT_RATIO_MAX = 2.5

# Proxy geometry. Coarse, deliberately conservative, and stated in the reply.
BODY_RADIUS_M = 0.035  # corridor half-width treated as swept by the gripper
OBS_RADIUS_M = 0.06  # a sample with no cloud point inside this is UNOBSERVED
SELF_RADIUS_M = 0.09  # cloud points this close to the measured fingertip may be
# the gripper's own surface; excluded so the robot cannot be reported as an
# obstacle in front of itself. The exclusion is a blunt sphere: it also deletes
# any real object inside it, so a sample whose own neighbourhood overlaps that
# sphere is reported UNKNOWN and can never contribute a "no obstruction" reading.
MAX_SPAN_M = 0.08  # jaw span used for a span comparison only, not graspability
ATTACH_RADIUS_M = 0.06
PATH_SAMPLES = 25

LIMITS = (
    "Coarse geometry on the currently visible RGB-D cloud only. Distances are "
    "measured to sampled points of a downsampled cloud, so they bound nothing: "
    "the real surface can be nearer than the nearest sampled return, and regions "
    "with no returns are unobserved rather than free. No collision proof, no "
    "contact dynamics, no grasp claim, no object identity."
)


# ── browser-side extraction (evaluated by the MCP server) ────────────────────

# Identity of the observation plus the robot's own telemetry. Identity comes from
# the producer's frame_meta below and from nothing else. An earlier version also
# hashed a strided subsample of the position buffer and claimed the hash proved
# freshness; it does not, and cannot. Two observations of a still scene hash
# alike, so an unchanged hash is not evidence of a reused cloud and a changed one
# is not evidence of a new frame. The hash was computed on every extraction and
# read by no caller, so it is gone rather than kept as a misleading field.
OBSERVE_JS = """() => {
    if (typeof pcGeo === 'undefined' || !pcGeo || !pcGeo.attributes.position)
        return null;
    const pos = pcGeo.attributes.position.array;
    let dirs = null;
    if (typeof lastEeEulerUI !== 'undefined' && lastEeEulerUI) {
        // Same convention snapSelectedToRealGripper() uses to place the blue
        // target on the real gripper, so measured and virtual poses are
        // expressed in one frame and can be compared component by component.
        const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(
            lastEeEulerUI[0], lastEeEulerUI[2], -lastEeEulerUI[1], 'YZX'));
        const toRobot = v => [v.x, -v.z, v.y];
        dirs = {
            approach: toRobot(new THREE.Vector3(0, -1, 0).applyQuaternion(q)),
            opening: toRobot(new THREE.Vector3(0, 0, 1).applyQuaternion(q)),
        };
    }
    const tip = (typeof lastFingertipUI !== 'undefined' && lastFingertipUI
                 && lastFingertipUI.toArray) ? lastFingertipUI.toArray() : null;
    return {
        cloud_points: pos.length / 3,
        // Reported for accounting only. It advances while a cloud is being
        // reused, so it can neither mint nor confirm an observation version.
        waypoint_done_count: window.__waypointDoneCount || 0,
        fingertip_ui: tip,
        ee_dirs_robot: dirs,
        gripper_action: (typeof lastGripperAction !== 'undefined' && lastGripperAction)
            ? Array.from(lastGripperAction) : null,
        cam_labels: (typeof camInfo !== 'undefined' && camInfo) ? Object.keys(camInfo) : [],
        // Producer-assigned pairing metadata (record_sim frame_meta). Absent on
        // a UI old enough not to send it — then nothing here is verifiable and
        // the caller must report unknown rather than fall back to contents.
        frame_meta: (typeof window.__frameMeta !== 'undefined' && window.__frameMeta)
            ? window.__frameMeta : null,
        // Which camera images are actually on screen, and which are still
        // decoding, at the instant this extraction ran.
        cam_feed_seq: (typeof window.__camFeedSeq !== 'undefined' && window.__camFeedSeq)
            ? JSON.parse(JSON.stringify(window.__camFeedSeq)) : null,
    };
}"""

# Neighbourhood of a UI-frame centre. Two passes so a dense neighbourhood is
# strided rather than truncated in raster order (truncation would drop a spatial
# band and bias the centroid).
SAMPLE_LOCAL_JS = """args => {
    const pos = pcGeo.attributes.position.array;
    const n = pos.length / 3;
    const [cx, cy, cz] = args.center;
    const r2 = args.radius * args.radius;
    let support = 0;
    for (let i = 0; i < n; i++) {
        const dx = pos[i*3] - cx, dy = pos[i*3+1] - cy, dz = pos[i*3+2] - cz;
        if (dx*dx + dy*dy + dz*dz <= r2) support++;
    }
    const stride = Math.max(1, Math.ceil(support / args.cap));
    const points = [];
    let seen = 0;
    for (let i = 0; i < n; i++) {
        const dx = pos[i*3] - cx, dy = pos[i*3+1] - cy, dz = pos[i*3+2] - cz;
        if (dx*dx + dy*dy + dz*dz > r2) continue;
        if (seen++ % stride === 0) points.push(pos[i*3], pos[i*3+1], pos[i*3+2]);
    }
    return {support: support, stride: stride, points: points, cloud_points: n};
}"""

# Per-sample nearest cloud distance along a candidate path. One pass over the
# cloud for all samples; distances are UI units.
CORRIDOR_JS = """args => {
    const pos = pcGeo.attributes.position.array;
    const n = pos.length / 3;
    const m = args.samples.length / 3;
    const best = new Array(m).fill(Infinity);
    const near = new Array(m).fill(0);
    const obs = new Array(m).fill(0);
    const br2 = args.body_radius * args.body_radius;
    const or2 = args.obs_radius * args.obs_radius;
    const sc = args.self_center, sr2 = args.self_radius * args.self_radius;
    let excluded = 0;
    for (let i = 0; i < n; i++) {
        const px = pos[i*3], py = pos[i*3+1], pz = pos[i*3+2];
        if (sc) {
            const dx = px - sc[0], dy = py - sc[1], dz = pz - sc[2];
            if (dx*dx + dy*dy + dz*dz <= sr2) { excluded++; continue; }
        }
        for (let j = 0; j < m; j++) {
            const dx = px - args.samples[j*3];
            const dy = py - args.samples[j*3+1];
            const dz = pz - args.samples[j*3+2];
            const d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best[j]) best[j] = d2;
            if (d2 <= br2) near[j]++;
            if (d2 <= or2) obs[j]++;
        }
    }
    return {
        min_distance: best.map(d => d === Infinity ? null : Math.sqrt(d)),
        body_support: near, observed_support: obs,
        excluded_self_points: excluded, cloud_points: n,
    };
}"""

# Project a UI-frame point to browser pixels so the overlay marks the extracted
# centre (not the raw click) and a reader can check the two agree.
PROJECT_JS = """p => {
    if (typeof camera === 'undefined' || !camera || typeof renderer === 'undefined')
        return null;
    const v = new THREE.Vector3(p[0], p[1], p[2]).project(camera);
    if (!isFinite(v.x) || !isFinite(v.y) || v.z < -1 || v.z > 1) return null;
    const rect = renderer.domElement.getBoundingClientRect();
    return {
        x: rect.left + (v.x * 0.5 + 0.5) * rect.width,
        y: rect.top + (-v.y * 0.5 + 0.5) * rect.height,
        inside: v.x >= -1 && v.x <= 1 && v.y >= -1 && v.y <= 1,
    };
}"""


# ── frames and small linear algebra ──────────────────────────────────────────

def ui_to_robot(p, ui_z_offset):
    """UI Three.js units -> robot metres. Mirrors mcp_server.ToolContext."""
    return [p[0] / UI_PER_M, -p[2] / UI_PER_M, p[1] / UI_PER_M + ui_z_offset]


def robot_to_ui(p, ui_z_offset):
    return [p[0] * UI_PER_M, (p[2] - ui_z_offset) * UI_PER_M, -p[1] * UI_PER_M]


def _finite(value, name, length=3):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} finite numbers")
    out = []
    for x in value:
        if type(x) not in (int, float) or not math.isfinite(x):
            raise ValueError(f"{name} must contain {length} finite numbers (not booleans)")
        out.append(float(x))
    return out


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


def jacobi_eigh(m):
    """Eigen decomposition of a symmetric 3x3 matrix, descending by eigenvalue.

    Pure Python (cyclic Jacobi) so this module needs no numpy and can be
    imported by the bare-script MCP server as cheaply as by the tests.
    """
    a = [row[:] for row in m]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(24):
        off = sum(a[i][j] ** 2 for i, j in ((0, 1), (0, 2), (1, 2)))
        if off <= 1e-24:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) <= 1e-18:
                continue
            theta = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
            c, s = math.cos(theta), math.sin(theta)
            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
            for k in range(3):
                apk, aqk = a[p][k], a[q][k]
                a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    order = sorted(range(3), key=lambda i: a[i][i], reverse=True)
    values = [a[i][i] for i in order]
    vectors = [[v[r][i] for r in range(3)] for i in order]
    return values, vectors


# ── observation identity ─────────────────────────────────────────────────────

CAM_LABEL_TO_ELEMENT = {"agentview": "cam-agentview", "wrist": "cam-wrist"}


def frame_meta(observation):
    """The producer's pairing metadata, or None when this UI does not send it.

    Returned as-is only after checking the fields the freshness logic needs are
    present and well typed. A UI without it cannot support any freshness claim,
    so callers must report unknown instead of inferring one from contents.
    """
    meta = (observation or {}).get("frame_meta")
    if not isinstance(meta, dict):
        return None
    try:
        epoch = meta["epoch"]
        seq = int(meta["seq"])
        cloud_seq = None if meta.get("cloud_seq") is None else int(meta["cloud_seq"])
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(epoch, str) or not epoch:
        return None
    return {"epoch": epoch, "seq": seq, "cloud_seq": cloud_seq,
            "cam_labels": list(meta.get("cam_labels") or []),
            "cam_info_seq": meta.get("cam_info_seq"),
            "proprio_seq": meta.get("proprio_seq"),
            # Transport-side fields. Reported for diagnosis; deliberately not
            # part of any version or freshness decision below, because a message
            # count says nothing about when the sensors were last read.
            "delivery_seq": meta.get("delivery_seq"),
            "redelivery": bool(meta.get("redelivery"))}


def observation_version(observation):
    """Stable id of one *frame*, from producer metadata only.

    ``e<epoch>-s<seq>-c<cloud_seq>``. `seq` numbers the producer's *observation*
    — one reading of the sensors — not the messages carrying it. It advances
    whenever the sim steps and a new snapshot is taken, including a streamed
    skip_pcl frame that only moved the end effector, so any real change between
    two calls makes a same-version check fail. It does not advance while the sim
    is idle and the producer redelivers the observation it already sent, because
    re-rendering cached sensor state measures nothing new; that is what lets a
    reference survive the seconds between two tool calls. `frame_meta`'s
    `delivery_seq` counts messages and is excluded here for the same reason.
    Camera moves, virtual-target edits and screenshots do not go through the
    producer at all, so they cannot change it.

    None when the UI sends no metadata: unverifiable, never a fabricated id.
    Cloud *contents* are deliberately not part of this. A hash cannot show a
    cloud is new (two observations of a still scene hash alike) and neither can
    the executed-waypoint count (it advances while the cloud is being reused).
    """
    meta = frame_meta(observation)
    if meta is None:
        return None
    return f"e{meta['epoch']}-s{meta['seq']}-c{meta['cloud_seq']}"


def cloud_version(observation):
    """Id of the cloud alone: which producer frame actually measured it."""
    meta = frame_meta(observation)
    if meta is None or meta["cloud_seq"] is None:
        return None
    return f"e{meta['epoch']}-c{meta['cloud_seq']}"


def parse_observation_version(version):
    """Split ``e<epoch>-s<seq>-c<cloud_seq>`` back into its parts, or None.

    Parsed from the right, because an epoch is an opaque producer string and may
    itself contain a dash. Returns None for anything that is not a version this
    module wrote, so a caller can never compare two ids it cannot interpret.
    """
    if not isinstance(version, str) or not version.startswith("e"):
        return None
    head, sep, cloud = version.rpartition("-c")
    if not sep:
        return None
    epoch, sep, seq = head.rpartition("-s")
    if not sep or len(epoch) < 2:
        return None
    try:
        return {"epoch": epoch[1:], "seq": int(seq),
                "cloud_seq": None if cloud == "None" else int(cloud)}
    except ValueError:
        return None


def frame_advance(before, after):
    """Why ``after`` is not a strictly later frame of the same episode, or None.

    Used to tell "a new observation measured this execution" from "the same frame
    read twice" and from "a frame belonging to a different episode". Both are ways
    an endpoint comparison could be arithmetic on unrelated states.
    """
    b, a = parse_observation_version(before), parse_observation_version(after)
    if b is None or a is None:
        return "unreadable_observation_version"
    if a["epoch"] != b["epoch"]:
        return "different_episode_epoch"
    if a["seq"] <= b["seq"]:
        return "not_a_later_frame"
    return None


def pairing_state(observation, *, require_cam_labels=()):
    """Whether this frame's parts were measured together.

    ``paired`` requires the cloud to come from this very frame (`cloud_seq ==
    seq`) and the proprioception and calibration to be this frame's too. When
    camera labels are named (a click resolved on a camera feed), the image on
    screen for each must also be this frame's: an image still decoding, or a
    previous one still displayed, cannot be combined with this frame's
    extrinsics. Anything short of that is ``unknown`` with a reason.
    """
    meta = frame_meta(observation)
    if meta is None:
        return {"status": "unknown", "reason": "no_frame_metadata",
                "note": "this UI does not report frame pairing; freshness is "
                        "unverifiable and no observation can be treated as new"}
    out = {"status": "paired", "epoch": meta["epoch"], "seq": meta["seq"],
           "cloud_seq": meta["cloud_seq"], "cam_info_seq": meta["cam_info_seq"],
           "proprio_seq": meta["proprio_seq"], "reason": None}
    if meta["cloud_seq"] is None:
        out.update(status="unknown", reason="no_cloud_yet")
        return out
    if meta["cloud_seq"] != meta["seq"]:
        out.update(status="unknown", reason="stale_cloud_reused_by_streamed_frame",
                   note="the cloud in the browser was measured by an earlier "
                        "frame than this end-effector/calibration state")
        return out
    if meta["proprio_seq"] != meta["seq"]:
        out.update(status="unknown", reason="stale_proprioception")
        return out
    feeds = (observation or {}).get("cam_feed_seq") or {}
    for label in require_cam_labels:
        if meta["cam_info_seq"] != meta["seq"]:
            out.update(status="unknown", reason="stale_camera_calibration")
            return out
        track = feeds.get(CAM_LABEL_TO_ELEMENT.get(label, label))
        if not isinstance(track, dict):
            out.update(status="unknown", reason=f"no_display_record_for_{label}")
            return out
        displayed, pending = track.get("displayed"), track.get("pending")
        if displayed != meta["seq"]:
            out.update(status="unknown",
                       reason=("camera_image_decode_pending" if pending == meta["seq"]
                               else "stale_camera_image_displayed"),
                       note=f"the {label} image on screen is from frame "
                            f"{displayed}, not {meta['seq']}")
            return out
    out["cam_labels_checked"] = list(require_cam_labels)
    return out


def gripper_state_class(observation):
    """Binary open/closed *classification* from robot telemetry.

    record_sim derives gripper_action from a threshold on the simulator's
    gripper_open signal, so this is a class label, never a measured opening
    width and never evidence that something is held.
    """
    action = (observation or {}).get("gripper_action")
    if not action:
        return "unknown"
    return "closed" if float(action[0]) > 0.5 else "open"


def measured_end_effector(observation, ui_z_offset):
    """The robot's own measured end effector, kept separate from the virtual target."""
    if not observation or not observation.get("fingertip_ui"):
        return {"status": "unknown", "reason": "no fingertip telemetry in this frame"}
    dirs = observation.get("ee_dirs_robot") or {}
    out = {
        "status": "ok",
        "source": "robot_telemetry",
        "frame": "robot",
        "fingertip_position": _round3(ui_to_robot(observation["fingertip_ui"], ui_z_offset)),
        "gripper_state_class": gripper_state_class(observation),
        "gripper_state_note": ("threshold classification from telemetry; not a measured "
                               "opening width and not grasp evidence"),
    }
    for key in ("approach", "opening"):
        out[key] = _round3(dirs[key], 9) if dirs.get(key) else None
    return out


def _round3(p, digits=4):
    return {k: round(float(x), digits) for k, x in zip("xyz", p)}


# ── local geometry summary ───────────────────────────────────────────────────

def summarize_points(points_robot, *, radius_m, support, stride, source):
    """Sparse local geometry with explicit validity, from cloud points alone."""
    summary = {
        "source": source,
        "frame": "robot",
        "radius_m": round(float(radius_m), 4),
        "support_points": int(support),
        "sampled_points": len(points_robot),
        "sample_stride": int(stride),
        "min_support_points": MIN_SUPPORT,
    }
    if support < MIN_SUPPORT or len(points_robot) < 3:
        summary.update(status="unknown", reasons=["insufficient local support"],
                       center=None, extent_m=None, principal_axis=None,
                       principal_extent_m=None, axis_ratio=None, keypoints=None)
        return summary

    n = len(points_robot)
    center = [sum(p[i] for p in points_robot) / n for i in range(3)]
    lo = [min(p[i] for p in points_robot) for i in range(3)]
    hi = [max(p[i] for p in points_robot) for i in range(3)]
    cov = [[sum((p[i] - center[i]) * (p[j] - center[j]) for p in points_robot) / n
            for j in range(3)] for i in range(3)]
    values, vectors = jacobi_eigh(cov)
    reasons = []
    axis = None
    axis_extent = None
    # Exactly collinear returns give values[1] == 0 (or a tiny negative from
    # round-off): that is the *strongest* possible elongation evidence, so the
    # ratio is treated as unbounded rather than rejected as degenerate. Only
    # values[0] itself, the variance along the candidate axis, can disqualify one.
    degenerate_second = values[1] <= AXIS_VARIANCE_MIN_M2
    ratio = None if degenerate_second else round(values[0] / values[1], 3)
    if values[0] < AXIS_VARIANCE_MIN_M2:
        reasons.append("neighbourhood too small for an evidence-backed axis")
        ratio = None
    elif not degenerate_second and ratio < AXIS_RATIO_MIN:
        reasons.append("neighbourhood not elongated enough for an evidence-backed axis")
    else:
        axis = vectors[0]
        length = _norm(axis)
        if length > 0:
            axis = [x / length for x in axis]
        projections = [_dot(_sub(p, center), axis) for p in points_robot]
        axis_extent = round(max(projections) - min(projections), 4)
        if degenerate_second:
            reasons.append("second-moment variance at or below the noise floor: the "
                           "returns are effectively collinear, so the axis is "
                           "reported and axis_ratio is unbounded")

    highest = max(points_robot, key=lambda p: p[2])
    lowest = min(points_robot, key=lambda p: p[2])
    summary.update(
        status="ok",
        reasons=reasons,
        center=_round3(center),
        extent_m={k: round(hi[i] - lo[i], 4) for i, k in enumerate("xyz")},
        # Directions get more digits than positions: 4 decimals is 1e-4 of unit
        # length, enough to break a caller's own unit-vector tolerance.
        principal_axis=_round3(axis, 9) if axis else None,
        principal_extent_m=axis_extent,
        axis_ratio=ratio,
        axis_ratio_note=(None if not axis or ratio is not None else
                         "unbounded: collinear returns, second moment ~ 0"),
        # Keypoints are actual cloud returns, not fitted or assumed features.
        keypoints={"highest_point": _round3(highest), "lowest_point": _round3(lowest)},
    )
    return summary


def span_along(points_robot, axis):
    """Extent of the sampled points along a unit axis, or None without points."""
    if not points_robot:
        return None
    projections = [_dot(p, axis) for p in points_robot]
    return round(max(projections) - min(projections), 4)


# ── reference validity ──────────────────────────────────────────────────────

STALE_NOTE = ("last successfully extracted centre; NOT the current location of "
              "anything")

# Why a reference stopped being usable. Kept as constants so the proxy's refusal
# and the refresh reply cannot drift apart in wording.
INVALID_NEW_OBSERVATION = "new_observation_since_binding_association_unverified"
INVALID_REFRESH_FAILED = "refresh_could_not_re_establish_the_reference"
# A refresh that could not run at all retires the reference too. Not because the
# frame is evidence the referent moved — it is not — but because after a refresh
# attempt that produced no current geometry, the stored coordinates are a record of
# where something was and nothing local can show they are still where anything is.
# They stay readable under the stale names until a visual rebind.
INVALID_UNVERIFIABLE_FRAME = "refresh_frame_could_not_be_verified_rebind_visually"
INVALID_FRAME_CHANGED_DURING_REFRESH = "frame_changed_during_refresh_rebind_visually"


def reference_validity(stored, current_version):
    """Whether a stored reference may still be used, and why not when it may not.

    Three outcomes, and the middle one is the point of this function:

    * ``valid``   — live geometry, extracted from *this very frame*.
    * ``stale``   — live geometry, but from an earlier frame. Nothing local can
      show the referent did not move or was not replaced in between, so this is
      not a usable location; it is a record of where something was.
    * ``invalid`` — no live geometry at all (a refresh failed, or a new
      observation retired it). Only the stale fields remain.

    Deliberately version equality, not proximity: a similar centre and a similar
    extent are what a *different* object of the same shape also produces, so
    matching them would be assuming the association this code cannot establish.
    """
    if not stored.get("geometry") or stored.get("validity") == "invalid":
        return "invalid", stored.get("invalid_reason") or INVALID_REFRESH_FAILED
    version = stored.get("geometry_observation_version")
    if version is None or current_version is None or version != current_version:
        return "stale", "geometry_extracted_from_an_earlier_observation"
    return "valid", None


def stale_fields(stored, *, reason):
    """The stale-only view of a reference: coordinates that may not be used."""
    return {
        "validity": "invalid",
        "invalid_reason": reason,
        "stale_stored_center": (stored.get("stale_geometry") or {}).get("center")
                               if not stored.get("geometry")
                               else stored["geometry"].get("center"),
        "stale_observation_version": (stored.get("geometry_observation_version")
                                      or stored.get("stale_observation_version")),
        "stale_note": STALE_NOTE,
    }


def invalidate_reference(stored, *, reason):
    """Retire a reference's live geometry into stale-only storage, in place.

    After this, ``reference_validity`` reports ``invalid`` and the proxy refuses
    the reference. The coordinates survive only under ``stale_*`` names, so a
    later reader cannot pick them up as a current location by accident.
    """
    if stored.get("geometry"):
        stored["stale_geometry"] = stored["geometry"]
        stored["stale_observation_version"] = stored.get("geometry_observation_version")
    stored["geometry"] = None
    stored["points_robot"] = []
    stored["geometry_observation_version"] = None
    stored["validity"] = "invalid"
    stored["invalid_reason"] = reason
    return stored


# ── reference refresh ────────────────────────────────────────────────────────

def refresh_verdict(previous, current):
    """Compare a re-extraction against the stored summary; never fake tracking.

    Returns (status, reasons, shift_m). ``unknown`` means the reference could not
    be re-established under the stated gates — the caller must report the stored
    coordinates as stale rather than as the current location.
    """
    reasons = []
    if current["status"] != "ok":
        return "unknown", list(current.get("reasons") or ["local geometry unavailable"]), None
    if previous.get("center") is None:
        return "unknown", ["no stored centre to match against"], None
    prev_c = [previous["center"][k] for k in "xyz"]
    cur_c = [current["center"][k] for k in "xyz"]
    shift = round(_norm(_sub(cur_c, prev_c)), 4)
    if shift > MAX_TRACK_SHIFT_M:
        return "unknown", [f"local match moved {shift:.4f} m, beyond the "
                           f"{MAX_TRACK_SHIFT_M:g} m match gate"], shift
    prev_e = previous.get("extent_m")
    if prev_e:
        for k in "xyz":
            a, b = float(prev_e[k]), float(current["extent_m"][k])
            if max(a, b) > 0.005 and max(a, b) / max(min(a, b), 1e-6) > EXTENT_RATIO_MAX:
                reasons.append(f"extent along {k} changed by more than "
                               f"{EXTENT_RATIO_MAX:g}x; match is ambiguous")
                return "unknown", reasons, shift
    return "ok", reasons, shift


# ── coarse geometric proxy ───────────────────────────────────────────────────

def path_samples_robot(start, end, count=PATH_SAMPLES):
    """Straight-line samples from the measured fingertip to the candidate pose.

    The controller interpolates linearly between poses, so a straight segment is
    the honest first-order approximation — it is not the executed trajectory and
    the reply says so.
    """
    count = max(2, int(count))
    return [[a + (b - a) * i / (count - 1) for a, b in zip(start, end)]
            for i in range(count)]


def self_blinded_samples(samples_robot, self_center_robot):
    """Indices whose own observation neighbourhood overlaps the self-exclusion sphere.

    CORRIDOR_JS drops every cloud point within SELF_RADIUS_M of the measured
    fingertip, including points belonging to real objects. A sample closer than
    SELF_RADIUS_M + OBS_RADIUS_M to that centre therefore had part of its own
    neighbourhood deleted before it was measured, so both its distance reading and
    its "no returns nearby" reading are uninformative: the region is unknown, and
    saying anything else would report the blind spot as free space.
    """
    if self_center_robot is None:
        return []
    limit = SELF_RADIUS_M + OBS_RADIUS_M
    return [i for i, s in enumerate(samples_robot)
            if _norm(_sub(s, self_center_robot)) <= limit]


def summarize_corridor(corridor, samples_robot, self_center_robot=None):
    """Turn per-sample nearest-sampled-point readings into a bounded verdict.

    No output of this function is a clearance bound or a claim of free space: the
    cloud is downsampled, self-excluded near the fingertip, and blind behind every
    visible surface. It reports where sampled returns were found and where nothing
    can be said.
    """
    distances = corridor["min_distance"]
    body = corridor["body_support"]
    observed = corridor["observed_support"]
    blinded = set(self_blinded_samples(samples_robot, self_center_robot))
    entries = []
    for i, sample in enumerate(samples_robot):
        d = distances[i]
        blind = i in blinded
        entry = {
            "index": i,
            "position": _round3(sample),
            # Distance to the nearest *sampled* return, not to the nearest surface.
            "min_sampled_point_distance_m": None if d is None else round(d / UI_PER_M, 4),
            "sampled_points_within_body_radius": int(body[i]),
            "self_exclusion_overlap": blind,
        }
        # A blinded sample's emptiness is an artefact of the exclusion sphere, so
        # it is never counted as observed.
        entry["observation"] = ("unknown_self_excluded" if blind
                                else "observed" if observed[i] else "unobserved")
        entries.append(entry)
    measured = [e for e in entries
                if e["min_sampled_point_distance_m"] is not None
                and not e["self_exclusion_overlap"]]
    unknown = [e["index"] for e in entries if e["observation"] != "observed"]
    nearest = (min(measured, key=lambda e: e["min_sampled_point_distance_m"])
               if measured else None)
    # A retained return inside the body radius is positive evidence of something
    # there, whatever the exclusion sphere deleted around it: the mask removes
    # points, so it can only ever hide obstacles, never invent them. So a
    # body-radius hit counts even on a self-blinded sample. The reverse is not
    # symmetric and must never be relaxed — a blinded sample with no returns is
    # still unknown, not clear (see the `observation` field above).
    contacts = [e["index"] for e in entries
                if e["sampled_points_within_body_radius"] > 0]
    if contacts:
        verdict = "sampled_points_within_body_radius"
    elif nearest is None:
        verdict = "unknown"
    elif unknown:
        verdict = "no_sampled_points_within_body_radius_with_unknown_samples"
    else:
        verdict = "no_sampled_points_within_body_radius"
    return {
        "verdict": verdict,
        "verdict_note": ("describes sampled cloud returns along the segment only; "
                         "no verdict here means clear, safe or collision-free"),
        "body_radius_m": BODY_RADIUS_M,
        "observation_radius_m": OBS_RADIUS_M,
        "samples": len(entries),
        "min_sampled_point_distance_m":
            None if nearest is None else nearest["min_sampled_point_distance_m"],
        "min_sampled_point_distance_at": None if nearest is None else nearest["position"],
        "samples_with_points_inside_body_radius": contacts,
        "unknown_sample_indices": unknown,
        "self_excluded_sample_indices": sorted(blinded),
        "excluded_self_points": int(corridor["excluded_self_points"]),
        "self_exclusion_radius_m": SELF_RADIUS_M,
        "notes": [
            "min_sampled_point_distance_m is the distance to the nearest sampled "
            "return of a downsampled cloud. It is not a clearance lower bound: the "
            "true surface may be nearer, and unsampled or occluded geometry is not "
            "in it at all.",
            "Cloud points within the self-exclusion radius of the measured "
            "fingertip were dropped to avoid reporting the gripper as its own "
            "obstacle. That sphere also deletes real objects, so samples marked "
            "unknown_self_excluded are blind spots, not free space. Masking only "
            "removes returns, so a return that survived inside the body radius "
            "is still reported as a hit even on such a sample; the absence of "
            "returns there remains unknown either way.",
            "Unobserved samples have no sampled returns nearby: unknown, not free.",
        ],
        "per_sample": entries,
    }


def attachment_evidence(reference_summary, measured, state_class):
    """Observations bearing on 'this object moves with the gripper' — never a verdict.

    ``status`` is always ``unknown``. Attachment is a claim about how the scene
    behaves *over time*: it can only be supported by observing the reference and
    the fingertip displace together across observations, which a single frame
    cannot show. A closed state class plus a nearby reference is exactly what an
    object resting untouched beside the closed jaws also looks like, so those cues
    are listed as ``consistent_with`` clues and nothing is upgraded by them.
    """
    out = {"claim": "object_moves_with_gripper", "status": "unknown",
           "consistent_with": [], "missing": [],
           "evidence_required": ("co-displacement of the reference centre and the "
                                 "measured fingertip across two observations"),
           "note": ("single-observation cues only; a closed gripper next to an "
                    "object is indistinguishable here from a held object, so this "
                    "stays unknown by construction")}
    if state_class != "closed":
        out["missing"].append(f"gripper state class is {state_class}, not closed")
    else:
        out["consistent_with"].append("gripper state class is closed")
    if not reference_summary or reference_summary.get("center") is None:
        out["missing"].append("no bound reference with a current centre")
    elif measured.get("status") != "ok":
        out["missing"].append("no measured fingertip to compare against")
    else:
        centre = [reference_summary["center"][k] for k in "xyz"]
        tip = [measured["fingertip_position"][k] for k in "xyz"]
        distance = round(_norm(_sub(centre, tip)), 4)
        out["reference_to_fingertip_m"] = distance
        if distance <= ATTACH_RADIUS_M:
            out["consistent_with"].append(
                f"reference centre {distance:.4f} m from the measured fingertip "
                f"(<= {ATTACH_RADIUS_M:g} m), which proximity alone cannot "
                f"distinguish from an untouched neighbour")
        else:
            out["missing"].append(f"reference centre {distance:.4f} m from the "
                                  f"measured fingertip, beyond {ATTACH_RADIUS_M:g} m")
    return out


def span_verdict(reference_summary, points_robot, opening_axis):
    """Compare a reference's extent along the candidate jaw axis to the jaw span."""
    if not points_robot or not reference_summary or reference_summary["status"] != "ok":
        return {"status": "unknown", "reason": "no valid reference geometry"}
    span = span_along(points_robot, opening_axis)
    verdict = "within_jaw_span" if span is not None and span <= MAX_SPAN_M else "exceeds_jaw_span"
    return {
        "status": "ok",
        "reference_span_along_opening_m": span,
        "jaw_span_m": MAX_SPAN_M,
        "verdict": verdict,
        "note": ("span comparison against sampled cloud returns only; it does not "
                 "predict contact, force or a successful grasp"),
    }


def pose_delta(candidate, measured):
    """Separate the virtual candidate target from the measured end effector."""
    out = {"candidate_frame": "robot", "measured_status": measured.get("status")}
    if measured.get("status") != "ok":
        out["reason"] = measured.get("reason", "measured end effector unavailable")
        return out
    tip = [measured["fingertip_position"][k] for k in "xyz"]
    out["translation_m"] = _round3(_sub(candidate["position"], tip))
    out["distance_m"] = round(_norm(_sub(candidate["position"], tip)), 4)
    if measured.get("approach"):
        a = [measured["approach"][k] for k in "xyz"]
        cos = max(-1.0, min(1.0, _dot(a, candidate["approach"])))
        out["approach_angle_deg"] = round(math.degrees(math.acos(cos)), 2)
    else:
        out["approach_angle_deg"] = None
    return out


# Gates for "the waypoint that ran is the one this prediction was about". Sized to
# the round-trip precision of the two things being compared, and no wider: a
# looser gate accepts a *different* nearby intent as the predicted one, which is
# exactly the confusion the check exists to prevent.
#
# Position: the prediction stores the candidate rounded to 4 decimals (metres) and
# gripper_pose reports the executed target through ui_to_robot, also rounded to 4
# decimals. Each component can therefore differ by up to 1e-4 for one identical
# pose, so the worst-case norm is sqrt(3) * 1e-4 = 1.74e-4 m. 2e-4 clears that and
# nothing else: a 1 mm difference in intent is five times the gate.
# Orientation: both sides are unit vectors — unrounded from the page, rounded to 9
# decimals in the record — so the round-trip angle is below 1e-6 degrees. 0.01
# degrees is far above the noise and far below any deliberate reorientation.
TARGET_MATCH_M = 0.0002
TARGET_MATCH_DEG = 0.01


def prediction_record(*, prediction_id, observation_version_id, candidate, corridor,
                      attachment, reference_id, execute_call_ordinal):
    """The falsifiable part of a proxy reply, stored so execution can check it.

    Recorded for every candidate the proxy actually evaluated, including one with
    no reference: the predicted endpoint is checkable on its own, and a candidate
    that left no record could never be shown to have been wrong.

    Orientation is part of the prediction, not decoration. Without it, a waypoint
    that reached the right point facing elsewhere would count as a hit.
    ``execute_call_ordinal`` is how many execute_waypoint calls this server had
    already seen when this was written — calls, not completions, because a call
    that completes nothing still advances it. It only distinguishes "the very next
    execution attempt" from "some later one"; it is never a count of finished
    waypoints. ``observation_version`` names the frame the candidate was evaluated
    against, so a check can require the execution to have started from that frame.
    """
    return {
        "prediction_id": prediction_id,
        "observation_version": observation_version_id,
        "reference_id": reference_id,
        "predicted_fingertip_position": _round3(candidate["position"]),
        "predicted_approach": _round3(candidate["approach"], 9),
        "predicted_opening": _round3(candidate["opening"], 9),
        "min_sampled_point_distance_m": corridor["min_sampled_point_distance_m"],
        "corridor_verdict": corridor["verdict"],
        "object_follows_gripper": (attachment or {}).get("status", "unknown"),
        "execute_call_ordinal_at_prediction": execute_call_ordinal,
        "checked": False,
        "endpoint_checked": False,
        # Filled in by the execution hook: the one check of this prediction that
        # read the target that actually ran. Refresh may cite it, never redo it.
        "execution_check": None,
    }


def _direction_angle_deg(a, b):
    """Angle between two direction dicts/lists, or None when either is missing."""
    if not a or not b:
        return None
    va = [float(a[k]) for k in "xyz"] if isinstance(a, dict) else [float(x) for x in a]
    vb = [float(b[k]) for k in "xyz"] if isinstance(b, dict) else [float(x) for x in b]
    na, nb = _norm(va), _norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return None
    cos = max(-1.0, min(1.0, _dot(va, vb) / (na * nb)))
    return round(math.degrees(math.acos(cos)), 3)


def execution_check(prediction, *, executed_target, measured,
                    execute_calls_seen_before, observation_paired,
                    version_before, before_paired, version_after, done_delta):
    """Check one prediction against the waypoint that actually ran.

    Every way this can fail to be a test of the prediction is named, because the
    alternative is arithmetic on mismatched things: subtracting the measured
    endpoint of a *different* waypoint from a predicted one produces a number that
    looks like an error and means nothing. The order below is deliberate — a check
    is refused on the first ground that makes the comparison meaningless.

    Frame identity carries the weight the local ordinal cannot. ``version_before``
    must be a paired frame equal to the one the prediction was written against —
    otherwise the scene the candidate was evaluated in is not the scene the
    execution started from, however the local counter looks. ``version_after`` must
    be a strictly later frame of the same epoch, and ``done_delta`` must be exactly
    one: zero means nothing finished, more than one means this frame cannot be
    attributed to the predicted waypoint alone.

    Every refusal is ``not_verifiable``. None of these are errors — not knowing
    whether a prediction held is an ordinary outcome, and reporting it as a failure
    would be a claim of its own.
    """
    out = {"prediction_id": prediction["prediction_id"],
           "reference_id": prediction["reference_id"],
           "predicted_fingertip_position": prediction["predicted_fingertip_position"],
           "predicted_approach": prediction.get("predicted_approach"),
           "prediction_observation_version": prediction.get("observation_version"),
           "observation_version_before_execution": version_before,
           "observation_version_after_execution": version_after,
           # Proximity plus a closed gripper is not attachment evidence in one
           # frame, and reaching the predicted endpoint does not make it any. This
           # stays unknown no matter how well the endpoint matches.
           "object_follows_gripper": "unknown",
           "object_follows_gripper_note": ("still unknown: an endpoint match says "
                                           "nothing about whether anything is held"),
           }

    def refuse(reason, **extra):
        out.update(status="not_verifiable", reason=reason, **extra)
        return out

    if prediction["execute_call_ordinal_at_prediction"] != execute_calls_seen_before:
        return refuse("another execute_waypoint call ran between the prediction and "
                      "this one, so this is not the execution it described",
                      execute_calls_seen_at_prediction=
                          prediction["execute_call_ordinal_at_prediction"],
                      execute_calls_seen_before_this_execution=
                          execute_calls_seen_before)
    if not before_paired or version_before is None:
        return refuse("no internally paired observation before the execution, so "
                      "the frame it started from cannot be identified")
    if version_before != prediction.get("observation_version"):
        # The local call ordinal cannot see this: an outside frame can arrive
        # between the proxy call and the execution without either the counter or
        # the completion count moving. The candidate was evaluated against a scene
        # that is no longer the scene the controller started from.
        return refuse("the execution started from a different observation than the "
                      "one this prediction was evaluated against")
    if executed_target is None:
        return refuse("the target that was executed could not be read, so it "
                      "cannot be matched against the prediction")
    # Compared unrounded, reported rounded: rounding before the comparison would
    # widen the gate by half a display digit.
    gap = _norm(_sub([float(executed_target["position"][k]) for k in "xyz"],
                     [prediction["predicted_fingertip_position"][k] for k in "xyz"]))
    position_gap = round(gap, 6)
    approach_gap = _direction_angle_deg(executed_target.get("approach"),
                                       prediction.get("predicted_approach"))
    opening_gap = _direction_angle_deg(executed_target.get("opening"),
                                       prediction.get("predicted_opening"))
    mismatched = (gap > TARGET_MATCH_M
                  or approach_gap is None or approach_gap > TARGET_MATCH_DEG
                  or opening_gap is None or opening_gap > TARGET_MATCH_DEG)
    if mismatched:
        return refuse("a different target was executed than the candidate this "
                      "prediction was about",
                      executed_target_position=executed_target["position"],
                      target_position_gap_m=position_gap,
                      target_approach_gap_deg=approach_gap,
                      target_opening_gap_deg=opening_gap)
    out["executed_the_predicted_target"] = True
    if done_delta is None:
        return refuse("the waypoint completion count could not be read across this "
                      "execution, so it is unknown whether one waypoint finished")
    out["waypoint_done_count_delta"] = done_delta
    if done_delta != 1:
        return refuse("exactly one completed waypoint has to separate the two "
                      "observations for this endpoint to be the predicted "
                      "waypoint's; the count moved by "
                      f"{done_delta}")
    if not observation_paired:
        return refuse("no internally paired observation after the execution, so "
                      "the endpoint could not be measured against it")
    advance = frame_advance(version_before, version_after)
    if advance is not None:
        # A repeated version would mean the endpoint is being read off the frame
        # that preceded the motion; a different epoch means the episode restarted
        # and the two states are not comparable at all.
        return refuse("the observation after the execution is not a later frame of "
                      "the same episode, so it cannot show where this waypoint "
                      "ended", observation_advance_problem=advance)
    if (measured or {}).get("status") != "ok":
        return refuse("no end-effector telemetry after the execution")
    predicted = [prediction["predicted_fingertip_position"][k] for k in "xyz"]
    actual = [measured["fingertip_position"][k] for k in "xyz"]
    out.update(
        status="measured",
        observed_fingertip_position=measured["fingertip_position"],
        position_error_m=round(_norm(_sub(actual, predicted)), 4),
        per_axis_error_m=_round3(_sub(actual, predicted)),
        approach_error_deg=_direction_angle_deg(measured.get("approach"),
                                                prediction.get("predicted_approach")),
        note=("measured endpoint of the waypoint this prediction described; it "
              "says nothing about clearance along the way, which was sampled, or "
              "about contact"))
    return out


def region_comparison(*, center_before, center_after):
    """How far the *neighbourhood centre* moved between two local extractions.

    Deliberately not a prediction check and deliberately not about an object. Two
    extractions near the same place in two frames can summarize different things —
    a neighbour of similar size, or another patch of one surface — so the identity
    of what is being compared is unknown and the number is offered as a region
    observation only.

    An endpoint comparison is not made here. The only check that can compare a
    predicted endpoint against reality is ``execution_check``, because it is the
    only one that reads the target the controller actually received; computing an
    endpoint error from a later observation alone would produce a second, unfounded
    figure for the same question.
    """
    if center_before is None or center_after is None:
        return {"status": "not_verifiable",
                "identity": "unknown",
                "reason": "a neighbourhood centre is missing on one of the two sides"}
    before = [center_before[k] for k in "xyz"]
    after = [center_after[k] for k in "xyz"]
    return {
        "status": "measured",
        "identity": "unknown",
        "region_center_before": _round3(before),
        "region_center_after": _round3(after),
        "region_center_shift_m": round(_norm(_sub(after, before)), 4),
        "note": ("displacement between two local extractions near the same place; "
                 "it does not establish attachment, does not prove the same "
                 "referent was re-found, and is not an endpoint error"),
    }

"""MCP tools for VIA_CONTROL_INTERFACE=coarse_fine_policy.

Three tools plus end_episode: ``cf_look``, ``cf_policy``, ``cf_state``. The model
authors the policy; this file is the browser and execution edge that runs it and
returns a fresh fine observation.

What it deliberately does NOT contain: any named skill (no approach_grasp, no
align_over, no descend_to), any candidate ranking, any recovery, any retry. A
program either validates and runs as written, or is refused with the evidence.

Execution reuses ``fast_geometry_tools``' primitives, so a waypoint from here is
physically the same kind of waypoint as every earlier interface's: a validated
``target_edit`` applied in the browser, then the record button. The one thing not
reused is the rotation splitter — see ``coarse_fine_policy.orientation_steps``.
"""

from __future__ import annotations

import base64
import json
import os
import time

import httpx
import mcp.types as types
import numpy as np

try:
    from . import coarse_fine_policy as cfp
    from . import fast_geometry as fg
    from . import fast_geometry_tools as fgt
    from . import geometry_ref as geom
    from .utils import deproject
except ImportError:  # direct import
    import coarse_fine_policy as cfp
    import fast_geometry as fg
    import fast_geometry_tools as fgt
    import geometry_ref as geom
    from utils import deproject


LOG_VARIABLE = "VIA_COARSE_FINE_LOG_FILE"

# record_sim's broadcast server, same derivation mcp_server uses. Kept local
# rather than imported from mcp_server, which imports this module.
_BASE_PORT = int(os.environ.get("SPHINX_BASE_PORT", "8100"))
SIM_HTTP_BASE = f"http://localhost:{_BASE_PORT + 1}"
SIM_HTTP_TIMEOUT_S = 5.0  # the depth payload is ~270 kB, not a small sidecar


async def fetch_sim_json(path: str) -> dict | None:
    """Best-effort GET of a JSON sidecar from record_sim's broadcast server."""
    try:
        async with httpx.AsyncClient(timeout=SIM_HTTP_TIMEOUT_S) as client:
            resp = await client.get(f"{SIM_HTTP_BASE}/{path}")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


def log_event(event: dict) -> None:
    """Append one JSONL event to the interface's own log, if configured.

    Best-effort, and a separate variable from the fast interface's: an exported
    path must not redirect a different interface's logging. The detailed evidence
    goes here; the model's reply stays bounded.
    """
    path = os.environ.get(LOG_VARIABLE)
    if not path:
        return
    event.setdefault("t", time.time())
    try:
        with open(path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass


class CoarseFineState:
    """Proxies and per-episode accounting. Module-level singleton.

    Physical actions and the waypoint count live in ``fast_geometry_tools.STATE``,
    because that is where the primitives executing them keep their books. This
    class deliberately keeps NO second copy: two counters incremented on different
    paths (one by the pose primitive, one by the dwell) reported whichever the
    reply happened to read, and under-reported real motion. ``waypoints`` and
    ``committed`` here are single-source views of that one set of books.
    """

    def __init__(self) -> None:
        self.proxies: dict[str, dict] = {}
        self.policy_calls = 0
        self.sim_steps_held = 0
        self.last_observation: dict | None = None

    @property
    def waypoints(self) -> int:
        return fgt.STATE.waypoints

    @property
    def committed(self) -> list[dict]:
        return fgt.STATE.committed

    def reset(self) -> None:
        self.__init__()


STATE = CoarseFineState()


def reset_episode() -> None:
    STATE.reset()
    # The execution primitives live in fast_geometry_tools and keep their own
    # per-episode singleton (budget accounting, committed actions). Reset it too,
    # or a second episode in one server would inherit the first one's counters.
    fgt.reset_episode()


# ── the fine observation ─────────────────────────────────────────────────────
#
# Real wrist depth, not a crop of the global cloud. SimEnv.get_point_cloud
# excludes eye_in_hand, so a "wrist region" taken from that cloud is a
# reprojection of the third-person cameras into a wrist-shaped window — the same
# measurement at the same resolution, relabelled. The wrist camera's own depth
# comes from record_sim's /cam/wrist_depth.json, which serves the flip- and
# rotation-matched array with the K/E of the SAME observation.

WRIST_DEPTH_PATH = "cam/wrist_depth.json"
MAX_DEPTH_SAMPLES = 12    # metric samples returned to the model
DEPTH_VALID_MIN_M = 0.02  # nearer than this is the camera housing, not the scene
DEPTH_VALID_MAX_M = 3.0


async def wrist_depth_cloud(ctx):
    """Wrist-camera depth reprojected into the robot frame. (points, meta).

    ``points`` is an (N, 3) array in robot coordinates or None. ``meta`` always
    says which frame the depth belonged to and why it is unusable if it is, so a
    caller can refuse rather than silently fall back to the global cloud.
    """
    payload = await fetch_sim_json(WRIST_DEPTH_PATH)
    if not payload:
        return None, {"status": "unavailable",
                      "reason": "record_sim served no wrist depth"}
    if payload.get("status") != "ok":
        return None, {"status": "unavailable",
                      "reason": payload.get("reason") or "wrist depth unavailable"}
    try:
        raw = base64.b64decode(payload["depth_b64"])
        height, width = int(payload["height"]), int(payload["width"])
        depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        K = np.asarray(payload["K"], dtype=np.float64).reshape(3, 3)
        E = np.asarray(payload["E"], dtype=np.float64).reshape(4, 4)
    except Exception as exc:
        return None, {"status": "unavailable",
                      "reason": f"the wrist depth payload could not be decoded: {exc}"}
    meta = {"status": "ok", "source": "robot0_eye_in_hand_depth",
            "depth_frame": f"{payload.get('epoch')}#{payload.get('seq')}",
            "depth_epoch": payload.get("epoch"), "depth_seq": payload.get("seq"),
            "camera": payload.get("camera"), "height": height, "width": width}
    # base_units=0 because the served array is already in metres.
    points = deproject(depth, K, E, base_units=0)
    finite = np.isfinite(points).all(axis=1)
    flat = depth.reshape(-1)
    keep = finite & (flat > DEPTH_VALID_MIN_M) & (flat < DEPTH_VALID_MAX_M)
    meta["returns_total"] = int(flat.size)
    meta["returns_valid"] = int(keep.sum())
    return points[keep], meta


def depth_samples(points, center, radius_m):
    """Bounded metric samples of the wrist cloud around a robot-frame centre.

    Returns the sample list plus its support and residual, so the model can see
    how much evidence the numbers rest on rather than only the numbers. Sampling
    is a fixed stride through the in-radius returns, not the nearest few: the
    nearest few of a noisy patch all agree with each other.
    """
    if points is None or len(points) == 0:
        return {"status": "unknown", "reason": "no wrist depth returns"}
    center = np.asarray(center, dtype=np.float64)
    delta = points - center
    distance = np.linalg.norm(delta, axis=1)
    inside = points[distance <= radius_m]
    if len(inside) == 0:
        return {"status": "unknown", "reason": "no wrist depth returns within the "
                                              "requested radius",
                "radius_m": radius_m, "support_points": 0,
                "nearest_return_m": round(float(distance.min()), 4)}
    stride = max(1, len(inside) // MAX_DEPTH_SAMPLES)
    sampled = inside[::stride][:MAX_DEPTH_SAMPLES]
    centroid = inside.mean(axis=0)
    residual = float(np.linalg.norm(inside - centroid, axis=1).std())
    return {
        "status": "ok",
        "source": "wrist_depth_reprojected",
        "frame": "robot",
        "radius_m": radius_m,
        "support_points": int(len(inside)),
        "sample_stride": int(stride),
        "residual_rms_m": round(residual, 5),
        "centroid": [round(float(v), 5) for v in centroid],
        "z_min": round(float(inside[:, 2].min()), 5),
        "z_max": round(float(inside[:, 2].max()), 5),
        "nearest_return_m": round(float(np.linalg.norm(inside - center,
                                                       axis=1).min()), 5),
        "samples_xyz": [[round(float(v), 5) for v in p] for p in sampled],
    }


async def fine_observation(ctx, *, radius_m, target=None, what="observation"):
    """The paired fine observation every cf_policy motion returns.

    One reading, and everything in it comes from that reading: the frame id, the
    robot's measured pose, the real wrist depth with its own frame id, the local
    metric samples taken from that depth, and the actual-versus-commanded pose. A
    caller cannot substitute an older global cloud for the depth, because the depth
    carries the frame it was measured on and this reply reports both.
    """
    frame = await fgt.read_frame(ctx, cam_labels=("wrist",))
    usable, reason = fgt.frame_usable(frame)
    measured = frame["measured"]
    out = {
        "what": what,
        "frame": frame["version"],
        "cloud_frame": frame["cloud_version"],
        "paired": usable,
        "measured_end_effector": measured,
        "measured_gripper": measured.get("gripper_state_class") if measured else None,
        "sim_steps_used": None,
    }
    if not usable:
        out["pairing"] = frame["pairing"]
        out["note"] = (f"this frame is not usable as geometry ({reason}); the depth "
                       f"sample below is reported with its own frame id so you can "
                       f"see whether it belongs to this moment")
    env = await fetch_sim_json("env.json")
    if env:
        out["sim_steps_used"] = env.get("sim_steps_used")
        out["sim_steps_budget"] = env.get("sim_steps_budget")
    points, meta = await wrist_depth_cloud(ctx)
    out["wrist_depth"] = meta
    if points is not None and measured and measured.get("status") == "ok":
        center = [measured["fingertip_position"][k] for k in "xyz"]
        out["local_depth"] = depth_samples(points, center, radius_m)
        out["local_depth"]["at"] = [round(v, 5) for v in center]
        # The depth array and the telemetry must be the same moment. They are
        # numbered independently, so compare the producer's (epoch, seq) rather
        # than the two strings: the depth payload carries `epoch#seq` and the frame
        # id is `e<epoch>-s<seq>-c<cloud_seq>`, so a string comparison of the two is
        # false even when they ARE the same observation.
        parsed = geom.parse_observation_version(frame["version"])
        out["depth_paired_with_frame"] = (
            None if parsed is None or meta.get("depth_seq") is None
            else (str(meta.get("depth_epoch")) == str(parsed["epoch"])
                  and int(meta["depth_seq"]) == int(parsed["seq"])))
    elif points is not None:
        out["local_depth"] = {"status": "unknown",
                              "reason": "no measured end effector to sample around"}
    else:
        out["local_depth"] = {"status": "unknown",
                              "reason": meta.get("reason")}
    if target is not None:
        out["commanded_pose"] = {"position": target["position"],
                                 "approach": target["approach"],
                                 "opening": target["opening"],
                                 "tolerance_m": target["tolerance_m"]}
        out["pose_error"] = cfp.pose_error(target, measured)
    STATE.last_observation = out
    return out


# ── execution ────────────────────────────────────────────────────────────────

MAX_EDITS_PER_STEP = 12


async def apply_target(ctx, target):
    """Walk the virtual target onto the resolved pose. Browser-only, no waypoint.

    Same primitive as every other interface (``target_edit``'s compare-and-set),
    but the orientation is split by ``coarse_fine_policy.orientation_steps`` rather
    than ``fast_geometry``'s: that one interpolates the two axes independently and
    re-orthogonalizes, which can emit a step whose measured angle exceeds the
    editor's 90 deg cap and be refused after earlier waypoints have already run.
    """
    edits = 0
    pose = await ctx.gripper_pose()
    if pose is None:
        raise fgt.Aborted("target_unavailable",
                          "the blue target gripper is not selected, so no pose "
                          "can be set")
    current = ([pose["robot_approach"][k] for k in "xyz"],
               [pose["robot_opening"][k] for k in "xyz"])
    steps = cfp.orientation_steps(current, (target["approach"], target["opening"]))
    if steps is None:
        raise fgt.Aborted("orientation_unusable",
                          "the current or requested gripper axes are not a usable "
                          "orthogonal pair, so nothing was changed")
    for step in steps:
        await fgt._one_edit(ctx, {"frame": "robot", "approach": step["approach"],
                                  "opening": step["opening"]})
        edits += 1
    for _ in range(MAX_EDITS_PER_STEP):
        pose = await ctx.gripper_pose()
        if pose is None:
            raise fgt.Aborted("target_unavailable",
                              "the blue target gripper is not selected")
        at = [pose["robot_position"][k] for k in "xyz"]
        position_steps = fg.edit_steps(at, target["position"])
        if not position_steps:
            break
        await fgt._one_edit(ctx, {"frame": "robot", "position": position_steps[0]})
        edits += 1
        if len(position_steps) == 1:
            break
    else:
        raise fgt.Aborted("target_not_reachable_by_editing",
                          f"the virtual target did not reach the commanded pose "
                          f"within {MAX_EDITS_PER_STEP} edits", edits=edits)
    return edits


HOLD_SETTLE_S = 0.2
HOLD_TIMEOUT_S = 60.0


async def run_dwell(ctx, budget, sim_steps):
    """Advance the sim in place for ``sim_steps``, and report the ACTUAL delta.

    Not a wait and not a re-command of the current pose: record_sim's hold path
    steps the simulator with a zero pose delta and the current gripper command, so
    a closed grasp is really held against gravity and contact. What is reported is
    the measured `sim_steps_used` delta from /env.json, never the requested count —
    a hold cut short by the step budget or a terminal state must say so.
    """
    budget.spend(f"a {sim_steps}-step hold")
    before = await fetch_sim_json("env.json")
    started = None if not before else before.get("sim_steps_used")
    action = fgt.STATE.commit({"action": "hold", "kind": "dwell",
                               "sim_steps_requested": sim_steps,
                               "submitted": True, "completed": None})
    prev = await ctx.page.evaluate("() => window.__waypointDoneCount || 0")
    sent = await ctx.page.evaluate(
        "n => window.__cfSendHold ? window.__cfSendHold(n) : "
        "({ok: false, reason: 'the hold path is not available in this UI'})",
        int(sim_steps))
    if not sent or not sent.get("ok"):
        action["completed"] = False
        raise fgt.Aborted("hold_not_submitted",
                          f"the hold was not submitted: "
                          f"{(sent or {}).get('reason')}", action=action)
    # The same counter execute_once increments, not a second one: a hold is a
    # physically executed waypoint and must be accounted for in one place.
    fgt.STATE.waypoints += 1
    try:
        await ctx.page.wait_for_function(
            "prev => (window.__waypointDoneCount || 0) > prev", arg=prev,
            timeout=int(min(HOLD_TIMEOUT_S, max(1.0, budget.remaining_s())) * 1000))
        action["completed"] = True
    except Exception as exc:
        action["completed"] = None
        action["completion_note"] = str(exc)[:200]
        raise fgt.Aborted("hold_completion_unknown",
                          f"a {sim_steps}-step hold was submitted but no completion "
                          f"signal arrived, so how long the sim actually advanced is "
                          f"unknown; do not re-submit it", action=action)
    await ctx.page.wait_for_timeout(int(HOLD_SETTLE_S * 1000))
    after = await fetch_sim_json("env.json")
    ended = None if not after else after.get("sim_steps_used")
    used = (None if started is None or ended is None else int(ended) - int(started))
    if used is not None:
        STATE.sim_steps_held += used
    action["sim_steps_used"] = used
    return {"sim_steps_requested": sim_steps, "sim_steps_used": used,
            "sim_steps_total_after": ended,
            "note": ("sim_steps_used is the measured step delta, not the request; a "
                     "shorter one means the episode's step budget or a terminal "
                     "state ended the hold early")
            if used != sim_steps else None}


async def run_program(ctx, program, budget, records=None):
    """Execute a validated program step by step. Returns the per-step records.

    Fail-closed at every boundary: a step is resolved against the frame measured
    immediately before it, and a resolution that cannot be made refuses instead of
    substituting anything. No step is retried and no correction is inserted — if a
    pose lands outside its tolerance, that is reported and the program stops,
    because the model authored the next step on the assumption that this one
    arrived.

    ``records`` may be supplied by the caller, and then it is appended to in place.
    That is how an abort keeps its evidence: an ``Aborted`` raised from here unwinds
    this frame, so a records list owned only by this function would be lost exactly
    in the case where the model most needs to see which steps ran and what they
    measured.
    """
    if records is None:
        records = []
    for index, step in enumerate(program["steps"]):
        kind = step["kind"]
        if kind == "observe":
            observation = await fine_observation(
                ctx, radius_m=step["radius_m"], what="observe")
            records.append({"step": index, "kind": "observe", "physical": False,
                            "observation": observation})
            continue
        if kind == "gripper":
            entry = await fgt.set_gripper(ctx, budget, step["action"] == "open")
            observation = await fine_observation(
                ctx, radius_m=cfp.DEFAULT_OBSERVE_RADIUS_M,
                what=f"after {step['action']}")
            records.append({"step": index, "kind": "gripper", "physical": True,
                            "requested": step["action"],
                            "already": entry.get("already", False),
                            "observation": observation})
            continue
        if kind == "dwell":
            entry = await run_dwell(ctx, budget, step["sim_steps"])
            observation = await fine_observation(
                ctx, radius_m=cfp.DEFAULT_OBSERVE_RADIUS_M, what="after dwell")
            records.append({"step": index, "kind": "dwell", "physical": True,
                            **entry, "observation": observation})
            continue

        # A pose step. Resolve against the frame measured NOW, not against the
        # frame the program was authored on.
        frame = await fgt.read_frame(ctx)
        proxy = STATE.proxies.get(step["proxy"]) if step["proxy"] else None
        target = cfp.resolve_pose(step, measured=frame["measured"], proxy=proxy)
        if target["status"] != "ok":
            records.append({"step": index, "kind": "pose", "physical": False,
                            "status": "refused", "reason": target["reason"],
                            "detail": target.get("detail"),
                            "frame": frame["version"]})
            raise fgt.Aborted("pose_not_resolvable",
                              f"step {index} could not be resolved against the "
                              f"current measurement ({target['reason']}), so it was "
                              f"not executed", step=index, resolved=target)
        record = {"step": index, "kind": "pose", "physical": True,
                  "resolved_from_frame": frame["version"], "target": target}
        records.append(record)
        record["edits"] = await apply_target(ctx, target)
        await fgt.execute_once(
            ctx, budget,
            what=f"moving to the pose of step {index}",
            detail={"kind": "pose", "step": index, "target": target})
        observation = await fine_observation(
            ctx, radius_m=cfp.DEFAULT_OBSERVE_RADIUS_M, target=target,
            what=f"after step {index}")
        record["observation"] = observation
        error = observation.get("pose_error") or {}
        record["within_tolerance"] = error.get("within_tolerance")
        # Only an explicit True continues. False is a miss and None is "arrival
        # could not be verified" — neither licenses a later step, which may be a
        # close, that was written assuming this one landed. Continuing on None was
        # the failure this branch exists to prevent: a reading with no measured
        # orientation, or none at all, would have read as arrival.
        if record["within_tolerance"] is not True:
            raise fgt.Aborted(
                "pose_not_verified_within_tolerance",
                f"step {index} did not verify as arrived "
                f"({_arrival_detail(error, target)}); the program stopped rather "
                f"than running a later step that assumed this one arrived",
                step=index, pose_error=error or {"status": "unknown"})
    return records


def _arrival_detail(error, target):
    """One clause saying which part of arrival failed or could not be checked."""
    if not error or error.get("status") != "ok":
        return (f"no measured end effector to compare against "
                f"({(error or {}).get('reason')})")
    parts = []
    if error.get("position_within_tolerance") is False:
        parts.append(f"position off by {error['position_error_m']:.4f} m, tolerance "
                     f"{target['tolerance_m']:g} m")
    if error.get("orientation_within_tolerance") is False:
        parts.append(f"orientation off by {error['orientation_error_deg']:.2f} deg, "
                     f"tolerance {target['orientation_tolerance_deg']:g} deg")
    if error.get("orientation_within_tolerance") is None:
        parts.append("orientation could not be measured, so arrival is unknown "
                     "rather than reached")
    return "; ".join(parts) or "reason unavailable"


# ── proxies ──────────────────────────────────────────────────────────────────

async def bind_proxy(ctx, request):
    """Fit one region into a geometric proxy and store it under its name.

    Deliberately thin: the fit is ``fast_geometry``'s, unchanged. What this adds is
    only that the result is stored under a model-chosen NAME, because a program's
    steps name their frame before the bind has run. No candidate list is derived
    from it and no grasp is proposed — the proxy is a centre, an up axis and a size,
    and the policy decides what to do with them.
    """
    frame = await fgt.read_frame(ctx)
    usable, reason = fgt.frame_usable(frame)
    if not usable:
        return {"status": "unknown", "reason": reason, "frame": frame["version"],
                "pairing": frame["pairing"],
                "message": ("this frame's cloud and telemetry were not measured "
                            "together, so nothing was bound and no step ran")}
    await fgt.clear_overlay(ctx)
    cards, _, error = await fgt.bind_objects(ctx, request, frame)
    if error:
        return {"status": "unknown", "reason": "occluded", "message": error}
    after = geom.observation_version(await ctx.observe_geometry())
    if after != frame["version"]:
        return {"status": "unknown", "reason": "no_paired_frame",
                "frame": frame["version"], "frame_after": after,
                "message": ("a new frame arrived while the region was being "
                            "extracted, so the proxy would mix two observations; "
                            "nothing was bound")}
    card = cards[0]
    name = card.get("name") or "proxy"
    card["bound_frame"] = frame["version"]
    STATE.proxies[name] = card
    view = fgt.compact_card(card)
    view["proxy"] = name
    view["frame_name"] = f"{cfp.PROXY_PREFIX}{name}"
    if not card.get("valid"):
        view["message"] = ("the proxy did not fit, so any step naming it is "
                           "refused; re-draw the region")
    return {"status": "ok", "frame": frame["version"], "proxy": view}


# ── what the model sees ──────────────────────────────────────────────────────
#
# The log gets every sample and every intermediate; the model gets what the next
# decision turns on. They differ on purpose — a full fine observation is ~12 xyz
# samples per step, and a 12-step program of those is most of the reply.

OBSERVATION_FIELDS = ("what", "frame", "cloud_frame", "paired", "sim_steps_used",
                      "sim_steps_budget", "measured_end_effector",
                      "measured_gripper", "local_depth", "wrist_depth",
                      "depth_paired_with_frame", "commanded_pose", "pose_error",
                      "pairing", "note")
DEPTH_KEEP = ("status", "reason", "radius_m", "support_points", "residual_rms_m",
              "centroid", "z_min", "z_max", "nearest_return_m", "at",
              "samples_xyz", "source")
WRIST_KEEP = ("status", "reason", "depth_frame", "returns_valid", "returns_total",
              "camera")


def observation_view(observation):
    if not isinstance(observation, dict):
        return observation
    out = {k: observation[k] for k in OBSERVATION_FIELDS if k in observation}
    if isinstance(out.get("local_depth"), dict):
        out["local_depth"] = {k: v for k, v in out["local_depth"].items()
                              if k in DEPTH_KEEP}
    if isinstance(out.get("wrist_depth"), dict):
        out["wrist_depth"] = {k: v for k, v in out["wrist_depth"].items()
                              if k in WRIST_KEEP}
    return out


def model_view(payload):
    out = dict(payload)
    if isinstance(out.get("steps"), list):
        out["steps"] = [dict(s, observation=observation_view(s["observation"]))
                        if isinstance(s.get("observation"), dict) else s
                        for s in out["steps"]]
    if isinstance(out.get("observation"), dict):
        out["observation"] = observation_view(out["observation"])
    return out


# ── tools ────────────────────────────────────────────────────────────────────

CF_PREAMBLE = (
    "Coordinates are the robot frame in metres: +x away from the base, +y left, "
    "+z up. Regions are fractions of the surface you are looking at, the same "
    "(u, v) convention as a screenshot: u across from 0.0 left, v down from 0.0 "
    "top.\n\n"
)


class CoarseFineTool:
    """Base for the coarse-to-fine tools. mcp_server adapts these; not a ToolBase."""

    name: str
    description: str
    input_schema: dict

    async def reply(self, ctx, payload, *, images=None):
        payload["timing_id"] = ctx.timing_id
        log_event({"tool": self.name, "payload": payload})
        view = model_view(payload)
        blocks = [types.TextContent(type="text", text=json.dumps(view))]
        return blocks + list(images or [])

    async def snap(self, ctx, text=None):
        try:
            return await ctx.snap(text) if text else await ctx.snap()
        except Exception:
            return []


class CfLookTool(CoarseFineTool):
    name = "cf_look"
    description = (
        CF_PREAMBLE +
        "Look at the scene: one screenshot, the id of the frame it belongs to, and "
        "the same fine observation cf_policy returns after a motion — the measured "
        "gripper pose, the wrist camera's own depth returns near the fingertips, and "
        "whether that depth belongs to this frame.\n\n"
        "It moves nothing and records no waypoint. Use it to see where things are "
        "before writing a program, and to re-read the fine observation without "
        "spending a step.\n\n"
        "'paired: true' is what makes a frame usable as geometry. During motion the "
        "producer streams frames that reuse an older cloud; those are reported "
        "unpaired. The depth carries its own frame id, so you can always see whether "
        "the depth and the telemetry are the same moment."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "radius_m": {
                "type": "number",
                "description": ("Radius around the fingertips to sample the wrist "
                                f"depth in. Default {cfp.DEFAULT_OBSERVE_RADIUS_M:g}, "
                                f"max {cfp.MAX_OBSERVE_RADIUS_M:g}."),
            },
        },
        "additionalProperties": False,
    }

    async def __call__(self, ctx, arguments):
        radius = arguments.get("radius_m", cfp.DEFAULT_OBSERVE_RADIUS_M)
        try:
            radius = float(radius)
        except (TypeError, ValueError):
            return await self.reply(ctx, {"status": "rejected",
                                         "message": "radius_m must be a number"})
        if not 0.005 <= radius <= cfp.MAX_OBSERVE_RADIUS_M:
            return await self.reply(ctx, {
                "status": "rejected",
                "message": (f"radius_m must be within [0.005, "
                            f"{cfp.MAX_OBSERVE_RADIUS_M:g}]")})
        await fgt.clear_overlay(ctx)
        observation = await fine_observation(ctx, radius_m=radius, what="cf_look")
        return await self.reply(ctx, {
            "status": "ok", "observation": observation,
            "proxies": sorted(STATE.proxies),
            "waypoints_executed": STATE.waypoints,
        }, images=await self.snap(ctx))


POSE_SCHEMA = {
    "type": "object",
    "description": (
        "Where to put the gripper. Position OR offset, never both. "
        "frame 'robot' is absolute; frame 'gripper' takes an offset in the "
        "gripper's own axes [along approach, along opening, along the third]; "
        "frame 'proxy:<name>' is relative to that proxy's centre. Orientation is "
        "either explicit approach+opening axes, or azimuth_deg/tilt_deg about a "
        "proxy's up axis (continuous, any value), or omitted to keep the current "
        "orientation."),
    "properties": {
        "frame": {"type": "string",
                  "description": "'robot', 'gripper' or 'proxy:<name>'."},
        "position": {"type": "array", "items": {"type": "number"},
                     "minItems": 3, "maxItems": 3},
        "offset": {"type": "array", "items": {"type": "number"},
                   "minItems": 3, "maxItems": 3},
        "approach": {"type": "array", "items": {"type": "number"},
                     "minItems": 3, "maxItems": 3,
                     "description": "Direction the fingers point. With opening."},
        "opening": {"type": "array", "items": {"type": "number"},
                    "minItems": 3, "maxItems": 3,
                    "description": "Direction the jaws separate along. With approach."},
        "azimuth_deg": {"type": "number",
                        "description": ("Rotation about the proxy's up axis. Any "
                                        "value; not chosen from a menu.")},
        "tilt_deg": {"type": "number",
                     "description": ("Tilt away from straight down along the "
                                     "proxy's up axis. 0 approaches along -up.")},
        "tolerance_m": {"type": "number",
                        "description": (f"How close counts as arrived. Default "
                                        f"{cfp.DEFAULT_TOLERANCE_M:g}, within "
                                        f"[{cfp.MIN_TOLERANCE_M:g}, "
                                        f"{cfp.MAX_TOLERANCE_M:g}]. The program "
                                        f"stops if a step misses it.")},
        "orientation_tolerance_deg": {
            "type": "number",
            "description": (f"How close the ORIENTATION must be to count as "
                            f"arrived. Default "
                            f"{cfp.DEFAULT_ORIENTATION_TOLERANCE_DEG:g}, within "
                            f"[{cfp.MIN_ORIENTATION_TOLERANCE_DEG:g}, "
                            f"{cfp.MAX_ORIENTATION_TOLERANCE_DEG:g}]. Both the "
                            f"position and the orientation must be verified and "
                            f"met, or the program stops.")},
    },
    "additionalProperties": False,
}

STEP_SCHEMA = {
    "type": "object",
    "description": ("Exactly one of pose / gripper / dwell / observe."),
    "properties": {
        "pose": POSE_SCHEMA,
        "gripper": {"type": "string", "enum": ["open", "close"],
                    "description": "Its own waypoint; never combined with a pose."},
        "dwell": {
            "type": "object",
            "description": ("Advance the simulation in place, holding the current "
                            "pose and gripper command. Really steps the physics — "
                            "the reply reports the measured step delta."),
            "properties": {"sim_steps": {"type": "integer"}},
            "required": ["sim_steps"],
            "additionalProperties": False,
        },
        "observe": {
            "type": "object",
            "description": "A fine observation without moving. Costs no waypoint.",
            "properties": {"at": {"type": "string", "enum": ["gripper"]},
                           "radius_m": {"type": "number"}},
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


class CfPolicyTool(CoarseFineTool):
    name = "cf_policy"
    description = (
        CF_PREAMBLE +
        "Run a short program of steps and return a fresh fine observation after "
        f"every one. At most {cfp.MAX_STEPS} steps per call; write a few, look at "
        "what came back, then write the next few.\n\n"
        "Steps are pose / gripper / dwell / observe. A pose is resolved against the "
        "robot's telemetry measured immediately before it — not against the frame "
        "you wrote the program on — so a program stays correct if an earlier step "
        "landed slightly off. After each motion you get: the new frame id, the "
        "measured gripper pose and open/closed state, the wrist camera's own depth "
        "reprojected near the fingertips, and the commanded-versus-measured error.\n\n"
        "Optionally bind one region into a proxy first, so later steps can name "
        "'proxy:<name>' and use its centre and up axis. The proxy is a centre, an "
        "up axis and a size. Nothing here proposes a grasp, ranks candidates, "
        "retries a step or recovers from a failure — if a step is not resolvable, "
        "or lands outside its tolerance, the program stops and reports where it "
        "stopped and what was already executed. Decide the next move yourself.\n\n"
        "The whole program is validated before anything moves, so a malformed step "
        "costs no motion. A dwell really advances the physics: 'sim_steps_used' is "
        "the measured delta, and a smaller one than requested means the episode's "
        "step budget or a terminal state ended it early."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "bind": {
                "type": "object",
                "description": ("Optional: fit one region into a named proxy before "
                                "the steps run."),
                "properties": {
                    "name": {"type": "string",
                             "description": "The name steps refer to as 'proxy:<name>'."},
                    "surface": {"type": "string",
                                "enum": ["canvas", "agentview", "wrist"],
                                "description": "Which view the region is drawn on."},
                    "region": fgt.REGION_SCHEMA,
                    # The enum is fast_geometry's own SHAPES, not a copy of it: the
                    # schema advertised 'cylinder' while validate_bind has only ever
                    # accepted SHAPES, so every model that believed the example lost
                    # its whole program to a Rejected before anything moved.
                    "shape": {"type": "string",
                              "enum": list(fg.SHAPES),
                              "description": ("Fit shape, one of "
                                              + ", ".join(f"'{s}'" for s in fg.SHAPES)
                                              + ". Defaults to 'blob', which claims "
                                                "only a centre, an AABB and an axis.")},
                },
                "required": ["name", "region"],
                "additionalProperties": False,
            },
            "steps": {"type": "array", "items": STEP_SCHEMA,
                      "minItems": 1, "maxItems": cfp.MAX_STEPS},
            "budget": {
                "type": "object",
                "description": "Caps for this call. Refused before anything moves.",
                "properties": {"waypoints": {"type": "integer"},
                               "seconds": {"type": "number"}},
                "additionalProperties": False,
            },
        },
        "required": ["steps"],
        "additionalProperties": False,
    }

    async def __call__(self, ctx, arguments):
        started = time.perf_counter()
        try:
            program = cfp.validate_program(arguments)
        except cfp.Rejected as exc:
            return await self.reply(ctx, {"status": "rejected", "message": str(exc),
                                         "steps": []})
        STATE.policy_calls += 1
        payload = {"status": "ok", "steps": [],
                   "physical_steps": program["physical_steps"]}
        if program["bind"]:
            outcome = await bind_proxy(ctx, program["bind"])
            payload["bind"] = outcome
            if outcome["status"] != "ok":
                payload["status"] = outcome["status"]
                payload["message"] = outcome.get("message")
                return await self.reply(ctx, payload, images=await self.snap(ctx))
        budget = fgt.Budget(program["budget"]["waypoints"],
                            program["budget"]["seconds"])
        # The records list is owned here, not inside run_program, so an abort keeps
        # the steps that did run and the observations they measured. The marker is
        # taken before execution so `executed` can be sliced down to THIS call's
        # actions: returning the whole episode's history made every reply look like
        # the earlier waypoints had just been re-executed.
        payload["steps"] = records = []
        committed_before = len(STATE.committed)
        try:
            await run_program(ctx, program, budget, records=records)
        except fgt.Aborted as exc:
            payload["status"] = "stopped"
            payload["reason"] = exc.reason
            payload["message"] = exc.message
            payload["detail"] = exc.detail
        payload["executed"] = STATE.committed[committed_before:]
        payload["executed_note"] = ("the physical actions THIS call submitted; they "
                                    "are already done, so do not re-issue them. "
                                    "cf_state has the whole episode's history.")
        payload["budget"] = budget.report()
        payload["waypoints_executed"] = STATE.waypoints
        payload["cost_s"] = round(time.perf_counter() - started, 3)
        return await self.reply(ctx, payload, images=await self.snap(ctx))


class CfStateTool(CoarseFineTool):
    name = "cf_state"
    description = (
        CF_PREAMBLE +
        "Report what earlier calls established: every bound proxy with its last "
        "measurement and validity, the last fine observation, how many waypoints "
        "and simulation steps this episode has spent, and every physical action "
        "already executed.\n\n"
        "No images, no motion, no new measurement — a proxy bound several frames "
        "ago is reported with the frame it was bound on rather than re-measured. "
        "Use cf_look when you need fresh geometry."
    )
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, ctx, arguments):
        proxies = {}
        for name, card in sorted(STATE.proxies.items()):
            entry = fgt.compact_card(card)
            entry["bound_frame"] = card.get("bound_frame")
            entry["frame_name"] = f"{cfp.PROXY_PREFIX}{name}"
            proxies[name] = entry
        return await self.reply(ctx, {
            "status": "ok", "proxies": proxies,
            "waypoints_executed": STATE.waypoints,
            "sim_steps_held": STATE.sim_steps_held,
            "cf_policy_calls": STATE.policy_calls,
            "executed": STATE.committed,
            "executed_note": ("every physical action of the WHOLE episode, oldest "
                              "first — not just the last call's"),
            "last_observation": STATE.last_observation,
        })


COARSE_FINE_TOOLS = (CfLookTool(), CfPolicyTool(), CfStateTool())

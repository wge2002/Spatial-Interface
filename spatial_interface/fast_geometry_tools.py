"""MCP tools for VIA_CONTROL_INTERFACE=fast_geometry.

Seven tools (fg_look, fg_bind, fg_check, fg_run, fg_state, plus the existing
gripper_toggle and end_episode, which mcp_server keeps owning). The geometry,
solving and validation all live in ``spatial_interface/fast_geometry.py``; this file is the
browser and execution edge:

* it reads the live cloud, the calibration and the robot's own telemetry;
* it drives the SAME execution primitives the legacy interface uses — a target
  edit (target_edit.APPLY_EDIT_JS) followed by the record button — so a waypoint
  from here is physically the same kind of waypoint as before;
* it consumes intermediate frames locally: one fg_run call may execute several
  waypoints and re-measure between them without any model round trip.

Kept out of ``mcp_server.py`` deliberately: that file already carries the frozen
18 tools plus the compact and geometry surfaces, and the legacy path must stay
readable and unchanged.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time

import mcp.types as types

try:
    from . import fast_geometry as fg
    from . import geometry_ref as geom
    from .target_edit import APPLY_EDIT_JS, prepare_edit, validate_edit
except ImportError:  # direct import
    import fast_geometry as fg
    import geometry_ref as geom
    from target_edit import APPLY_EDIT_JS, prepare_edit, validate_edit


# ── episode state ────────────────────────────────────────────────────────────

class FastGeometryState:
    """Refs, attachment and the event log for one episode.

    Module-level singleton, reset by ``reset_episode``, matching how the geometry
    interface holds its references. One MCP server owns one episode.
    """

    def __init__(self) -> None:
        self.refs: dict[str, dict] = {}
        self.attachment: dict | None = None
        self.attachment_ref: str | None = None
        self.counter = 0
        self.execute_calls = 0
        self.waypoints = 0
        # Measured observations per ref, for calibrate_attachment. Only entries
        # whose object centre was RE-MEASURED on a paired frame go in here, each
        # tagged with the frame it came from and the gripper state and axes that
        # frame's own telemetry reported. fit_attachment refuses a predicted
        # centre, so a hint-derived position must never be appended.
        self.observations: dict[str, list[dict]] = {}
        # Which grasp attempt each observation belongs to. Calibration may only use
        # observations from the CURRENT attempt, and this counter is what makes that
        # window explicit rather than a guess about recency:
        #
        # * taking the last few observations regardless would include the open
        #   frames from the approach, and fit_attachment (rightly) refuses an
        #   unclosed gripper — so an ordinary grasp could never calibrate;
        # * filtering all history for closed frames instead would mix the previous
        #   attempt's evidence into a re-grasp, calibrating the new hold from the
        #   old one's offset.
        #
        # The counter advances on every executed open OR close, so the window is
        # exactly "since the gripper last changed state", and an open retires the
        # attachment outright.
        self.grasp_epoch = 0
        self.closed_since: str | None = None  # frame the current close began on
        # Physical actions this episode has actually submitted, so an aborted run
        # can say what was already done rather than leaving the model to re-issue it.
        self.committed: list[dict] = []

    def observe(self, ref: str, entry: dict) -> None:
        """Record one measured observation, newest last, one per frame."""
        entry["grasp_epoch"] = self.grasp_epoch
        history = self.observations.setdefault(ref, [])
        if history and history[-1].get("frame") == entry.get("frame"):
            history[-1] = entry
        else:
            history.append(entry)
        del history[:-OBSERVATION_HISTORY]

    def calibration_window(self, ref: str):
        """Observations calibration may use, and why the others were excluded.

        Exactly the distinct paired frames observed since the gripper last changed
        state. Not "the last few": those cover the open approach. Not "every closed
        frame in history": that spans a re-grasp.
        """
        history = list(self.observations.get(ref) or [])
        usable, excluded = [], []
        seen = set()
        for entry in history:
            if entry.get("grasp_epoch") != self.grasp_epoch:
                excluded.append({"frame": entry.get("frame"),
                                 "why": "before the current gripper close/open",
                                 "grasp_epoch": entry.get("grasp_epoch")})
            elif entry.get("frame") in seen:
                excluded.append({"frame": entry.get("frame"),
                                 "why": "same frame already counted"})
            else:
                seen.add(entry.get("frame"))
                usable.append(entry)
        return usable, excluded

    def gripper_changed(self, *, now_open: bool, frame: str | None) -> None:
        """A gripper open/close executed: start a new attempt, retire stale state.

        Opening drops the attachment: the offset described a hold that no longer
        exists, and keeping it would let the next relation place an object the
        gripper released. Closing does not create one — only measured
        co-displacement does.
        """
        self.grasp_epoch += 1
        self.closed_since = None if now_open else frame
        if now_open:
            self.attachment = None
            self.attachment_ref = None

    def commit(self, action: dict) -> dict:
        action["index"] = len(self.committed)
        self.committed.append(action)
        return action

    def reset(self) -> None:
        self.__init__()

    def new_ref_id(self, name: str) -> str:
        self.counter += 1
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:20]
        return f"{safe}#{self.counter}"


OBSERVATION_HISTORY = 6  # how many measured observations per ref are kept

STATE = FastGeometryState()


def reset_episode() -> None:
    STATE.reset()


def log_event(event: dict) -> None:
    """Append one JSONL event to VIA_FAST_GEOMETRY_LOG_FILE, if configured.

    Best-effort: a log that cannot be written must not fail a robot call. The
    detailed evidence goes here rather than into the model's reply, which is the
    point — the model gets a compact answer, the run stays auditable.
    """
    path = os.environ.get("VIA_FAST_GEOMETRY_LOG_FILE")
    if not path:
        return
    event.setdefault("t", time.time())
    try:
        with open(path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass


# ── shared helpers ───────────────────────────────────────────────────────────

async def read_frame(ctx, *, cam_labels=()):
    """One observation with its version, pairing state and measured end effector."""
    observation = await ctx.observe_geometry()
    return {
        "observation": observation,
        "version": geom.observation_version(observation),
        "pairing": geom.pairing_state(observation, require_cam_labels=cam_labels),
        "measured": geom.measured_end_effector(observation, ctx.ui_z_offset),
        "cloud_version": geom.cloud_version(observation),
    }


def frame_usable(frame) -> tuple[bool, str | None]:
    """Whether this frame may be used as geometric evidence.

    A frame whose cloud was measured by an earlier observation than its
    telemetry (a ``skip_pcl`` motion frame) is the specific case that must never
    count: the old cloud would be paired with a new end-effector pose, and every
    residual computed from it would be arithmetic across two moments.
    """
    if frame["version"] is None:
        return False, "no_paired_frame"
    if frame["pairing"]["status"] != "paired":
        return False, frame["pairing"].get("reason") or "no_paired_frame"
    return True, None


WINDOW_STRIDE = 6  # xyz + rgb, matching fg.WINDOW_POINTS_JS
REGION_STRIDE = 8  # xyz + pixel uv + rgb, matching fg.REGION_POINTS_JS


def _decode(kept, stride, ui_z_offset, *, color_at):
    """Split a flat browser return into robot points and parallel per-point colour.

    ``color_at`` is where RGB starts within one record. The two lists are built in
    the same pass so index i of each always describes the same return — the
    alternative, re-deriving colour by looking a point up afterwards, is what makes
    a stride change silently mis-pair colour with geometry.
    """
    points, colors = [], []
    for i in range(0, len(kept) - stride + 1, stride):
        points.append(geom.ui_to_robot(kept[i:i + 3], ui_z_offset))
        rgb = [kept[i + color_at], kept[i + color_at + 1], kept[i + color_at + 2]]
        colors.append(None if any(v is None or v < 0 for v in rgb) else rgb)
    return points, colors


async def window_points_robot(ctx, center_robot, radius_m):
    """Cloud points in a sphere of a robot-frame centre, with their own colour.

    Returns ``(points, colors)`` in the same order, so a re-fit can correspond on
    appearance against the same returns it fitted. Colour is ``None`` per point
    where the cloud carries none, never dropped: a shorter colour list would
    mis-align with the points and the appearance check would compare a different
    part of the object.
    """
    center_ui = geom.robot_to_ui(center_robot, ctx.ui_z_offset)
    result = await ctx._timed_evaluate(
        fg.WINDOW_POINTS_JS,
        {"center": center_ui, "radius": radius_m * geom.UI_PER_M})
    if not result:
        return [], []
    kept = result.get("kept") or []
    stride = int(result.get("stride") or WINDOW_STRIDE)
    if stride != WINDOW_STRIDE:
        # The browser and this decoder disagree about the record layout, so every
        # coordinate would be read from the wrong slot. Report nothing rather than
        # numbers derived from a misread buffer.
        log_event({"event": "window_stride_mismatch", "stride": stride,
                   "expected": WINDOW_STRIDE})
        return [], []
    points, colors = _decode(kept, stride, ctx.ui_z_offset, color_at=3)
    ctx.geometry_sampled_points += len(points)
    return points, colors


def stored_card_for_solve(ref: str):
    """The card a relation should be solved against, or (None, reason).

    A ref whose last check came back unknown is not usable, and this is where that
    is enforced: the executor asks for a card and gets nothing, so the relation is
    refused rather than solved against coordinates that describe the past.
    """
    stored = STATE.refs.get(ref)
    if stored is None:
        return None, f"unknown reference {ref!r}"
    if not stored.get("valid"):
        return None, (f"reference {ref!r} is not currently valid: "
                      f"{stored.get('invalid_reason') or stored.get('reasons')}")
    return stored, None


# ── overlay ──────────────────────────────────────────────────────────────────

async def project_points(ctx, points_robot):
    """Project robot-frame points to viewport pixels for the overlay."""
    out = []
    for p in points_robot:
        pixel = await ctx.page.evaluate(geom.PROJECT_JS,
                                        geom.robot_to_ui(p, ctx.ui_z_offset))
        out.append(None if not pixel else [round(pixel["x"], 1), round(pixel["y"], 1)])
    return out


async def draw_fit_overlay(ctx, cards):
    """Draw each card's fitted geometry over the canvas.

    Drawn from the *fit*, not from the model's region: a ring is rendered by
    projecting points on the fitted circle, so a circle that does not follow the
    visible rim is visible as a wrong fit rather than hidden behind good numbers.
    """
    shapes = []
    for card in cards:
        if card.get("center") is None:
            continue
        colour = "#00ff88" if card.get("valid") else "#ff5555"
        centre = [card["center"][k] for k in "xyz"]
        radius = card.get("radius_m")
        if radius:
            ring = []
            for i in range(37):
                a = 2 * math.pi * i / 36
                ring.append([centre[0] + radius * math.cos(a),
                             centre[1] + radius * math.sin(a), centre[2]])
            pixels = [p for p in await project_points(ctx, ring) if p]
            if len(pixels) > 1:
                shapes.append({"kind": "polyline", "points": pixels, "color": colour,
                               "width": 2})
        else:
            extent = card.get("extent_m") or {}
            half = [(extent.get("x") or 0.02) / 2, (extent.get("y") or 0.02) / 2]
            box = [[centre[0] - half[0], centre[1] - half[1], centre[2]],
                   [centre[0] + half[0], centre[1] - half[1], centre[2]],
                   [centre[0] + half[0], centre[1] + half[1], centre[2]],
                   [centre[0] - half[0], centre[1] + half[1], centre[2]],
                   [centre[0] - half[0], centre[1] - half[1], centre[2]]]
            pixels = [p for p in await project_points(ctx, box) if p]
            if len(pixels) > 1:
                shapes.append({"kind": "polyline", "points": pixels, "color": colour,
                               "width": 2, "dash": "4 3"})
        if card.get("up") and card.get("up_source", "").startswith("fitted"):
            up = [card["up"][k] for k in "xyz"]
            tip = [centre[i] + up[i] * 0.05 for i in range(3)]
            pixels = [p for p in await project_points(ctx, [centre, tip]) if p]
            if len(pixels) == 2:
                shapes.append({"kind": "polyline", "points": pixels,
                               "color": "#66ccff", "width": 2})
        label_at = (await project_points(ctx, [centre]))[0]
        if label_at:
            text = f"{card['name']} {card['shape']}"
            if card.get("fit", {}).get("residual_rms_m") is not None:
                text += f" rms={card['fit']['residual_rms_m']*1000:.1f}mm"
            if not card.get("valid"):
                text += " INVALID"
            shapes.append({"kind": "text", "at": [label_at[0] + 6, label_at[1] - 6],
                           "text": text, "color": colour})
    if not shapes:
        return 0
    result = await ctx.page.evaluate(fg.OVERLAY_JS, {"shapes": shapes})
    return (result or {}).get("drawn", 0)


async def clear_overlay(ctx):
    try:
        await ctx.page.evaluate(fg.CLEAR_OVERLAY_JS)
    except Exception:
        pass


# ── binding ──────────────────────────────────────────────────────────────────

async def extract_region(ctx, spec, surface):
    """Kept cloud returns inside one region, for the canvas or a camera feed.

    Two different projections, because the two surfaces are different cameras: the
    canvas region is tested against the Three.js view the model is looking at, a
    camera region against that feed's own K/E. Fractions mean the same thing on
    both, which is what lets the model draw a region on whichever view shows the
    object best.
    """
    if surface == "canvas":
        probe = await ctx._timed_evaluate(fg.REGION_POINTS_JS, {
            "region": {"kind": "box", "x0": -1e9, "y0": -1e9, "x1": -1e9, "y1": -1e9}})
        if not probe:
            return None, None, ("the point cloud or renderer is not available in "
                                "the browser")
        region_px = fg.region_to_pixels(spec["region"], probe["viewport"])
        extracted = await ctx._timed_evaluate(fg.REGION_POINTS_JS,
                                              {"region": region_px})
        return extracted, region_px, None
    cam = "agentview" if surface == "agentview" else "wrist"
    extracted = await ctx._timed_evaluate(fg.CAM_REGION_POINTS_JS, {
        "cam": cam, "region": spec["region"], "ui_z_offset": ctx.ui_z_offset})
    if not extracted:
        return None, None, (f"the {cam} feed's calibration is not available in the "
                            f"browser, so a region on it cannot be projected")
    region_px = fg.region_to_pixels(spec["region"], extracted["viewport"])
    return extracted, region_px, None


async def bind_objects(ctx, request, frame):
    """Fit every requested object from the region-selected returns of one frame."""
    cards = []
    fit_s = 0.0
    for spec in request["objects"]:
        extracted, region_px, error = await extract_region(ctx, spec,
                                                           request["surface"])
        if error:
            return [], fit_s, error
        if not extracted:
            cards.append({"name": spec["name"], "shape": spec["shape"], "valid": False,
                          "reasons": ["the browser returned no cloud for this region"],
                          "center": None, "fit": {}})
            continue
        kept = extracted["kept"] or []
        stride = int(extracted.get("stride") or REGION_STRIDE)
        if stride != REGION_STRIDE:
            log_event({"event": "region_stride_mismatch", "stride": stride,
                       "expected": REGION_STRIDE})
            return [], fit_s, (f"the browser returned {stride}-value region records "
                               f"but this server decodes {REGION_STRIDE}; nothing was "
                               f"fitted rather than reading coordinates from the "
                               f"wrong slots")
        ctx.geometry_sampled_points += len(kept) // stride
        points, colors = _decode(kept, stride, ctx.ui_z_offset, color_at=5)
        pixels = [[kept[i + 3], kept[i + 4]]
                  for i in range(0, len(kept) - stride + 1, stride)]
        start = time.perf_counter()
        split = fg.split_support(points)
        clusters = fg.cluster_points(split["object"])
        keep = clusters[0] if clusters else []
        # Pixels and colour must both follow the KEPT cluster: the region's own
        # pixels would report clipping whenever the region touched the table, and
        # the region's own colour would describe the table as well as the object.
        # Both are looked up by rounded position, the same key ``fg._key3`` uses.
        by_point = {}
        for p, px, c in zip(points, pixels, colors):
            by_point.setdefault(fg._key3(p), (px, c))
        keep_pixels = [by_point[fg._key3(p)][0] for p in keep
                       if fg._key3(p) in by_point]
        keep_colors = [by_point.get(fg._key3(p), (None, None))[1] for p in keep]
        support_colors = [by_point.get(fg._key3(p), (None, None))[1]
                          for p in split["support"]]
        card = fg.build_card(
            name=spec["name"], shape=spec["shape"], points=keep,
            pixels=keep_pixels, region_px=region_px, support=split["support"],
            clusters=clusters, from_frame=frame["version"],
            surface=request["surface"], colors=keep_colors,
            support_colors=support_colors)
        fit_s += time.perf_counter() - start
        card["fit"]["support_fraction"] = split["support_fraction"]
        card["fit"]["support_z"] = split["support_z"]
        card["region"] = spec["region"]
        cards.append(card)
    return cards, fit_s, None


# ── tools ────────────────────────────────────────────────────────────────────

FG_PREAMBLE = (
    "Coordinates are the robot frame in metres: +x away from the base, +y left, "
    "+z up. Regions are fractions of the surface you are looking at, the same "
    "(u, v) convention as a screenshot: u across from 0.0 left, v down from 0.0 "
    "top.\n\n"
)


# ── what the model is sent, versus what is logged ────────────────────────────
#
# One rule: a SUCCESS is a number, a FAILURE is an explanation. A matched ref that
# moved 3 mm needs its centre and nothing else — the gate values it passed are
# constants, and sending them per ref per stage crowded out the fields the next
# decision actually turns on. An unknown or refused one keeps everything, because
# there the reason IS the result. The full nested record always reaches the log.

# Kept from a re-check that MATCHED. Everything else (the gates it passed, the
# candidates it ranked, the window it searched) is in the log.
CHECK_OK_FIELDS = ("status", "center", "center_shift_m")
# Kept from a stage record. `verdict` is the one that decides; `arrival` and
# `relation_residual` are summarised beside it so the gripper-arrived /
# object-did-not case stays visible without two full nested records.
STAGE_FIELDS = ("stage", "kind", "relation", "verified_by", "within_tolerance",
                "waypoints", "edits", "already", "frame", "ref", "note",
                "motion_plan", "pre_move_frame", "physical",
                # calibration: what evidence the fit was allowed to use, and what
                # it refused. A hold accepted on two frames the model cannot see is
                # exactly the thing it should be able to question.
                "observations_used", "observation_frames", "observations_excluded",
                "grasp_epoch", "retired_attachment_for", "retired_previous_attachment")
# A residual, reduced to the numbers a correction would be based on.
RESIDUAL_FIELDS = ("status", "error_m", "tolerance_m", "within_tolerance",
                   "orientation_error_deg", "per_axis_m", "horizontal_error_m",
                   "reason", "subject", "note")


def _residual_view(residual):
    if not isinstance(residual, dict):
        return residual
    return {k: residual[k] for k in RESIDUAL_FIELDS if residual.get(k) is not None}


def _checks_view(checked):
    """Matched refs shrink to their measurement; unmatched keep their reason."""
    if not isinstance(checked, dict):
        return checked
    out = {}
    for ref, entry in checked.items():
        if not isinstance(entry, dict):
            out[ref] = entry
        elif entry.get("status") == "matched":
            small = {k: entry[k] for k in CHECK_OK_FIELDS if entry.get(k) is not None}
            hold = entry.get("attachment")
            if isinstance(hold, dict):
                # A confirmed hold keeps its residual: "still holding, 4 mm from
                # where the offset predicts" is a different fact from "still
                # holding", and it is the one that shows a slip developing. A hold
                # in doubt is never trimmed at all.
                small["attachment"] = (
                    {k: hold[k] for k in ("status", "residual_m", "gate_m")
                     if hold.get(k) is not None}
                    if hold.get("status") == "attached" else hold)
            out[ref] = small
        else:
            out[ref] = entry
    return out


def _stage_view(record):
    if not isinstance(record, dict):
        return record
    out = {k: record[k] for k in STAGE_FIELDS if record.get(k) is not None}
    for key in ("verdict", "arrival", "relation_residual"):
        if record.get(key) is not None:
            out[key] = _residual_view(record[key])
    for key in ("checked", "post_move_check", "pre_move_check"):
        if record.get(key) is not None:
            out[key] = _checks_view(record[key])
    # Solving: what the goal was and any approximation in it, not the derivation.
    solved = record.get("solved")
    if isinstance(solved, dict):
        out["solved"] = {k: solved[k] for k in
                         ("relation", "grasp", "goal_point", "object_goal",
                          "approximation", "status", "reason",
                          # the offset the goal was derived from, and whether it had
                          # to be carried through a rotation to get there
                          "attachment_offset_m", "attachment_offset_mapping")
                         if solved.get(k) is not None}
    for key in ("refine_stopped", "attachment"):
        if record.get(key):
            out[key] = record[key]
    if record.get("check") is not None:
        # calibrate_attachment's own re-measurement, same rule as any other.
        out["check"] = _checks_view({"_": record["check"]})["_"]
    basis_is_relation = record.get("verified_by") == "relation"
    if record.get("corrections") is not None:
        # One residual per attempt — the sequence is what shows whether correcting
        # was converging — without each attempt's own nested check results.
        out["corrections"] = [
            {"attempt": c.get("attempt"), "frame": c.get("frame"),
             "residual": _residual_view(c.get("relation_residual")
                                        if basis_is_relation else c.get("arrival")),
             **({"stop": c["stop"]} if c.get("stop") else {})}
            for c in record["corrections"]]
    if record.get("segments") is not None:
        # Kept as a list, without each segment's target vector: how many physical
        # waypoints a stage actually spent is a budget fact the model needs, and the
        # coordinates it drove through are not.
        out["segments"] = [{k: s[k] for k in ("index", "completed", "submitted")
                            if isinstance(s, dict) and s.get(k) is not None}
                           for s in record["segments"]]
    # Boundaries matter when one of them stopped or corrected something; a clean
    # multi-segment drive is adequately described by its count.
    for entry in record.get("boundaries") or []:
        if any((v or {}).get("status") not in (None, "matched")
               for v in (entry.get("checked") or {}).values()):
            out["boundaries"] = [
                {"after_segment": e.get("after_segment"),
                 "remaining_distance_m": e.get("remaining_distance_m"),
                 "checked": _checks_view(e.get("checked") or {})}
                for e in record["boundaries"]]
            break
    return out


def model_view(payload):
    """The model-facing form of a tool payload. The log keeps the full record."""
    view = {k: v for k, v in payload.items() if k not in ("stages", "bound",
                                                          "checked", "refs")}
    if payload.get("status") != "ok":
        # A refusal or an abort is the case where the long text is the useful part.
        view["limits"] = fg.LIMITS
    if isinstance(payload.get("stages"), list):
        view["stages"] = [_stage_view(s) for s in payload["stages"]]
    if isinstance(payload.get("checked"), dict):
        view["checked"] = _checks_view(payload["checked"])
    if payload.get("bound") is not None:
        view["bound"] = payload["bound"]
    if payload.get("refs") is not None:
        view["refs"] = payload["refs"]
    return view


class FastGeometryTool:
    """Base for the fast-geometry tools. Deliberately not a mcp_server ToolBase
    subclass at import time — mcp_server owns the registry and adapts these."""

    name: str
    description: str
    input_schema: dict

    async def reply(self, ctx, payload, *, images=None):
        payload["timing_id"] = ctx.timing_id
        # The LOG gets everything: every candidate a match considered, every
        # threshold it applied, every boundary of every segment. That is what a
        # failure is diagnosed from afterwards.
        log_event({"tool": self.name, "payload": payload})
        # The MODEL gets what it has to act on. The two differ on purpose: a
        # successful stage's match evidence and gate values are the same numbers
        # every time and say nothing the model can act on, and repeating them for
        # each ref of each stage was the bulk of the reply. Anything unknown,
        # refused or out of tolerance keeps its full detail, because that is the
        # case where the reason is the point. fg_state re-reads the rest on demand.
        view = model_view(payload)
        blocks = [types.TextContent(type="text", text=json.dumps(view))]
        return blocks + list(images or [])

    async def snap(self, ctx, text=None):
        try:
            return await ctx.snap(text) if text else await ctx.snap()
        except Exception:
            return []


class FgLookTool(FastGeometryTool):
    name = "fg_look"
    description = (
        FG_PREAMBLE +
        "Look at the scene: one screenshot plus the id of the frame it belongs to, "
        "and whether that frame's cloud, telemetry and camera images were measured "
        "together.\n\n"
        "Use this before fg_bind so you can draw regions on a view you have just "
        "seen. It moves nothing, records no waypoint and creates no frame — fg_run "
        "is the only tool in this interface that moves the robot.\n\n"
        "'paired: true' is what makes the frame usable as geometry. During motion "
        "the producer streams frames that reuse an older cloud; those are reported "
        "unpaired and fg_bind/fg_check refuse them rather than mixing two moments."
    )
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, ctx, arguments):
        await clear_overlay(ctx)
        frame = await read_frame(ctx)
        usable, reason = frame_usable(frame)
        payload = {"status": "ok", "frame": frame["version"],
                   "cloud_frame": frame["cloud_version"],
                   "paired": usable, "pairing": frame["pairing"],
                   "measured_end_effector": frame["measured"],
                   "refs": sorted(STATE.refs),
                   "attachment": (STATE.attachment or {}).get("status", "none")}
        if not usable:
            payload["note"] = (f"this frame is not usable as geometry ({reason}); "
                               f"fg_bind and fg_check will refuse it")
        return await self.reply(ctx, payload, images=await self.snap(ctx))


REGION_SCHEMA = {
    "type": "object",
    "description": ("Where the object is on the surface, in fractions. "
                    "kind='box' with u0,v0,u1,v1; kind='point' with u,v,radius_px; "
                    "kind='polygon' with points [[u,v],...] up to 12 vertices."),
    "properties": {
        "kind": {"type": "string", "enum": ["box", "point", "polygon"]},
        "u0": {"type": "number"}, "v0": {"type": "number"},
        "u1": {"type": "number"}, "v1": {"type": "number"},
        "u": {"type": "number"}, "v": {"type": "number"},
        "radius_px": {"type": "number"},
        "points": {"type": "array", "items": {"type": "array",
                                             "items": {"type": "number"},
                                             "minItems": 2, "maxItems": 2}},
    },
    "required": ["kind"],
    "additionalProperties": False,
}


class FgBindTool(FastGeometryTool):
    name = "fg_bind"
    description = (
        FG_PREAMBLE +
        "Bind up to four task objects in one call by drawing a region around each "
        "and naming the shape to fit. Returns one short card per object plus a "
        "screenshot with the fits drawn over the view.\n\n"
        "shape: 'ring' (a rim: bowl, mug, opening), 'disc' (a flat round top: "
        "plate, lid), 'plane_patch' (a flat surface or the table), 'blob' (anything "
        "else — you get a centroid and a box, no invented pose).\n\n"
        "Each card carries what was measured and what it cost to believe: the "
        "fitted centre, radius and height, the rim and base points kept separate, "
        "grasp candidates on the fitted geometry, the residual of the fit, how many "
        "returns supported it, and how much bearing the returns covered. "
        "valid=false comes with reasons — too few returns, residual over the gate, "
        "an arc instead of a circle, two candidate clusters in one region, or a "
        "cluster running off the region edge. An invalid card cannot be named by "
        "fg_run, so re-draw the region instead of acting on it.\n\n"
        "Points outside your region cannot enter a fit however close they are in "
        "3-D, and the table is removed before fitting, so a region that clips the "
        "tabletop is fine. Binding needs a paired frame; on an unpaired one nothing "
        "is stored."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "surface": {"type": "string", "enum": ["canvas", "agentview", "wrist"],
                        "description": ("Which view your regions are drawn on. "
                                        "Default 'canvas' (the 3-D point cloud).")},
            "objects": {
                "type": "array", "minItems": 1, "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "maxLength": 40,
                                 "description": "Your label. Stored unverified."},
                        "region": REGION_SCHEMA,
                        "shape": {"type": "string",
                                  "enum": list(fg.SHAPES),
                                  "description": "Shape to fit. Default 'blob'."},
                    },
                    "required": ["name", "region"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["objects"],
        "additionalProperties": False,
    }

    async def __call__(self, ctx, arguments):
        started = time.perf_counter()
        try:
            request = fg.validate_bind(arguments)
        except fg.Rejected as exc:
            return await self.reply(ctx, {"status": "rejected", "message": str(exc),
                                         "bound": []})
        cam_labels = ((request["surface"],)
                      if request["surface"] in geom.CAM_LABEL_TO_ELEMENT else ())
        frame = await read_frame(ctx, cam_labels=cam_labels)
        usable, reason = frame_usable(frame)
        if not usable:
            return await self.reply(ctx, {
                "status": "unknown", "reason": reason, "frame": frame["version"],
                "pairing": frame["pairing"], "bound": [],
                "message": ("this frame's cloud, telemetry and images were not "
                            "measured together, so nothing was bound; call fg_look "
                            "again")}, images=await self.snap(ctx))
        await clear_overlay(ctx)
        cards, fit_s, error = await bind_objects(ctx, request, frame)
        if error:
            return await self.reply(ctx, {"status": "unknown", "reason": "occluded",
                                          "message": error, "bound": []})
        # Extraction spans several browser calls, so a frame arriving mid-way would
        # leave earlier cards fitted from a cloud the later ones did not see.
        after = geom.observation_version(await ctx.observe_geometry())
        if after != frame["version"]:
            return await self.reply(ctx, {
                "status": "unknown", "reason": "no_paired_frame",
                "frame": frame["version"], "frame_after": after, "bound": [],
                "message": ("a new frame arrived while the regions were being "
                            "extracted, so the cards would mix two observations; "
                            "nothing was stored. Retry.")},
                images=await self.snap(ctx))
        bound = []
        for card in cards:
            ref = STATE.new_ref_id(card["name"])
            card["ref"] = ref
            card["bound_frame"] = frame["version"]
            card["frames_since_bind"] = 0
            STATE.refs[ref] = card
            bound.append(compact_card(card))
        drawn = await draw_fit_overlay(ctx, cards)
        images = await self.snap(ctx)
        await clear_overlay(ctx)
        payload = {"status": "ok", "frame": frame["version"],
                   "surface": request["surface"], "bound": bound,
                   "overlay_shapes": drawn,
                   "cost_s": {"total": round(time.perf_counter() - started, 3),
                              "fit": round(fit_s, 4)}}
        invalid = [c["ref"] for c in bound if not c["valid"]]
        if invalid:
            payload["unusable_refs"] = invalid
            payload["note"] = ("these refs cannot be named by fg_run until they fit; "
                               "read their reasons and re-draw the region")
        return await self.reply(ctx, payload, images=images)


def compact_card(card):
    """What the model sees of a card: measurements and validity, no point lists.

    The old geometry replies grew with the cloud. Here the reply is a fixed handful
    of numbers per object and the per-point evidence stays in the local log, which
    is what keeps a bind cheap in tokens as well as in time.
    """
    fit = card.get("fit") or {}
    out = {"ref": card.get("ref"), "name": card.get("name"),
           "name_note": card.get("name_note"), "shape": card.get("shape"),
           "from_frame": card.get("from_frame"), "valid": bool(card.get("valid")),
           "center": card.get("center"), "base_center": card.get("base_center"),
           "top_z": card.get("top_z"), "radius_m": card.get("radius_m"),
           "height_m": card.get("height_m"), "extent_m": card.get("extent_m"),
           "up": card.get("up"), "up_source": card.get("up_source"),
           "fit": {k: fit.get(k) for k in
                   ("residual_rms_m", "object_points", "support_points",
                    "bearing_coverage_deg", "clipped_by_region",
                    "second_cluster_points")},
           "grasp_candidates": [c["id"] for c in card.get("grasp_candidates") or []]}
    if not card.get("valid"):
        out["reasons"] = card.get("reasons")
    return out


class FgCheckTool(FastGeometryTool):
    name = "fg_check"
    description = (
        FG_PREAMBLE +
        "Re-measure bound references against the current frame. Use it after "
        "anything that may have moved an object, and before trusting a stored "
        "position.\n\n"
        "For each ref the same shape is re-fitted from the returns near where the "
        "object was, and the result is accepted only if it still looks like the same "
        "thing: comparable residual and size, a shift inside the search window, no "
        "rival cluster of similar size. You get matched with the new centre, how far "
        "it moved, the new residual and how many frames since it was bound — or "
        "unknown with one reason: no_paired_frame, same_frame_no_new_evidence, "
        "insufficient_support, fit_degraded, shift_exceeds_window, "
        "ambiguous_two_candidates, occluded, no_stored_geometry.\n\n"
        "unknown makes the ref unusable and any fg_run stage naming it is refused; "
        "its old numbers stay readable as last_known_* but describe where the object "
        "was. Re-reading the same frame is not new evidence and returns "
        "same_frame_no_new_evidence, so this cannot be used to talk a stale "
        "reference back into validity. A ref believed held is re-measured against "
        "the attachment offset's prediction, and the prediction error is reported "
        "next to the measurement, never instead of it."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "refs": {"type": "array", "items": {"type": "string"},
                     "minItems": 1, "maxItems": 6,
                     "description": "Reference ids from fg_bind. Default: all."},
            "image": {"type": "boolean",
                      "description": "Return a screenshot too. Default false."},
        },
        "additionalProperties": False,
    }

    async def __call__(self, ctx, arguments):
        refs = arguments.get("refs") or sorted(STATE.refs)
        if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
            return await self.reply(ctx, {"status": "rejected",
                                         "message": "refs must be a list of ids"})
        if not refs:
            return await self.reply(ctx, {"status": "ok", "checked": {},
                                          "message": "no references are bound yet"})
        frame = await read_frame(ctx)
        results, _ = await check_refs(ctx, refs, frame)
        images = await self.snap(ctx) if arguments.get("image") else []
        return await self.reply(ctx, {
            "status": "ok", "frame": frame["version"], "checked": results,
            "attachment": (STATE.attachment or {}).get("status", "none")},
            images=images)


class FgStateTool(FastGeometryTool):
    name = "fg_state"
    description = (
        FG_PREAMBLE +
        "Report what is currently known — every reference with its last "
        "measurement and validity, the attachment state and its evidence, and how "
        "many waypoints this episode has used. No images, no robot motion, no new "
        "measurement: this reads back what earlier calls established, so a ref last "
        "seen several frames ago is reported with that age rather than re-measured. "
        "Call fg_check when you need fresh geometry."
    )
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, ctx, arguments):
        refs = {}
        for ref, card in sorted(STATE.refs.items()):
            entry = compact_card(card)
            entry["frames_since_bind"] = card.get("frames_since_bind")
            entry["last_checked_frame"] = card.get("last_checked_frame")
            if not card.get("valid"):
                entry["invalid_reason"] = card.get("invalid_reason")
                entry["last_known_center"] = card.get("last_known_center")
            refs[ref] = entry
        return await self.reply(ctx, {
            "status": "ok", "refs": refs,
            "attachment": STATE.attachment or {"status": "none"},
            "attachment_ref": STATE.attachment_ref,
            "waypoints_executed": STATE.waypoints,
            "fg_run_calls": STATE.execute_calls})


# ── re-association ───────────────────────────────────────────────────────────

async def check_refs_now(ctx, refs):
    """Read a fresh frame and re-measure ``refs`` on it. Returns (frame, results)."""
    frame = await read_frame(ctx)
    results, _ = await check_refs(ctx, refs, frame)
    return frame, results


async def check_refs(ctx, refs, frame):
    """Re-measure each ref on ``frame``; update STATE. Returns (results, any_new).

    The freshness rule lives here so it cannot be bypassed: a ref is only updated
    from a frame strictly later than the one it was last measured on, and only from
    a paired frame. Everything else leaves the stored geometry alone and reports
    unknown, which is what makes ``valid`` mean "measured recently" rather than
    "measured once".
    """
    results = {}
    usable, reason = frame_usable(frame)
    for ref in refs:
        stored = STATE.refs.get(ref)
        if stored is None:
            results[ref] = {"status": "unknown", "reason": "no_stored_geometry",
                            "message": f"no reference {ref!r} is bound"}
            continue
        if not usable:
            invalidate(stored, reason)
            results[ref] = {"status": "unknown", "reason": reason,
                            "last_known_center": stored.get("last_known_center")}
            continue
        last = stored.get("last_checked_frame") or stored.get("bound_frame")
        advance = geom.frame_advance(last, frame["version"]) if last else None
        if advance is not None:
            results[ref] = {
                "status": "unknown", "reason": "same_frame_no_new_evidence",
                "frame": frame["version"], "last_measured_frame": last,
                "frame_advance": advance,
                "message": ("this is not a strictly later frame of this episode, so "
                            "it carries no new evidence about this reference; the "
                            "stored geometry was left as it was"),
                "valid": bool(stored.get("valid")),
                "last_known_center": stored.get("center")}
            continue
        expected = None
        offset_mapping = None
        if (STATE.attachment_ref == ref and STATE.attachment
                and STATE.attachment["status"] == "attached"
                and frame["measured"].get("status") == "ok"):
            # The search hint must use the offset carried into the gripper's
            # CURRENT orientation, not the raw world vector: after a rotation the
            # world offset points somewhere the object is not, so the window would
            # be centred off the object and the re-fit would report unknown for a
            # reason that is the hint's fault. When the rotation is unmeasurable
            # the mapping refuses and the search falls back to the last known
            # centre, which is recorded in the reply.
            offset, offset_mapping = fg.offset_in_current_orientation(
                STATE.attachment, frame["measured"])
            if offset is not None:
                tip = [frame["measured"]["fingertip_position"][k] for k in "xyz"]
                expected = [tip[i] + offset[i] for i in range(3)]
        if expected is not None:
            anchor = expected
        elif stored.get("center"):
            anchor = [stored["center"][k] for k in "xyz"]
        else:
            anchor = None
        if anchor is None:
            results[ref] = {"status": "unknown", "reason": "no_stored_geometry"}
            invalidate(stored, "no_stored_geometry")
            continue
        radius = fg.check_window_radius(stored)
        window, window_colors = await window_points_robot(ctx, anchor, radius)
        outcome = fg.reassociate(stored, window, from_frame=frame["version"],
                                 expected_shift=expected,
                                 window_colors=window_colors)
        apply_check(stored, outcome, frame)
        results[ref] = compact_check(outcome, stored)
        if offset_mapping is not None:
            results[ref]["attachment_offset_mapping"] = offset_mapping
        if outcome["status"] == "matched":
            record_observation(ref, outcome["center"], frame)
        hold = verify_attachment(ref, outcome, frame)
        if hold is not None:
            results[ref]["attachment"] = hold
    log_event({"event": "check", "frame": frame["version"], "results": results})
    return results, usable


def record_observation(ref, center, frame):
    """Keep one measured (object centre, fingertip, gripper state) triple.

    Everything ``fit_attachment`` gates on comes from here, so each field is the
    frame's own measurement: the centre is the re-fit's, the fingertip and axes are
    that frame's telemetry, and ``gripper_state_class`` is what telemetry reported —
    including "unknown", which the fit now refuses rather than skips. Nothing
    derived from the attachment offset may be written here; that would let a
    prediction become the evidence for the offset that produced it.
    """
    measured = frame["measured"]
    if measured.get("status") != "ok" or not center:
        return
    STATE.observe(ref, {
        "frame": frame["version"],
        "object_center": [center[k] for k in "xyz"],
        "object_center_source": "re_measured_refit_in_search_window",
        "fingertip": [measured["fingertip_position"][k] for k in "xyz"],
        "gripper_state_class": measured.get("gripper_state_class"),
        "approach": ([measured["approach"][k] for k in "xyz"]
                     if measured.get("approach") else None),
        "opening": ([measured["opening"][k] for k in "xyz"]
                    if measured.get("opening") else None),
    })


def verify_attachment(ref, outcome, frame):
    """Check the cached hold against this frame's own measurement of the held ref.

    Until now the calibration was fitted once and then reused for the rest of the
    episode, so a slip or a release between waypoints left every later target solved
    from an offset that no longer described anything. The offset is only ever
    checked here, against a re-measured centre and the CURRENT gripper axes, and a
    failed check clears it: a placement that needs an attachment then refuses
    instead of aiming at a predicted centre.

    Returns the verdict for ``ref``, or None when ``ref`` is not the held object.
    """
    if ref != STATE.attachment_ref or not STATE.attachment:
        return None
    measured = frame["measured"]
    if outcome.get("status") != "matched" or measured.get("status") != "ok":
        # No new evidence either way. The offset is NOT dropped on a single
        # unusable frame — that would discard a good hold for a bad look — but it is
        # not confirmed either, and the reply says which.
        return {"status": "unconfirmed",
                "reason": ("the held object could not be re-measured on this frame, "
                           "so the stored offset was neither confirmed nor cleared")}
    hold = fg.attachment_still_holds(
        STATE.attachment,
        object_center=[outcome["center"][k] for k in "xyz"],
        fingertip=[measured["fingertip_position"][k] for k in "xyz"],
        measured=measured)
    gripper_class = measured.get("gripper_state_class")
    if hold.get("status") == "attached" and gripper_class == "open":
        # An offset that still predicts the centre while the jaws are open means the
        # object is merely sitting where it was left, not held.
        hold = {"status": "lost", "residual_m": hold.get("residual_m"),
                "gripper_state_class": gripper_class,
                "reason": ("the gripper is open, so the object is resting there, not "
                           "held; the offset was cleared")}
    elif hold.get("status") == "attached" and gripper_class != "closed":
        # Open is not the only non-closed state. A missing or "unknown" class is not
        # the absence of a problem: the same co-located centre supports "still held"
        # and "released and left exactly where it was", and only the gripper state
        # tells those apart. ``fit_attachment`` refuses an unknown class when it
        # ACCEPTS an offset (gripper_state_unknown), so confirming one here on the
        # same missing evidence would reopen the gap it closes.
        #
        # Unconfirmed, not lost: unlike the open case this is absent evidence rather
        # than contrary evidence, and the rule one level up is that a single bad look
        # must not discard a good hold. The offset survives and is not confirmed, so
        # a later frame that does report the class decides it.
        hold = {"status": "unconfirmed", "residual_m": hold.get("residual_m"),
                "gripper_state_class": gripper_class,
                "reason": (f"the gripper state is "
                           f"{gripper_class or 'not reported'} on this frame, so "
                           f"nothing here shows the jaws were closed while the object "
                           f"stayed with them; the stored offset was neither confirmed "
                           f"nor cleared")}
        return hold
    if hold.get("status") != "attached":
        STATE.attachment = None
        STATE.attachment_ref = None
        hold["cleared"] = True
        hold["consequence"] = ("relations that place a held object now refuse until "
                               "calibrate_attachment measures a new hold")
        log_event({"event": "attachment_lost", "ref": ref, "frame": frame["version"],
                   "verdict": hold})
    return hold


def invalidate(stored, reason):
    """Mark a ref unusable, preserving its last measurement as last_known_*."""
    if stored.get("valid") and stored.get("center"):
        stored["last_known_center"] = stored["center"]
        stored["last_known_frame"] = stored.get("last_checked_frame") \
            or stored.get("bound_frame")
    stored["valid"] = False
    stored["invalid_reason"] = reason


def apply_check(stored, outcome, frame):
    """Fold one re-association outcome into the stored card."""
    if outcome["status"] != "matched":
        invalidate(stored, outcome["reason"])
        return
    for key in ("center", "base_center", "top_z", "radius_m", "height_m",
                "extent_m", "keypoints", "grasp_candidates", "fit"):
        if outcome.get(key) is not None:
            stored[key] = outcome[key]
    stored["valid"] = True
    stored["invalid_reason"] = None
    stored["last_checked_frame"] = frame["version"]
    bound = stored.get("bound_frame")
    if bound:
        b = geom.parse_observation_version(bound)
        n = geom.parse_observation_version(frame["version"])
        if b and n and b["epoch"] == n["epoch"]:
            stored["frames_since_bind"] = n["seq"] - b["seq"]


def compact_check(outcome, stored):
    """The short form of a re-association, for the model."""
    keep = ("status", "reason", "center", "center_shift_m", "radius_change_m",
            "residual_rms_m", "predicted_center", "prediction_error_m",
            "search_window_m", "object_points", "top_z", "match_evidence",
            "cluster_sizes", "fit_reasons")
    out = {k: outcome[k] for k in keep if outcome.get(k) is not None}
    out["status"] = outcome["status"]
    out["frames_since_bind"] = stored.get("frames_since_bind")
    if outcome["status"] != "matched":
        out["last_known_center"] = stored.get("last_known_center")
        out["last_known_note"] = ("where this object was when last measured, not "
                                  "where it is now")
    return out


# ── execution ────────────────────────────────────────────────────────────────
#
# The executor is the reason this interface exists: one model turn declares the
# whole sequence, and the loop below spends the waypoints, takes the intermediate
# observations and does the arithmetic locally. Three properties are load-bearing:
#
# * nothing physical happens until the WHOLE request validates, so a schema error
#   in a later stage cannot be discovered after earlier stages have moved the arm;
# * every physical action is appended to STATE.committed BEFORE it is submitted, so
#   an abort reports what was already done and the model does not re-issue it;
# * arrival, the relation being satisfied, and the benchmark terminating are three
#   separate records. The gripper reaching the solved pose is not the object being
#   placed, and neither is the task succeeding.

class Aborted(Exception):
    """Stop the run and report it, with everything already committed."""

    def __init__(self, reason, message, **detail):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.detail = detail


class Budget:
    """Waypoints and wall clock for one fg_run call.

    The seconds budget is checked before each physical action rather than only at
    the top: a run that has spent its time must stop and report, not start another
    waypoint whose controller time would land after the harness's own tool timeout
    and get the whole call cancelled mid-motion.
    """

    def __init__(self, waypoints, seconds, *, now=None):
        self._now = now or time.monotonic
        self.waypoints = waypoints
        self.seconds = seconds
        self.started = self._now()
        self.used = 0

    def elapsed(self):
        return self._now() - self.started

    def remaining_s(self):
        return self.seconds - self.elapsed()

    def spend(self, what):
        if self.used >= self.waypoints:
            raise Aborted("waypoint_budget_exhausted",
                          f"the {self.waypoints}-waypoint budget is spent, and "
                          f"{what} would need another one",
                          waypoints_used=self.used)
        # A physical waypoint can take seconds of controller time, so refuse to
        # start one the deadline cannot contain.
        if self.remaining_s() <= MIN_WAYPOINT_HEADROOM_S:
            raise Aborted("deadline_reached",
                          f"{self.elapsed():.1f} s of the {self.seconds:g} s budget "
                          f"are gone, too little left to start {what} and observe "
                          f"the result",
                          seconds_elapsed=round(self.elapsed(), 2))
        self.used += 1
        return self.used

    def report(self):
        return {"waypoints_used": self.used, "waypoints_allowed": self.waypoints,
                "seconds_elapsed": round(self.elapsed(), 2),
                "seconds_allowed": self.seconds}


MIN_WAYPOINT_HEADROOM_S = 12.0  # controller time one waypoint plus its observation needs


async def apply_target(ctx, position, *, approach=None, opening=None):
    """Walk the virtual target to a pose with as many edits as the caps require.

    Browser-only: this is ``target_edit``'s own compare-and-set, called several
    times when one edit's 0.1 m / 90 deg cap cannot cover the displacement. No
    record button, so none of it costs a physical waypoint — which is the
    distinction the protocol doc had backwards.
    """
    edits = 0
    if approach is not None and opening is not None:
        pose = await ctx.gripper_pose()
        if pose is None:
            raise Aborted("target_unavailable", "the blue target gripper is not "
                                                "selected, so no pose can be set")
        current = {"approach": [pose["robot_approach"][k] for k in "xyz"],
                   "opening": [pose["robot_opening"][k] for k in "xyz"]}
        steps = fg.orientation_steps(current, {"approach": approach,
                                               "opening": opening})
        if steps is None:
            raise Aborted("orientation_unusable",
                          "the current or requested gripper axes are not a usable "
                          "orthogonal pair, so the orientation was not changed")
        for step in steps:
            await _one_edit(ctx, {"frame": "robot", "approach": step["approach"],
                                  "opening": step["opening"]})
            edits += 1
    if position is not None:
        for _ in range(MAX_EDITS_PER_STAGE):
            pose = await ctx.gripper_pose()
            if pose is None:
                raise Aborted("target_unavailable", "the blue target gripper is not "
                                                    "selected, so no pose can be set")
            at = [pose["robot_position"][k] for k in "xyz"]
            steps = fg.edit_steps(at, position)
            if not steps:
                break
            await _one_edit(ctx, {"frame": "robot", "position": steps[0]})
            edits += 1
            if len(steps) == 1:
                break
        else:
            raise Aborted("target_not_reachable_by_editing",
                          f"the virtual target did not reach the solved pose within "
                          f"{MAX_EDITS_PER_STAGE} edits", edits=edits)
    return edits


MAX_EDITS_PER_STAGE = 12


async def _one_edit(ctx, arguments):
    """One validated target edit through the same primitive edit_target uses."""
    if not await ctx.ensure_translation_mode():
        raise Aborted("target_not_editable", "the target is not in translation mode")
    pose = await ctx.gripper_pose()
    if pose is None:
        raise Aborted("target_unavailable", "the blue target gripper is not selected")
    try:
        prepared = prepare_edit(validate_edit(arguments), pose, ctx.ui_z_offset)
    except ValueError as exc:
        raise Aborted("edit_rejected", f"the target edit was rejected: {exc}")
    result = await ctx.page.evaluate(APPLY_EDIT_JS, prepared)
    if not result.get("ok"):
        raise Aborted("edit_rejected",
                      f"the target edit was refused by the browser: "
                      f"{result.get('reason')}")
    return prepared


async def execute_once(ctx, budget, *, what, detail):
    """Submit the current target and wait for the controller. One waypoint.

    Committed before the click, because a click that lands and then loses its
    completion signal has still actuated the robot: reporting it only on success
    would let an abort claim the arm never moved.
    """
    budget.spend(what)
    action = STATE.commit({"action": what, **detail, "submitted": True,
                           "completed": None})
    prev = await ctx.page.evaluate("() => window.__waypointDoneCount || 0")
    await ctx.page.locator("#btn-record").click()
    STATE.waypoints += 1
    try:
        await ctx.page.wait_for_function(
            "prev => (window.__waypointDoneCount || 0) > prev", arg=prev,
            timeout=int(min(EXECUTE_TIMEOUT_S, max(1.0, budget.remaining_s())) * 1000))
        action["completed"] = True
    except Exception as exc:
        # Unknown, not failed: the actuation was submitted and the arm may still
        # be moving. The run stops rather than stacking another waypoint on a
        # controller whose state nothing here knows.
        action["completed"] = None
        action["completion_note"] = str(exc)[:200]
        raise Aborted("waypoint_completion_unknown",
                      f"{what} was submitted to the controller but no completion "
                      f"signal arrived, so whether the robot reached the target is "
                      f"unknown; do not re-submit it", action=action)
    await asyncio.sleep(POST_WAYPOINT_SETTLE_S)
    return action


EXECUTE_TIMEOUT_S = 30.0
POST_WAYPOINT_SETTLE_S = 0.15


async def set_gripper(ctx, budget, want_open):
    """Open or close the gripper in its own waypoint, pose unchanged.

    Deliberately not the legacy ``gripper_toggle`` tool: that one only flips the
    *virtual* target's mesh and leaves the model to execute it. Here the mesh swap
    and its execution are one stage, so an open/close is a real action with a real
    cost and never rides along with a pose change.
    """
    pose = await ctx.gripper_pose()
    if pose is None or pose.get("gripper_open") is None:
        raise Aborted("gripper_state_unknown",
                      "the target gripper's open/closed state is not readable, so a "
                      "toggle could leave it either way")
    if bool(pose["gripper_open"]) == want_open:
        return {"action": "gripper", "requested": "open" if want_open else "close",
                "already": True, "note": "already in that state; no waypoint spent"}
    await ctx.page.keyboard.press("g")
    swapped = None
    for _ in range(GRIPPER_POLL_ATTEMPTS):
        await asyncio.sleep(GRIPPER_POLL_DELAY_S)
        now = await ctx.gripper_pose()
        state = None if now is None else now.get("gripper_open")
        if state is not None and bool(state) != bool(pose["gripper_open"]):
            swapped = bool(state)
            break
    if swapped is None or swapped != want_open:
        raise Aborted("gripper_mesh_did_not_swap",
                      f"the target gripper did not become "
                      f"{'open' if want_open else 'closed'} after the mesh swap, so "
                      f"nothing was executed")
    # The epoch advances BEFORE the waypoint is executed, so evidence from either
    # side of the change is never mixed even if the execution then aborts with the
    # completion unknown: on an unknown completion the gripper may be in either
    # state, and the old attempt's observations are not evidence about this one.
    retired = STATE.attachment_ref if want_open else None
    STATE.gripper_changed(now_open=want_open, frame=None)
    action = await execute_once(
        ctx, budget, what=("opening" if want_open else "closing") + " the gripper",
        detail={"kind": "gripper", "requested": "open" if want_open else "close"})
    if not want_open:
        # Record which frame the closed hold began on, once it actually executed.
        STATE.closed_since = (await read_frame(ctx))["version"]
    out = dict(action, virtual_state_after=swapped,
               grasp_epoch=STATE.grasp_epoch)
    if retired:
        out["retired_attachment_for"] = retired
        out["retired_note"] = ("opening released the object, so the measured offset "
                               "no longer describes a hold and was dropped; "
                               "re-calibrate after any new close")
    if not want_open:
        out["calibration_note"] = (
            "calibrate_attachment can now use observations taken from here on; move "
            "the object and check it at least twice so two distinct paired frames "
            "carry a re-measured centre")
    return out


GRIPPER_POLL_ATTEMPTS = 20
GRIPPER_POLL_DELAY_S = 0.1


async def solved_target_for(ctx, move, frame):
    """Solve one move against the geometry measured on ``frame``."""
    card = held = None
    if move.get("ref"):
        card, reason = stored_card_for_solve(move["ref"])
        if card is None:
            raise Aborted("reference_not_usable", reason, ref=move["ref"])
    if STATE.attachment_ref and STATE.attachment_ref in STATE.refs:
        held = STATE.refs[STATE.attachment_ref]
    solved = fg.solve_relation(move, card=card, measured=frame["measured"],
                               attachment=STATE.attachment, held_card=held)
    if solved["status"] != "ok":
        raise Aborted("relation_refused", solved.get("reason"), solved=solved)
    return solved


def arrival(frame, target, *, tolerance_m):
    """End-effector arrival at the solved pose: position AND orientation.

    Distinct from the relation residual on purpose. Orientation is part of arriving,
    not a detail: a grasp approach that reaches the right point with the jaws turned
    is not at the pose it was sent to, and closing there closes across a different
    axis.
    """
    out = fg.endpoint_residual(target, frame["measured"], tolerance_m=tolerance_m)
    out["note"] = ("how close the GRIPPER came to the solved pose, in position and "
                   "orientation; says nothing about where the object ended up")
    return out


async def run_stage(ctx, stage, index, budget, report):
    """One validated stage. Appends its own record to ``report`` and returns it."""
    kind = stage["kind"]
    if kind == "gripper":
        entry = await set_gripper(ctx, budget, stage["action"] == "open")
        record = {"stage": index, "kind": "gripper", "physical": True, **entry}
    elif kind == "check":
        frame, results = await check_refs_now(ctx, stage["refs"])
        record = {"stage": index, "kind": "check", "frame": frame["version"],
                  "physical": False, "checked": results}
    elif kind == "calibrate_attachment":
        record = {"stage": index, "kind": "calibrate_attachment",
                  "physical": False, **await calibrate(ctx, stage["ref"])}
    else:
        # A move appends its own record BEFORE it can abort, and then mutates it in
        # place. Returning the record and appending it here loses the whole stage
        # whenever the move raises — which is precisely the case where the segments
        # already driven and the boundary that refused are what the model needs to
        # see. The waypoints were still committed; the record of them must survive
        # with them.
        return await run_move(ctx, stage, index, budget, report)
    report.append(record)
    return record


def stage_blocks_continuation(record):
    """Whether a later PHYSICAL stage may still run after this one. (bool, why).

    This is the gate finding 2 was about. ``run_move`` used to return
    ``within_tolerance=False`` — or a residual it could not measure at all — and the
    loop simply appended it and carried on, so a failed approach was still followed
    by a close and a failed descend by a release. Both are physical dependencies:
    the close only means anything if the fingertips arrived, and the release only if
    the object got where it was going.

    A read-only stage never blocks: ``check`` and ``calibrate_attachment`` may
    honestly report unknown, and that is information, not a failure. What must not
    happen is a *physical* stage proceeding on top of one whose outcome is unknown
    or out of tolerance.
    """
    if record.get("kind") != "move":
        return False, None
    verdict = record.get("verdict") or {}
    if verdict.get("status") != "measured":
        why = verdict.get("reason") or "the residual could not be measured"
        return True, (f"stage {record['stage']} ({record.get('relation')}) could not "
                      f"establish whether it arrived: {why}")
    if not verdict.get("within_tolerance"):
        return True, (f"stage {record['stage']} ({record.get('relation')}) ended "
                      f"{verdict.get('error_m')} m from its goal, outside the "
                      f"{verdict.get('tolerance_m')} m tolerance it was given")
    return False, None


async def calibrate(ctx, ref):
    """Fit the object-to-gripper offset from this ref's own measured observations.

    The window is ``STATE.calibration_window``: only distinct paired frames from the
    current grasp attempt. Which observations were excluded, and why, is reported —
    a calibration that failed for lack of evidence must be distinguishable from one
    that failed because the object is not held.
    """
    # Noted BEFORE the check, because the check itself now verifies the stored hold
    # and may clear it. Either path is a retirement of the offset this call was
    # about to replace, and the reply has to say so however it happened.
    held_before = STATE.attachment_ref == ref and bool(STATE.attachment)
    frame, results = await check_refs_now(ctx, [ref])
    cleared_by_check = held_before and STATE.attachment_ref != ref
    history, excluded = STATE.calibration_window(ref)
    fit = fg.fit_attachment(history) if len(history) >= 2 else {
        "status": "unknown", "reason_code": "insufficient_observations",
        "reason": (f"only {len(history)} measured observation(s) of {ref!r} since the "
                   f"gripper last opened or closed; move the gripper with the object "
                   f"held and check again, so two distinct paired frames each carry a "
                   f"re-measured centre")}
    out = {"ref": ref, "frame": frame["version"], "check": results.get(ref),
           "attachment": fit, "observations_used": len(history),
           "observation_frames": [o["frame"] for o in history],
           "grasp_epoch": STATE.grasp_epoch, "closed_since_frame": STATE.closed_since,
           "observation_window": ("distinct paired frames re-measured since the "
                                  "gripper last changed state")}
    if excluded:
        out["observations_excluded"] = excluded
    if fit["status"] == "attached":
        STATE.attachment = fit
        STATE.attachment_ref = ref
    elif held_before:
        # A failed re-calibration must retire the old offset. Keeping it would let
        # a later relation be solved from an offset this evidence just refused.
        STATE.attachment = None
        STATE.attachment_ref = None
        out["retired_previous_attachment"] = True
        if cleared_by_check:
            out["previous_attachment_cleared_by"] = (
                "the re-measurement in this call, before the new fit was attempted")
    log_event({"event": "calibrate", "ref": ref, "attachment": fit,
               "observations": history})
    return out


async def run_move(ctx, move, index, budget, report=None):
    """One move relation: solve, execute in segments, re-measure, correct.

    The correction loop is the closed-loop part, and it is bounded twice over — by
    ``MAX_CORRECTIONS`` and by ``fg.progress_made``, so a residual the geometry
    cannot reduce ends the stage instead of consuming the whole budget in motion
    that looks like progress.

    ``report`` is the caller's stage list. The record is put into it up front and
    then filled in, so an abort mid-move leaves the stage visible with whatever it
    had established — its segments, its boundaries and the check that refused —
    rather than dropping it and leaving committed waypoints with no stage to
    explain them.
    """
    record = {"stage": index, "kind": "move", "relation": move["relation"],
              "physical": True, "segments": [], "corrections": [],
              "edits": 0, "waypoints": 0}
    if report is not None:
        report.append(record)
    frame = await read_frame(ctx)
    usable, reason = frame_usable(frame)
    if not usable:
        raise Aborted("no_paired_frame",
                      f"this stage needs geometry measured together with telemetry, "
                      f"and the current frame is {reason}")
    # A placement may only be planned from a frame that itself reports the jaws
    # KNOWN CLOSED. Unconditional, and before any re-check, because the hold verdict
    # below is not always produced: fg_check can return matched+unconfirmed on an
    # unknown-class frame and record it as this ref's last checked frame, and the next
    # fg_run on that same frame then gets `same_frame_no_new_evidence` — no verdict to
    # read, no boundary reached yet, and the placement used to be solved from an offset
    # on a frame where nothing showed the object was in the jaws at all. So the
    # requirement is stated about the frame, not about whether a check happened to run.
    #
    # `open` is not the only non-closed state; an unknown or missing class is absent
    # evidence, which is why the stored offset is KEPT here rather than cleared. Not
    # forgetting a hold and being allowed to carry with it are different permissions.
    #
    # Only when an offset exists. With no attachment at all there is nothing to
    # confirm and `solved_target_for` already refuses with the more useful reason
    # (relation_refused: no measured object-to-gripper offset), which names what the
    # model has to do rather than what this frame failed to show.
    if move["relation"] in fg.NEEDS_ATTACHMENT and STATE.attachment:
        gripper_class = frame["measured"].get("gripper_state_class")
        if gripper_class != "closed":
            raise Aborted("attachment_not_confirmed",
                          f"this frame reports the gripper as "
                          f"{gripper_class or 'not reported'}, not known closed, so it "
                          f"is not evidence that {STATE.attachment_ref or 'the object'} "
                          f"is in the jaws and a placement cannot be planned from it; "
                          f"any stored offset was kept, not cleared",
                          frame=frame["version"],
                          gripper_state_class=gripper_class,
                          advice=ATTACH_RECHECK_ADVICE)
    # Re-measure both ends before solving: the SOURCE (the held object, whose
    # offset the target position depends on) and the TARGET reference. Solving
    # against a card measured several waypoints ago is what makes a placement land
    # where the object used to be.
    refs = [r for r in {move.get("ref"), STATE.attachment_ref} if r in STATE.refs]
    if refs:
        pre, checks = await check_refs(ctx, refs, frame)
        record["pre_move_check"] = pre
        # One frame for both ends, named in the reply: a placement solved from a
        # source and a target measured on different frames is arithmetic across two
        # moments, and this is the field that shows it was not.
        record["pre_move_frame"] = frame["version"]
        # "No new evidence" is not the same as "stale". A ref measured on THIS very
        # frame — the ordinary case of binding and then moving in the same turn —
        # has nothing to re-measure and is already current; refusing it would make
        # every first move after a bind impossible. Anything else unmatched means
        # the stored geometry describes a moment that is no longer this one.
        stale = [r for r, v in pre.items()
                 if v.get("status") != "matched"
                 and not (v.get("reason") == "same_frame_no_new_evidence"
                          and v.get("last_measured_frame") == frame["version"]
                          and v.get("valid"))]
        if move.get("ref") in stale:
            raise Aborted("reference_not_usable",
                          f"{move['ref']!r} could not be re-measured on this frame "
                          f"({pre[move['ref']].get('reason')}), so the relation would "
                          f"be solved against where it used to be",
                          checked=pre)
        if STATE.attachment_ref in stale and move["relation"] in fg.NEEDS_ATTACHMENT:
            raise Aborted("held_object_not_usable",
                          f"the held object {STATE.attachment_ref!r} could not be "
                          f"re-measured ({pre[STATE.attachment_ref].get('reason')}), "
                          f"so where it hangs in the grasp is unknown", checked=pre)
        # And the hold verdict this frame's re-measurement produced, when there is one:
        # `unconfirmed` keeps the offset but is not evidence the object is in the jaws.
        held_now = pre.get(STATE.attachment_ref) if STATE.attachment_ref else None
        if (move["relation"] in fg.NEEDS_ATTACHMENT and held_now
                and held_now.get("status") == "matched"
                and (held_now.get("attachment") or {}).get("status") != "attached"):
            hold = held_now.get("attachment") or {}
            raise Aborted("attachment_not_confirmed",
                          f"the hold on {STATE.attachment_ref!r} is not confirmed "
                          f"on this frame ({hold.get('reason') or 'no verdict'}), "
                          f"so this placement would be planned for an object whose "
                          f"place in the grasp is not established; the stored "
                          f"offset was kept, not cleared",
                          checked=pre, hold=hold or None, advice=ATTACH_RECHECK_ADVICE)
    solved = await solved_target_for(ctx, move, frame)
    record["solved"] = {k: solved[k] for k in
                        ("relation", "grasp", "explain", "goal_point", "object_goal",
                         "attachment_offset_m", "attachment_offset_mapping",
                         "approximation") if solved.get(k) is not None}
    target = solved["target"]
    # Which quantity decides this relation, and therefore which one refinement
    # corrects against and which one blocks a dependent physical stage. A reach is
    # about the fingertips; only a placement is about the held object.
    basis = fg.RELATION_VERIFIED_BY[move["relation"]]
    record["verified_by"] = basis
    record["verified_by_note"] = ENDPOINT_BASIS_NOTE if basis == "endpoint" \
        else RELATION_BASIS_NOTE
    # What must still correspond at every observation boundary for the remaining
    # travel to be travel towards this goal.
    #
    # The EXTERNAL reference the goal was solved against is always here: if it stops
    # corresponding mid-travel, the rest of the journey is towards a point derived
    # from geometry that no longer exists, so travel stops there.
    #
    # For a placement the HELD object belongs here too. The comment that used to
    # stand in this place claimed its correspondence was checked at the same
    # boundaries by ``verify_attachment`` inside ``check_refs`` — but check_refs only
    # examines the refs it is GIVEN, and the held ref was not one of them, so nothing
    # looked at the hold between segments. A grip lost during the first segment of a
    # descend was then discovered only by the post-move verdict, after the object had
    # been carried the whole way and the arm had driven to a pose solved from an
    # offset that no longer described anything. Passing it here is what makes the
    # docstring's claim true: the hold is re-verified at each boundary, and travel
    # stops at the boundary that refuses it.
    #
    # Only for a relation that depends on the attachment. A reach with something
    # incidentally held is not made wrong by that object shifting in the jaws, and
    # stopping it would refuse a legitimate move for a fact it does not use.
    depends_on = [move["ref"]] if move.get("ref") in STATE.refs else []
    needs_hold = (move["relation"] in fg.NEEDS_ATTACHMENT
                  and STATE.attachment_ref in STATE.refs
                  and STATE.attachment_ref not in depends_on)
    held_ref = STATE.attachment_ref if needs_hold else None
    verify_refs = depends_on + ([held_ref] if held_ref else [])
    record.update(await drive_to(ctx, target, budget, record,
                                 verify_refs=verify_refs, held_ref=held_ref))
    frame = await read_frame(ctx)

    previous = None
    for attempt in range(fg.MAX_CORRECTIONS + 1 if move["refine"] else 1):
        # Both quantities are always reported; which one is the VERDICT depends on
        # the relation. Keeping the other visible is what lets a model see that the
        # gripper arrived and the object did not, or the reverse.
        endpoint = arrival(frame, target, tolerance_m=move["tolerance_m"])
        relation, checked = await relation_error(ctx, move, solved, frame)
        verdict = endpoint if basis == "endpoint" else relation
        entry = {"attempt": attempt, "frame": frame["version"], "arrival": endpoint,
                 "relation_residual": relation, "checked": checked}
        if attempt == 0:
            record["arrival"] = endpoint
            record["relation_residual"] = relation
            record["post_move_check"] = checked
        else:
            record["corrections"].append(entry)
        record["verdict"] = verdict
        if verdict.get("status") != "measured":
            entry["stop"] = "verdict_unknown"
            record["refine_stopped"] = (
                f"the {basis} residual could not be measured "
                f"({verdict.get('reason')}), so no correction could be justified")
            break
        error = verdict["error_m"]
        if verdict.get("within_tolerance"):
            record["within_tolerance"] = True
            break
        if not move["refine"] or attempt == fg.MAX_CORRECTIONS:
            break
        if previous is not None and not fg.progress_made(previous, error):
            record["refine_stopped"] = (
                f"the previous correction moved the residual from {previous:.4f} m to "
                f"{error:.4f} m, under the "
                f"{int(fg.PROGRESS_FRACTION * 100)}% improvement this loop requires, "
                f"so it stopped rather than spend more waypoints oscillating")
            break
        previous = error
        # Re-solve from the geometry just measured, not from the original numbers:
        # the correction must be against the CURRENT error, or it repeats the
        # displacement that already missed.
        solved = await solved_target_for(ctx, move, frame)
        target = solved["target"]
        record.update(await drive_to(ctx, target, budget, record,
                                     verify_refs=verify_refs, held_ref=held_ref))
        frame = await read_frame(ctx)
    record["within_tolerance"] = bool(record.get("within_tolerance"))
    return record


ATTACH_RECHECK_ADVICE = (
    "re-measure with fg_check on a frame that reports the gripper state; the stored "
    "offset is still there, so one confirming frame is enough and no re-grasp is "
    "implied. if the hold is genuinely gone, re-grasp and calibrate again")
ENDPOINT_BASIS_NOTE = (
    "this relation places the GRIPPER, so its verdict is the measured fingertip pose "
    "against the solved target, position and orientation. relation_residual is "
    "reported too but is not what this stage was asked to achieve")
RELATION_BASIS_NOTE = (
    "this relation places the HELD OBJECT, so its verdict is the object's own "
    "re-measured centre against the goal. arrival is reported too: the gripper "
    "reaching its pose is not the object arriving")


async def measured_start(ctx, frame):
    """Where the ROBOT is, for planning physical travel. Not the virtual target.

    Segmenting from ``ctx.gripper_pose()`` — the blue target mesh — was wrong in a
    specific and silent way: after an execute the mesh already sits at the goal, so
    a re-solve that wanted the robot to travel further computed a zero-length plan
    and the correction executed no motion at all, while reporting a segment. The
    virtual target is an EDITING coordinate (it bounds one 0.1 m browser edit); the
    robot's own telemetry is what physical travel has to be measured from.

    There is deliberately NO fallback. This used to read the mesh whenever
    telemetry was missing and label the result ``virtual_target_fallback``, which
    named the substitution without fixing it: a physical segment plan built from the
    mesh is a plan from where the robot was *told* to go, and after any execute that
    is the previous goal, not the current position. Labelling it does not make the
    distance real, and every residual measured against that plan would be arithmetic
    across two different things. An unmeasurable start is an unknown start, so the
    stage stops here and says which frame could not be read.
    """
    measured = frame["measured"]
    if measured.get("status") == "ok":
        return [measured["fingertip_position"][k] for k in "xyz"], "measured_fingertip"
    raise Aborted("start_position_unknown",
                  f"the robot's own fingertip position is not measurable on frame "
                  f"{frame['version']} ({measured.get('reason') or 'no telemetry'}), "
                  f"so physical travel cannot be planned: the only other position "
                  f"available is the virtual target mesh, which after an execute "
                  f"holds the previous goal rather than where the robot is",
                  frame=frame["version"])


async def drive_to(ctx, target, budget, record, *, verify_refs=(), held_ref=None):
    """Drive to a solved target, re-observing at every physical segment boundary.

    A segment boundary is an OBSERVATION boundary — that is the entire reason
    segmenting exists. Previously every planned segment was executed back to back
    with no frame read between them, so a 0.6 m travel ran fully open-loop against
    geometry measured before it started while the reply described it as segmented
    for re-measurement. Here each boundary reads a fresh frame, re-plans the
    remainder from the newly measured fingertip, and re-checks the references the
    move depends on, stopping rather than continuing on geometry that no longer
    corresponds.

    ``held_ref``, when given, is the carried object this move's goal was solved
    against. It is checked in the same pass, but its failure is a different failure:
    a stale target reference means the goal no longer refers to anything, while a
    lost hold means there is no longer a held object to place. Both stop the travel;
    the reply must not describe one as the other.
    """
    frame = await read_frame(ctx)
    start, start_source = await measured_start(ctx, frame)
    plan = fg.plan_motion(start, target["position"])
    edits = record.get("edits", 0)
    waypoints = record.get("waypoints", 0)
    total = len(plan["segments"])
    boundaries = []
    index = 0
    first = True
    while True:
        segment = plan["segments"][0]
        # Orientation is asserted with the first segment only: re-asserting it at
        # every boundary would spend edits on a pose already reached. An
        # orientation-only move still executes — the plan always carries one
        # segment, whose distance may be zero, so a pure turn is a real waypoint.
        edits += await apply_target(
            ctx, segment["target"],
            approach=target["approach"] if first else None,
            opening=target["opening"] if first else None)
        first = False
        index += 1
        action = await execute_once(
            ctx, budget, what=f"segment {index} of {max(total, index)}",
            detail={"kind": "move_segment", "target": segment["target"]})
        waypoints += 1
        record["segments"].append({"target": segment["target"],
                                   "distance_m": segment["distance_m"],
                                   "edits": segment["edits"],
                                   "completed": action["completed"]})
        frame = await read_frame(ctx)
        at, source = await measured_start(ctx, frame)
        remaining = math.sqrt(sum((target["position"][i] - at[i]) ** 2
                                 for i in range(3)))
        boundary = {"after_segment": index, "frame": frame["version"],
                    "measured_at": fg._r(at), "measured_source": source,
                    "remaining_distance_m": round(remaining, 4)}
        boundaries.append(boundary)
        if remaining <= fg.SEGMENT_ARRIVED_M:
            boundary["stop"] = "at_target"
            break
        if index >= fg.MAX_SEGMENTS_PER_MOVE:
            boundary["stop"] = "segment_limit"
            record["segment_limit_reached"] = (
                f"{index} physical segments were executed and the target is still "
                f"{remaining:.3f} m away; the stage stopped rather than keep driving")
            break
        # The boundary is only usable as a decision point if it is paired, and the
        # references the target was solved against must still correspond — otherwise
        # continuing means driving the remaining distance towards a goal computed
        # from geometry this frame no longer supports.
        usable, reason = frame_usable(frame)
        if not usable:
            boundary["stop"] = reason
            raise Aborted("no_paired_frame",
                          f"the observation boundary after segment {index} is "
                          f"{reason}, so the remaining {remaining:.3f} m cannot be "
                          f"driven against measured geometry",
                          boundaries=boundaries)
        if verify_refs:
            checks, _ = await check_refs(ctx, list(verify_refs), frame)
            boundary["checked"] = checks
            # The hold first, because a lost grip is the more specific fact: the
            # held ref may well have been re-measured perfectly and still no longer
            # be in the jaws, and reporting that as "the goal's geometry moved" would
            # send the model to re-bind a reference that is fine.
            if held_ref and STATE.attachment_ref != held_ref:
                hold = (checks.get(held_ref) or {}).get("attachment") or {}
                why = hold.get("reason") or "the stored offset was refused"
                boundary["stop"] = "attachment_lost"
                raise Aborted("attachment_lost",
                              f"after segment {index} of this move, the hold on "
                              f"{held_ref!r} no longer verifies ({why}), so the "
                              f"remaining {remaining:.3f} m would carry nothing to a "
                              f"goal computed for a carried object",
                              boundaries=boundaries, checked=checks)
            lost = [r for r, v in checks.items()
                    if v.get("status") != "matched" and r != held_ref]
            if lost:
                boundary["stop"] = "reference_not_usable"
                raise Aborted("reference_not_usable",
                              f"after segment {index} of this move, "
                              f"{', '.join(lost)} could not be re-measured, so the "
                              f"remaining {remaining:.3f} m would be driven towards a "
                              f"goal derived from geometry that no longer corresponds",
                              boundaries=boundaries, checked=checks)
            if held_ref and (checks.get(held_ref) or {}).get("status") != "matched":
                # The hold still stands (checked above) but the object itself could
                # not be re-measured on this frame, so where it hangs in the grasp is
                # no longer observed and the placement goal cannot be maintained.
                boundary["stop"] = "held_object_not_usable"
                raise Aborted("held_object_not_usable",
                              f"after segment {index} of this move, the held object "
                              f"{held_ref!r} could not be re-measured "
                              f"({(checks.get(held_ref) or {}).get('reason')}), so "
                              f"where it sits in the grasp is unknown for the "
                              f"remaining {remaining:.3f} m",
                              boundaries=boundaries, checked=checks)
            if held_ref:
                # Last, because it is the least specific of the three and the two
                # above name a concrete thing that went wrong. Reached when the held
                # object WAS re-measured and the offset still stands, and the hold is
                # still not confirmed on this frame — in practice an unknown or
                # missing gripper class.
                #
                # "The offset survived" and "the hold is confirmed" are different
                # facts, and treating the first as the second was a hole. An unknown
                # class leaves the offset in place and reports `unconfirmed`, because
                # absent evidence must not discard a good hold (see
                # verify_attachment); a gate that only asked whether
                # STATE.attachment_ref survived therefore let the remaining segments
                # carry on with nothing having shown the jaws closed around anything.
                # Keeping an offset is a refusal to forget; continuing to carry with
                # it needs positive evidence. So this asks for `attached`.
                hold = (checks.get(held_ref) or {}).get("attachment") or {}
                if hold.get("status") != "attached":
                    boundary["stop"] = "attachment_not_confirmed"
                    raise Aborted("attachment_not_confirmed",
                                  f"after segment {index} of this move, the hold on "
                                  f"{held_ref!r} is not confirmed on this frame "
                                  f"({hold.get('reason') or 'no verdict'}), so the "
                                  f"remaining {remaining:.3f} m would carry an object "
                                  f"whose place in the grasp is not established; the "
                                  f"stored offset was kept, not cleared",
                                  boundaries=boundaries, checked=checks,
                                  hold=hold or None,
                                  advice=ATTACH_RECHECK_ADVICE)
        plan = fg.plan_motion(at, target["position"])
        total = max(total, index + len(plan["segments"]))
    record["boundaries"] = boundaries
    return {"edits": edits, "waypoints": waypoints, "motion_plan": {
        "physical_waypoints": index, "edits": edits,
        "planned_from": start_source,
        "total_distance_m": round(math.sqrt(sum(
            (target["position"][i] - start[i]) ** 2 for i in range(3))), 4),
        "segment_limit_m": fg.ACTION_SEGMENT_M,
        "why_segmented": plan["why_segmented"] if index > 1 else None}}


async def relation_error(ctx, move, solved, frame):
    """Re-measure the relation's own error. (residual, checks)."""
    if move["relation"] == "retreat":
        return {"status": "not_applicable",
                "reason": "retreat has no object relation to satisfy; see arrival"}, {}
    if fg.RELATION_VERIFIED_BY[move["relation"]] != "relation":
        # This is a reach: the thing that had to arrive is the gripper, and the only
        # object in play is the reference the goal was expressed against. Comparing
        # that reference's new centre to a goal derived from itself was the earlier
        # defect — for `above` it is the object against itself (always zero, even
        # with the gripper in the wrong place) and for `approach_grasp` it is a
        # centre against a rim point (about one radius of error, even with the
        # fingertip exactly on the rim). So no relation residual is claimed here;
        # the reference is still re-measured, because a target solved against
        # geometry that has since moved or vanished is not a target.
        ref = move.get("ref")
        checks, _ = await check_refs(ctx, [ref], frame) if ref in STATE.refs else ({}, 0)
        entry = checks.get(ref) or {}
        return {"status": "not_applicable",
                "reason": ("this relation positions the gripper, so its error is the "
                           "arrival residual; the reference was re-measured only to "
                           "confirm the goal still refers to observed geometry"),
                "reference_recheck": entry.get("status", "not_checked")}, checks
    subject = STATE.attachment_ref
    if subject is None or subject not in STATE.refs:
        return {"status": "unknown",
                "reason": ("no calibrated attachment, so there is no held object whose "
                           "arrival could be measured")}, {}
    # A placement's goal was computed from the TARGET's geometry, so a new centre
    # for the source measured against a target position from several waypoints ago
    # is not a newly measured relation — it is a difference across two moments. Both
    # ends are therefore re-measured on this one frame, and if the target cannot be
    # re-measured the residual is unknown rather than quietly assuming it held still.
    target_ref = move.get("ref")
    wanted = [subject] + ([target_ref] if target_ref and target_ref != subject
                          and target_ref in STATE.refs else [])
    checks, _ = await check_refs(ctx, wanted, frame)
    entry = checks.get(subject) or {}
    if entry.get("status") != "matched":
        return {"status": "unknown", "subject": subject,
                "reason": (f"{subject!r} could not be re-measured on this frame "
                           f"({entry.get('reason')}), so whether the relation is "
                           f"satisfied is unknown")}, checks
    resolved = solved
    if len(wanted) > 1:
        target_entry = checks.get(target_ref) or {}
        if target_entry.get("status") != "matched":
            return {"status": "unknown", "subject": subject,
                    "target_ref": target_ref,
                    "reason": (f"the target {target_ref!r} could not be re-measured on "
                               f"this frame ({target_entry.get('reason')}); comparing "
                               f"{subject!r}'s new centre against where the target used "
                               f"to be would not be a measured relation")}, checks
        # Re-solve the goal against the target as it is NOW, so the residual is one
        # frame's source against the same frame's target. Refusals here are reported
        # as unknown: they mean the relation cannot currently be evaluated.
        try:
            resolved = await solved_target_for(ctx, move, frame)
        except Aborted as exc:
            return {"status": "unknown", "subject": subject,
                    "reason": (f"the relation could not be re-solved against the "
                               f"re-measured target ({exc.reason}: {exc.message}), so "
                               f"its residual is unknown")}, checks
    residual = fg.relation_residual(resolved, object_center=entry["center"])
    residual["subject"] = subject
    residual["tolerance_m"] = move["tolerance_m"]
    if len(wanted) > 1:
        residual["measured_against"] = {
            "source_ref": subject, "target_ref": target_ref,
            "frame": frame["version"],
            "note": ("both ends re-measured on this frame and the goal re-solved "
                     "from them, so this is a relation observed now, not a new "
                     "source position against an older target")}
    else:
        residual["static_target_assumption"] = (
            "the goal was expressed against this reference itself, so there is no "
            "second object whose position had to be assumed unchanged")
    if residual.get("status") == "measured":
        residual["within_tolerance"] = residual["error_m"] <= move["tolerance_m"]
    return residual, checks


class FgRunTool(FastGeometryTool):
    name = "fg_run"
    description = (
        FG_PREAMBLE +
        "Execute a short sequence of stages in one call. This is the only tool that "
        "moves the robot in this interface, and it does the intermediate work "
        "itself: it re-measures between waypoints, solves each relation against the "
        "geometry just measured, and reports what happened — you are not asked to "
        "poll in between.\n\n"
        "Stages, each exactly one key:\n"
        "  {'move': {...}}                  one relation, below\n"
        "  {'gripper': 'open'|'close'}      open/close in its OWN waypoint\n"
        "  {'calibrate_attachment': {'ref': id}}  measure the held object's offset\n"
        "  {'check': {'refs': [id, ...]}}   re-measure without acting\n\n"
        "Relations, solved from measured geometry, never from a guessed pose:\n"
        "  approach_grasp  fingertips standoff_m short of a grasp candidate on ref\n"
        "  above           fingertip height_m above ref's centre, orientation kept\n"
        "  align_over      the HELD object height_m above ref's top surface\n"
        "  descend_to      the HELD object down to height_m above ref's top\n"
        "  retreat         back off distance_m along the measured approach axis\n"
        "align_over and descend_to move the held object, so they need a measured "
        "object-to-gripper offset: run calibrate_attachment after closing, with real "
        "motion between its observations, or they are refused.\n\n"
        "How to compose stages is your decision. A grasp is usually: approach_grasp, "
        "close, a small retreat or above, calibrate_attachment. A place is usually: "
        "align_over, descend_to, open. Splitting that across several fg_run calls is "
        "fine and lets you look in between.\n\n"
        "The whole request is validated before anything moves, so a bad later stage "
        "costs no motion. Set refine on a move to let the executor correct up to "
        f"{fg.MAX_CORRECTIONS} times against the re-measured residual; it stops early "
        "when a correction does not improve it.\n\n"
        "The reply separates three things that are easy to conflate: 'arrival' is how "
        "close the GRIPPER came to the solved pose, 'relation_residual' is how far "
        "the OBJECT is from the goal, and a benchmark result is reported only if the "
        "harness declared one. A run that stops early returns aborted with a reason "
        "and 'committed' — the physical actions already submitted. Do not re-issue "
        "those."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "stages": {
                "type": "array", "minItems": 1, "maxItems": fg.MAX_STAGES,
                "description": "Stages in order. Each object has exactly one key.",
                "items": {
                    "type": "object",
                    "properties": {
                        "move": {
                            "type": "object",
                            "properties": {
                                "relation": {"type": "string",
                                             "enum": list(fg.RELATIONS)},
                                "ref": {"type": "string",
                                        "description": "Reference id from fg_bind."},
                                "grasp": {"type": "string",
                                          "description": ("Grasp candidate id; "
                                                          "default the nearest.")},
                                "standoff_m": {"type": "number",
                                               "description": "approach_grasp, [0, 0.2]."},
                                "height_m": {"type": "number",
                                             "description": ("above/align_over/"
                                                             "descend_to, [-0.02, 0.3].")},
                                "distance_m": {"type": "number",
                                               "description": "retreat, [0.01, 0.2]."},
                                "tolerance_m": {"type": "number",
                                                "description": "[0.002, 0.05]. Default 0.01."},
                                "refine": {"type": "boolean",
                                           "description": ("Correct against the "
                                                           "re-measured residual.")},
                            },
                            "required": ["relation"],
                            "additionalProperties": False,
                        },
                        "gripper": {"type": "string", "enum": ["open", "close"]},
                        "calibrate_attachment": {
                            "type": "object",
                            "properties": {"ref": {"type": "string"}},
                            "required": ["ref"], "additionalProperties": False},
                        "check": {
                            "type": "object",
                            "properties": {"refs": {"type": "array",
                                                    "items": {"type": "string"},
                                                    "minItems": 1}},
                            "required": ["refs"], "additionalProperties": False},
                    },
                    "additionalProperties": False,
                },
            },
            "budget": {
                "type": "object",
                "description": ("Caps for this call. waypoints 1.."
                                f"{fg.MAX_WAYPOINT_BUDGET} (default "
                                f"{fg.DEFAULT_WAYPOINT_BUDGET}), seconds 5.."
                                f"{fg.MAX_DEADLINE_S:g} (default "
                                f"{fg.DEFAULT_DEADLINE_S:g})."),
                "properties": {"waypoints": {"type": "integer"},
                               "seconds": {"type": "number"}},
                "additionalProperties": False,
            },
        },
        "required": ["stages"],
        "additionalProperties": False,
    }

    async def __call__(self, ctx, arguments):
        try:
            request = fg.validate_run(arguments)
        except fg.Rejected as exc:
            # Nothing ran: this is the point of validating the whole request.
            return await self.reply(ctx, {
                "status": "rejected", "message": str(exc), "executed_stages": 0,
                "committed": [], "note": "nothing was executed"})
        STATE.execute_calls += 1
        budget = Budget(request["budget"]["waypoints"], request["budget"]["seconds"])
        committed_before = len(STATE.committed)
        report: list[dict] = []
        payload = {"stages": report, "physical": request["physical"]}
        await clear_overlay(ctx)
        # The stage an abort happened in, named explicitly rather than inferred from
        # the report's length: a move records itself before it can abort (see
        # run_move), so counting entries would report the stage after the one that
        # failed.
        current = 0
        try:
            for index, stage in enumerate(request["stages"]):
                current = index
                record = await run_stage(ctx, stage, index, budget, report)
                # A stage that did not arrive, or could not tell whether it did,
                # must not be followed by a stage that depends on it having
                # arrived. Closing after a failed approach grips air; opening after
                # a failed descend drops the object from height. The already
                # executed stages and their committed actions are kept: this stops
                # the run, it does not discard what happened.
                blocked, why = stage_blocks_continuation(record)
                if blocked and index + 1 < len(request["stages"]):
                    raise Aborted(
                        "dependent_stage_blocked", why,
                        blocked_stage=index,
                        not_executed=[s["kind"] for s
                                      in request["stages"][index + 1:]],
                        advice=("re-measure with fg_check, then issue a corrected "
                                "sequence; the stages above already ran"))
                if blocked:
                    # The same failure, with nothing after it. It used to be reported
                    # under status ok because the blocking check asked only whether a
                    # LATER stage existed — so a run whose single move ended outside
                    # tolerance, or whose residual could not be measured at all, was
                    # topped with the same word as a run that did what it was asked.
                    # `within_tolerance: false` was there to be read, but the status
                    # is what says whether the request was carried out, and this one
                    # was not. Nothing is discarded: every executed stage, its
                    # measurements and its committed actions are already in the
                    # payload and stay there.
                    raise Aborted(
                        "move_did_not_arrive", why,
                        blocked_stage=index, not_executed=[],
                        incomplete=("this was the last stage, so nothing was stopped "
                                    "— the request itself was not carried out"),
                        advice=("re-measure with fg_check and re-issue this move "
                                "(a tighter standoff, refine, or a re-bind if the "
                                "reference has drifted); the actions above already "
                                "ran and must not be re-issued"))
            payload["status"] = "ok"
        except Aborted as exc:
            payload.update(status="aborted", reason=exc.reason, message=exc.message,
                           aborted_at_stage=current, **exc.detail)
        except Exception as exc:  # a browser fault mid-run is an abort, not a crash
            payload.update(status="aborted", reason="execution_error",
                           message=f"{type(exc).__name__}: {exc}"[:300],
                           aborted_at_stage=current)
        payload["budget"] = budget.report()
        payload["committed"] = STATE.committed[committed_before:]
        payload["committed_note"] = ("physical actions this call submitted; already "
                                     "done, do not re-issue them")
        payload["attachment"] = (STATE.attachment or {"status": "none"}).get("status")
        payload["refs"] = {r: bool(c.get("valid")) for r, c in STATE.refs.items()}
        payload["waypoints_this_episode"] = STATE.waypoints
        # The benchmark's own verdict, when the harness has published one. Kept
        # apart from every measurement above: task success is the harness's
        # judgement, not something these fits can establish, and a run that
        # satisfied its relation may still not have satisfied the task.
        payload["benchmark"] = await benchmark_state(ctx)
        images = await self.snap(ctx)
        return await self.reply(ctx, payload, images=images)


async def benchmark_state(ctx):
    """Whether the harness has declared this episode over. Never inferred."""
    try:
        info = await _sim_success()
    except Exception:
        info = None
    if not isinstance(info, dict) or not isinstance(info.get("success"), bool):
        return {"status": "not_reported",
                "note": ("the harness has not published a verdict for this episode; "
                         "this is not evidence either way")}
    if not info["success"]:
        return {"status": "running", "success": False,
                "note": "the harness has not judged the task complete"}
    return {"status": "succeeded", "success": True,
            "note": ("the harness declared the task goal satisfied and the UI is "
                     "closing; stop here, further tool calls will fail")}


async def _sim_success():
    """success.json from record_sim, or None. Import-late so tests can patch it."""
    try:
        from . import mcp_server
    except ImportError:
        import mcp_server
    return await mcp_server._fetch_sim_json("success.json")

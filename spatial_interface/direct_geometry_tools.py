"""Astra-authored geometric programs over measured evidence, direct to controller."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
import os
import time
import uuid

import numpy as np
import mcp.types as types

try:
    from . import coarse_fine_tools as cf
    from . import coarse_fine_policy as cfp
    from . import fast_geometry_tools as fgt
    from . import geometry_ref as geom
    from . import geometry_workspace as gw
    from . import direct_control as dc
    from . import dg_feedback as feedback
    from . import dg_receipt as receipt
except ImportError:
    import coarse_fine_tools as cf
    import coarse_fine_policy as cfp
    import fast_geometry_tools as fgt
    import geometry_ref as geom
    import geometry_workspace as gw
    import direct_control as dc
    import dg_feedback as feedback
    import dg_receipt as receipt

LOG_VARIABLE = "VIA_DIRECT_GEOMETRY_LOG_FILE"


def registry():
    # These describe actual adapter dependencies. Observation IDs record when the
    # paired K/E and pixels were sampled; they are not simulator object-state IDs.
    return gw.SourceRegistry({
        "depth": gw.Source("depth_sensor", record="fixed-camera RGB-D plus wrist RGB-D"),
        "K": gw.Source("camera_intrinsic_calibration", record="record_sim paired cam_info.K"),
        "E": gw.Source("provided_camera_extrinsics", record="record_sim paired cam_info.E"),
        "cloud": gw.Source("derived", ("depth", "K", "E")),
        "robot": gw.Source("robot_pose_sensor", record="paired observe_proprio EEF pose"),
        "jaws": gw.Source("gripper_sensor", record="finger joint width"),
    })


class State:
    def __init__(self):
        self.workspace = gw.Workspace(registry())
        self.frame = None
        self.proxies = {}
        self.refs = {}
        self.executed = []
        self.policy_calls = 0
        self.last_observation = None
        self.program_ids = set()
        # Structured history of every dg_policy call of this episode, rejected
        # ones included: dg_state reads it, and no other tool may drop it.
        self.programs = []


STATE = State()


def reset_episode():
    global STATE
    STATE = State()


def log(event):
    path = os.environ.get(LOG_VARIABLE)
    if path:
        with open(path, "a") as stream:
            stream.write(json.dumps(dict(event, t=time.time()), default=str) + "\n")


def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k != "grasp_candidates"}
    if isinstance(value, (list, tuple)):
        return [public(v) for v in value]
    return value


async def observe(ctx, radius=0.08, min_seq=None):
    """Wait for transport pairing, without stepping or reissuing any command."""
    deadline = time.monotonic() + 10
    while True:
        view = await cf.fine_observation(ctx, radius_m=radius)
        robot = await cf.fetch_sim_json("robot_state.json")
        version = geom.parse_observation_version(view.get("frame"))
        okay = (robot and version and view.get("paired") and view.get("depth_paired_with_frame")
                and str(version["epoch"]) == str(robot.get("epoch"))
                and version["seq"] == robot.get("seq")
                and (min_seq is None or robot["seq"] >= min_seq))
        if okay:
            break
        if time.monotonic() >= deadline:
            raise gw.BoundaryError("paired_robot_rgbd_unavailable")
        await asyncio.sleep(0.1)
    view["measured_end_effector"] = dc.measured_view(robot)
    view["measured_gripper"] = view["measured_end_effector"]["gripper_state_class"]
    view["sim_steps_used"] = robot["sim_steps"]
    if STATE.frame != view["frame"]:
        flat = await ctx.page.evaluate("""() => {
            const a = pcGeo.attributes.position.array, out = [];
            const stride = Math.max(1, Math.ceil(a.length / 3 / 4096));
            for (let i=0; i<a.length/3; i+=stride) out.push([a[3*i],a[3*i+1],a[3*i+2]]);
            return out;
        }""")
        if geom.observation_version(await ctx.observe_geometry()) != view["frame"]:
            raise gw.BoundaryError("cloud_changed_during_capture")
        points = tuple(tuple(geom.ui_to_robot(p, ctx.ui_z_offset)) for p in flat)
        STATE.workspace.publish(gw.Observation(view["frame"], "robot", points, ("cloud",),
                                                robot["sim_steps"] / robot["control_freq"]))
        STATE.frame = view["frame"]
    # dg_state keeps its own copy, so a later call cannot rewrite what the model
    # was already shown.
    STATE.last_observation = deepcopy(view)
    return view, robot


def propose(name, pose, frame, card):
    hypothesis = STATE.workspace.propose(name, frame, pose)
    STATE.refs[name] = hypothesis
    STATE.proxies[name] = dict(card, bound_frame=frame)
    return {"reference": asdict(hypothesis),
            "support": asdict(STATE.workspace.check(hypothesis.ref, 0.08)),
            "note": "hypothesis only; point support does not verify identity, contact or grasp"}


def compile_program(program, robot, frame):
    """Freeze model-authored targets once; later relative poses use prior targets.

    This is documented straight-line program semantics, not a prediction of actual
    motion. Executor checks each measured arrival before it permits the next step.
    """
    cursor = dc.measured_view(robot)
    commands, targets = [], []
    for step in program["steps"]:
        kind = step["kind"]
        if kind == "observe":
            continue
        if kind == "gripper":
            commands.append(gw.Gripper(step["action"]))
            targets.append(None)
        elif kind == "dwell":
            commands.append(gw.Hold(step["sim_steps"] / robot["control_freq"]))
            targets.append(None)
        else:
            name = step["proxy"]
            card = STATE.proxies.get(name) if name else None
            hypothesis = STATE.refs.get(name) if name else None
            if name and (hypothesis is None or hypothesis.observation_id != frame):
                raise gw.BoundaryError("proxy_from_older_frame_rebind_or_propose_explicitly")
            target = cfp.resolve_pose(step, measured=cursor, proxy=card)
            if target["status"] != "ok":
                raise gw.BoundaryError(target["reason"])
            pose = dc.pose_from_axes(target["position"], target["approach"], target["opening"])
            reference = None
            request_pose = pose
            if hypothesis:
                reference = hypothesis.ref
                request_pose = (np.linalg.inv(hypothesis.sensor_from_proxy) @ np.asarray(pose)).tolist()
            commands.append(gw.Move(request_pose, reference, target["tolerance_m"],
                                    target["orientation_tolerance_deg"]))
            targets.append(target)
            cursor = dict(cursor, fingertip_position=dict(zip("xyz", target["position"])),
                          approach=dict(zip("xyz", target["approach"])),
                          opening=dict(zip("xyz", target["opening"])))
    return commands, targets


class DirectTool:
    #: Pure payload -> model-facing receipt. The raw payload is logged first and
    #: in full; only this projection decides what the reply spends characters on.
    project = staticmethod(receipt.receipt)

    async def reply(self, ctx, payload, images=False, feedback_images=()):
        payload = public(dict(payload, timing_id=ctx.timing_id))
        log({"tool": self.name, "payload": payload})
        view = type(self).project(payload)
        # Canonical json.dumps with default formatting: a client that
        # de-duplicates repeated observations re-serializes the text it received
        # and compares it byte for byte, so custom separators would disable it.
        blocks = [types.TextContent(type="text", text=json.dumps(view, default=str))]
        if images:
            blocks += await ctx.snap()
        for label, image in feedback_images:
            ctx._save_screenshot(image)
            blocks.extend([types.TextContent(type="text", text=label), ctx._image_content(image)])
        return blocks


class DgLookTool(DirectTool):
    name = "dg_look"
    description = "Observe paired RGB-D, cloud and actual robot EEF pose. No actuation. Robot metres; +Z up."
    input_schema = deepcopy(cf.CfLookTool.input_schema)
    project = staticmethod(receipt.look_receipt)

    async def __call__(self, ctx, arguments):
        mode = feedback.variant()
        radius = cfp.validate_step({"observe": arguments})["radius_m"]
        view, _ = await observe(ctx, radius)
        payload = {"status": "ok", "observation": cf.observation_view(view)}
        extra = await feedback.finish(ctx, payload, mode) if mode != "baseline" else []
        return await self.reply(ctx, payload, True, extra)


class DgPolicyTool(DirectTool):
    name = "dg_policy"
    description = ("Optionally fit a model-selected ROI or propose an explicit geometric reference, then "
                   "commit and execute YOUR short pose/gripper/dwell program directly on the robot controller. "
                   "No blue target editing. Robot metres; approach=EEF +Z, opening=EEF +Y. "
                   "All targets are compiled once: the first offset is based on measured EEF, subsequent "
                   "offsets on the preceding requested pose. frame=robot offsets use robot XYZ; "
                   "frame=gripper offsets use approach/opening/approach-cross-opening; proxy offsets use "
                   "robot XYZ about its centre. Each measured arrival is checked before the next command. "
                   "observe is optional and must be LAST; every call returns a fresh final observation. "
                   "To only fit/edit/check a proxy use steps=[{observe:{}}], which emits zero motion. "
                   "References retire after motion; rebind or explicitly propose a new hypothesis to reuse "
                   "them. No implicit tracking, grasp candidate selection, recovery or replay. "
                   "The reply reports, per command, the measured start and end, the committed absolute "
                   "target and the position/orientation residuals against your own tolerances; completed "
                   "commands are summarised and dg_state holds every command in full. "
                   "End the call at a desired Astra feedback boundary; decide the next program yourself.")
    input_schema = deepcopy(cf.CfPolicyTool.input_schema)
    project = staticmethod(receipt.policy_receipt)
    input_schema["properties"]["program_id"] = {"type": "string", "description": "Optional unique ID; a reused ID never runs twice."}
    input_schema["properties"]["propose"] = {
        "type": "object", "description": "Model hypothesis, not a new object measurement. Alternative to bind.",
        "properties": {"name": {"type": "string"},
                       "pose": {"type": "array", "minItems": 4, "maxItems": 4,
                                "items": {"type": "array", "minItems": 4, "maxItems": 4,
                                          "items": {"type": "number"}}}},
        "required": ["name", "pose"], "additionalProperties": False}

    async def __call__(self, ctx, arguments):
        mode = feedback.variant()  # Invalid experiment configuration never actuates.
        started = time.monotonic()
        STATE.policy_calls += 1
        payload = {"status": "rejected", "executed": []}
        backend = None
        before, selection, cards, feedback_errors = None, None, [], []
        try:
            args = dict(arguments)
            pid = args.pop("program_id", "p"+uuid.uuid4().hex[:16])
            proposal = args.pop("propose", None)
            program = cfp.validate_program(args)
            if proposal and program["bind"]:
                raise gw.BoundaryError("choose_bind_or_propose")
            if any(s["kind"] == "observe" for s in program["steps"][:-1]):
                raise gw.BoundaryError("observe_must_end_program")
            if not isinstance(pid, str) or not pid or pid in STATE.program_ids:
                raise gw.BoundaryError("program_id_invalid_or_reused")
            STATE.program_ids.add(pid)
            payload["program_id"] = pid
            view, robot = await observe(ctx)
            frame = view["frame"]
            payload["observation_before"] = cf.observation_view(view)
            if mode == "paired":
                try:
                    before = await feedback.capture(ctx, view)
                except Exception as exc:
                    feedback_errors.append("before_capture: "+str(exc))
            if program["bind"]:
                if mode != "baseline":
                    try:
                        selection = await feedback.selected_returns(ctx, program["bind"], frame)
                    except Exception as exc:
                        feedback_errors.append("selection: "+str(exc))
                binding = await cf.bind_proxy(ctx, program["bind"])
                payload["bind"] = public(binding)
                name = program["bind"]["objects"][0]["name"]
                card = cf.STATE.proxies.get(name)
                if card:
                    cards = [card]
                if binding.get("status") != "ok" or not card or not card.get("valid"):
                    raise gw.BoundaryError("proxy_fit_invalid_no_motion")
                center = card["center"]
                center = [center[k] for k in "xyz"] if isinstance(center, dict) else center
                pose = np.eye(4)
                pose[:3, 3] = center
                payload["geometry_check"] = propose(name, pose.tolist(), frame, card)
            elif proposal is not None:
                if set(proposal) != {"name", "pose"}:
                    raise gw.BoundaryError("invalid_proposal_fields")
                pose = gw.rigid(proposal["pose"])
                card = {"name": proposal["name"], "valid": True,
                        "center": [pose[i][3] for i in range(3)], "up": [pose[i][2] for i in range(3)],
                        "source": "explicit_model_hypothesis_not_object_measurement"}
                cards = [dict(card, bound_frame=frame)]
                payload["geometry_check"] = propose(proposal["name"], pose, frame, card)
            commands, targets = compile_program(program, robot, frame)
            payload["compiled_targets"] = targets
            if mode != "baseline" and not cards:
                names = {step.get("proxy") for step in program["steps"]}
                cards = [card for name, card in STATE.proxies.items() if name in names]
            if commands:
                committed = STATE.workspace.commit(pid, frame, commands)
                payload["commit"] = asdict(committed)
                backend = dc.RpcBackend(seconds=program["budget"]["seconds"]-(time.monotonic()-started))
                result = await asyncio.to_thread(gw.Executor(STATE.workspace, backend).execute, pid)
                payload["status"] = result.status
                payload["execution"] = asdict(result)
                STATE.executed.extend(backend.records)
                payload["executed"] = backend.records
                min_seq = max((r.get("reply", {}).get("robot", {}).get("seq", 0) for r in backend.records), default=0)
                radius = program["steps"][-1].get("radius_m", 0.08)
                view, robot = await observe(ctx, radius, min_seq)
            else:
                payload["status"] = "geometry_only"
            payload["observation"] = cf.observation_view(view)
        except (gw.BoundaryError, cfp.Rejected, ValueError, KeyError) as exc:
            payload["reason"] = str(exc)
            if backend is not None:
                payload["status"] = "stopped"
                payload["executed"] = backend.records
                if mode != "baseline":
                    try:
                        final, _ = await observe(ctx)
                        payload["observation"] = cf.observation_view(final)
                    except Exception as capture_error:
                        feedback_errors.append("stopped_observation: "+str(capture_error))
        extra = (await feedback.finish(ctx,payload,mode,before,selection,cards,feedback_errors)
                 if mode != "baseline" else [])
        payload["cost_s"] = round(time.monotonic()-started, 3)
        # Structured history, including rejected and stopped programs, kept as
        # this module's own copy so a later call can neither mutate nor lose it.
        STATE.programs.append(receipt.program_record(public(payload), STATE.policy_calls))
        return await self.reply(ctx, payload, True, extra)


def proxy_entries():
    """Every stored geometric hypothesis with its binding frame and staleness."""
    return [receipt.proxy_state_entry(
                name, revision=hypothesis.ref.revision,
                bound_frame=hypothesis.observation_id,
                current=hypothesis.observation_id == STATE.frame,
                card=STATE.proxies.get(name))
            for name, hypothesis in STATE.refs.items()]


_PAGE_SCHEMA = {"type": "object", "additionalProperties": False,
                "properties": {"offset": {"type": "integer", "minimum": 0},
                               "limit": {"type": "integer", "minimum": 1}}}


def page_schema(maximum, description):
    schema = deepcopy(_PAGE_SCHEMA)
    schema["properties"]["limit"]["maximum"] = maximum
    schema["description"] = description
    return schema


class DgStateTool(DirectTool):
    name = "dg_state"
    description = (
        "Read what earlier calls already produced: your program history with its measured command "
        "diagnostics, your stored geometric hypotheses and the last observation. No new sensing, no "
        "motion, nothing re-measured. Default: the most recent 8 program summaries, newest first. "
        "Ask for one program's full per-command diagnostics with program_id; page it with "
        f"commands:{{offset,limit}} (limit up to {receipt.MAX_COMMAND_PAGE}, the maximum program length). "
        f"programs:{{offset,limit}} (limit up to {receipt.MAX_PROGRAM_PAGE}) and "
        f"proxies:{{offset,limit}} (limit up to {receipt.MAX_PROXY_PAGE}) page the other lists. "
        "include_depth_samples returns that program's stored local-depth returns. "
        "An invalid query is refused and changes nothing.")
    input_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "programs": page_schema(receipt.MAX_PROGRAM_PAGE,
                                    "Page of program summaries, newest first."),
            "program_id": {"type": "string",
                           "description": "Return this one program's full command diagnostics."},
            "commands": page_schema(receipt.MAX_COMMAND_PAGE,
                                    "Command page within that program. Requires program_id."),
            "proxies": page_schema(receipt.MAX_PROXY_PAGE, "Page of stored hypotheses."),
            "include_depth_samples": {
                "type": "boolean",
                "description": "Stored local-depth returns of that program's observations. Requires program_id."}}}
    project = staticmethod(lambda payload: payload)

    async def __call__(self, ctx, arguments):
        try:
            query = receipt.validate_state_query(arguments)
        except receipt.StateQueryError as exc:
            # A malformed read is refused here: no sensing, no motion, no state change.
            return await self.reply(ctx, {"status": "rejected", "reason": str(exc),
                                          "note": "dg_state read refused; nothing was sensed or moved"})
        return await self.reply(ctx, receipt.state_receipt(
            policy_calls=STATE.policy_calls, records=STATE.programs,
            proxies=proxy_entries(), last_observation=STATE.last_observation, query=query))

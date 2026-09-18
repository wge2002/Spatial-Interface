"""Typed robot-frame transport. No browser actions, object state or task policy."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
import re
import time

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from . import geometry_workspace as gw
except ImportError:
    import geometry_workspace as gw

PROTOCOL = "spatial_interface.direct.v1"


def pose_from_axes(position, approach, opening):
    """EEF +Z points toward the fingers; EEF +Y is the jaw opening axis."""
    a, o = np.asarray(approach, float), np.asarray(opening, float)
    a = a / np.linalg.norm(a)
    o = o - a * np.dot(a, o)
    o = o / np.linalg.norm(o)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((np.cross(o, a), o, a))
    pose[:3, 3] = position
    return gw.rigid(pose.tolist())


def measured_view(state):
    p = np.asarray(state["pose"])
    return {"status": "ok", "source": "robot_proprioception", "frame": "robot",
            "fingertip_position": dict(zip("xyz", p[:3, 3].tolist())),
            "approach": dict(zip("xyz", p[:3, 2].tolist())),
            "opening": dict(zip("xyz", p[:3, 1].tolist())),
            "gripper_width_m": state["gripper_width_m"],
            "gripper_state_class": "open" if state["commanded_gripper_open"] else "closed",
            "gripper_state_note": "command state and measured width; neither proves a grasp"}


def sensor_state(obs, *, seq, sim_steps, width_max_m, commanded_open, control_freq):
    """Only robot proprioception from the paired observation enters this channel."""
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(obs["ee_quat"]).as_matrix()
    pose[:3, 3] = obs["ee_pos"]
    return {"pose": pose.tolist(), "seq": int(seq), "sim_steps": int(sim_steps),
            "gripper_width_m": float(np.asarray(obs["gripper_open"]).item()) * width_max_m,
            "commanded_gripper_open": int(commanded_open), "control_freq": float(control_freq),
            "sources": ["robot_pose_sensor", "gripper_sensor"],
            "pose_convention": "robot-frame EEF: +Z approach, +Y jaw opening; metres"}


def encode(command_id, command):
    kind = {gw.Move: "pose", gw.Gripper: "gripper", gw.Hold: "hold"}[type(command)]
    return {"protocol": PROTOCOL, "id": command_id, "kind": kind,
            "value": (command.pose if kind == "pose" else
                      command.state if kind == "gripper" else command.seconds)}


def decode(request):
    if not isinstance(request, dict) or set(request) != {"protocol", "id", "kind", "value"}:
        raise ValueError("invalid_command_fields")
    if request["protocol"] != PROTOCOL or not isinstance(request["id"], str) or not re.fullmatch(
            r"[A-Za-z0-9_./-]{1,120}", request["id"]):
        raise ValueError("invalid_command_identity")
    kind, value = request["kind"], request["value"]
    if kind == "pose":
        return gw.Move(value)
    if kind == "gripper":
        return gw.Gripper(value)
    if kind == "hold":
        return gw.Hold(value)
    raise ValueError("invalid_command_kind")


class RpcBackend:
    """One command, one acknowledgement. Never retries a possibly submitted ID."""
    def __init__(self, *, seconds=120):
        self.deadline = time.monotonic() + seconds
        self.records = []

    def execute(self, command_id, command):
        from websockets.sync.client import connect
        request = encode(command_id, command)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("program_budget_exhausted")
        base = int(os.environ.get("SPHINX_BASE_PORT", "8100"))
        record = {"command_id": command_id, "request": request,
                  "submitted": False, "completed": None}
        self.records.append(record)
        start = time.monotonic()
        with connect(f"ws://localhost:{base+2}", open_timeout=min(5, remaining),
                     close_timeout=1) as socket:
            socket.send(json.dumps(request))
            record["submitted"] = True
            reply = json.loads(socket.recv(timeout=max(0.01, self.deadline-time.monotonic())))
        record.update(reply=reply, wall_s=time.monotonic()-start)
        if reply.get("id") != command_id or reply.get("status") not in ("completed", "stopped"):
            raise RuntimeError("command_completion_unverified")
        state = reply["robot"]
        record["completed"] = reply["status"] == "completed"
        return gw.Acknowledgement(command_id, reply["status"], ("robot", "jaws"),
                                  measured_pose=state["pose"],
                                  gripper_width_m=state["gripper_width_m"],
                                  elapsed_s=reply["sim_steps_used"] / state["control_freq"])

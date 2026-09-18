"""Geometry hypotheses and explicit robot programs; no live device adapter.

Trusted adapters supply observations and the source registry. A registry checks
declared dependencies, not whether an adapter tells the truth. Synthetic inputs
are disabled by default. Nothing here imports the UI, simulator or evaluator.
Execution claims last for this workspace's lifetime, not across process restarts.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import math
from numbers import Real
import re
from threading import RLock
from types import MappingProxyType
from typing import Protocol


class BoundaryError(ValueError):
    pass


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BoundaryError("expected_finite_number")
    value = float(value)
    if not math.isfinite(value):
        raise BoundaryError("expected_finite_number")
    return value


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w.:-]{1,96}", value):
        raise BoundaryError("invalid_identifier")
    return value


def _ids(values):
    if isinstance(values, str):
        raise BoundaryError("source_ids_must_be_sequence")
    return tuple(_name(v) for v in values)


def rigid(value):
    """Own an immutable, finite, proper 4x4 rigid transform (metres)."""
    try:
        m = tuple(tuple(_number(x) for x in row) for row in value)
    except (TypeError, ValueError) as exc:
        raise BoundaryError("invalid_rigid_transform") from exc
    if len(m) != 4 or any(len(row) != 4 for row in m):
        raise BoundaryError("invalid_rigid_transform")
    if any(abs(m[3][j] - (1 if j == 3 else 0)) > 1e-8 for j in range(4)):
        raise BoundaryError("invalid_homogeneous_row")
    if any(abs(sum(m[k][i] * m[k][j] for k in range(3)) - (i == j)) > 1e-6
           for i in range(3) for j in range(3)):
        raise BoundaryError("rotation_not_orthonormal")
    det = (m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1])
           - m[0][1] * (m[1][0]*m[2][2] - m[1][2]*m[2][0])
           + m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0]))
    if abs(det - 1) > 1e-6:
        raise BoundaryError("rotation_not_proper")
    return m


IDENTITY = rigid(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))


def compose(a, b):
    return rigid(tuple(tuple(sum(a[i][k] * b[k][j] for k in range(4))
                             for j in range(4)) for i in range(4)))


# ── arrival geometry ─────────────────────────────────────────────────────────
#
# Extracted verbatim from Executor._validate_ack so the model-facing diagnostics
# and the gate that actually stops a program cannot drift apart. These are pure
# functions of two poses: they read no state, decide nothing and never round.
# A caller that displays a rounded figure must still gate on the raw value.

def position_error_m(measured, target):
    """Straight-line distance between two poses' translations, in metres."""
    return math.dist([measured[i][3] for i in range(3)],
                     [target[i][3] for i in range(3)])


def orientation_error_deg(measured, target):
    """Geodesic angle between two poses' rotations, in degrees."""
    cosine = (sum(measured[i][j] * target[i][j]
                  for i in range(3) for j in range(3)) - 1) / 2
    return math.degrees(math.acos(max(-1, min(1, cosine))))


def arrival_error(measured, target):
    """Both residuals as one pair; the exact quantities the gate compares."""
    return position_error_m(measured, target), orientation_error_deg(measured, target)


def arrival_exceeds(measured, target, position_tolerance_m, orientation_tolerance_deg):
    """True when this arrival misses either tolerance -- the gate's own test."""
    distance, angle = arrival_error(measured, target)
    return distance > position_tolerance_m or angle > orientation_tolerance_deg


@dataclass(frozen=True)
class Source:
    kind: str
    parents: tuple[str, ...] = ()
    record: str | None = None
    synthetic: bool = False

    def __post_init__(self):
        if not isinstance(self.kind, str) or type(self.synthetic) is not bool:
            raise BoundaryError("invalid_source_declaration")
        if self.record is not None and not isinstance(self.record, str):
            raise BoundaryError("invalid_source_record")
        object.__setattr__(self, "parents", _ids(self.parents))


class SourceRegistry:
    """Application-owned registry; never construct it from model tool arguments.

    Records identify the sensor setup/source; they need not be independently
    measured calibration artifacts. The agreed protocol allows provided exact
    camera extrinsics, including the simulator's known rig/wrist camera pose.
    No file is read or registered here; actual adapter dependencies need audit.
    """
    EXTRINSICS = frozenset({"camera_extrinsic_calibration", "provided_camera_extrinsics",
                           "sim_camera_world_pose"})
    ROOTS = frozenset({"rgb_sensor", "depth_sensor", "joint_encoder",
                      "gripper_sensor", "robot_model", "robot_pose_sensor",
                      "camera_intrinsic_calibration"}) | EXTRINSICS
    FORBIDDEN = frozenset({"sim_object_pose", "sim_instance_id", "sim_segmentation",
                          "sim_object_mesh", "sim_sdf", "sim_contacts", "sim_reward",
                          "sim_success", "sim_semantic_terminal"})

    def __init__(self, sources: dict[str, Source], *, allow_synthetic=False):
        self._sources = MappingProxyType({_name(k): v for k, v in sources.items()})
        if any(type(v) is not Source for v in self._sources.values()):
            raise BoundaryError("adapter_must_register_typed_sources")
        self._allow_synthetic = allow_synthetic

    def audit(self, source_ids):
        source_ids = _ids(source_ids)
        if not source_ids:
            raise BoundaryError("missing_source")
        visiting, kinds = set(), set()

        def visit(key):
            if key in visiting:
                raise BoundaryError("source_cycle")
            source = self._sources.get(key)
            if source is None:
                raise BoundaryError("unregistered_source")
            if source.kind in self.FORBIDDEN:
                raise BoundaryError("privileged_source:" + source.kind)
            if source.kind not in self.ROOTS | {"derived"}:
                raise BoundaryError("unknown_source_kind")
            if source.synthetic and not self._allow_synthetic:
                raise BoundaryError("synthetic_source_disabled")
            if source.kind == "derived" and not source.parents:
                raise BoundaryError("derived_source_without_dependencies")
            if (source.kind.endswith("_calibration") or source.kind in self.EXTRINSICS) and not (
                    isinstance(source.record, str) and source.record.strip()):
                raise BoundaryError("missing_calibration_record")
            visiting.add(key)
            kinds.add(source.kind)
            for parent in source.parents:
                visit(parent)
            visiting.remove(key)

        for key in source_ids:
            visit(key)
        return frozenset(kinds)

    def require(self, source_ids, required):
        kinds = self.audit(source_ids)
        if not set(required) <= kinds:
            raise BoundaryError("missing_source_dependency")
        return kinds

    def require_extrinsics(self, source_ids):
        kinds = self.audit(source_ids)
        if not kinds & self.EXTRINSICS:
            raise BoundaryError("missing_source_dependency")
        return kinds


@dataclass(frozen=True)
class Calibration:
    robot_from_sensor: tuple
    sources: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "robot_from_sensor", rigid(self.robot_from_sensor))
        object.__setattr__(self, "sources", _ids(self.sources))


@dataclass(frozen=True)
class Observation:
    id: str
    frame: str
    points: tuple
    sources: tuple[str, ...]
    timestamp_s: float
    calibration: Calibration | None = None

    def __post_init__(self):
        _name(self.id)
        _name(self.frame)
        points = tuple(tuple(_number(x) for x in p) for p in self.points)
        if any(len(p) != 3 for p in points):
            raise BoundaryError("invalid_point_shape")
        if self.calibration is not None and type(self.calibration) is not Calibration:
            raise BoundaryError("invalid_calibration_type")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "sources", _ids(self.sources))
        object.__setattr__(self, "timestamp_s", _number(self.timestamp_s))


@dataclass(frozen=True)
class ProxyRef:
    name: str
    revision: int

    def __post_init__(self):
        _name(self.name)
        if type(self.revision) is not int or self.revision < 1:
            raise BoundaryError("invalid_revision")


@dataclass(frozen=True)
class Hypothesis:
    ref: ProxyRef
    observation_id: str
    sensor_from_proxy: tuple


@dataclass(frozen=True)
class Move:
    pose: tuple
    reference: ProxyRef | None = None
    position_tolerance_m: float = 0.006
    orientation_tolerance_deg: float = 10.0

    def __post_init__(self):
        object.__setattr__(self, "pose", rigid(self.pose))
        if self.reference is not None and type(self.reference) is not ProxyRef:
            raise BoundaryError("invalid_reference")
        for key in ("position_tolerance_m", "orientation_tolerance_deg"):
            value = _number(getattr(self, key))
            if value <= 0 or (key == "orientation_tolerance_deg" and value > 180):
                raise BoundaryError("invalid_tolerance")
            object.__setattr__(self, key, value)


@dataclass(frozen=True)
class Gripper:
    state: str

    def __post_init__(self):
        if self.state not in ("open", "close"):
            raise BoundaryError("invalid_gripper_command")


@dataclass(frozen=True)
class Hold:
    seconds: float

    def __post_init__(self):
        seconds = _number(self.seconds)
        if not 0 < seconds <= 60:
            raise BoundaryError("hold_out_of_bounds")
        object.__setattr__(self, "seconds", seconds)


@dataclass(frozen=True)
class CommittedProgram:
    id: str
    observation_id: str
    requested: tuple
    resolved: tuple
    bindings: tuple[Hypothesis, ...]


@dataclass(frozen=True)
class SupportCheck:
    reference: ProxyRef
    observation_id: str
    frame: str
    sources: tuple[str, ...]
    total_points: int
    support_points: int
    radius_m: float
    nearest_distance_m: float | None
    support_distance_rms_m: float | None
    scope: str = "observed_points_to_proxy_origin_only"


@dataclass(frozen=True)
class Acknowledgement:
    command_id: str
    status: str
    sources: tuple[str, ...]
    measured_pose: tuple | None = None
    gripper_width_m: float | None = None
    elapsed_s: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "sources", _ids(self.sources))
        if self.status not in ("completed", "stopped"):
            raise BoundaryError("invalid_ack_status")
        if self.measured_pose is not None:
            object.__setattr__(self, "measured_pose", rigid(self.measured_pose))
        for key in ("gripper_width_m", "elapsed_s"):
            if getattr(self, key) is not None:
                value = _number(getattr(self, key))
                if value < 0:
                    raise BoundaryError("invalid_measurement")
                object.__setattr__(self, key, value)


@dataclass(frozen=True)
class CommandRecord:
    command_id: str
    status: str = "not_submitted"
    acknowledgement: Acknowledgement | None = None
    error: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    program_id: str
    status: str
    records: tuple[CommandRecord, ...]


class Backend(Protocol):
    def execute(self, command_id: str, command: Move | Gripper | Hold) -> Acknowledgement:
        ...


class Workspace:
    """Editable hypotheses over immutable evidence. No backend is accepted here."""
    def __init__(self, registry: SourceRegistry):
        self.registry = registry
        self._observations = {}
        self._current = None
        self._proxies = {}
        self._programs = {}
        self._results = {}
        self._lock = RLock()

    def publish(self, observation: Observation):
        if type(observation) is not Observation:
            raise BoundaryError("adapter_must_publish_typed_observation")
        required = {"depth_sensor", "camera_intrinsic_calibration"}
        if observation.frame == "robot":
            self.registry.require_extrinsics(observation.sources)
            if observation.calibration is not None:
                raise BoundaryError("robot_frame_must_not_be_transformed_twice")
        self.registry.require(observation.sources, required)
        if observation.calibration is not None:
            self.registry.require_extrinsics(observation.calibration.sources)
        with self._lock:
            if observation.id in self._observations:
                raise BoundaryError("observation_id_reused")
            if self._current is not None and observation.timestamp_s < self._observations[self._current].timestamp_s:
                raise BoundaryError("out_of_order_observation")
            self._observations[observation.id] = observation
            self._current = observation.id

    def observation(self, observation_id):
        return self._observations[observation_id]

    def propose(self, name, observation_id, sensor_from_proxy):
        _name(name)
        pose = rigid(sensor_from_proxy)
        with self._lock:
            self._current_observation(observation_id)
            old = self._proxies.get(name)
            ref = ProxyRef(name, old.ref.revision + 1 if old else 1)
            proposal = Hypothesis(ref, observation_id, pose)
            self._proxies[name] = proposal
            return proposal

    def _current_observation(self, observation_id):
        if observation_id != self._current or observation_id not in self._observations:
            raise BoundaryError("stale_or_missing_observation")
        return self._observations[observation_id]

    def _proxy(self, ref, observation_id):
        proxy = self._proxies.get(ref.name)
        if proxy is None or proxy.ref != ref or proxy.observation_id != observation_id:
            raise BoundaryError("stale_or_missing_proxy")
        return proxy

    def check(self, reference: ProxyRef, radius_m: float):
        radius = _number(radius_m)
        if radius <= 0:
            raise BoundaryError("invalid_radius")
        with self._lock:
            observation = self._current_observation(self._current)
            proxy = self._proxy(reference, observation.id)
            origin = tuple(proxy.sensor_from_proxy[i][3] for i in range(3))
            distances = [math.dist(p, origin) for p in observation.points]
            if any(not math.isfinite(d) for d in distances):
                raise BoundaryError("nonfinite_derived_distance")
            inside = [d for d in distances if d <= radius]
            scale = max(inside, default=0)
            rms = (scale * math.sqrt(sum((d/scale)**2 for d in inside) / len(inside))
                   if scale else (0.0 if inside else None))
            return SupportCheck(reference, observation.id, observation.frame,
                                observation.sources, len(distances), len(inside), radius,
                                min(distances) if distances else None,
                                rms)

    def commit(self, program_id, observation_id, commands):
        _name(program_id)
        commands = tuple(islice(iter(commands), 13))
        if not 1 <= len(commands) <= 12:
            raise BoundaryError("program_length_out_of_bounds")
        with self._lock:
            observation = self._current_observation(observation_id)
            if program_id in self._programs:
                raise BoundaryError("program_id_reused")
            resolved, bindings = [], []
            for command in commands:
                if type(command) not in (Move, Gripper, Hold):
                    raise BoundaryError("unknown_command_type")
                if type(command) is Move and command.reference is not None:
                    proxy = self._proxy(command.reference, observation_id)
                    if observation.frame == "robot":
                        robot_from_sensor = IDENTITY
                    elif observation.calibration is not None:
                        robot_from_sensor = observation.calibration.robot_from_sensor
                    else:
                        raise BoundaryError("missing_robot_camera_calibration")
                    target = compose(compose(robot_from_sensor, proxy.sensor_from_proxy), command.pose)
                    resolved.append(Move(target, None, command.position_tolerance_m,
                                         command.orientation_tolerance_deg))
                    bindings.append(proxy)
                else:
                    resolved.append(command)
            program = CommittedProgram(program_id, observation_id, commands,
                                       tuple(resolved), tuple(bindings))
            self._programs[program_id] = program
            return program

    def result(self, program_id):
        with self._lock:
            return self._results.get(program_id)


class Executor:
    """Only this explicit entry point can call the injected backend.

    Claims are shared by all executors of one Workspace. Device-side durable
    deduplication, limits and restart recovery remain adapter integration work.
    """
    def __init__(self, workspace: Workspace, backend: Backend):
        self.workspace, self.backend = workspace, backend

    def _validate_ack(self, ack, command_id, command):
        if type(ack) is not Acknowledgement or ack.command_id != command_id:
            raise BoundaryError("invalid_ack")
        kinds = self.workspace.registry.audit(ack.sources)
        if ack.measured_pose is not None and not (
                "robot_pose_sensor" in kinds or {"joint_encoder", "robot_model"} <= kinds):
            raise BoundaryError("invalid_pose_measurement_source")
        if ack.gripper_width_m is not None and "gripper_sensor" not in kinds:
            raise BoundaryError("invalid_gripper_measurement_source")
        if ack.status == "stopped":
            return "backend_stopped"
        if type(command) is Move:
            if ack.measured_pose is None:
                raise BoundaryError("missing_measured_pose")
            if arrival_exceeds(ack.measured_pose, command.pose,
                               command.position_tolerance_m,
                               command.orientation_tolerance_deg):
                return "arrival_not_reached"
        if type(command) is Gripper and ack.gripper_width_m is None:
            raise BoundaryError("missing_gripper_measurement")
        if type(command) is Hold and (ack.elapsed_s is None or ack.measured_pose is None):
            raise BoundaryError("missing_hold_measurement")
        if type(command) is Hold and ack.elapsed_s < command.seconds:
            return "hold_not_completed"
        return None

    def execute(self, program_id):
        workspace = self.workspace
        with workspace._lock:
            program = workspace._programs.get(program_id)
            if program is None:
                raise BoundaryError("program_not_committed")
            if program_id in workspace._results:
                raise BoundaryError("program_already_claimed")
            workspace._current_observation(program.observation_id)
            records = [CommandRecord(f"{program_id}/{i}") for i in range(len(program.resolved))]
            workspace._results[program_id] = ExecutionResult(program_id, "running", tuple(records))
        for i, command in enumerate(program.resolved):
            command_id = records[i].command_id
            records[i] = CommandRecord(command_id, "submitted")
            self._save(program_id, "running", records)
            try:
                ack = self.backend.execute(command_id, command)
            except Exception:
                # Exception text is not a sensor channel and may contain hidden state.
                records[i] = CommandRecord(command_id, "uncertain", error="backend_exception")
                return self._save(program_id, "stopped", records)
            try:
                stop = self._validate_ack(ack, command_id, command)
            except Exception:
                # Do not forward arbitrary dictionaries, fields or exception text.
                records[i] = CommandRecord(command_id, "uncertain", error="invalid_acknowledgement")
                return self._save(program_id, "stopped", records)
            records[i] = CommandRecord(command_id, "stopped" if stop else "completed", ack, stop)
            if stop:
                return self._save(program_id, "stopped", records)
        return self._save(program_id, "completed", records)

    def _save(self, program_id, status, records):
        result = ExecutionResult(program_id, status, tuple(records))
        with self.workspace._lock:
            self.workspace._results[program_id] = result
        return result

"""Synthetic contract checks, not robot/sensor/calibration experiment evidence."""
from dataclasses import FrozenInstanceError, asdict
import json
import threading
import unittest

from spatial_interface.geometry_workspace import (
    Acknowledgement, BoundaryError, Calibration, Executor, Gripper, Hold,
    IDENTITY, Move, Observation, ProxyRef, Source, SourceRegistry, Workspace, rigid,
)


def pose(x=0, y=0, z=0):
    return ((1, 0, 0, x), (0, 1, 0, y), (0, 0, 1, z), (0, 0, 0, 1))


def sources():
    # Every allowed fixture leaf is explicitly synthetic. Production rejects it.
    return {
        "depth": Source("depth_sensor", synthetic=True),
        "K": Source("camera_intrinsic_calibration", record="SYNTHETIC:K", synthetic=True),
        "E": Source("camera_extrinsic_calibration", record="SYNTHETIC:E", synthetic=True),
        "cloud": Source("derived", ("depth", "K")),
        "robot_cloud": Source("derived", ("cloud", "E")),
        "eef": Source("robot_pose_sensor", synthetic=True),
        "width": Source("gripper_sensor", synthetic=True),
        "gt": Source("sim_object_pose", record="Forbidden object state used as a negative fixture"),
    }


def fixture(*, calibrated=True, points=((0, 0, 0), (.1, 0, 0), (1, 0, 0))):
    workspace = Workspace(SourceRegistry(sources(), allow_synthetic=True))
    observation = Observation("obs1", "camera", points, ("cloud",), 1,
                              Calibration(IDENTITY, ("E",)) if calibrated else None)
    workspace.publish(observation)
    return workspace


class RecordingBackend:
    def __init__(self):
        self.calls = []
        self.measured = IDENTITY

    def execute(self, command_id, command):
        self.calls.append((command_id, command))
        if type(command) is Move:
            self.measured = command.pose
        return Acknowledgement(command_id, "completed", ("eef", "width"),
                               self.measured, .04,
                               command.seconds if type(command) is Hold else .1)


class WorkspaceContract(unittest.TestCase):
    def test_edits_and_checks_do_not_actuate_or_mutate_evidence(self):
        points = [[0, 0, 0], [.1, 0, 0], [1, 0, 0]]
        workspace = fixture(points=points)
        backend = RecordingBackend()
        Executor(workspace, backend)
        original = workspace.observation("obs1")
        points[0][0] = 100
        candidate = [list(row) for row in IDENTITY]
        first = workspace.propose("bowl", "obs1", candidate)
        candidate[0][3] = 100
        check = workspace.check(first.ref, .2)
        self.assertEqual((check.total_points, check.support_points), (3, 2))
        self.assertAlmostEqual(check.support_distance_rms_m, (.01/2)**.5)
        second = workspace.propose("bowl", "obs1", pose(.2))
        self.assertEqual(second.ref.revision, 2)
        self.assertEqual(first.sensor_from_proxy, IDENTITY)
        self.assertEqual(original.points[0], (0, 0, 0))
        self.assertEqual(workspace.observation("obs1"), original)
        self.assertEqual(backend.calls, [])
        self.assertNotIn("success", asdict(check))
        self.assertNotIn("collision_free", asdict(check))
        with self.assertRaises(FrozenInstanceError):
            first.observation_id = "other"

    def test_full_noncommuting_reference_pose_and_frozen_request(self):
        workspace = fixture()
        # Independent closed-form oracle: Rz(90) @ Rx(90), including translation.
        proxy_pose = ((0,-1,0,1),(1,0,0,2),(0,0,1,3),(0,0,0,1))
        local = [[1,0,0,.1],[0,0,-1,.2],[0,1,0,.3],[0,0,0,1]]
        ref = workspace.propose("part", "obs1", proxy_pose).ref
        commands = [Move(local, ref)]
        program = workspace.commit("p", "obs1", commands)
        expected = ((0,0,1,.8),(1,0,0,2.1),(0,1,0,3.3),(0,0,0,1))
        self.assertEqual(program.resolved[0].pose, expected)
        local[0][3] = 9
        commands.clear()
        workspace.propose("part", "obs1", pose(9))
        backend = RecordingBackend()
        result = Executor(workspace, backend).execute("p")
        self.assertEqual(result.status, "completed")
        self.assertEqual(backend.calls[0][1].pose, expected)
        self.assertEqual(program.requested[0].pose[0][3], .1)
        self.assertEqual(program.bindings[0].ref, ref)

    def test_calibration_is_composed_before_proxy_transform(self):
        workspace = fixture()
        workspace.publish(Observation("obs2", "camera", (), ("cloud",), 2,
                                      Calibration(pose(10,20,30), ("E",))))
        ref = workspace.propose("part", "obs2", pose(1,2,3)).ref
        program = workspace.commit("p", "obs2", [Move(pose(.1,.2,.3), ref)])
        self.assertEqual(tuple(program.resolved[0].pose[i][3] for i in range(3)),
                         (11.1,22.2,33.3))

    def test_missing_calibration_preserves_camera_check_but_refuses_resolution(self):
        workspace = fixture(calibrated=False)
        ref = workspace.propose("part", "obs1", IDENTITY).ref
        self.assertEqual(workspace.check(ref, .2).frame, "camera")
        with self.assertRaisesRegex(BoundaryError, "missing_robot_camera_calibration"):
            workspace.commit("p", "obs1", [Move(IDENTITY, ref)])
        # A model-authored robot-frame target does not require a proxy/UI step.
        workspace.commit("direct", "obs1", [Move(IDENTITY)])

    def test_stale_references_and_observations_are_refused(self):
        workspace = fixture()
        first = workspace.propose("part", "obs1", IDENTITY).ref
        workspace.propose("part", "obs1", pose(.1))
        with self.assertRaisesRegex(BoundaryError, "stale_or_missing_proxy"):
            workspace.commit("p", "obs1", [Move(IDENTITY), Move(IDENTITY, first)])
        workspace.commit("old", "obs1", [Move(IDENTITY)])
        workspace.publish(Observation("obs2", "camera", (), ("cloud",), 2))
        with self.assertRaisesRegex(BoundaryError, "stale_or_missing_observation"):
            workspace.commit("p", "obs1", [Move(IDENTITY)])
        backend = RecordingBackend()
        with self.assertRaises(BoundaryError):
            Executor(workspace, backend).execute("old")
        self.assertEqual(backend.calls, [])

    def test_privileged_dependencies_and_unregistered_claims_are_refused(self):
        declared = sources()
        declared["bad_cloud"] = Source("derived", ("cloud", "gt"))
        registry = SourceRegistry(declared, allow_synthetic=True)
        workspace = Workspace(registry)
        with self.assertRaisesRegex(BoundaryError, "privileged_source:sim_object_pose"):
            workspace.publish(Observation("bad", "camera", (), ("bad_cloud",), 1))
        with self.assertRaisesRegex(BoundaryError, "unregistered_source"):
            registry.audit(("model_says_trusted",))
        with self.assertRaises(BoundaryError):
            SourceRegistry({"x": {"kind": "depth_sensor", "trusted": True}})
        with self.assertRaisesRegex(BoundaryError, "missing_source_dependency"):
            workspace.publish(Observation("renamed", "robot", (), ("cloud",), 1))
        with self.assertRaisesRegex(BoundaryError, "privileged_source"):
            workspace.publish(Observation("badE", "camera", (), ("cloud",), 1,
                                          Calibration(IDENTITY, ("gt",))))

    def test_unknown_cycles_missing_calibration_records_and_synthetic_default(self):
        cases = [({"x": Source("invented")}, "unknown_source_kind"),
                 ({"x": Source("derived", ("x",))}, "source_cycle"),
                 ({"x": Source("derived")}, "derived_source_without_dependencies"),
                 ({"x": Source("camera_extrinsic_calibration")}, "missing_calibration_record")]
        for declared, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(BoundaryError, reason):
                SourceRegistry(declared).audit(("x",))
        with self.assertRaisesRegex(BoundaryError, "synthetic_source_disabled"):
            SourceRegistry(sources()).audit(("cloud",))

    def test_pose_and_program_validation_precedes_any_backend_call(self):
        bad_poses = [((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,1)),
                     pose(float("nan")), pose(float("inf")), [[1]*4]*4]
        for bad in bad_poses:
            with self.subTest(bad=bad), self.assertRaises(BoundaryError):
                Move(bad)
        for seconds in (-1, 0, float("inf"), True, 61):
            with self.subTest(seconds=seconds), self.assertRaises(BoundaryError):
                Hold(seconds)
        workspace = fixture()
        backend = RecordingBackend()
        executor = Executor(workspace, backend)
        with self.assertRaises(BoundaryError):
            workspace.commit("p", "obs1", [Move(IDENTITY), {"skill": "grasp"}])
        with self.assertRaises(BoundaryError):
            workspace.commit("p", "obs1", [Hold(1)]*13)
        with self.assertRaisesRegex(BoundaryError, "program_not_committed"):
            executor.execute("p")
        self.assertEqual(backend.calls, [])

    def test_failure_preserves_partial_records_and_never_replays(self):
        workspace = fixture()
        workspace.commit("p", "obs1", [Move(IDENTITY), Gripper("close"), Hold(1)])

        class FailsSecond(RecordingBackend):
            def execute(self, command_id, command):
                if len(self.calls) == 1:
                    self.calls.append((command_id, command))
                    raise RuntimeError("task_success=True; object_state=SECRET")
                return super().execute(command_id, command)

        backend = FailsSecond()
        result = Executor(workspace, backend).execute("p")
        self.assertEqual([r.status for r in result.records],
                         ["completed", "uncertain", "not_submitted"])
        self.assertEqual(len(backend.calls), 2)
        self.assertNotIn("SECRET", json.dumps(asdict(result)))
        with self.assertRaisesRegex(BoundaryError, "program_already_claimed"):
            Executor(workspace, backend).execute("p")
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(workspace.result("p"), result)

    def test_arbitrary_or_privileged_acknowledgements_do_not_enter_feedback(self):
        bad = [dict(task_success=True, reward=1, object_state="SECRET"),
               Acknowledgement("p/0", "completed", ("gt",), IDENTITY)]
        for acknowledgement in bad:
            with self.subTest(acknowledgement=type(acknowledgement).__name__):
                workspace = fixture()
                workspace.commit("p", "obs1", [Move(IDENTITY), Hold(1)])

                class BadBackend:
                    def execute(self, command_id, command):
                        return acknowledgement

                result = Executor(workspace, BadBackend()).execute("p")
                self.assertEqual(result.status, "stopped")
                self.assertIsNone(result.records[0].acknowledgement)
                self.assertEqual(result.records[1].status, "not_submitted")
                text = json.dumps(asdict(result))
                for forbidden in ("task_success", "reward", "object_state", "SECRET"):
                    self.assertNotIn(forbidden, text)

    def test_orientation_failure_stops_next_command_even_when_position_matches(self):
        workspace = fixture()
        workspace.commit("p", "obs1", [Move(IDENTITY), Gripper("close")])

        class WrongOrientation(RecordingBackend):
            def execute(self, command_id, command):
                self.calls.append((command_id, command))
                rotated = ((0,-1,0,0),(1,0,0,0),(0,0,1,0),(0,0,0,1))
                return Acknowledgement(command_id, "completed", ("eef",), rotated)

        backend = WrongOrientation()
        result = Executor(workspace, backend).execute("p")
        self.assertEqual(result.records[0].error, "arrival_not_reached")
        self.assertEqual(result.records[1].status, "not_submitted")
        self.assertEqual(len(backend.calls), 1)

    def test_concurrent_duplicate_execution_is_claimed_before_backend_call(self):
        workspace = fixture()
        workspace.commit("p", "obs1", [Move(IDENTITY)])
        started, release = threading.Event(), threading.Event()

        class BlockingBackend(RecordingBackend):
            def execute(self, command_id, command):
                started.set()
                if not release.wait(2):
                    raise RuntimeError("test synchronization timeout")
                return super().execute(command_id, command)

        backend = BlockingBackend()
        thread = threading.Thread(target=Executor(workspace, backend).execute, args=("p",))
        thread.start()
        try:
            self.assertTrue(started.wait(1))
            with self.assertRaisesRegex(BoundaryError, "program_already_claimed"):
                Executor(workspace, backend).execute("p")
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(workspace.result("p").status, "completed")

    def test_short_hold_stops_and_valid_program_preserves_all_commands(self):
        workspace = fixture()
        workspace.commit("valid", "obs1", [Move(pose(.1)), Gripper("close"), Hold(2)])
        backend = RecordingBackend()
        result = Executor(workspace, backend).execute("valid")
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(result.records[-1].acknowledgement.elapsed_s, 2)
        with self.assertRaisesRegex(BoundaryError, "program_id_reused"):
            workspace.commit("valid", "obs1", [Hold(1)])

        class ShortHold(RecordingBackend):
            def execute(self, command_id, command):
                self.calls.append((command_id, command))
                return Acknowledgement(command_id, "completed", ("eef",), IDENTITY,
                                       elapsed_s=.5)

        workspace.commit("short", "obs1", [Hold(2), Gripper("open")])
        short = ShortHold()
        result = Executor(workspace, short).execute("short")
        self.assertEqual(result.records[0].error, "hold_not_completed")
        self.assertEqual(result.records[1].status, "not_submitted")
        self.assertEqual(len(short.calls), 1)

    def test_no_observed_points_is_unknown_and_source_mapping_is_owned(self):
        declared = sources()
        registry = SourceRegistry(declared, allow_synthetic=True)
        declared["E"] = Source("sim_object_pose")
        self.assertIn("camera_extrinsic_calibration", registry.audit(("E",)))
        with self.assertRaises(TypeError):
            registry._sources["E"] = declared["E"]
        workspace = fixture(points=())
        ref = workspace.propose("part", "obs1", IDENTITY).ref
        check = workspace.check(ref, .2)
        self.assertEqual(check.support_points, 0)
        self.assertIsNone(check.nearest_distance_m)
        self.assertIsNone(check.support_distance_rms_m)

    def test_provided_extrinsics_keep_actual_source_and_resolve_targets(self):
        for kind in SourceRegistry.EXTRINSICS:
            with self.subTest(kind=kind):
                declared = sources()
                declared["providedE"] = Source(kind, record="SYNTHETIC:known_camera_setup",
                                               synthetic=True)
                registry = SourceRegistry(declared, allow_synthetic=True)
                self.assertEqual(registry.require_extrinsics(("providedE",)), {kind})
                workspace = Workspace(registry)
                workspace.publish(Observation("cam", "camera", (), ("cloud",), 1,
                                              Calibration(pose(1,2,3), ("providedE",))))
                ref = workspace.propose("part", "cam", pose(.1,.2,.3)).ref
                program = workspace.commit("p", "cam", [Move(IDENTITY, ref)])
                self.assertEqual(program.resolved[0].pose, pose(1.1,2.2,3.3))
                workspace.publish(Observation("robot_cloud", "robot", (),
                                              ("cloud", "providedE"), 2))
                # Calling something calibration cannot legitimize object GT.
                declared["taintedE"] = Source(kind, ("gt",), record="known_camera_setup")
                with self.assertRaisesRegex(BoundaryError, "privileged_source:sim_object_pose"):
                    SourceRegistry(declared, allow_synthetic=True).require_extrinsics(("taintedE",))

    def test_moving_wrist_extrinsics_update_with_each_observation(self):
        declared = sources()
        declared["wristE"] = Source("sim_camera_world_pose", ("eef",),
                                    record="SYNTHETIC:known_wrist_mount_and_current_robot_pose",
                                    synthetic=True)
        workspace = Workspace(SourceRegistry(declared, allow_synthetic=True))
        targets = []
        for index, x in enumerate((1, 2), 1):
            observation_id = f"wrist{index}"
            workspace.publish(Observation(observation_id, "wrist", (), ("cloud",), index,
                                          Calibration(pose(x), ("wristE",))))
            ref = workspace.propose("part", observation_id, pose(.1)).ref
            program = workspace.commit(f"p{index}", observation_id, [Move(IDENTITY, ref)])
            targets.append(program.resolved[0].pose[0][3])
        self.assertEqual(targets, [1.1, 2.1])


if __name__ == "__main__":
    unittest.main()

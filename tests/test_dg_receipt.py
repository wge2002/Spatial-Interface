"""Model-facing receipt projection: honest diagnostics, bounded state, no invention.

Every case here is a pure-projection case: a payload in, a receipt out. Nothing
in this file senses, actuates or infers, and two of the tests assert exactly that
by making any call into the transport an error.
"""
import copy
import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
import asyncio

import numpy as np
from scipy.spatial.transform import Rotation

from spatial_interface import dg_receipt as receipt
from spatial_interface import direct_control as dc, direct_geometry_tools as dg, geometry_workspace as gw


def pose(position, rotation=None):
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix.tolist()


def measured(position, rotation=None, width=0.08, status="ok",
             source="robot_proprioception"):
    matrix = np.asarray(pose(position, rotation))
    return {"status": status, "source": source, "frame": "robot",
            "fingertip_position": dict(zip("xyz", matrix[:3, 3].tolist())),
            "approach": dict(zip("xyz", matrix[:3, 2].tolist())),
            "opening": dict(zip("xyz", matrix[:3, 1].tolist())),
            "gripper_width_m": width, "gripper_state_class": "open"}


def ack(command_id, target, *, status="completed", width=0.08, elapsed=None):
    out = {"command_id": command_id, "status": status, "sources": ["robot", "jaws"],
           "measured_pose": target, "gripper_width_m": width}
    if elapsed is not None:
        out["elapsed_s"] = elapsed
    return out


def move(target, tolerance=0.005, orientation_tolerance=5.0):
    return {"pose": target, "reference": None, "position_tolerance_m": tolerance,
            "orientation_tolerance_deg": orientation_tolerance}


def payload_of(resolved, records, *, before, requested=None, executed=None, status="completed",
               program_id="p1", observation=None):
    out = {"status": status, "program_id": program_id,
           "observation_before": {"what": "observation", "frame": "f1",
                                  "measured_end_effector": before},
           "commit": {"id": program_id, "requested": requested or [{} for _ in resolved],
                      "resolved": resolved, "bindings": []},
           "executed": executed if executed is not None else
                       [{"command_id": r["command_id"], "request": {"kind": "pose"},
                         "submitted": True, "completed": True,
                         "reply": {"sim_steps_used": 50}, "wall_s": 1.0}
                        for r in records]}
    if records is not None:
        out["execution"] = {"program_id": program_id, "status": status, "records": records}
    if observation is not None:
        out["observation"] = observation
    return out


class ArrivalNumberTests(unittest.TestCase):
    def test_known_translation_and_rotation_give_the_expected_residuals(self):
        target = pose([0.1, 0.2, 1.0])
        actual = pose([0.13, 0.24, 1.0])  # 3-4-5 triangle: exactly 5 cm away
        out = receipt.arrival(actual, target, 0.01, 5.0)
        self.assertAlmostEqual(out["position_error_m"], 0.05, places=9)
        self.assertEqual(out["orientation_error_deg"], 0.0)
        self.assertFalse(out["position_within"])
        self.assertTrue(out["orientation_within"])

    def test_rotation_only_error_is_the_geodesic_angle_and_position_is_zero(self):
        rotation = Rotation.from_euler("z", 30, degrees=True).as_matrix()
        out = receipt.arrival(pose([0.1, 0.2, 1.0], rotation), pose([0.1, 0.2, 1.0]),
                              0.005, 5.0)
        self.assertEqual(out["position_error_m"], 0.0)
        self.assertAlmostEqual(out["orientation_error_deg"], 30.0, places=6)
        self.assertTrue(out["position_within"])
        self.assertFalse(out["orientation_within"])

    def test_identity_arrival_is_zero_and_within_both_tolerances(self):
        out = receipt.arrival(gw.IDENTITY, gw.IDENTITY, 0.001, 0.5)
        self.assertEqual((out["position_error_m"], out["orientation_error_deg"]), (0.0, 0.0))
        self.assertTrue(out["position_within"] and out["orientation_within"])

    def test_displayed_verdict_matches_the_executor_gate_at_the_boundary(self):
        """Rounding is display only: the boolean comes from the unrounded value."""
        executor = gw.Executor(gw.Workspace(dg.registry()), None)
        for delta in (-1e-9, 0.0, 1e-9, 4e-6):
            for angle_delta in (-1e-9, 0.0, 1e-9):
                tolerance, angle = 0.002, 3.0
                rotation = Rotation.from_euler(
                    "z", angle + angle_delta, degrees=True).as_matrix()
                actual = pose([tolerance + delta, 0, 0], rotation)
                command = gw.Move(gw.IDENTITY, position_tolerance_m=tolerance,
                                  orientation_tolerance_deg=angle)
                stop = executor._validate_ack(
                    gw.Acknowledgement("c", "completed", ("robot", "jaws"),
                                       measured_pose=actual, gripper_width_m=0.05),
                    "c", command)
                out = receipt.arrival(actual, gw.IDENTITY, tolerance, angle)
                reached = out["position_within"] and out["orientation_within"]
                self.assertEqual(reached, stop is None, (delta, angle_delta, out))

    def test_arrival_is_unknown_rather_than_zero_when_a_pose_is_missing(self):
        self.assertIsNone(receipt.arrival(None, gw.IDENTITY, 0.01, 1.0))
        self.assertIsNone(receipt.arrival(gw.IDENTITY, None, 0.01, 1.0))


class CommandDiagnosticTests(unittest.TestCase):
    def test_later_command_starts_from_the_previous_actual_not_the_requested_pose(self):
        first_target, second_target = pose([0.2, 0, 1.0]), pose([0.3, 0, 1.0])
        landed = pose([0.18, 0, 1.0])  # 2 cm short of the first requested target
        records = [{"command_id": "p1/0", "status": "completed",
                    "acknowledgement": ack("p1/0", landed), "error": None},
                   {"command_id": "p1/1", "status": "completed",
                    "acknowledgement": ack("p1/1", pose([0.29, 0, 1.0])), "error": None}]
        out = receipt.command_diagnostics(payload_of(
            [move(first_target, 0.05), move(second_target, 0.05)], records,
            before=measured([0.1, 0, 1.0])))
        self.assertEqual(out[0]["start_source"], "measured_pre_program")
        self.assertEqual(out[0]["start_eef_position"], [0.1, 0.0, 1.0])
        self.assertAlmostEqual(out[0]["after"]["position_error_m"], 0.02, places=9)
        # The second command's measured start is where the robot ACTUALLY was,
        # 0.18, not the 0.2 that was requested. 0.3 - 0.18 = 0.12.
        self.assertEqual(out[1]["start_eef_position"], [0.18, 0.0, 1.0])
        self.assertNotIn("start_source", out[1])
        self.assertAlmostEqual(out[1]["before"]["position_error_m"], 0.12, places=9)
        self.assertEqual(out[1]["target_position"], [0.3, 0.0, 1.0])

    def test_target_is_the_committed_absolute_pose_even_for_a_proxy_request(self):
        absolute = pose([0.4, -0.1, 1.05])
        records = [{"command_id": "p1/0", "status": "completed",
                    "acknowledgement": ack("p1/0", absolute), "error": None}]
        requested = [{"pose": pose([0.02, 0, 0.05]), "reference": {"name": "item", "revision": 1},
                      "position_tolerance_m": 0.01, "orientation_tolerance_deg": 5.0}]
        out = receipt.command_diagnostics(payload_of(
            [move(absolute)], records, before=measured([0, 0, 1.0]), requested=requested))
        self.assertEqual(out[0]["target_position"], [0.4, -0.1, 1.05])
        self.assertTrue(out[0]["target_resolved_from_proxy_request"])
        self.assertEqual(out[0]["after"]["position_error_m"], 0.0)

    def test_steps_come_from_the_reply_and_the_stop_reason_from_the_record(self):
        target = pose([0.2, 0, 1.0])
        records = [{"command_id": "p1/0", "status": "stopped",
                    "acknowledgement": ack("p1/0", pose([0.15, 0, 1.0])),
                    "error": "arrival_not_reached"},
                   {"command_id": "p1/1", "status": "not_submitted",
                    "acknowledgement": None, "error": None}]
        payload = payload_of([move(target), {"state": "close"}], records,
                             before=measured([0.1, 0, 1.0]), status="stopped",
                             executed=[{"command_id": "p1/0", "request": {"kind": "pose"},
                                        "submitted": True, "completed": True,
                                        "reply": {"sim_steps_used": 50}, "wall_s": 2.0}])
        out = receipt.command_diagnostics(payload)
        self.assertEqual(out[0]["status"], "stopped")
        self.assertEqual(out[0]["stop_reason"], "arrival_not_reached")
        self.assertEqual(out[0]["sim_steps_used"], 50)
        self.assertAlmostEqual(out[0]["after"]["position_error_m"], 0.05, places=9)
        self.assertEqual(out[0]["target_minus_actual_m"], [0.05, 0.0, 0.0])
        self.assertEqual(receipt.not_executed_tail(out),
                         {"from_index": 1, "count": 1,
                          "note": "these commands did not run; do not assume their effect"})
        # The command that never ran has no measurements of any kind.
        self.assertEqual(set(out[1]), {"index", "command_id", "kind", "status"})

    def test_a_submitted_command_without_an_execution_record_is_uncertain(self):
        """The exception path: transport submitted, the arrival gate never ran."""
        payload = {"status": "stopped", "program_id": "p1", "reason": "backend_unreachable",
                   "observation_before": {"what": "observation", "frame": "f1",
                                          "measured_end_effector": measured([0.1, 0, 1.0])},
                   "commit": {"id": "p1", "requested": [{}, {}],
                              "resolved": [move(pose([0.2, 0, 1.0])), {"state": "close"}],
                              "bindings": []},
                   "executed": [{"command_id": "p1/0", "request": {"kind": "pose"},
                                 "submitted": True, "completed": True,
                                 "reply": {"sim_steps_used": 31}, "wall_s": 1.0}]}
        out = receipt.command_diagnostics(payload)
        self.assertEqual(out[0]["status"], "uncertain")
        self.assertEqual(out[0]["stop_reason"], receipt.NO_ARRIVAL_CHECK)
        self.assertEqual(out[0]["sim_steps_used"], 31)
        self.assertEqual(out[0]["measured_end"], receipt.UNKNOWN_END)
        self.assertEqual(out[0]["after"], receipt.UNKNOWN_END)
        self.assertEqual(out[1]["status"], "not_submitted")
        self.assertNotIn("measured_start", out[1])
        tail = receipt.not_executed_tail(out)
        self.assertEqual(tail["from_index"], 1)

    def test_attempted_commands_survive_a_payload_with_no_committed_program(self):
        payload = {"status": "stopped", "reason": "program_not_committed",
                   "executed": [{"command_id": "x/0", "request": {"kind": "pose"},
                                 "submitted": True, "completed": None, "wall_s": 0.4},
                                {"command_id": "x/1", "request": {"kind": "gripper"},
                                 "submitted": False, "completed": None}]}
        out = receipt.command_diagnostics(payload)
        self.assertEqual([e["status"] for e in out], ["uncertain", "not_submitted"])
        self.assertEqual([e["submitted"] for e in out], [True, False])
        self.assertEqual(out[0]["measured_end"], receipt.UNKNOWN_END)
        self.assertNotIn("measured_start", out[0])

    def test_backend_uncertainty_is_never_upgraded_to_completed(self):
        records = [{"command_id": "p1/0", "status": "uncertain",
                    "acknowledgement": ack("p1/0", pose([0.2, 0, 1.0])),
                    "error": "invalid_acknowledgement"}]
        payload = payload_of([move(pose([0.2, 0, 1.0]))], records,
                             before=measured([0.1, 0, 1.0]), status="stopped",
                             executed=[{"command_id": "p1/0", "request": {"kind": "pose"},
                                        "submitted": True, "completed": True,
                                        "reply": {"sim_steps_used": 12}}])
        out = receipt.command_diagnostics(payload)
        self.assertEqual(out[0]["status"], "uncertain")
        self.assertEqual(out[0]["stop_reason"], "invalid_acknowledgement")
        self.assertEqual(out[0]["measured_end"], receipt.UNKNOWN_END)

    def test_an_unverified_pre_program_reading_is_unknown_not_zero(self):
        records = [{"command_id": "p1/0", "status": "completed",
                    "acknowledgement": ack("p1/0", pose([0.2, 0, 1.0])), "error": None}]
        for before in (measured([0.1, 0, 1.0], status="unknown"),
                       measured([0.1, 0, 1.0], source="robot_telemetry"),
                       {"status": "ok", "source": "robot_proprioception"},
                       None):
            out = receipt.command_diagnostics(payload_of(
                [move(pose([0.2, 0, 1.0]))], records, before=before))
            self.assertEqual(out[0]["measured_start"], receipt.UNKNOWN_START, before)
            self.assertEqual(out[0]["before"], receipt.UNKNOWN_START)
            self.assertNotIn("start_eef_position", out[0])
            # The end of that same command is still measured and still shown.
            self.assertEqual(out[0]["end_eef_position"], [0.2, 0.0, 1.0])

    def test_a_missing_endpoint_breaks_the_pose_and_the_width_chain(self):
        records = [{"command_id": "p1/0", "status": "uncertain", "acknowledgement": None,
                    "error": "backend_exception"},
                   {"command_id": "p1/1", "status": "completed",
                    "acknowledgement": {"command_id": "p1/1", "status": "completed",
                                        "sources": ["robot", "jaws"], "measured_pose": None,
                                        "gripper_width_m": None},
                    "error": None},
                   {"command_id": "p1/2", "status": "completed",
                    "acknowledgement": ack("p1/2", pose([0.3, 0, 1.0]), width=0.02),
                    "error": None}]
        out = receipt.command_diagnostics(payload_of(
            [move(pose([0.2, 0, 1.0])), {"state": "close"}, move(pose([0.3, 0, 1.0]))],
            records, before=measured([0.1, 0, 1.0], width=0.079)))
        self.assertEqual(out[0]["measured_end"], receipt.UNKNOWN_END)
        # Command 1 follows an unverified endpoint: no start pose, and the jaw
        # width before it is unknown rather than the stale 0.079.
        self.assertEqual(out[1]["measured_start"], receipt.UNKNOWN_START)
        self.assertEqual(out[1]["gripper_width_before_m"], "unknown")
        self.assertEqual(out[1]["gripper_width_after_m"], "unknown")
        self.assertEqual(out[2]["measured_start"], receipt.UNKNOWN_START)
        self.assertEqual(out[2]["before"], receipt.UNKNOWN_START)
        self.assertEqual(out[2]["after"]["position_error_m"], 0.0)

    def test_gripper_widths_are_measurements_and_hold_reports_actual_elapsed(self):
        records = [{"command_id": "p1/0", "status": "completed",
                    "acknowledgement": ack("p1/0", pose([0.1, 0, 1.0]), width=0.021),
                    "error": None},
                   {"command_id": "p1/1", "status": "completed",
                    "acknowledgement": ack("p1/1", pose([0.1, 0, 1.0]), width=0.021,
                                           elapsed=2.5), "error": None}]
        out = receipt.command_diagnostics(payload_of(
            [{"state": "close"}, {"seconds": 3.0}], records,
            before=measured([0.1, 0, 1.0], width=0.0788)))
        self.assertEqual(out[0]["gripper_command"], "close")
        self.assertEqual(out[0]["gripper_width_before_m"], 0.0788)
        self.assertEqual(out[0]["gripper_width_after_m"], 0.021)
        self.assertNotIn("grasped", json.dumps(out))
        self.assertEqual(out[1]["hold_requested_s"], 3.0)
        self.assertEqual(out[1]["hold_elapsed_s"], 2.5)


class ProjectionShapeTests(unittest.TestCase):
    def test_closing_after_motion_keeps_the_last_pose_residual(self):
        target, landed = pose([0.2, 0, 1]), pose([0.196, 0, 1])
        records = [{"command_id": f"p1/{i}", "status": "completed",
                    "acknowledgement": ack(f"p1/{i}", landed, width=width), "error": None}
                   for i, width in enumerate((0.08, 0.02))]
        result = receipt.policy_receipt(payload_of(
            [move(target), {"state": "close"}], records,
            before=measured([0.1, 0, 1])))
        self.assertAlmostEqual(result["commands"][0]["after"]["position_error_m"], 0.004)
        self.assertEqual(result["commands"][0]["target_position"], [0.2, 0.0, 1.0])
        self.assertEqual(result["commands"][1]["gripper_width_after_m"], 0.02)

    def test_stale_measurements_and_frame_provenance_remain_explicit(self):
        bad = measured([9, 8, 7], status="unknown")
        bad["reason"] = "telemetry_unpaired"
        observation = receipt.observation_receipt({
            "what": "observation", "measured_end_effector": bad,
            "wrist_depth": {"status": "unknown", "depth_frame": "old", "reason": "stale"}})
        self.assertIsNone(observation["measured_eef"]["eef_position"])
        self.assertEqual(observation["measured_eef"]["gripper_width_m"], "unknown")
        self.assertEqual(observation["measured_eef"]["source"], "robot_proprioception")
        self.assertEqual(observation["measured_eef"]["frame"], "robot")
        self.assertEqual(observation["wrist_depth"]["depth_frame"], "old")
        bind = {"status": "unknown", "reason": "no_paired_frame", "frame": "a",
                "frame_after": "b", "message": "frame changed; nothing bound"}
        temporal = {"status": "unknown", "reason": "frame_advance_invalid",
                    "before_frame": "a", "after_frame": "b"}
        result = receipt.policy_receipt({"status": "rejected", "bind": bind,
                                          "feedback": {"variant": "paired", "temporal": temporal}})
        self.assertEqual(result["bind"], bind)
        self.assertEqual(result["feedback"]["temporal"], temporal)
        stored = receipt.program_record({"status": "rejected", "bind": bind}, 1)
        self.assertEqual(stored["bind"]["reason"], "no_paired_frame")
        self.assertEqual(stored["bind"]["frame_after"], "b")

    def test_completed_intermediates_are_summarised_and_the_decisive_ones_are_not(self):
        targets = [pose([0.2, 0, 1.0]), pose([0.25, 0, 1.0]), pose([0.3, 0, 1.0])]
        records = [{"command_id": f"p1/{i}", "status": status,
                    "acknowledgement": ack(f"p1/{i}", landed), "error": error}
                   for i, (status, landed, error) in enumerate(
                       (("completed", targets[0], None),
                        ("completed", targets[1], None),
                        ("stopped", pose([0.26, 0, 1.0]), "arrival_not_reached")))]
        diagnostics = receipt.command_diagnostics(payload_of(
            [move(t) for t in targets], records, before=measured([0.1, 0, 1.0]),
            status="stopped"))
        projected = receipt.project_commands(diagnostics)
        self.assertEqual(set(projected[0]), {"index", "kind", "status", "sim_steps_used"})
        self.assertEqual(set(projected[1]), set(projected[0]))
        self.assertIn("after", projected[2])
        self.assertIn("target_position", projected[2])
        self.assertEqual(projected[2]["stop_reason"], "arrival_not_reached")
        # Nothing is lost: the full record is still what dg_state holds.
        self.assertIn("target_position", diagnostics[0])

    def test_the_last_executed_command_keeps_its_full_endpoint_evidence(self):
        targets = [pose([0.2, 0, 1.0]), pose([0.25, 0, 1.0])]
        records = [{"command_id": f"p1/{i}", "status": "completed",
                    "acknowledgement": ack(f"p1/{i}", t), "error": None}
                   for i, t in enumerate(targets)]
        projected = receipt.project_commands(receipt.command_diagnostics(payload_of(
            [move(t) for t in targets], records, before=measured([0.1, 0, 1.0]))))
        self.assertNotIn("after", projected[0])
        self.assertEqual(projected[1]["after"]["position_error_m"], 0.0)
        self.assertEqual(projected[1]["end_eef_position"], [0.25, 0.0, 1.0])

    def test_projection_does_not_mutate_the_payload_it_reads(self):
        records = [{"command_id": "p1/0", "status": "completed",
                    "acknowledgement": ack("p1/0", pose([0.2, 0, 1.0])), "error": None}]
        payload = payload_of([move(pose([0.2, 0, 1.0]))], records,
                             before=measured([0.1, 0, 1.0]),
                             observation={"what": "observation", "frame": "f2",
                                          "measured_end_effector": measured([0.2, 0, 1.0]),
                                          "local_depth": {"status": "ok",
                                                          "samples_xyz": [[0, 0, 1]],
                                                          "at": [0.2, 0, 1.0]}})
        payload["feedback"] = {"variant": "grounded", "identity": "model_assigned_not_verified",
                               "errors": [], "spatial": {"status": "paired"}}
        original = copy.deepcopy(payload)
        out = receipt.policy_receipt(payload)
        record = receipt.program_record(payload, 1)
        self.assertEqual(payload, original)
        # And the cached history owns its copy: editing it cannot reach back.
        record["commands"][0]["status"] = "tampered"
        self.assertEqual(payload["execution"]["records"][0]["status"], "completed")
        self.assertEqual(out["observation"]["local_depth"]["samples"],
                         "omitted from this summary")

    def test_observation_keeps_the_marker_and_the_unusable_frame_reasons(self):
        out = receipt.observation_receipt({
            "what": "observation", "frame": "e1-s2-c3", "cloud_frame": "e1-c3",
            "paired": False, "pairing": {"status": "unpaired", "missing": ["wrist"]},
            "note": "this frame is not usable as geometry (stale wrist)",
            "sim_steps_used": 120, "sim_steps_budget": 2000,
            "measured_end_effector": {"status": "unknown", "reason": "no telemetry"},
            "wrist_depth": {"status": "unknown", "reason": "no depth frame"}})
        self.assertEqual(out["what"], "observation")
        self.assertEqual(out["sim_steps_remaining"], 1880)
        self.assertEqual(out["pairing"]["missing"], ["wrist"])
        self.assertIn("not usable", out["note"])
        self.assertEqual(out["measured_eef"]["status"], "unknown")
        self.assertEqual(out["measured_eef"]["reason"], "no telemetry")
        self.assertEqual(out["measured_eef"]["gripper_width_m"], "unknown")
        self.assertEqual(out["wrist_depth"]["reason"], "no depth frame")
        # The cloud id restates the frame id's epoch and cloud counter; a real
        # disagreement is shown, an agreement is not repeated.
        self.assertNotIn("cloud_frame", out)
        self.assertEqual(receipt.observation_receipt(
            {"frame": "e1-s2-c3", "cloud_frame": "e1-c2"})["cloud_frame"], "e1-c2")

    def test_a_call_that_ran_nothing_says_so_with_an_empty_command_list(self):
        for payload in ({"status": "rejected", "reason": "proxy_fit_invalid_no_motion",
                         "executed": []},
                        {"status": "geometry_only", "executed": []}):
            out = receipt.policy_receipt(payload)
            self.assertEqual(out["commands"], [], payload)
            self.assertNotIn("not_executed", out)

    def test_a_refused_fit_keeps_its_reasons_quality_numbers_and_provenance(self):
        payload = {"status": "rejected", "reason": "proxy_fit_invalid_no_motion",
                   "executed": [],
                   "bind": {"status": "ok",
                            "proxy": {"name": "handle", "shape": "ring", "valid": False,
                                      "from_frame": "e1-s2-c3", "up": [0, 0, 1],
                                      "up_source": "assumed_world_up",
                                      "center": [0.1, 0.2, 1.0],
                                      "reasons": ["insufficient_bearing_coverage"],
                                      "fit": {"residual_rms_m": 0.01, "object_points": 42,
                                              "bearing_coverage_deg": 31.5,
                                              "clipped_by_region": True}}}}
        out = receipt.policy_receipt(payload)
        proxy = out["bind"]["proxy"]
        self.assertEqual(out["reason"], "proxy_fit_invalid_no_motion")
        self.assertEqual(proxy["reasons"], ["insufficient_bearing_coverage"])
        self.assertEqual(proxy["fit_quality"], payload["bind"]["proxy"]["fit"])
        self.assertEqual(proxy["up_source"], "assumed_world_up")
        self.assertFalse(proxy["valid"])
        self.assertIn("not a verified object identity", proxy["identity_note"])

    def test_the_feedback_envelope_keeps_provenance_and_errors(self):
        payload = {"status": "completed", "executed": [],
                   "feedback": {"variant": "paired", "identity": "model_assigned_not_verified",
                                "errors": ["before_capture: camera_capture_not_paired"],
                                "spatial": {"status": "paired", "image_frame": "e1-s2-c3",
                                            "selection_source_frame": "e1-s1-c1",
                                            "selection_historical": True,
                                            "selection_surface": "agentview",
                                            "regions": [{"name": "handle",
                                                         "region": {"kind": "box"},
                                                         "returns": 812}],
                                            "references": [
                                                {"name": "handle", "fit_valid": None,
                                                 "source": "explicit_model_hypothesis",
                                                 "source_frame": "e1-s1-c1",
                                                 "historical": True,
                                                 "identity_verified": False}]},
                                "temporal": {"status": "unknown",
                                             "reason": "nonpositive_sim_step_delta"}}}
        out = receipt.policy_receipt(payload)["feedback"]
        self.assertEqual(out["variant"], "paired")
        self.assertEqual(out["spatial"]["selection_source_frame"], "e1-s1-c1")
        self.assertTrue(out["spatial"]["selection_historical"])
        reference = out["spatial"]["references"][0]
        self.assertEqual(reference["source"], "explicit_model_hypothesis")
        self.assertIsNone(reference["fit_valid"])
        self.assertFalse(reference["identity_verified"])
        self.assertEqual(out["spatial"]["regions"][0]["returns"], 812)
        self.assertEqual(out["temporal"]["reason"], "nonpositive_sim_step_delta")
        self.assertEqual(out["errors"], payload["feedback"]["errors"])


class StateQueryTests(unittest.TestCase):
    def test_published_schema_rejects_page_sizes_the_backend_refuses(self):
        import jsonschema
        for field, maximum in (("programs", 20), ("proxies", 20), ("commands", 12)):
            query = {field: {"limit": maximum}}
            if field == "commands":
                query["program_id"] = "p1"
            jsonschema.validate(query, dg.DgStateTool.input_schema)
            receipt.validate_state_query(query)
            query[field]["limit"] += 1
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(query, dg.DgStateTool.input_schema)
            with self.assertRaises(receipt.StateQueryError):
                receipt.validate_state_query(query)

    def setUp(self):
        dg.reset_episode()

    def records(self, count=10):
        out = []
        for index in range(count):
            target = pose([0.1 + index / 100, 0, 1.0])
            records = [{"command_id": f"p{index}/0", "status": "completed",
                        "acknowledgement": ack(f"p{index}/0", target), "error": None}]
            payload = payload_of([move(target)], records, before=measured([0.1, 0, 1.0]),
                                 program_id=f"p{index}")
            payload["observation"] = {"what": "observation", "frame": "f2",
                                      "local_depth": {"status": "ok", "at": [0, 0, 1],
                                                      "radius_m": 0.08,
                                                      "samples_xyz": [[0.0, 0.0, 1.0]]}}
            out.append(receipt.program_record(payload, index))
        return out

    def test_the_default_query_returns_the_recent_eight_programs_newest_first(self):
        out = receipt.state_receipt(policy_calls=10, records=self.records(), proxies=[],
                                    last_observation=None,
                                    query=receipt.validate_state_query(None))
        self.assertEqual(len(out["programs"]), 8)
        self.assertEqual(out["programs"][0]["program_id"], "p9")
        self.assertEqual(out["programs_total"], 10)
        self.assertTrue(out["has_more_programs"])
        self.assertNotIn("commands", out["programs"][0])

    def test_pages_are_bounded_and_bad_queries_are_refused(self):
        for bad in ({"programs": {"limit": 21}}, {"programs": {"limit": 0}},
                    {"programs": {"offset": -1}}, {"commands": {"limit": 5}},
                    {"program_id": "p1", "commands": {"limit": 13}},
                    {"include_depth_samples": True}, {"program_id": ""},
                    {"program_id": 7}, {"proxies": {"limit": 21}},
                    {"unknown": 1}, {"programs": {"page": 2}}, {"programs": 3},
                    {"program_id": "p1", "include_depth_samples": "yes"}, []):
            with self.assertRaises(receipt.StateQueryError, msg=bad):
                receipt.validate_state_query(bad)
        query = receipt.validate_state_query({"programs": {"offset": 2, "limit": 20}})
        self.assertEqual(query["programs"], (2, 20))

    def test_detail_needs_a_program_id_and_pages_its_commands(self):
        records = self.records(3)
        records[1]["commands"] = records[1]["commands"] * 6
        query = receipt.validate_state_query(
            {"program_id": "p1", "commands": {"offset": 2, "limit": 3},
             "include_depth_samples": True})
        out = receipt.state_receipt(policy_calls=3, records=records, proxies=[],
                                    last_observation=None, query=query)
        self.assertEqual(out["program"]["program_id"], "p1")
        self.assertEqual(out["program"]["command_count"], 6)
        self.assertEqual(len(out["program"]["commands"]), 3)
        self.assertTrue(out["program"]["has_more_commands"])
        self.assertEqual(out["program"]["depth_samples"]["after"]["samples_xyz"], [[0.0, 0.0, 1.0]])
        self.assertNotIn("programs", out)

    def test_an_unknown_program_id_is_refused_without_touching_anything(self):
        out = receipt.state_receipt(policy_calls=1, records=self.records(2), proxies=[],
                                    last_observation=None,
                                    query=receipt.validate_state_query({"program_id": "nope"}))
        self.assertEqual(out["status"], "rejected")
        self.assertEqual(out["reason"], "unknown_program_id")
        self.assertIn("p1", out["known_program_ids"])

    def test_rejected_programs_stay_in_the_history(self):
        rejected = receipt.program_record(
            {"status": "rejected", "program_id": "bad1", "reason": "proxy_fit_invalid_no_motion",
             "executed": [], "bind": {"status": "ok", "proxy": {"name": "x", "valid": False}}}, 0)
        out = receipt.state_receipt(policy_calls=1, records=[rejected], proxies=[],
                                    last_observation=None,
                                    query=receipt.validate_state_query({}))
        summary = out["programs"][0]
        self.assertEqual(summary["status"], "rejected")
        self.assertEqual(summary["reason"], "proxy_fit_invalid_no_motion")
        self.assertEqual(summary["bind_status"], "ok")

    def test_a_stale_proxy_is_marked_and_the_stored_observation_is_labelled(self):
        current = receipt.proxy_state_entry("a", revision=1, bound_frame="f2", current=True,
                                            card={"center": [0, 0, 1], "valid": True})
        stale = receipt.proxy_state_entry("b", revision=2, bound_frame="f1", current=False,
                                          card={"center": [0, 0, 1], "valid": True})
        out = receipt.state_receipt(
            policy_calls=1, records=[], proxies=[current, stale],
            last_observation={"what": "observation", "frame": "f2"},
            query=receipt.validate_state_query({"proxies": {"limit": 1}}))
        self.assertEqual(out["proxies"], [current])
        self.assertTrue(out["has_more_proxies"])
        self.assertIn("historical", stale["note"])
        self.assertEqual(out["last_observation"]["what"], "observation")
        self.assertIn("stored copy", out["last_observation_note"])


class ToolWiringTests(unittest.TestCase):
    """The tools' own boundaries: no actuation, canonical text, history preserved."""

    def setUp(self):
        dg.reset_episode()

    def context(self):
        return SimpleNamespace(timing_id="t1", snap=AsyncMock(return_value=[]))

    def text_of(self, blocks):
        return next(b.text for b in blocks if b.type == "text")

    def call(self, tool, arguments):
        with patch.object(dc.RpcBackend, "execute",
                          side_effect=AssertionError("unexpected actuation")), \
             patch.object(dg, "log"):
            return asyncio.run(tool(self.context(), arguments))

    def test_state_is_read_only_canonical_and_keeps_program_history(self):
        target = pose([0.2, 0, 1.0])
        records = [{"command_id": "p1/0", "status": "completed",
                    "acknowledgement": ack("p1/0", target), "error": None}]
        payload = payload_of([move(target)], records, before=measured([0.1, 0, 1.0]))
        dg.STATE.programs.append(receipt.program_record(payload, 0))
        dg.STATE.last_observation = {"what": "observation", "frame": "f2",
                                     "measured_end_effector": measured([0.2, 0, 1.0])}
        text = self.text_of(self.call(dg.DgStateTool(), {}))
        value = json.loads(text)
        self.assertEqual(json.dumps(value), text)  # canonical: client dedup depends on it
        self.assertEqual(value["last_observation"]["what"], "observation")
        self.assertEqual(value["programs"][0]["program_id"], "p1")
        self.assertEqual(len(dg.STATE.programs), 1)

    def test_an_invalid_state_query_is_refused_and_actuates_nothing(self):
        value = json.loads(self.text_of(self.call(dg.DgStateTool(), {"programs": {"limit": 99}})))
        self.assertEqual(value["status"], "rejected")
        self.assertIn("limit_must_be_between_1_and_20", value["reason"])
        self.assertNotIn("programs", value)

    def test_a_look_reply_does_not_wipe_the_structured_program_history(self):
        dg.STATE.programs.append(receipt.program_record(
            {"status": "rejected", "program_id": "p0", "reason": "x", "executed": []}, 0))
        observation = {"what": "observation", "frame": "f1", "paired": True,
                       "measured_end_effector": measured([0.1, 0, 1.0])}
        blocks = asyncio.run(dg.DgLookTool().reply(
            self.context(), {"status": "ok", "observation": observation}))
        value = json.loads(self.text_of(blocks))
        self.assertEqual(value["observation"]["what"], "observation")
        self.assertEqual(value["observation"]["measured_eef"]["eef_position"], [0.1, 0.0, 1.0])
        self.assertEqual(len(dg.STATE.programs), 1)

    def test_the_public_eef_name_is_a_rename_only_and_carries_no_offset(self):
        state = dc.sensor_state({"ee_pos": [0.2, 0.1, 1.1],
                                 "ee_quat": Rotation.from_euler("z", 0.3).as_quat(),
                                 "gripper_open": np.array([0.5]), "sim_state": [1]},
                                seq=1, sim_steps=2, width_max_m=0.08,
                                commanded_open=1, control_freq=20)
        view = dc.measured_view(state)
        out = receipt.observation_receipt({"what": "observation",
                                           "measured_end_effector": view})
        self.assertEqual(out["measured_eef"][receipt.EEF_KEY], [0.2, 0.1, 1.1])
        self.assertEqual(view[receipt.LEGACY_EEF_KEY], dict(zip("xyz", [0.2, 0.1, 1.1])))


if __name__ == "__main__":
    unittest.main()

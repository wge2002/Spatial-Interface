"""Validity lifecycle of a geometry reference, and of the predictions made from it.

Each case is a way the previous version could report a number that looked like a
measurement and was not: stale coordinates reaching the proxy after a failed
refresh, a re-extraction near an old centre passing for object tracking, and an
error figure computed against a waypoint that was never the one predicted.
"""

import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from spatial_interface import geometry_ref as geom
from spatial_interface import mcp_server as server
from tests.test_geometry_frames_mcp import context, payload_of
from tests.test_geometry_ref import frame


def stored_reference(version="e1-s4-c4", center=(0.0, 0.0, 1.0)):
    """A reference as bind stores it, live geometry included."""
    summary = {"status": "ok", "center": dict(zip("xyz", center)),
               "extent_m": {"x": 0.04, "y": 0.04, "z": 0.04},
               "source": "pointcloud_via_pointcloud", "frame": "robot",
               "radius_m": 0.05, "support_points": 300}
    return {"reference_id": "ref1", "bound_observation_version": version,
            "radius_m": 0.05, "model_label": "cube", "center_ui": [0.0, 2.0, 0.0],
            "validity": "valid", "invalid_reason": None, "geometry": summary,
            "geometry_observation_version": version, "stale_geometry": None,
            "stale_observation_version": None, "points_robot": [[0, 0, 1]],
            "history": [{"observation_version": version, "status": "ok",
                         "center": summary["center"]}]}


class ReferenceValidityTests(unittest.TestCase):
    def test_geometry_from_this_very_frame_is_valid(self):
        self.assertEqual(geom.reference_validity(stored_reference(), "e1-s4-c4"),
                         ("valid", None))

    def test_geometry_from_an_earlier_frame_is_stale_not_valid(self):
        validity, reason = geom.reference_validity(stored_reference(), "e1-s5-c5")
        self.assertEqual(validity, "stale")
        self.assertIn("earlier_observation", reason)

    def test_a_similar_centre_and_extent_do_not_make_it_valid_again(self):
        # The whole failure mode: a neighbouring object of the same size, or a
        # different patch of one surface, re-extracts to the same numbers. Identity
        # is decided by which frame measured it, never by resemblance.
        ref = stored_reference(center=(0.0, 0.0, 1.0))
        twin = stored_reference(center=(0.0, 0.0, 1.0))
        twin["geometry_observation_version"] = "e1-s9-c9"
        self.assertEqual(ref["geometry"]["center"], twin["geometry"]["center"])
        self.assertEqual(geom.reference_validity(twin, "e1-s4-c4")[0], "stale")

    def test_no_live_geometry_is_invalid_with_the_stored_reason(self):
        ref = geom.invalidate_reference(stored_reference(),
                                       reason=geom.INVALID_REFRESH_FAILED)
        self.assertEqual(geom.reference_validity(ref, "e1-s4-c4"),
                         ("invalid", geom.INVALID_REFRESH_FAILED))

    def test_an_unverifiable_current_frame_cannot_be_valid(self):
        # observation_version None means freshness is unverifiable, so a match
        # against it must not be claimed either.
        self.assertEqual(geom.reference_validity(stored_reference(), None)[0], "stale")


class InvalidationTests(unittest.TestCase):
    def test_retired_geometry_survives_only_under_stale_names(self):
        ref = stored_reference()
        before = dict(ref["geometry"])
        geom.invalidate_reference(ref, reason=geom.INVALID_NEW_OBSERVATION)
        self.assertIsNone(ref["geometry"])
        self.assertEqual(ref["points_robot"], [])
        self.assertIsNone(ref["geometry_observation_version"])
        self.assertEqual(ref["stale_geometry"], before)
        self.assertEqual(ref["validity"], "invalid")

    def test_stale_fields_name_the_reason_and_never_a_current_centre(self):
        ref = stored_reference()
        geom.invalidate_reference(ref, reason=geom.INVALID_NEW_OBSERVATION)
        fields = geom.stale_fields(ref, reason=geom.INVALID_NEW_OBSERVATION)
        self.assertEqual(fields["validity"], "invalid")
        self.assertEqual(fields["stale_stored_center"], ref["stale_geometry"]["center"])
        self.assertNotIn("center", fields)
        self.assertNotIn("geometry", fields)
        self.assertIn("NOT the current location", fields["stale_note"])


def prediction(position=(0.0, 0.0, 1.08), approach=(0, 0, -1), opening=(0, 1, 0),
               ordinal=0, reference_id="ref1", version="e1-s4-c4"):
    return geom.prediction_record(
        prediction_id="pred1", observation_version_id=version,
        candidate={"position": list(position), "approach": list(approach),
                   "opening": list(opening)},
        corridor={"min_sampled_point_distance_m": 0.04,
                  "verdict": "no_sampled_points_within_body_radius"},
        attachment={"status": "unknown"}, reference_id=reference_id,
        execute_call_ordinal=ordinal)


def target(position=(0.0, 0.0, 1.08), approach=(0, 0, -1), opening=(0, 1, 0)):
    return {"position": dict(zip("xyz", position)),
            "approach": dict(zip("xyz", approach)),
            "opening": dict(zip("xyz", opening)), "gripper_open": True}


def measured(position=(0.0, 0.0, 1.085), approach=(0, 0, -1)):
    return {"status": "ok", "fingertip_position": dict(zip("xyz", position)),
            "approach": dict(zip("xyz", approach)), "opening": {"x": 0, "y": 1, "z": 0},
            "gripper_state_class": "closed"}


_DEFAULT = object()  # so a test can pass executed_target=None deliberately


def check(prediction_obj=None, *, executed_target=_DEFAULT, done_delta=1,
          measured_ee=None, before=0, paired=True, version_before="e1-s4-c4",
          before_paired=True, version_after="e1-s5-c5"):
    return geom.execution_check(
        prediction_obj if prediction_obj is not None else prediction(),
        executed_target=target() if executed_target is _DEFAULT else executed_target,
        measured=measured() if measured_ee is None else measured_ee,
        execute_calls_seen_before=before, observation_paired=paired,
        version_before=version_before, before_paired=before_paired,
        version_after=version_after, done_delta=done_delta)


class ExecutionCheckTests(unittest.TestCase):
    def test_the_predicted_waypoint_yields_a_measured_endpoint_error(self):
        out = check()
        self.assertEqual(out["status"], "measured")
        self.assertTrue(out["executed_the_predicted_target"])
        self.assertAlmostEqual(out["position_error_m"], 0.005, places=4)
        self.assertEqual(out["approach_error_deg"], 0.0)

    def test_a_reference_free_prediction_is_checked_the_same_way(self):
        out = check(prediction(reference_id=None))
        self.assertEqual(out["status"], "measured")
        self.assertIsNone(out["reference_id"])

    def test_a_different_target_position_is_not_verifiable_not_an_error(self):
        # The number 0.1 m below is the distance between two *different* intents.
        # Reporting it as a position error would read as the controller missing.
        out = check(executed_target=target(position=(0.0, 0.0, 0.98)))
        self.assertEqual(out["status"], "not_verifiable")
        self.assertIn("different target", out["reason"])
        self.assertNotIn("position_error_m", out)

    def test_a_matching_position_with_a_different_orientation_is_refused(self):
        # Without the orientation gate this passed: right point, wrong facing.
        out = check(executed_target=target(approach=(0, -1, 0)))
        self.assertEqual(out["status"], "not_verifiable")
        self.assertIn("different target", out["reason"])
        self.assertEqual(out["target_position_gap_m"], 0.0)
        self.assertEqual(out["target_approach_gap_deg"], 90.0)

    def test_a_different_opening_axis_about_the_approach_is_refused(self):
        out = check(executed_target=target(opening=(1, 0, 0)))
        self.assertEqual(out["status"], "not_verifiable")
        self.assertEqual(out["target_opening_gap_deg"], 90.0)

    def test_an_intervening_execute_call_makes_it_not_the_predicted_execution(self):
        out = check(prediction(ordinal=0), before=1)
        self.assertEqual(out["status"], "not_verifiable")
        self.assertIn("between the prediction", out["reason"])
        self.assertEqual(out["execute_calls_seen_at_prediction"], 0)
        self.assertEqual(out["execute_calls_seen_before_this_execution"], 1)

    def test_the_ordinal_is_never_reported_as_a_completion_count(self):
        # It increments once per execute_waypoint call, including calls that finish
        # nothing, so naming it a completion count would overstate what it knows.
        out = check(prediction(ordinal=0), before=1)
        self.assertNotIn("completed", json.dumps(out))

    def test_no_completed_waypoint_leaves_the_endpoint_unknown(self):
        out = check(done_delta=0)
        self.assertEqual(out["status"], "not_verifiable")
        self.assertIn("exactly one completed waypoint", out["reason"])
        self.assertEqual(out["waypoint_done_count_delta"], 0)
        self.assertNotIn("position_error_m", out)

    def test_two_completed_waypoints_cannot_be_attributed_to_this_prediction(self):
        out = check(done_delta=2)
        self.assertEqual(out["status"], "not_verifiable")
        self.assertIn("exactly one completed waypoint", out["reason"])
        self.assertEqual(out["waypoint_done_count_delta"], 2)
        self.assertNotIn("position_error_m", out)

    def test_an_unreadable_completion_count_is_not_verifiable(self):
        out = check(done_delta=None)
        self.assertEqual(out["status"], "not_verifiable")
        self.assertIn("could not be read", out["reason"])
        self.assertNotIn("position_error_m", out)

    def test_a_before_frame_other_than_the_predictions_source_is_refused(self):
        # The failure the local ordinal cannot see: an outside frame arrives between
        # the proxy call and the execution, so the candidate was evaluated against a
        # scene the controller never started from.
        out = check(prediction(version="e1-s4-c4"), version_before="e1-s6-c6",
                    version_after="e1-s7-c7")
        self.assertEqual(out["status"], "not_verifiable")
        self.assertIn("different observation", out["reason"])
        self.assertEqual(out["prediction_observation_version"], "e1-s4-c4")
        self.assertEqual(out["observation_version_before_execution"], "e1-s6-c6")
        self.assertNotIn("position_error_m", out)

    def test_an_unpaired_or_unreadable_before_frame_is_refused(self):
        for kwargs in ({"before_paired": False}, {"version_before": None}):
            out = check(**kwargs)
            self.assertEqual(out["status"], "not_verifiable")
            self.assertIn("before the execution", out["reason"])
            self.assertNotIn("position_error_m", out)

    def test_the_same_frame_after_the_execution_cannot_show_the_endpoint(self):
        out = check(version_after="e1-s4-c4")
        self.assertEqual(out["status"], "not_verifiable")
        self.assertEqual(out["observation_advance_problem"], "not_a_later_frame")
        self.assertNotIn("position_error_m", out)

    def test_an_after_frame_from_another_epoch_is_refused(self):
        out = check(version_after="e2-s1-c1")
        self.assertEqual(out["status"], "not_verifiable")
        self.assertEqual(out["observation_advance_problem"], "different_episode_epoch")
        self.assertNotIn("position_error_m", out)

    def test_a_missing_after_frame_version_is_refused(self):
        out = check(version_after=None)
        self.assertEqual(out["status"], "not_verifiable")
        self.assertEqual(out["observation_advance_problem"],
                         "unreadable_observation_version")
        self.assertNotIn("position_error_m", out)

    def test_a_millimetre_of_target_difference_is_a_different_candidate(self):
        # 1 mm is five times the round-trip gate: two poses this far apart are two
        # intents, and calling the gap a position error would read as the controller
        # having missed.
        out = check(executed_target=target(position=(0.0, 0.0, 1.081)))
        self.assertEqual(out["status"], "not_verifiable")
        self.assertIn("different target", out["reason"])
        self.assertEqual(out["target_position_gap_m"], 0.001)
        self.assertNotIn("position_error_m", out)

    def test_a_degree_of_orientation_difference_is_a_different_candidate(self):
        radians = math.radians(1.0)
        out = check(executed_target=target(
            approach=(0.0, -math.sin(radians), -math.cos(radians))))
        self.assertEqual(out["status"], "not_verifiable")
        self.assertIn("different target", out["reason"])
        self.assertAlmostEqual(out["target_approach_gap_deg"], 1.0, places=2)
        self.assertNotIn("position_error_m", out)

    def test_ordinary_output_rounding_still_round_trips_as_the_same_target(self):
        # The two sides are rounded independently — the prediction to 4 decimals in
        # metres, the executed target through ui_to_robot's own 4 decimals — so one
        # identical pose can differ by 1e-4 per component. That has to stay a match.
        out = check(executed_target=target(position=(0.0001, -0.0001, 1.0801),
                                          approach=(0.0, 1e-9, -1.0)))
        self.assertEqual(out["status"], "measured")
        self.assertTrue(out["executed_the_predicted_target"])

    def test_an_unpaired_observation_after_execution_cannot_measure_it(self):
        out = check(paired=False)
        self.assertEqual(out["status"], "not_verifiable")
        self.assertIn("paired observation", out["reason"])

    def test_missing_telemetry_after_execution_is_not_verifiable(self):
        out = check(measured_ee={"status": "unknown", "reason": "no telemetry"})
        self.assertEqual(out["status"], "not_verifiable")

    def test_an_unreadable_executed_target_is_not_verifiable(self):
        out = check(executed_target=None)
        self.assertEqual(out["status"], "not_verifiable")
        self.assertIn("could not be read", out["reason"])

    def test_a_perfect_endpoint_match_still_proves_nothing_is_held(self):
        # Close and closed is the case the old attachment note warned about; an
        # exact endpoint hit must not upgrade it.
        out = check(measured_ee=measured(position=(0.0, 0.0, 1.08)))
        self.assertEqual(out["status"], "measured")
        self.assertEqual(out["position_error_m"], 0.0)
        self.assertEqual(out["object_follows_gripper"], "unknown")

    def test_orientation_is_part_of_the_written_prediction(self):
        record = prediction()
        self.assertEqual(record["predicted_approach"], {"x": 0.0, "y": 0.0, "z": -1.0})
        self.assertEqual(record["predicted_opening"], {"x": 0.0, "y": 1.0, "z": 0.0})
        self.assertEqual(record["execute_call_ordinal_at_prediction"], 0)
        self.assertFalse(record["endpoint_checked"])

    def test_the_match_gates_are_no_wider_than_the_output_precision(self):
        # Both sides are rounded to 4 decimals in metres, so sqrt(3) * 1e-4 is the
        # largest gap one identical pose can produce. The gate must clear that and
        # stay well under a millimetre of real difference.
        self.assertGreater(geom.TARGET_MATCH_M, math.sqrt(3) * 1e-4)
        self.assertLess(geom.TARGET_MATCH_M, 0.001)
        # Directions are unit vectors rounded to 9 decimals; the round-trip angle is
        # orders of magnitude below a degree.
        self.assertLess(geom.TARGET_MATCH_DEG, 1.0)


class ReferenceLifecycleMcpTests(unittest.IsolatedAsyncioTestCase):
    """The stale-reference path end to end, through the tools themselves."""

    def setUp(self):
        server._geometry_reset_episode_state()

    async def bind(self, seq=4):
        out = payload_of(await server.GeometryBindReferenceTool()(
            context([frame(seq=seq, cloud_seq=seq)]), {"u": 0.5, "v": 0.5}))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["validity"], "valid")
        return out["reference_id"]

    async def refresh(self, ref, seq):
        return payload_of(await server.GeometryRefreshReferenceTool()(
            context([frame(seq=seq, cloud_seq=seq)]), {"reference_id": ref}))

    async def proxy(self, ref, seq):
        ctx = context([frame(seq=seq, cloud_seq=seq)])
        out = payload_of(await server.GeometryProxyTool()(ctx, {
            "frame": "robot", "delta_position": [0, 0, -0.02],
            **({"reference_id": ref} if ref else {})}))
        return out, ctx

    async def test_a_reference_from_an_earlier_frame_is_refused_by_the_proxy(self):
        ref = await self.bind(seq=4)
        out, ctx = await self.proxy(ref, seq=5)
        self.assertEqual(out["status"], "invalid")
        self.assertEqual(out["validity"], "stale")
        self.assertIn("stale_stored_center", out)
        self.assertNotIn("reference_span", out)
        self.assertNotIn("prediction", out)
        self.assertEqual(server.GEOMETRY_PREDICTIONS, {})
        ctx.corridor_clearance.assert_not_awaited()

    async def test_a_refresh_on_a_new_frame_retires_it_instead_of_tracking(self):
        ref = await self.bind(seq=4)
        out = await self.refresh(ref, seq=7)
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["validity"], "invalid")
        self.assertEqual(out["invalid_reason"], geom.INVALID_NEW_OBSERVATION)
        self.assertEqual(out["association"], "unverified_across_observations")
        self.assertIsNone(out["geometry"])
        # The fresh numbers are still offered — for inspection, under a name that
        # cannot be read as the object's location.
        self.assertEqual(out["nearby_region_geometry"]["status"], "ok")
        self.assertIn("not an object identity", out["nearby_region_note"])
        self.assertIsNone(server.GEOMETRY_REFERENCES[ref]["geometry"])

    async def test_a_retired_reference_cannot_be_revived_by_a_later_proxy_call(self):
        ref = await self.bind(seq=4)
        await self.refresh(ref, seq=7)
        out, ctx = await self.proxy(ref, seq=7)
        self.assertEqual(out["status"], "invalid")
        self.assertEqual(out["invalid_reason"], geom.INVALID_NEW_OBSERVATION)
        self.assertEqual(server.GEOMETRY_PREDICTIONS, {})
        ctx.corridor_clearance.assert_not_awaited()

    async def test_a_same_frame_refresh_keeps_the_reference_usable(self):
        ref = await self.bind(seq=4)
        out = await self.refresh(ref, seq=4)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["validity"], "valid")
        self.assertTrue(out["same_observation_as_stored_geometry"])
        proxied, _ = await self.proxy(ref, seq=4)
        self.assertEqual(proxied["status"], "ok")
        self.assertEqual(proxied["validity"], "valid")

    async def test_a_failed_same_frame_refresh_retires_the_stored_geometry(self):
        ref = await self.bind(seq=4)
        ctx = context([frame(seq=4, cloud_seq=4)])
        # Support collapses: the neighbourhood no longer summarizes, so the older
        # extraction cannot stand as current either.
        ctx.sample_local_cloud = AsyncMock(return_value={
            "support": 3, "stride": 1, "points": [0.0, 2.0, 0.0], "cloud_points": 10})
        out = payload_of(await server.GeometryRefreshReferenceTool()(
            ctx, {"reference_id": ref}))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["invalid_reason"], geom.INVALID_REFRESH_FAILED)
        self.assertIn("stale_stored_center", out)
        self.assertIsNone(server.GEOMETRY_REFERENCES[ref]["geometry"])
        after, _ = await self.proxy(ref, seq=4)
        self.assertEqual(after["status"], "invalid")

    async def test_a_reference_free_candidate_is_still_recorded(self):
        out, _ = await self.proxy(None, seq=4)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(len(server.GEOMETRY_PREDICTIONS), 1)
        record = server.GEOMETRY_PREDICTIONS["pred1"]
        self.assertIsNone(record["reference_id"])
        self.assertIn("predicted_approach", record)

    async def test_a_frame_that_cannot_be_verified_retires_the_reference(self):
        # The reference does not survive a refresh that could not run: nothing here
        # shows the stored centre still describes anything, so it goes stale-only
        # and the proxy refuses it until a visual rebind.
        ref = await self.bind(seq=4)
        out = payload_of(await server.GeometryRefreshReferenceTool()(
            context([frame(seq=6, cloud_seq=4)]), {"reference_id": ref}))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["validity"], "invalid")
        self.assertEqual(out["invalid_reason"], geom.INVALID_UNVERIFIABLE_FRAME)
        self.assertIsNone(server.GEOMETRY_REFERENCES[ref]["geometry"])
        self.assertIn("stale_stored_center", out)
        self.assertNotIn("geometry", out)
        after, ctx = await self.proxy(ref, seq=4)
        self.assertEqual(after["status"], "invalid")
        ctx.corridor_clearance.assert_not_awaited()

    async def test_a_frame_change_mid_refresh_retires_the_reference(self):
        ref = await self.bind(seq=4)
        out = payload_of(await server.GeometryRefreshReferenceTool()(
            context([frame(seq=4, cloud_seq=4), frame(seq=5, cloud_seq=5)]),
            {"reference_id": ref}))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["validity"], "invalid")
        self.assertEqual(out["invalid_reason"],
                         geom.INVALID_FRAME_CHANGED_DURING_REFRESH)
        self.assertIsNone(server.GEOMETRY_REFERENCES[ref]["geometry"])

    async def test_returning_to_the_old_frame_cannot_revive_a_retired_reference(self):
        # The frame the reference was extracted from can come back around — a later
        # refresh on that very version must not read the retired geometry as current.
        ref = await self.bind(seq=4)
        await server.GeometryRefreshReferenceTool()(
            context([frame(seq=6, cloud_seq=4)]), {"reference_id": ref})
        out = await self.refresh(ref, seq=4)
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["validity"], "invalid")
        self.assertIsNone(out["geometry"])
        self.assertIsNone(server.GEOMETRY_REFERENCES[ref]["geometry"])
        again, ctx = await self.proxy(ref, seq=4)
        self.assertEqual(again["status"], "invalid")
        ctx.corridor_clearance.assert_not_awaited()


def executed_frame(seq, done):
    """One paired frame with an explicit completed-waypoint count."""
    return dict(frame(seq=seq, cloud_seq=seq), waypoint_done_count=done)


class ExecutionHookMcpTests(unittest.IsolatedAsyncioTestCase):
    """The proxy → execute_waypoint → refresh path through the hooks themselves."""

    def setUp(self):
        server._geometry_reset_episode_state()

    async def bind_and_predict(self, seq=4):
        ref = payload_of(await server.GeometryBindReferenceTool()(
            context([frame(seq=seq, cloud_seq=seq)]), {"u": 0.5, "v": 0.5}))["reference_id"]
        out = payload_of(await server.GeometryProxyTool()(
            context([frame(seq=seq, cloud_seq=seq)]),
            {"frame": "robot", "delta_position": [0, 0, -0.02],
             "reference_id": ref}))
        self.assertEqual(out["status"], "ok")
        return ref, out["prediction"]

    async def execute(self, *, before_seq=4, before_done=3, after_seq=5, after_done=4,
                      target_position=None):
        """Run both hooks around a no-op execution and return the hook report.

        The virtual target before the execution defaults to the recorded prediction's
        own position, which is what "execute this exact candidate next" means.
        """
        before_ctx = context([executed_frame(before_seq, before_done)])
        if target_position is None:
            target_position = tuple(
                server.GEOMETRY_PREDICTIONS["pred1"]
                ["predicted_fingertip_position"][k] for k in "xyz")
        pose = dict(before_ctx.gripper_pose.return_value,
                    robot_position=dict(zip("xyz", target_position)))
        before_ctx.gripper_pose = AsyncMock(return_value=pose)
        state = await server._geometry_before_execution(before_ctx)
        after_ctx = context([executed_frame(after_seq, after_done)])
        blocks = await server._geometry_after_execution(after_ctx, state)
        return json.loads(blocks[0].text) if blocks else None

    async def test_the_predicted_target_yields_one_measured_endpoint_check(self):
        await self.bind_and_predict(seq=4)
        report = await self.execute()
        self.assertEqual(report["waypoint_done_count_delta"], 1)
        self.assertEqual(report["observation_before_execution"],
                         geom.observation_version(executed_frame(4, 3)))
        self.assertEqual(report["observation_after_execution"],
                         geom.observation_version(executed_frame(5, 4)))
        checks = report["geometry_prediction_endpoint_checks"]
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["status"], "measured")
        self.assertIn("position_error_m", checks[0])
        self.assertEqual(server.GEOMETRY_PREDICTIONS["pred1"]["execution_check"],
                         checks[0])

    async def test_a_frame_arriving_while_the_target_is_read_is_not_verifiable(self):
        # The pre-hook reads the frame id, then the target, then the frame id again.
        # A frame landing in between would pair this frame's version with the
        # previous frame's target — a mixture every later same-version check would
        # accept, so the before-frame is discarded instead.
        _, predicted = await self.bind_and_predict(seq=4)
        before_ctx = context([executed_frame(4, 3), executed_frame(5, 3)])
        pose = dict(before_ctx.gripper_pose.return_value,
                    robot_position=predicted["predicted_fingertip_position"])
        before_ctx.gripper_pose = AsyncMock(return_value=pose)
        state = await server._geometry_before_execution(before_ctx)
        self.assertTrue(state["frame_changed_while_reading_target"])
        self.assertIsNone(state["version"])
        report = json.loads((await server._geometry_after_execution(
            context([executed_frame(6, 4)]), state))[0].text)
        check = report["geometry_prediction_endpoint_checks"][0]
        self.assertEqual(check["status"], "not_verifiable")
        self.assertIn("before the execution", check["reason"])
        self.assertNotIn("position_error_m", check)

    async def test_the_report_counts_execute_calls_and_never_completions(self):
        await self.bind_and_predict(seq=4)
        report = await self.execute()
        self.assertEqual(report["execute_calls_seen"], 1)
        self.assertNotIn("waypoints_completed", report)

    async def test_an_execution_from_a_newer_frame_is_not_verifiable(self):
        # The prediction was evaluated against seq 4; the execution starts from
        # seq 6. No local counter moved, and this still must not verify.
        await self.bind_and_predict(seq=4)
        report = await self.execute(before_seq=6, after_seq=7)
        check = report["geometry_prediction_endpoint_checks"][0]
        self.assertEqual(check["status"], "not_verifiable")
        self.assertIn("different observation", check["reason"])
        self.assertNotIn("position_error_m", check)

    async def test_a_completion_count_that_does_not_move_is_not_verifiable(self):
        await self.bind_and_predict(seq=4)
        report = await self.execute(after_done=3)
        check = report["geometry_prediction_endpoint_checks"][0]
        self.assertEqual(check["status"], "not_verifiable")
        self.assertEqual(check["waypoint_done_count_delta"], 0)

    async def test_two_completions_across_one_call_are_not_verifiable(self):
        await self.bind_and_predict(seq=4)
        report = await self.execute(after_done=5)
        check = report["geometry_prediction_endpoint_checks"][0]
        self.assertEqual(check["status"], "not_verifiable")
        self.assertEqual(check["waypoint_done_count_delta"], 2)

    async def test_a_second_execution_cannot_check_the_same_prediction_again(self):
        await self.bind_and_predict(seq=4)
        first = await self.execute()
        self.assertEqual(first["geometry_prediction_endpoint_checks"][0]["status"],
                         "measured")
        second = await self.execute(before_seq=5, before_done=4,
                                    after_seq=6, after_done=5)
        self.assertIsNone(second)

    async def test_refresh_reports_the_stored_check_and_computes_no_second_figure(self):
        ref, _ = await self.bind_and_predict(seq=4)
        await self.execute()
        out = payload_of(await server.GeometryRefreshReferenceTool()(
            context([executed_frame(5, 4)]), {"reference_id": ref}))
        checks = out["prediction_checks"]
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["source"], "execution_check_at_execute_waypoint")
        self.assertEqual(checks[0], {"source": "execution_check_at_execute_waypoint",
                                     **server.GEOMETRY_PREDICTIONS["pred1"]
                                     ["execution_check"]})
        # The neighbourhood figure is named as a region, with identity unknown, and
        # is not a second answer to the endpoint question.
        self.assertEqual(out["region_comparison"]["identity"], "unknown")
        self.assertNotIn("position_error", json.dumps(out["region_comparison"]))

    async def test_a_candidate_that_was_never_executed_is_not_verifiable_in_refresh(self):
        ref, _ = await self.bind_and_predict(seq=4)
        out = payload_of(await server.GeometryRefreshReferenceTool()(
            context([executed_frame(5, 4)]), {"reference_id": ref}))
        check = out["prediction_checks"][0]
        self.assertEqual(check["status"], "not_verifiable")
        self.assertEqual(check["source"], "no_execution_check_exists")
        self.assertNotIn("position_error_m", check)

    async def test_refresh_after_a_different_candidate_ran_reports_no_endpoint_error(self):
        # The failure this replaces: refresh computed an endpoint error from a later
        # observation without ever reading the executed target, so a *different*
        # waypoint produced a number that read as the prediction having missed.
        ref, _ = await self.bind_and_predict(seq=4)
        report = await self.execute(target_position=(0.0, 0.0, 0.9))
        check = report["geometry_prediction_endpoint_checks"][0]
        self.assertEqual(check["status"], "not_verifiable")
        self.assertIn("different target", check["reason"])
        out = payload_of(await server.GeometryRefreshReferenceTool()(
            context([executed_frame(5, 4)]), {"reference_id": ref}))
        self.assertEqual(out["prediction_checks"][0]["status"], "not_verifiable")
        self.assertNotIn("position_error_m", json.dumps(out["prediction_checks"]))
        self.assertNotIn("fingertip_check", json.dumps(out["prediction_checks"]))

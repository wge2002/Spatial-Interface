"""Pure geometry/validity contracts for the reference + proxy prototype.

The cases here are behavioural counterexamples, not source-string checks: each one
constructs the observation that would make an over-claiming implementation return
the wrong verdict (an object hidden inside the self-exclusion sphere, a cube
resting untouched beside closed jaws, an exactly collinear neighbourhood), and
asserts the reply stays honest. Float comparisons use tolerances.
"""

import json
import math
import unittest

from spatial_interface import geometry_ref as geom


def blob(n=200, spread=0.01, elongate=0.0, center=(0.0, 0.0, 1.0)):
    """Deterministic pseudo-random neighbourhood; elongate stretches along x."""
    points = []
    for i in range(n):
        t = (i * 0.6180339887498949) % 1.0 - 0.5
        s = (i * 0.7548776662466927) % 1.0 - 0.5
        u = (i * 0.5698402909980532) % 1.0 - 0.5
        points.append([center[0] + t * (spread + elongate),
                       center[1] + s * spread,
                       center[2] + u * spread])
    return points


class FrameTests(unittest.TestCase):
    def test_ui_robot_round_trip_matches_server_convention(self):
        for p in ([0.0, 0.0, 0.0], [1.5, -2.25, 0.75], [-3.0, 4.0, -1.0]):
            back = geom.robot_to_ui(geom.ui_to_robot(p, 0.8), 0.8)
            for a, b in zip(p, back):
                self.assertAlmostEqual(a, b, places=9)

    def test_ui_to_robot_uses_the_documented_axis_map_and_scale(self):
        # robot = (ui_x/10, -ui_z/10, ui_y/10 + offset); 1 UI unit = 0.1 m.
        # Compared with a tolerance: (1.0 - 0.8) * 10 is not exactly 2.0 in binary
        # floating point, and an exact-equality assertion here would be testing
        # float representation rather than the axis map.
        for got, want in zip(geom.ui_to_robot([1.0, 2.0, 3.0], 0.8), [0.1, -0.3, 1.0]):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(geom.robot_to_ui([0.1, -0.3, 1.0], 0.8), [1.0, 2.0, 3.0]):
            self.assertAlmostEqual(got, want, places=12)

    def test_offset_other_than_default_is_honoured(self):
        self.assertAlmostEqual(geom.ui_to_robot([0, 0, 0], 0.5)[2], 0.5)


_SAME = object()


def frame(seq=7, cloud_seq=_SAME, *, epoch="e1", cam_seq=None, displayed=None,
          pending=None, proprio_seq=None, labels=("agentview", "wrist")):
    """One producer frame as OBSERVE_JS reports it, with pairing metadata.

    Defaults describe a fully paired frame; each argument spoils one pairing.
    """
    if cloud_seq is _SAME:
        cloud_seq = seq
    cam_seq = seq if cam_seq is None else cam_seq
    proprio_seq = seq if proprio_seq is None else proprio_seq
    feeds = {}
    for label in labels:
        feeds[geom.CAM_LABEL_TO_ELEMENT[label]] = {
            "displayed": seq if displayed is None else displayed,
            "pending": seq if pending is None else pending,
        }
    return {
        "cloud_points": 100000, "cloud_hash": "abc123", "waypoint_done_count": 3,
        "gripper_action": [0], "fingertip_ui": [1.0, 2.0, 3.0],
        "frame_meta": {"epoch": epoch, "seq": seq, "cloud_seq": cloud_seq,
                       "cloud_is_fresh": cloud_seq == seq,
                       "cam_labels": list(labels), "cam_info_seq": cam_seq,
                       "proprio_seq": proprio_seq},
        "cam_feed_seq": feeds,
    }


class ObservationVersionTests(unittest.TestCase):
    def test_version_comes_from_producer_metadata_not_cloud_contents(self):
        # A hash cannot show a cloud is new: two observations of a still scene
        # hash alike. Only the producer's own frame numbering can.
        a = frame(seq=7, cloud_seq=7)
        self.assertEqual(geom.observation_version(a),
                         geom.observation_version(dict(a, cloud_hash="def456",
                                                       cloud_points=99999)))
        self.assertNotEqual(geom.observation_version(a),
                            geom.observation_version(frame(seq=8, cloud_seq=8)))

    def test_waypoint_count_alone_cannot_mint_a_new_version(self):
        a = frame(seq=7, cloud_seq=7)
        self.assertEqual(geom.observation_version(a),
                         geom.observation_version(dict(a, waypoint_done_count=99)))

    def test_a_streamed_frame_reusing_the_cloud_is_a_different_version(self):
        # It genuinely is a different frame (new tip, new images, old cloud), so a
        # same-version check across evaluates must fail on it.
        self.assertNotEqual(geom.observation_version(frame(seq=7, cloud_seq=7)),
                            geom.observation_version(frame(seq=8, cloud_seq=7)))

    def test_cloud_version_identifies_the_frame_that_measured_the_cloud(self):
        # ... while the cloud itself is unchanged, and says so.
        self.assertEqual(geom.cloud_version(frame(seq=8, cloud_seq=7)),
                         geom.cloud_version(frame(seq=9, cloud_seq=7)))
        self.assertNotEqual(geom.cloud_version(frame(seq=9, cloud_seq=7)),
                            geom.cloud_version(frame(seq=9, cloud_seq=9)))

    def test_a_new_producer_process_never_reuses_an_old_frames_identity(self):
        self.assertNotEqual(geom.observation_version(frame(seq=7, epoch="e1")),
                            geom.observation_version(frame(seq=7, epoch="e2")))

    def test_missing_observation_is_none_not_a_fabricated_version(self):
        self.assertIsNone(geom.observation_version(None))
        self.assertIsNone(geom.observation_version({}))

    def test_ui_without_pairing_metadata_is_unverifiable_not_current(self):
        old_ui = {"cloud_points": 100, "cloud_hash": "abc123",
                  "waypoint_done_count": 3, "fingertip_ui": [1, 2, 3]}
        self.assertIsNone(geom.observation_version(old_ui))
        self.assertIsNone(geom.cloud_version(old_ui))
        state = geom.pairing_state(old_ui)
        self.assertEqual(state["status"], "unknown")
        self.assertEqual(state["reason"], "no_frame_metadata")

    def test_malformed_metadata_is_rejected_rather_than_half_trusted(self):
        for bad in ({"seq": 1}, {"epoch": "", "seq": 1, "cloud_seq": 1},
                    {"epoch": "e1", "seq": "x", "cloud_seq": 1}, {"epoch": "e1"}):
            obs = dict(frame(), frame_meta=bad)
            self.assertIsNone(geom.observation_version(obs), bad)
            self.assertEqual(geom.pairing_state(obs)["status"], "unknown", bad)


class PairingTests(unittest.TestCase):
    def test_a_fully_paired_frame_is_paired(self):
        state = geom.pairing_state(frame(), require_cam_labels=("wrist",))
        self.assertEqual(state["status"], "paired")
        self.assertIsNone(state["reason"])

    def test_cached_cloud_with_newer_proprioception_is_rejected(self):
        # The exact frame record_sim streams during execution: skip_pcl reuses the
        # previous cloud while the fingertip and images move on.
        state = geom.pairing_state(frame(seq=8, cloud_seq=7))
        self.assertEqual(state["status"], "unknown")
        self.assertEqual(state["reason"], "stale_cloud_reused_by_streamed_frame")

    def test_no_cloud_yet_is_unknown(self):
        self.assertEqual(geom.pairing_state(frame(seq=3, cloud_seq=None))["reason"],
                         "no_cloud_yet")

    def test_pending_camera_decode_cannot_be_paired_with_this_calibration(self):
        # The new JPEG was handed to the browser but has not finished decoding, so
        # the pixels on screen are still the previous frame's.
        state = geom.pairing_state(frame(seq=9, displayed=8, pending=9),
                                   require_cam_labels=("agentview",))
        self.assertEqual(state["status"], "unknown")
        self.assertEqual(state["reason"], "camera_image_decode_pending")

    def test_older_image_still_displayed_is_rejected_as_stale(self):
        state = geom.pairing_state(frame(seq=9, displayed=8, pending=8),
                                   require_cam_labels=("wrist",))
        self.assertEqual(state["reason"], "stale_camera_image_displayed")

    def test_camera_freshness_is_only_required_of_the_feed_actually_used(self):
        # A canvas click does not read any camera image, so a lagging feed must
        # not block it — and naming that feed must.
        obs = frame(seq=9, displayed=8, pending=9)
        self.assertEqual(geom.pairing_state(obs)["status"], "paired")
        self.assertEqual(geom.pairing_state(obs, require_cam_labels=("wrist",))["status"],
                         "unknown")

    def test_a_feed_with_no_display_record_is_unknown_not_assumed_fresh(self):
        obs = frame(seq=5)
        obs["cam_feed_seq"] = {}
        self.assertEqual(
            geom.pairing_state(obs, require_cam_labels=("wrist",))["reason"],
            "no_display_record_for_wrist")

    def test_stale_calibration_or_telemetry_is_rejected(self):
        self.assertEqual(geom.pairing_state(frame(seq=9, proprio_seq=8))["reason"],
                         "stale_proprioception")
        self.assertEqual(
            geom.pairing_state(frame(seq=9, cam_seq=8),
                               require_cam_labels=("wrist",))["reason"],
            "stale_camera_calibration")

    def test_unchanged_frame_is_stable_across_repeated_reads(self):
        # Screenshots, orbits and target edits do not go through the producer, so
        # re-reading the same frame must yield the same id and the same verdict.
        a, b = frame(seq=11), frame(seq=11)
        self.assertEqual(geom.observation_version(a), geom.observation_version(b))
        self.assertEqual(geom.pairing_state(a, require_cam_labels=("wrist",)),
                         geom.pairing_state(b, require_cam_labels=("wrist",)))

    def test_gripper_state_is_a_class_and_unknown_without_telemetry(self):
        self.assertEqual(geom.gripper_state_class({"gripper_action": [1]}), "closed")
        self.assertEqual(geom.gripper_state_class({"gripper_action": [0]}), "open")
        self.assertEqual(geom.gripper_state_class({}), "unknown")
        self.assertEqual(geom.gripper_state_class(None), "unknown")

    def test_measured_end_effector_is_unknown_without_fingertip(self):
        out = geom.measured_end_effector({"gripper_action": [1]}, 0.8)
        self.assertEqual(out["status"], "unknown")
        self.assertNotIn("fingertip_position", out)

    def test_measured_end_effector_reports_robot_frame_and_class_caveat(self):
        out = geom.measured_end_effector(
            {"fingertip_ui": [1.0, 2.0, 3.0], "gripper_action": [1],
             "ee_dirs_robot": {"approach": [0, 0, -1], "opening": [0, 1, 0]}}, 0.8)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["fingertip_position"], {"x": 0.1, "y": -0.3, "z": 1.0})
        self.assertEqual(out["gripper_state_class"], "closed")
        self.assertIn("not grasp evidence", out["gripper_state_note"])


class SummaryTests(unittest.TestCase):
    def test_thin_support_is_unknown_with_no_coordinates(self):
        out = geom.summarize_points(blob(5), radius_m=0.05, support=5, stride=1,
                                    source="pointcloud")
        self.assertEqual(out["status"], "unknown")
        self.assertIsNone(out["center"])
        self.assertIsNone(out["extent_m"])
        self.assertIsNone(out["principal_axis"])

    def test_isotropic_blob_reports_no_principal_axis(self):
        points = blob(200, spread=0.02)
        out = geom.summarize_points(points, radius_m=0.05, support=200, stride=1,
                                    source="pointcloud")
        self.assertEqual(out["status"], "ok")
        self.assertIsNone(out["principal_axis"])
        self.assertTrue(any("elongated" in r for r in out["reasons"]))

    def test_elongated_blob_reports_an_axis_along_the_long_direction(self):
        points = blob(300, spread=0.005, elongate=0.06)
        out = geom.summarize_points(points, radius_m=0.06, support=300, stride=1,
                                    source="pointcloud")
        self.assertEqual(out["status"], "ok")
        axis = [out["principal_axis"][k] for k in "xyz"]
        self.assertGreater(abs(axis[0]), 0.9)
        self.assertGreater(out["principal_extent_m"], 0.03)
        self.assertGreater(out["axis_ratio"], geom.AXIS_RATIO_MIN)

    def test_perfectly_collinear_points_still_yield_the_evidenced_axis(self):
        # values[1] == 0 for an exactly 1-D neighbourhood. That is the strongest
        # possible axis evidence, so it must not be rejected as degenerate.
        points = [[0.01 * i, 0.0, 1.0] for i in range(60)]
        out = geom.summarize_points(points, radius_m=0.4, support=60, stride=1,
                                    source="pointcloud")
        self.assertEqual(out["status"], "ok")
        self.assertIsNotNone(out["principal_axis"])
        self.assertGreater(abs(out["principal_axis"]["x"]), 0.99)
        self.assertAlmostEqual(out["principal_extent_m"], 0.59, places=3)
        # The ratio is unbounded rather than a number, and that is said explicitly
        # instead of being encoded as a huge float or a silent None.
        self.assertIsNone(out["axis_ratio"])
        self.assertIn("collinear", out["axis_ratio_note"])

    def test_collinear_but_microscopic_extent_still_has_no_axis(self):
        # Collinearity is not enough on its own: a 0.1 mm line is below the
        # variance floor, so the direction is sampling noise, not geometry.
        points = [[1e-5 * i, 0.0, 1.0] for i in range(60)]
        out = geom.summarize_points(points, radius_m=0.05, support=60, stride=1,
                                    source="pointcloud")
        self.assertEqual(out["status"], "ok")
        self.assertIsNone(out["principal_axis"])
        self.assertTrue(any("too small" in r for r in out["reasons"]))

    def test_reported_axis_is_a_unit_vector_to_the_digits_reported(self):
        # The reply is rounded; a caller re-checking |axis| == 1 must not be
        # tripped by the rounding itself.
        for points in (blob(300, spread=0.005, elongate=0.06),
                       [[0.01 * i, 0.0, 1.0] for i in range(60)]):
            out = geom.summarize_points(points, radius_m=0.4, support=len(points),
                                        stride=1, source="pointcloud")
            axis = [out["principal_axis"][k] for k in "xyz"]
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in axis)), 1.0, places=8)

    def test_keypoints_are_actual_cloud_returns(self):
        points = blob(200, spread=0.02)
        out = geom.summarize_points(points, radius_m=0.05, support=200, stride=1,
                                    source="pointcloud")
        highest = [out["keypoints"]["highest_point"][k] for k in "xyz"]
        self.assertTrue(any(all(abs(a - b) < 5e-5 for a, b in zip(highest, p))
                            for p in points))

    def test_eigh_matches_a_known_decomposition(self):
        values, vectors = geom.jacobi_eigh([[4.0, 1.0, 0.0], [1.0, 4.0, 0.0],
                                            [0.0, 0.0, 2.0]])
        self.assertAlmostEqual(values[0], 5.0, places=9)
        self.assertAlmostEqual(values[1], 3.0, places=9)
        self.assertAlmostEqual(values[2], 2.0, places=9)
        for vec in vectors:
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in vec)), 1.0, places=9)


class RefreshTests(unittest.TestCase):
    def summary(self, center, spread=0.02, support=200):
        return geom.summarize_points(blob(support, spread=spread, center=center),
                                     radius_m=0.05, support=support, stride=1,
                                     source="pointcloud")

    def test_small_motion_matches_and_reports_the_shift(self):
        status, reasons, shift = geom.refresh_verdict(
            self.summary((0, 0, 1)), self.summary((0.01, 0, 1)))
        self.assertEqual(status, "ok")
        self.assertLess(shift, geom.MAX_TRACK_SHIFT_M)
        self.assertEqual(reasons, [])

    def test_motion_beyond_the_gate_is_unknown_not_tracked(self):
        status, reasons, shift = geom.refresh_verdict(
            self.summary((0, 0, 1)), self.summary((0.2, 0, 1)))
        self.assertEqual(status, "unknown")
        self.assertGreater(shift, geom.MAX_TRACK_SHIFT_M)
        self.assertTrue(any("gate" in r for r in reasons))

    def test_occluded_reextraction_is_unknown(self):
        thin = geom.summarize_points(blob(4), radius_m=0.05, support=4, stride=1,
                                     source="pointcloud")
        status, reasons, shift = geom.refresh_verdict(self.summary((0, 0, 1)), thin)
        self.assertEqual(status, "unknown")
        self.assertIsNone(shift)
        self.assertTrue(reasons)

    def test_neighbourhood_replaced_by_a_different_shape_is_ambiguous(self):
        # A similar object swapped into the neighbourhood: the centre barely
        # moves, so only the extent change can reveal that the association is
        # not established.
        status, reasons, _ = geom.refresh_verdict(self.summary((0, 0, 1), spread=0.01),
                                                  self.summary((0, 0, 1), spread=0.09))
        self.assertEqual(status, "unknown")
        self.assertTrue(any("ambiguous" in r for r in reasons))


class ProxyTests(unittest.TestCase):
    def corridor(self, distances_ui, body=None, observed=None, excluded=7):
        n = len(distances_ui)
        return {"min_distance": distances_ui,
                "body_support": body if body is not None else [0] * n,
                "observed_support": observed if observed is not None else [3] * n,
                "excluded_self_points": excluded, "cloud_points": 100000}

    def test_path_samples_span_measured_start_to_candidate(self):
        samples = geom.path_samples_robot([0, 0, 1], [0, 0, 1.1], count=5)
        self.assertEqual(len(samples), 5)
        self.assertEqual(samples[0], [0, 0, 1])
        self.assertAlmostEqual(samples[-1][2], 1.1)
        self.assertAlmostEqual(samples[2][2], 1.05)

    def test_distance_is_named_and_documented_as_a_sampled_point_distance(self):
        samples = geom.path_samples_robot([0, 0, 1], [0, 0, 1.05], count=3)
        out = geom.summarize_corridor(self.corridor([2.0, 1.0, 3.0]), samples)
        self.assertEqual(out["verdict"], "no_sampled_points_within_body_radius")
        self.assertAlmostEqual(out["min_sampled_point_distance_m"], 0.1)
        self.assertEqual(out["excluded_self_points"], 7)
        # No key may promise a bound, clearance or free space. The word may appear
        # in a note only to deny it, which the next assertion pins down.
        text = json.dumps(out)
        self.assertNotIn("clearance", json.dumps(sorted(out)))
        self.assertNotIn("clear_in", text)
        self.assertTrue(any("not a clearance lower bound" in n for n in out["notes"]))
        self.assertIn("no verdict here means clear", out["verdict_note"])

    def test_points_inside_the_body_radius_are_reported_as_such(self):
        samples = geom.path_samples_robot([0, 0, 1], [0, 0, 1.05], count=3)
        out = geom.summarize_corridor(self.corridor([2.0, 0.1, 2.0], body=[0, 5, 0]),
                                      samples)
        self.assertEqual(out["verdict"], "sampled_points_within_body_radius")
        self.assertEqual(out["samples_with_points_inside_body_radius"], [1])

    def test_samples_without_returns_are_unknown_not_free(self):
        samples = geom.path_samples_robot([0, 0, 1], [0, 0, 1.05], count=3)
        out = geom.summarize_corridor(
            self.corridor([2.0, 2.0, 2.0], observed=[3, 0, 3]), samples)
        self.assertEqual(out["verdict"],
                         "no_sampled_points_within_body_radius_with_unknown_samples")
        self.assertEqual(out["unknown_sample_indices"], [1])
        self.assertEqual(out["per_sample"][1]["observation"], "unobserved")

    def test_no_cloud_measurement_at_all_is_unknown(self):
        samples = geom.path_samples_robot([0, 0, 1], [0, 0, 1.05], count=3)
        out = geom.summarize_corridor(
            self.corridor([None, None, None], observed=[0, 0, 0]), samples)
        self.assertEqual(out["verdict"], "unknown")
        self.assertIsNone(out["min_sampled_point_distance_m"])

    def test_self_exclusion_region_is_unknown_never_clear(self):
        # The real counterexample: an object sitting inside the 9 cm sphere around
        # the fingertip has its returns deleted before measurement, so those
        # samples report no points nearby. Reading that as free space would let the
        # proxy approve a path straight into the object it just blinded itself to.
        start = [0.0, 0.0, 1.0]
        samples = geom.path_samples_robot(start, [0.0, 0.0, 1.1], count=3)
        out = geom.summarize_corridor(
            self.corridor([9.0, 9.0, 9.0], observed=[0, 0, 0]), samples, start)
        self.assertEqual(out["verdict"], "unknown")
        # Every sample of this short path is inside SELF_RADIUS_M + OBS_RADIUS_M.
        self.assertEqual(out["self_excluded_sample_indices"], [0, 1, 2])
        self.assertEqual(out["unknown_sample_indices"], [0, 1, 2])
        for entry in out["per_sample"]:
            self.assertEqual(entry["observation"], "unknown_self_excluded")
            self.assertTrue(entry["self_exclusion_overlap"])
        self.assertTrue(any("blind spots, not free space" in n for n in out["notes"]))

    def test_self_excluded_samples_never_carry_the_no_obstruction_verdict(self):
        # Far samples are clean, near ones are blinded: the verdict must degrade to
        # "with unknown samples", and the blinded readings must not set the
        # reported distance.
        start = [0.0, 0.0, 1.0]
        samples = geom.path_samples_robot(start, [0.0, 0.0, 1.6], count=5)
        out = geom.summarize_corridor(
            self.corridor([9.0, 9.0, 4.0, 5.0, 6.0]), samples, start)
        self.assertEqual(out["verdict"],
                         "no_sampled_points_within_body_radius_with_unknown_samples")
        self.assertEqual(out["self_excluded_sample_indices"], [0, 1])
        self.assertAlmostEqual(out["min_sampled_point_distance_m"], 0.4)

    def test_a_surviving_return_inside_the_body_radius_is_reported_even_if_blinded(self):
        # Counterexample to dropping these: the mask only ever *removes* points, so
        # a return that survived it is real evidence of something in the corridor.
        # Suppressing it because the sample's neighbourhood overlaps the sphere
        # would turn measured returns into silence at exactly the samples closest
        # to the robot.
        start = [0.0, 0.0, 1.0]
        samples = geom.path_samples_robot(start, [0.0, 0.0, 1.05], count=3)
        out = geom.summarize_corridor(
            self.corridor([0.2, 0.2, 0.2], body=[4, 4, 4]), samples, start)
        self.assertEqual(out["samples_with_points_inside_body_radius"], [0, 1, 2])
        self.assertEqual(out["verdict"], "sampled_points_within_body_radius")

    def test_the_relaxation_is_one_directional_absence_stays_unknown(self):
        # The other half of the same counterexample: with no surviving returns the
        # blinded region must still be unknown. Positive evidence is reportable;
        # masked-away absence is never safety.
        start = [0.0, 0.0, 1.0]
        samples = geom.path_samples_robot(start, [0.0, 0.0, 1.05], count=3)
        out = geom.summarize_corridor(
            self.corridor([9.0, 9.0, 9.0], body=[0, 0, 0], observed=[0, 0, 0]),
            samples, start)
        self.assertEqual(out["samples_with_points_inside_body_radius"], [])
        self.assertEqual(out["verdict"], "unknown")
        self.assertEqual(out["unknown_sample_indices"], [0, 1, 2])
        self.assertTrue(any("still reported as a hit" in n for n in out["notes"]))

    def test_blinding_is_decided_by_observed_distance_to_the_measured_tip(self):
        start = [0.0, 0.0, 1.0]
        far = [0.0, 0.0, 1.0 + geom.SELF_RADIUS_M + geom.OBS_RADIUS_M + 0.01]
        self.assertEqual(geom.self_blinded_samples([far], start), [])
        near = [0.0, 0.0, 1.0 + geom.SELF_RADIUS_M + geom.OBS_RADIUS_M - 0.01]
        self.assertEqual(geom.self_blinded_samples([near], start), [0])
        # Without a measured tip nothing is assumed to be excluded.
        self.assertEqual(geom.self_blinded_samples([near], None), [])

    def test_candidate_is_compared_against_the_measured_end_effector(self):
        measured = geom.measured_end_effector(
            {"fingertip_ui": [0.0, 2.0, 0.0], "gripper_action": [0],
             "ee_dirs_robot": {"approach": [0, 0, -1], "opening": [0, 1, 0]}}, 0.8)
        delta = geom.pose_delta({"position": [0.0, 0.0, 1.1], "approach": [0, 0, -1],
                                 "opening": [0, 1, 0]}, measured)
        self.assertAlmostEqual(delta["distance_m"], 0.1, places=6)
        self.assertAlmostEqual(delta["approach_angle_deg"], 0.0, places=4)

    def test_pose_delta_without_telemetry_reports_a_reason(self):
        delta = geom.pose_delta({"position": [0, 0, 1], "approach": [0, 0, -1],
                                 "opening": [0, 1, 0]},
                                {"status": "unknown", "reason": "no telemetry"})
        self.assertEqual(delta["measured_status"], "unknown")
        self.assertNotIn("translation_m", delta)


class AttachmentTests(unittest.TestCase):
    def measured(self, tip=(0.0, 0.0, 1.0)):
        return {"status": "ok", "fingertip_position": {k: v for k, v in zip("xyz", tip)}}

    def summary(self, center):
        return {"status": "ok", "center": {k: v for k, v in zip("xyz", center)}}

    def test_closed_and_near_is_still_unknown_not_assumed_attachment(self):
        # The real counterexample: a cube resting untouched beside closed jaws
        # produces exactly these two cues. Nothing in a single observation
        # separates it from a held cube, so proximity must not upgrade the status.
        out = geom.attachment_evidence(self.summary((0, 0, 1.01)), self.measured(),
                                       "closed")
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["missing"], [])
        self.assertEqual(len(out["consistent_with"]), 2)
        self.assertIn("co-displacement", out["evidence_required"])
        self.assertNotIn("assumed", json.dumps(out))

    def test_cues_are_reported_even_though_the_status_stays_unknown(self):
        out = geom.attachment_evidence(self.summary((0, 0, 1.01)), self.measured(),
                                       "closed")
        self.assertAlmostEqual(out["reference_to_fingertip_m"], 0.01, places=4)
        self.assertTrue(any("closed" in c for c in out["consistent_with"]))

    def test_open_gripper_is_unknown_and_says_what_is_missing(self):
        out = geom.attachment_evidence(self.summary((0, 0, 1.0)), self.measured(),
                                       "open")
        self.assertEqual(out["status"], "unknown")
        self.assertTrue(any("not closed" in m for m in out["missing"]))

    def test_far_reference_is_unknown_even_when_closed(self):
        out = geom.attachment_evidence(self.summary((0.4, 0, 1.0)), self.measured(),
                                       "closed")
        self.assertEqual(out["status"], "unknown")
        self.assertTrue(any("beyond" in m for m in out["missing"]))

    def test_no_reference_is_unknown(self):
        out = geom.attachment_evidence(None, self.measured(), "closed")
        self.assertEqual(out["status"], "unknown")

    def test_no_input_combination_can_produce_a_non_unknown_status(self):
        for summary in (None, self.summary((0, 0, 1.0)), self.summary((0.4, 0, 1.0))):
            for measured in (self.measured(), {"status": "unknown"}):
                for state in ("closed", "open", "unknown"):
                    self.assertEqual(
                        geom.attachment_evidence(summary, measured, state)["status"],
                        "unknown")

    def test_span_comparison_makes_no_grasp_claim(self):
        summary = {"status": "ok", "center": {"x": 0, "y": 0, "z": 1}}
        points = [[0.0, y * 0.01, 1.0] for y in range(4)]
        out = geom.span_verdict(summary, points, [0.0, 1.0, 0.0])
        self.assertEqual(out["verdict"], "within_jaw_span")
        self.assertIn("does not predict", out["note"])
        wide = [[0.0, y * 0.05, 1.0] for y in range(4)]
        self.assertEqual(geom.span_verdict(summary, wide, [0.0, 1.0, 0.0])["verdict"],
                         "exceeds_jaw_span")

    def test_span_without_valid_geometry_is_unknown(self):
        self.assertEqual(geom.span_verdict(None, [], [0, 1, 0])["status"], "unknown")


class PredictionTests(unittest.TestCase):
    def record(self, attachment_status="unknown"):
        return geom.prediction_record(
            prediction_id="pred1", observation_version_id="e1-s10-c10",
            candidate={"position": [0.0, 0.0, 1.1], "approach": [0, 0, -1],
                       "opening": [0, 1, 0]},
            corridor={"min_sampled_point_distance_m": 0.04,
                      "verdict": "no_sampled_points_within_body_radius"},
            attachment={"status": attachment_status}, reference_id="ref1",
            execute_call_ordinal=0)

    def test_written_prediction_is_not_marked_checked(self):
        record = self.record()
        self.assertFalse(record["checked"])
        self.assertFalse(record["endpoint_checked"])
        self.assertIsNone(record["execution_check"])

    def test_the_prediction_names_the_frame_it_was_evaluated_against(self):
        # The execution check matches this against the frame the execution started
        # from, so a record without it could only be ordered by a local counter.
        self.assertEqual(self.record()["observation_version"], "e1-s10-c10")

    def test_the_ordinal_counts_execute_calls_and_never_completions(self):
        record = self.record()
        self.assertEqual(record["execute_call_ordinal_at_prediction"], 0)
        self.assertNotIn("completed", json.dumps(record))

    def test_prediction_carries_the_sampled_point_distance_not_a_clearance(self):
        record = self.record()
        self.assertAlmostEqual(record["min_sampled_point_distance_m"], 0.04)
        self.assertNotIn("clearance", json.dumps(record))


class RegionComparisonTests(unittest.TestCase):
    """The before/after neighbourhood figure refresh reports instead of an error."""

    def test_the_shift_is_reported_with_the_identity_left_unknown(self):
        out = geom.region_comparison(center_before={"x": 0.0, "y": 0.0, "z": 1.0},
                                     center_after={"x": 0.0, "y": 0.0, "z": 1.06})
        self.assertEqual(out["status"], "measured")
        self.assertEqual(out["identity"], "unknown")
        self.assertAlmostEqual(out["region_center_shift_m"], 0.06, places=4)
        self.assertIn("does not establish attachment", out["note"])
        self.assertIn("same referent", out["note"])

    def test_a_region_figure_is_never_named_as_an_endpoint_error(self):
        out = geom.region_comparison(center_before={"x": 0.0, "y": 0.0, "z": 1.0},
                                     center_after={"x": 0.0, "y": 0.0, "z": 1.06})
        blob_text = json.dumps(out)
        self.assertNotIn("position_error", blob_text)
        self.assertNotIn("fingertip", blob_text)
        self.assertIn("is not an endpoint error", out["note"])

    def test_a_missing_centre_on_either_side_is_not_verifiable(self):
        for before, after in (({"x": 0, "y": 0, "z": 1}, None),
                              (None, {"x": 0, "y": 0, "z": 1}), (None, None)):
            out = geom.region_comparison(center_before=before, center_after=after)
            self.assertEqual(out["status"], "not_verifiable")
            self.assertEqual(out["identity"], "unknown")


class FrameAdvanceTests(unittest.TestCase):
    def test_a_later_seq_in_the_same_epoch_advances(self):
        self.assertIsNone(geom.frame_advance("e1-s4-c4", "e1-s5-c5"))

    def test_the_same_frame_read_twice_is_not_an_advance(self):
        self.assertEqual(geom.frame_advance("e1-s4-c4", "e1-s4-c4"),
                         "not_a_later_frame")
        self.assertEqual(geom.frame_advance("e1-s6-c6", "e1-s5-c5"),
                         "not_a_later_frame")

    def test_a_different_epoch_is_a_different_episode(self):
        self.assertEqual(geom.frame_advance("e1-s4-c4", "e2-s5-c5"),
                         "different_episode_epoch")

    def test_an_unreadable_version_never_counts_as_an_advance(self):
        for before, after in (("e1-s4-c4", None), (None, "e1-s5-c5"),
                              ("e1-s4-c4", "nonsense")):
            self.assertEqual(geom.frame_advance(before, after),
                             "unreadable_observation_version")

    def test_an_epoch_containing_a_dash_still_parses(self):
        parsed = geom.parse_observation_version("ea-b-s7-c7")
        self.assertEqual(parsed, {"epoch": "a-b", "seq": 7, "cloud_seq": 7})


class InputTests(unittest.TestCase):
    def test_non_finite_and_boolean_vectors_are_rejected(self):
        for bad in ([float("nan"), 0, 0], [float("inf"), 0, 0], [True, 0, 0],
                    [0, 0], "abc", None, [1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                geom._finite(bad, "v")

    def test_valid_vector_is_coerced_to_floats(self):
        self.assertEqual(geom._finite([1, 2, 3], "v"), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()

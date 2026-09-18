"""Counterexamples for the fast-geometry core.

Every case here is a way this interface could report a number that looks like a
measurement and is not: an empty close that fits a constant offset, a slipping
object that co-moves a little, a re-read of the same frame passing for new
evidence, another object of a different size or colour inheriting a stored id, an
arc reported as a circle, a stored offset reused across a rotation, and a request
whose later stage is invalid after its earlier stages have already moved the arm.

The point of each assertion is the *refusal*, so the positive cases are here too:
a gate that refuses everything is not evidence of anything either.
"""

import math
import unittest

from spatial_interface import fast_geometry as fg


# ── synthetic geometry ───────────────────────────────────────────────────────

def ring(center=(0.0, 0.0, 1.0), radius=0.06, height=0.05, n=240,
         arc_deg=360.0, tilt_deg=0.0):
    """Returns from an upright open cylinder: a bowl rim over its wall."""
    cx, cy, cz = center
    pts = []
    for i in range(n):
        a = math.radians(arc_deg) * i / n
        for k in range(6):
            z = cz - height * k / 5.0
            x, y = radius * math.cos(a), radius * math.sin(a)
            if tilt_deg:
                t = math.radians(tilt_deg)
                dz = x * math.sin(t)
                x = x * math.cos(t)
                z = z + dz
            pts.append([cx + x, cy + y, z])
    return pts


def table(center=(0.0, 0.0, 0.9), half=0.25, n=26):
    """A flat support patch under everything, so split_support has a table."""
    cx, cy, cz = center
    return [[cx - half + 2 * half * i / (n - 1), cy - half + 2 * half * j / (n - 1), cz]
            for i in range(n) for j in range(n)]


def disc(center=(0.2, 0.0, 0.92), radius=0.09, n=400):
    """A flat round top, e.g. a plate."""
    cx, cy, cz = center
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        for f in (1.0, 0.94, 0.88):
            pts.append([cx + radius * f * math.cos(a), cy + radius * f * math.sin(a), cz])
    return pts


def white(n):
    return [[0.9, 0.9, 0.9]] * n


def red(n):
    return [[0.85, 0.12, 0.12]] * n


def card_for(points, shape="ring", *, colors=None, name="bowl", support=()):
    """Build a card the way fg_bind does, with colour following the points."""
    split = fg.split_support(list(points) + list(support))
    clusters = fg.cluster_points(split["object"])
    keep = clusters[0] if clusters else []
    return fg.build_card(
        name=name, shape=shape, points=keep, pixels=[],
        region_px={"kind": "box", "x0": -1e9, "y0": -1e9, "x1": 1e9, "y1": 1e9},
        support=split["support"], clusters=clusters, from_frame="e1-s2-c2",
        surface="canvas",
        colors=(colors or white(len(keep)))[:len(keep)],
        support_colors=white(len(split["support"])))


def observation(frame, obj, tip, *, state="closed", turn_deg=0.0, predicted=False):
    """One attachment observation with measured axes, optionally rotated."""
    t = math.radians(turn_deg)
    entry = {"frame": frame, "object_center": list(obj), "fingertip": list(tip),
             "gripper_state_class": state,
             "approach": [0.0, 0.0, -1.0],
             "opening": [math.cos(t), math.sin(t), 0.0]}
    if predicted:
        entry["object_center_is_predicted"] = True
    return entry


# ── fitting: what a card may and may not claim ───────────────────────────────

class CardFitTests(unittest.TestCase):
    def test_a_full_rim_fits_and_offers_grasp_candidates(self):
        card = card_for(ring(), support=table())
        self.assertTrue(card["valid"], card["reasons"])
        self.assertAlmostEqual(card["radius_m"], 0.06, places=2)
        # The rim centre is the average over the upper band, so it sits a little
        # below the topmost return by construction, not by error.
        self.assertLessEqual(1.0 - card["center"]["z"], 0.01)
        self.assertLessEqual(card["center"]["z"], 1.0)
        self.assertEqual(len(card["grasp_candidates"]), 4)
        for cand in card["grasp_candidates"]:
            radial = math.hypot(cand["point"]["x"] - card["center"]["x"],
                                cand["point"]["y"] - card["center"]["y"])
            self.assertAlmostEqual(radial, card["radius_m"], places=3)

    def test_an_arc_is_refused_because_its_centre_is_an_extrapolation(self):
        card = card_for(ring(arc_deg=150.0), support=table())
        self.assertFalse(card["valid"])
        self.assertTrue(any("bearing" in r for r in card["reasons"]), card["reasons"])

    def test_a_tilted_rim_is_refused_rather_than_projected(self):
        card = card_for(ring(tilt_deg=40.0), support=table())
        self.assertFalse(card["valid"])
        self.assertTrue(any("tilted" in r for r in card["reasons"]), card["reasons"])
        self.assertIs(card["fit"]["upright_assumption"]["applicable"], False)

    def test_upright_is_an_assumption_not_a_fitted_normal(self):
        card = card_for(ring(), support=table())
        self.assertIn("assumed_world_up", card["up_source"])

    def test_too_few_returns_is_unknown_not_a_small_object(self):
        card = card_for(ring(n=4), support=table())
        self.assertFalse(card["valid"])
        self.assertTrue(any(str(fg.MIN_OBJECT_POINTS) in r for r in card["reasons"]))

    def test_two_comparable_clusters_in_one_region_are_ambiguous(self):
        both = ring() + ring(center=(0.25, 0.0, 1.0))
        card = card_for(both, support=table())
        self.assertFalse(card["valid"])
        self.assertTrue(any("ambiguous" in r for r in card["reasons"]), card["reasons"])

    def test_a_tabletop_plane_patch_fits_from_the_support_band(self):
        # The regression this guards: the table lands entirely in the support band,
        # so `object` is empty and the object-count reason described another shape.
        card = card_for([], shape="plane_patch", name="table", support=table())
        self.assertTrue(card["valid"], card["reasons"])
        self.assertEqual(card["fit"]["fitted_from"], "support_band")

    def test_a_blob_claims_a_centroid_and_no_pose(self):
        card = card_for(ring(), shape="blob", support=table())
        self.assertTrue(card["valid"], card["reasons"])
        self.assertIsNone(card["radius_m"])
        self.assertEqual(card["grasp_candidates"], [])

    def test_split_support_ignores_a_single_low_outlier(self):
        split = fg.split_support(table() + ring() + [[0.0, 0.0, 0.5]])
        self.assertAlmostEqual(split["support_z"], 0.9, places=2)

    def test_a_region_that_is_all_one_flat_band_reports_no_object(self):
        split = fg.split_support(table())
        self.assertEqual(split["object"], [])
        self.assertEqual(split["support_fraction"], 1.0)

    def test_clustering_keeps_a_bowl_and_an_adjacent_plate_apart(self):
        clusters = fg.cluster_points(ring() + disc(center=(0.3, 0.0, 1.0)))
        self.assertGreaterEqual(len(clusters), 2)


# ── appearance and size correspondence ───────────────────────────────────────

class CorrespondenceTests(unittest.TestCase):
    def test_a_colourless_cloud_yields_no_signature(self):
        self.assertIsNone(fg.color_signature([]))
        self.assertIsNone(fg.color_signature([[-1, -1, -1]] * 200))
        self.assertIsNone(fg.color_signature(white(fg.MIN_COLOR_POINTS - 1)))

    def test_missing_colour_is_unavailable_not_matched(self):
        self.assertEqual(fg.color_match(None, fg.color_signature(white(50)))["status"],
                         "unavailable")

    def test_a_red_bowl_does_not_match_a_white_plate(self):
        out = fg.color_match(fg.color_signature(red(80)),
                             fg.color_signature(white(80)))
        self.assertEqual(out["status"], "mismatch")

    def test_the_same_colour_under_shading_still_matches(self):
        shaded = [[0.86, 0.86, 0.86]] * 40 + [[0.94, 0.94, 0.94]] * 40
        out = fg.color_match(fg.color_signature(white(80)),
                             fg.color_signature(shaded))
        self.assertEqual(out["status"], "matched")

    def test_a_differently_sized_circle_is_a_size_mismatch(self):
        stored = card_for(ring(radius=0.06), support=table())
        other = card_for(ring(radius=0.11), support=table())
        self.assertEqual(fg.size_match(stored, other)["status"], "mismatch")

    def test_size_tolerance_is_relative_so_a_large_plate_survives_noise(self):
        stored = card_for(disc(radius=0.09), shape="disc", support=table())
        jittered = card_for(disc(radius=0.093), shape="disc", support=table())
        self.assertEqual(fg.size_match(stored, jittered)["status"], "matched")


class ReassociationTests(unittest.TestCase):
    def moved(self, stored, shift, **kw):
        pts = ring(center=(shift, 0.0, 1.0), **kw)
        colors = kw.pop("colors", None) or white(len(pts))
        return fg.reassociate(stored, pts + table(), from_frame="e1-s6-c6",
                              window_colors=colors + white(len(table())))

    def setUp(self):
        self.stored = card_for(ring(), support=table())
        self.stored["valid"] = True

    def test_the_same_object_a_little_moved_is_matched(self):
        out = self.moved(self.stored, 0.02)
        self.assertEqual(out["status"], "matched", out.get("reason"))
        self.assertLess(out["center_shift_m"], 0.03)
        self.assertEqual(out["appearance_match"]["status"], "matched")

    def test_a_shift_beyond_the_window_is_unknown_not_tracked(self):
        out = self.moved(self.stored, 0.40)
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["reason"], "shift_exceeds_window")

    def test_a_differently_sized_circle_nearby_does_not_inherit_the_id(self):
        pts = ring(radius=0.11) + table()
        out = fg.reassociate(self.stored, pts, from_frame="e1-s6-c6",
                             window_colors=white(len(pts)))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["reason"], "size_mismatch")

    def test_a_differently_coloured_object_does_not_inherit_the_id(self):
        pts = ring()
        out = fg.reassociate(self.stored, pts + table(), from_frame="e1-s6-c6",
                             window_colors=red(len(pts)) + white(len(table())))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["reason"], "appearance_mismatch")

    def test_two_rival_clusters_are_ambiguous_not_a_pick(self):
        pts = ring() + ring(center=(0.14, 0.0, 1.0)) + table()
        out = fg.reassociate(self.stored, pts, from_frame="e1-s6-c6",
                             window_colors=white(len(pts)))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["reason"], "ambiguous_two_candidates")

    def test_no_colour_and_no_decisive_geometry_refuses(self):
        # Colourless returns, and the shift is over half the window: the geometry
        # alone must not read as a match.
        stored = dict(self.stored)
        out = fg.reassociate(stored, ring(center=(0.10, 0.0, 1.0)) + table(),
                             from_frame="e1-s6-c6")
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["reason"], "no_correspondence_evidence")

    def test_a_thin_window_is_insufficient_support(self):
        out = fg.reassociate(self.stored, ring(n=3), from_frame="e1-s6-c6")
        self.assertEqual(out["reason"], "insufficient_support")

    def test_colour_must_be_parallel_to_the_points(self):
        with self.assertRaises(fg.Rejected):
            fg.reassociate(self.stored, ring(), from_frame="e1-s6-c6",
                           window_colors=white(3))

    def test_the_prediction_is_reported_beside_the_measurement(self):
        out = fg.reassociate(self.stored, ring(center=(0.02, 0, 1.0)) + table(),
                             from_frame="e1-s6-c6",
                             expected_shift=[0.02, 0.0, 1.0],
                             window_colors=white(len(ring()) + len(table())))
        self.assertEqual(out["status"], "matched")
        self.assertIsNotNone(out["center"])
        self.assertIsNotNone(out["prediction_error_m"])
        self.assertIn("not an observation", out["predicted_center_note"])


# ── attachment: the counterexamples that motivated the gates ─────────────────

class AttachmentTests(unittest.TestCase):
    def test_an_empty_close_beside_a_stationary_object_is_refused(self):
        # The documented counterexample: offsets (0,0,0) and (0,-0.02,0) have an
        # RMS spread of exactly 0.01 m, inside the 0.015 m offset gate. Only the
        # object-motion gates catch it.
        out = fg.fit_attachment([
            observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.0)),
            observation("e1-s4-c4", (0, 0, 1.0), (0, 0.03, 1.0))])
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["reason_code"], "object_did_not_move")

    def test_a_carried_object_is_accepted(self):
        out = fg.fit_attachment([
            observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02)),
            observation("e1-s4-c4", (0, 0.05, 1.0), (0, 0.05, 1.02))])
        self.assertEqual(out["status"], "attached", out.get("reason"))
        self.assertAlmostEqual(out["offset_m"]["z"], -0.02, places=3)
        self.assertIn("offset_local_m", out["measured_under"])

    def test_a_slipping_object_that_moves_a_fraction_is_refused(self):
        out = fg.fit_attachment([
            observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02)),
            observation("e1-s4-c4", (0, 0.015, 1.0), (0, 0.10, 1.02))])
        self.assertEqual(out["status"], "unknown")
        self.assertIn(out["reason_code"], ("not_co_moving", "displacement_too_small"))

    def test_an_unmoving_gripper_cannot_excite_the_test(self):
        out = fg.fit_attachment([
            observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02)),
            observation("e1-s4-c4", (0, 0.002, 1.0), (0, 0.002, 1.02))])
        self.assertEqual(out["reason_code"], "insufficient_excitation")

    def test_one_frame_twice_is_not_two_observations(self):
        same = observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02))
        out = fg.fit_attachment([same, dict(same, object_center=[0, 0.05, 1.0],
                                            fingertip=[0, 0.05, 1.02])])
        self.assertEqual(out["reason_code"], "insufficient_observations")

    def test_a_predicted_centre_cannot_justify_the_offset_that_produced_it(self):
        out = fg.fit_attachment([
            observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02)),
            observation("e1-s4-c4", (0, 0.05, 1.0), (0, 0.05, 1.02),
                        predicted=True)])
        self.assertEqual(out["reason_code"], "predicted_position_supplied")

    def test_an_open_gripper_cannot_be_carrying_anything(self):
        out = fg.fit_attachment([
            observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02), state="open"),
            observation("e1-s4-c4", (0, 0.05, 1.0), (0, 0.05, 1.02), state="open")])
        self.assertEqual(out["reason_code"], "gripper_not_closed")

    def test_a_missing_gripper_state_is_refused_not_skipped(self):
        # The fix this suite was written for: co-displacement with no gripper state
        # used to be accepted, so telemetry that failed to report read as consent.
        a = observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02))
        b = observation("e1-s4-c4", (0, 0.05, 1.0), (0, 0.05, 1.02))
        del a["gripper_state_class"]
        out = fg.fit_attachment([a, b])
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["reason_code"], "gripper_state_unknown")

    def test_an_unknown_gripper_state_class_is_refused_too(self):
        out = fg.fit_attachment([
            observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02), state="unknown"),
            observation("e1-s4-c4", (0, 0.05, 1.0), (0, 0.05, 1.02))])
        self.assertEqual(out["reason_code"], "gripper_state_unknown")

    def test_an_unmeasurable_orientation_is_refused(self):
        a = observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02))
        b = observation("e1-s4-c4", (0, 0.05, 1.0), (0, 0.05, 1.02))
        del a["approach"], a["opening"]
        out = fg.fit_attachment([a, b])
        self.assertEqual(out["reason_code"], "gripper_orientation_unknown")

    def test_a_rotating_grasp_cannot_share_one_world_offset(self):
        out = fg.fit_attachment([
            observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02)),
            observation("e1-s4-c4", (0, 0.05, 1.0), (0, 0.05, 1.02),
                        turn_deg=45.0)])
        self.assertEqual(out["reason_code"], "orientation_changed")


class OffsetOrientationTests(unittest.TestCase):
    def attached(self):
        return fg.fit_attachment([
            observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02)),
            observation("e1-s4-c4", (0, 0.05, 1.0), (0, 0.05, 1.02))])

    def measured(self, turn_deg=0.0, axes=True):
        t = math.radians(turn_deg)
        out = {"status": "ok",
               "fingertip_position": {"x": 0.0, "y": 0.0, "z": 1.02},
               "gripper_state_class": "closed"}
        if axes:
            out["approach"] = {"x": 0.0, "y": 0.0, "z": -1.0}
            out["opening"] = {"x": math.cos(t), "y": math.sin(t), "z": 0.0}
        return out

    def test_within_tolerance_the_measured_offset_applies_as_is(self):
        offset, detail = fg.offset_in_current_orientation(self.attached(),
                                                          self.measured(5.0))
        self.assertEqual(detail["mapping"], "world_offset_reused")
        self.assertAlmostEqual(offset[2], -0.02, places=3)

    def test_beyond_tolerance_the_offset_is_rotated_through_measured_axes(self):
        offset, detail = fg.offset_in_current_orientation(self.attached(),
                                                          self.measured(90.0))
        self.assertEqual(detail["mapping"], "rotated_through_measured_gripper_axes")
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in offset)), 0.02, places=3)

    def test_an_unmeasured_orientation_refuses_rather_than_reusing_the_vector(self):
        offset, detail = fg.offset_in_current_orientation(
            self.attached(), self.measured(axes=False))
        self.assertIsNone(offset)
        self.assertEqual(detail["mapping"], "refused")
        self.assertEqual(detail["missing"], "current_gripper_axes")

    def test_a_stored_offset_that_stops_predicting_is_reported_lost(self):
        out = fg.attachment_still_holds(self.attached(),
                                        object_center=[0.0, 0.20, 1.0],
                                        fingertip=[0.0, 0.0, 1.02],
                                        measured=self.measured())
        self.assertEqual(out["status"], "lost")

    def test_a_holding_offset_still_predicts_the_object(self):
        out = fg.attachment_still_holds(self.attached(),
                                        object_center=[0.0, 0.0, 1.0],
                                        fingertip=[0.0, 0.0, 1.02],
                                        measured=self.measured())
        self.assertEqual(out["status"], "attached")


# ── request validation: nothing runs unless all of it passes ─────────────────

class ValidateRunTests(unittest.TestCase):
    def test_a_gripper_stage_alone_is_a_real_action(self):
        out = fg.validate_run({"stages": [{"gripper": "close"}]})
        self.assertEqual(out["stages"], [{"kind": "gripper", "action": "close"}])
        self.assertTrue(out["physical"])

    def test_a_run_of_checks_alone_is_not_physical(self):
        out = fg.validate_run({"stages": [{"check": {"refs": ["bowl#1"]}}]})
        self.assertFalse(out["physical"])

    def test_a_later_invalid_stage_rejects_the_whole_request(self):
        with self.assertRaises(fg.Rejected):
            fg.validate_run({"stages": [
                {"move": {"relation": "above", "ref": "plate#2"}},
                {"gripper": "close"},
                {"move": {"relation": "descend_to", "ref": "plate#2",
                          "height_m": 9.0}}]})

    def test_a_stage_with_two_keys_is_rejected(self):
        with self.assertRaises(fg.Rejected):
            fg.validate_run({"stages": [{"gripper": "close",
                                         "move": {"relation": "retreat"}}]})

    def test_an_unknown_relation_is_rejected(self):
        with self.assertRaises(fg.Rejected):
            fg.validate_run({"stages": [{"move": {"relation": "grab_it"}}]})

    def test_a_relation_needing_a_reference_is_rejected_without_one(self):
        with self.assertRaises(fg.Rejected):
            fg.validate_run({"stages": [{"move": {"relation": "align_over"}}]})

    def test_retreat_has_nothing_to_refine_against(self):
        with self.assertRaises(fg.Rejected):
            fg.validate_run({"stages": [{"move": {"relation": "retreat",
                                                  "refine": True}}]})

    def test_budgets_outside_the_caps_are_rejected(self):
        for budget in ({"waypoints": 0}, {"waypoints": fg.MAX_WAYPOINT_BUDGET + 1},
                       {"waypoints": 2.5}, {"seconds": 1.0},
                       {"seconds": fg.MAX_DEADLINE_S + 1}, {"minutes": 2}):
            with self.assertRaises(fg.Rejected, msg=budget):
                fg.validate_run({"stages": [{"gripper": "close"}], "budget": budget})

    def test_a_boolean_waypoint_budget_is_not_an_integer(self):
        with self.assertRaises(fg.Rejected):
            fg.validate_run({"stages": [{"gripper": "close"}],
                             "budget": {"waypoints": True}})

    def test_too_many_stages_are_rejected(self):
        with self.assertRaises(fg.Rejected):
            fg.validate_run({"stages": [{"gripper": "close"}] * (fg.MAX_STAGES + 1)})

    def test_defaults_are_applied_and_reported(self):
        out = fg.validate_run({"stages": [{"move": {"relation": "above",
                                                    "ref": "plate#2"}}]})
        self.assertEqual(out["budget"], {"waypoints": fg.DEFAULT_WAYPOINT_BUDGET,
                                         "seconds": fg.DEFAULT_DEADLINE_S})
        self.assertEqual(out["stages"][0]["tolerance_m"], fg.DEFAULT_TOLERANCE_M)


# ── relations: solved from measurement, refused without it ───────────────────

class SolveRelationTests(unittest.TestCase):
    def setUp(self):
        self.card = card_for(ring(), support=table())
        self.plate = card_for(disc(), shape="disc", support=table())
        self.measured = {"status": "ok",
                         "fingertip_position": {"x": 0.0, "y": 0.0, "z": 1.20},
                         "approach": {"x": 0.0, "y": 0.0, "z": -1.0},
                         "opening": {"x": 1.0, "y": 0.0, "z": 0.0},
                         "gripper_state_class": "closed"}
        self.attachment = fg.fit_attachment([
            observation("e1-s2-c2", (0, 0, 1.0), (0, 0, 1.02)),
            observation("e1-s4-c4", (0, 0.05, 1.0), (0, 0.05, 1.02))])

    def move(self, **kw):
        return fg.validate_run({"stages": [{"move": kw}]})["stages"][0]

    def test_approach_grasp_stands_off_along_the_candidates_own_axis(self):
        out = fg.solve_relation(self.move(relation="approach_grasp", ref="bowl#1",
                                          grasp="rim_x+", standoff_m=0.04),
                                card=self.card, measured=self.measured)
        self.assertEqual(out["status"], "ok")
        goal = out["goal_point"]
        self.assertAlmostEqual(out["target"]["position"][2] - goal[2], 0.04, places=4)

    def test_an_unknown_grasp_candidate_is_refused_with_the_available_ones(self):
        out = fg.solve_relation(self.move(relation="approach_grasp", ref="bowl#1",
                                          grasp="handle"),
                                card=self.card, measured=self.measured)
        self.assertEqual(out["status"], "refused")
        self.assertIn("rim_x+", out["available"])

    def test_an_invalid_reference_cannot_be_solved_against(self):
        bad = dict(self.card, valid=False, reasons=["arc"])
        out = fg.solve_relation(self.move(relation="above", ref="bowl#1"),
                                card=bad, measured=self.measured)
        self.assertEqual(out["status"], "refused")

    def test_no_measured_end_effector_refuses_every_relation(self):
        out = fg.solve_relation(self.move(relation="retreat"), card=None,
                                measured={"status": "unknown"})
        self.assertEqual(out["status"], "refused")

    def test_placement_without_a_measured_offset_is_refused(self):
        out = fg.solve_relation(self.move(relation="align_over", ref="plate#2"),
                                card=self.plate, measured=self.measured)
        self.assertEqual(out["status"], "refused")
        self.assertIn("calibrate_attachment", out["reason"])

    def test_placement_targets_the_object_not_the_fingertip(self):
        out = fg.solve_relation(self.move(relation="align_over", ref="plate#2",
                                          height_m=0.06),
                                card=self.plate, measured=self.measured,
                                attachment=self.attachment)
        self.assertEqual(out["status"], "ok")
        self.assertAlmostEqual(out["object_goal"][2],
                               self.plate["top_z"] + 0.06, places=4)
        # The fingertip target is the object goal MINUS the measured offset.
        self.assertAlmostEqual(out["target"]["position"][2],
                               out["object_goal"][2] + 0.02, places=3)

    def test_descend_to_lands_the_held_objects_observed_base_not_its_midpoint(self):
        # A tall bowl, so the two candidate definitions are far apart: its fitted
        # centre is the average height of the RIM, while half its height is 2.5 cm.
        # Using height/2 dropped the bowl by most of its own depth.
        held = card_for(ring(height=0.05), support=table())
        separation = held["center"]["z"] - held["base_center"]["z"]
        self.assertGreater(separation, 0.04,
                           "the fixture must be a case where centre-to-base and "
                           "half-height genuinely disagree")
        clearance = 0.01
        out = fg.solve_relation(self.move(relation="descend_to", ref="plate#2",
                                          height_m=clearance),
                                card=self.plate, measured=self.measured,
                                attachment=self.attachment, held_card=held)
        self.assertEqual(out["status"], "ok", out.get("reason"))
        self.assertAlmostEqual(out["held_base_below_center_m"], separation, places=4)
        # The thing the model asked for: the object's own observed base ends at the
        # target's top plus the clearance it requested.
        object_base = out["object_goal"][2] - separation
        self.assertAlmostEqual(object_base, self.plate["top_z"] + clearance, places=4)
        self.assertNotAlmostEqual(out["object_goal"][2],
                                  self.plate["top_z"] + clearance
                                  + held["height_m"] / 2.0, places=3)

    def test_descend_to_is_refused_when_the_held_object_is_not_measured(self):
        out = fg.solve_relation(self.move(relation="descend_to", ref="plate#2"),
                                card=self.plate, measured=self.measured,
                                attachment=self.attachment)
        self.assertEqual(out["status"], "refused")
        self.assertIn("base", out["reason"])
        self.assertIn("align_over", out["held_ref_note"])

    def test_descend_to_is_refused_when_the_card_carries_no_base(self):
        held = dict(card_for(ring(), support=table()), base_center=None)
        out = fg.solve_relation(self.move(relation="descend_to", ref="plate#2"),
                                card=self.plate, measured=self.measured,
                                attachment=self.attachment, held_card=held)
        self.assertEqual(out["status"], "refused")

    def test_descend_to_is_refused_once_the_grasp_has_rotated(self):
        # The vertical separation is taken under the upright prior; after a rotation
        # the object's base is not below its centre and the scalar is meaningless.
        turned = {k: v for k, v in self.measured.items()}
        turned["opening"] = {"x": 0.0, "y": 1.0, "z": 0.0}
        out = fg.solve_relation(self.move(relation="descend_to", ref="plate#2"),
                                card=self.plate, measured=turned,
                                attachment=self.attachment,
                                held_card=card_for(ring(), support=table()))
        self.assertEqual(out["status"], "refused")
        self.assertIn("rotated", out["reason"])

    def test_align_over_positions_the_centre_and_needs_no_base(self):
        held = dict(card_for(ring(), support=table()), base_center=None)
        out = fg.solve_relation(self.move(relation="align_over", ref="plate#2",
                                          height_m=0.06),
                                card=self.plate, measured=self.measured,
                                attachment=self.attachment, held_card=held)
        self.assertEqual(out["status"], "ok")
        self.assertNotIn("held_base_below_center_m", out)

    def test_placement_refuses_when_the_rotation_is_unmeasurable(self):
        blind = {k: v for k, v in self.measured.items()
                 if k not in ("approach", "opening")}
        out = fg.solve_relation(self.move(relation="align_over", ref="plate#2"),
                                card=self.plate, measured=blind,
                                attachment=self.attachment)
        self.assertEqual(out["status"], "refused")

    def test_retreat_backs_off_along_the_measured_approach_axis(self):
        out = fg.solve_relation(self.move(relation="retreat", distance_m=0.05),
                                card=None, measured=self.measured)
        self.assertAlmostEqual(out["target"]["position"][2], 1.25, places=4)


class RelationResidualTests(unittest.TestCase):
    def test_the_relation_error_is_measured_from_the_objects_own_centre(self):
        out = fg.relation_residual({"object_goal": [0.2, 0.0, 1.0]},
                                   object_center={"x": 0.21, "y": 0.0, "z": 1.0})
        self.assertEqual(out["status"], "measured")
        self.assertAlmostEqual(out["error_m"], 0.01, places=4)
        self.assertAlmostEqual(out["horizontal_error_m"], 0.01, places=4)

    def test_no_re_measured_geometry_is_unknown_not_zero_error(self):
        out = fg.relation_residual({"object_goal": [0.2, 0.0, 1.0]},
                                   object_center=None)
        self.assertEqual(out["status"], "unknown")


# ── virtual edits vs physical waypoints ──────────────────────────────────────

class MotionPlanTests(unittest.TestCase):
    def test_a_twenty_centimetre_reach_is_one_waypoint_and_several_edits(self):
        plan = fg.plan_motion([0, 0, 1.0], [0.2, 0, 1.0])
        self.assertEqual(plan["physical_waypoints"], 1)
        self.assertEqual(plan["edits"], 3)
        self.assertIsNone(plan["why_segmented"])

    def test_a_short_reach_is_one_edit_and_one_waypoint(self):
        plan = fg.plan_motion([0, 0, 1.0], [0.05, 0, 1.0])
        self.assertEqual((plan["physical_waypoints"], plan["edits"]), (1, 1))

    def test_a_reach_beyond_the_segment_is_split_and_says_why(self):
        plan = fg.plan_motion([0, 0, 1.0], [0.6, 0, 1.0])
        self.assertEqual(plan["physical_waypoints"], 3)
        self.assertIn("re-measure", plan["why_segmented"])

    def test_each_segment_ends_exactly_on_its_endpoint(self):
        plan = fg.plan_motion([0, 0, 1.0], [0.6, 0, 1.0])
        self.assertEqual([round(v, 6) for v in plan["segments"][-1]["target"]],
                         [0.6, 0.0, 1.0])

    def test_edit_steps_never_exceed_the_editor_cap(self):
        steps = fg.edit_steps([0, 0, 1.0], [0.2, 0, 1.0])
        at = [0, 0, 1.0]
        for step in steps:
            self.assertLessEqual(
                math.sqrt(sum((a - b) ** 2 for a, b in zip(step, at))),
                fg.MAX_EDIT_STEP_M + 1e-9)
            at = step

    def test_a_large_turn_is_walked_in_several_orientation_edits(self):
        steps = fg.orientation_steps(
            {"approach": [0, 0, -1], "opening": [1, 0, 0]},
            {"approach": [0, 0, -1], "opening": [-1, 0, 0]})
        self.assertGreaterEqual(len(steps), 3)
        for step in steps:
            self.assertAlmostEqual(sum(a * b for a, b in zip(step["approach"],
                                                             step["opening"])),
                                   0.0, places=6)
        self.assertAlmostEqual(steps[-1]["opening"][0], -1.0, places=6)

    def test_an_unchanged_orientation_costs_no_edit(self):
        same = {"approach": [0, 0, -1], "opening": [1, 0, 0]}
        self.assertEqual(fg.orientation_steps(same, same), [])

    def test_an_unusable_orientation_pair_is_refused(self):
        self.assertIsNone(fg.orientation_steps(
            {"approach": [0, 0, -1], "opening": [0, 0, -1]},
            {"approach": [0, 0, -1], "opening": [1, 0, 0]}))

    def test_progress_needs_a_real_improvement(self):
        self.assertTrue(fg.progress_made(0.10, 0.05))
        self.assertFalse(fg.progress_made(0.10, 0.095))
        self.assertIsNone(fg.progress_made(None, 0.05))


# ── regions ──────────────────────────────────────────────────────────────────

class RegionTests(unittest.TestCase):
    def test_fractions_outside_the_surface_are_rejected(self):
        for region in ({"kind": "box", "u0": -0.1, "v0": 0, "u1": 0.5, "v1": 0.5},
                       {"kind": "box", "u0": 0, "v0": 0, "u1": 1.5, "v1": 0.5},
                       {"kind": "point", "u": 1.2, "v": 0.5}):
            with self.assertRaises(fg.Rejected, msg=region):
                fg.validate_region(region)

    def test_a_degenerate_box_is_rejected(self):
        with self.assertRaises(fg.Rejected):
            fg.validate_region({"kind": "box", "u0": 0.5, "v0": 0.5,
                                "u1": 0.5, "v1": 0.9})

    def test_a_polygon_needs_three_to_twelve_vertices(self):
        with self.assertRaises(fg.Rejected):
            fg.validate_region({"kind": "polygon", "points": [[0.1, 0.1], [0.2, 0.2]]})

    def test_a_region_resolves_against_the_live_viewport(self):
        px = fg.region_to_pixels(
            fg.validate_region({"kind": "box", "u0": 0.25, "v0": 0.5,
                                "u1": 0.75, "v1": 0.9}),
            {"left": 10, "top": 20, "width": 400, "height": 200})
        self.assertEqual((px["x0"], px["y0"], px["x1"], px["y1"]),
                         (110.0, 120.0, 310.0, 200.0))

    def test_duplicate_object_names_in_one_bind_are_rejected(self):
        with self.assertRaises(fg.Rejected):
            fg.validate_bind({"objects": [
                {"name": "bowl", "region": {"kind": "box", "u0": 0.1, "v0": 0.1,
                                            "u1": 0.3, "v1": 0.3}},
                {"name": "bowl", "region": {"kind": "box", "u0": 0.5, "v0": 0.1,
                                            "u1": 0.7, "v1": 0.3}}]})

    def test_an_unknown_surface_is_rejected(self):
        with self.assertRaises(fg.Rejected):
            fg.validate_bind({"surface": "overhead", "objects": [
                {"name": "bowl", "region": {"kind": "point", "u": 0.5, "v": 0.5}}]})


if __name__ == "__main__":
    unittest.main()

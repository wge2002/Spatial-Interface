"""Targeted checks for the coarse-to-fine policy layer.

Two things are worth a test here and nothing else is: the rotation splitter this
interface had to fix, and the fail-closed program validation. Both are pure, so
neither needs a browser or a simulator.
"""

import inspect
import random
import unittest
from unittest.mock import patch

from spatial_interface import coarse_fine_policy as cfp
from spatial_interface import fast_geometry as fg
from spatial_interface.target_edit import MAX_ROTATION_DEG


def random_basis(rng):
    while True:
        basis = cfp.basis_from_axes([rng.gauss(0, 1) for _ in range(3)],
                                    [rng.gauss(0, 1) for _ in range(3)])
        if basis is not None:
            return basis


def axes(basis):
    return basis["approach"], basis["opening"]


class RotationSplitterTests(unittest.TestCase):
    """§5 of the handoff: the fast splitter can emit a step the editor refuses.

    ``fast_geometry.orientation_steps`` interpolates approach and opening
    independently and re-orthogonalizes each intermediate pair. The projection
    moves the pose off the interpolation, so the angle actually measured between
    consecutive steps is not the requested fraction — and near 180 deg the two
    nearly-cancelling vectors make it much larger. ``prepare_edit`` then rejects
    the step mid-program, after earlier waypoints have already been executed.
    """

    def test_no_emitted_step_exceeds_the_editor_cap_at_100_to_180_deg(self):
        rng = random.Random(0)
        cases = 0
        worst = 0.0
        while cases < 400:
            start, want = random_basis(rng), random_basis(rng)
            total = cfp.rotation_between(axes(start), axes(want))[1]
            if total < 100.0:
                continue
            cases += 1
            steps = cfp.orientation_steps(axes(start), axes(want))
            self.assertIsNotNone(steps, f"{total:.1f} deg turn was refused outright")
            angles = cfp.step_angles_deg(axes(start), steps)
            self.assertNotIn(None, angles)
            worst = max(worst, max(angles))
            for i, angle in enumerate(angles):
                self.assertLessEqual(
                    angle, MAX_ROTATION_DEG,
                    f"step {i} of a {total:.1f} deg turn measures {angle:.2f} deg, "
                    f"which prepare_edit would refuse")
        self.assertLess(worst, MAX_ROTATION_DEG)

    def test_every_step_is_an_equal_share_of_the_total(self):
        rng = random.Random(7)
        for _ in range(200):
            start, want = random_basis(rng), random_basis(rng)
            total = cfp.rotation_between(axes(start), axes(want))[1]
            if total < 100.0:
                continue
            steps = cfp.orientation_steps(axes(start), axes(want))
            angles = cfp.step_angles_deg(axes(start), steps)
            share = total / len(steps)
            for angle in angles:
                self.assertAlmostEqual(angle, share, places=5)

    def test_the_last_step_lands_exactly_on_the_requested_axes(self):
        rng = random.Random(11)
        for _ in range(200):
            start, want = random_basis(rng), random_basis(rng)
            steps = cfp.orientation_steps(axes(start), axes(want))
            if not steps:
                continue
            last = (steps[-1]["approach"], steps[-1]["opening"])
            # The last step IS the requested axes; the residual is acos's own
            # conditioning near zero angle (a trace error of 1e-16 shows up as
            # ~1e-6 deg), not a difference in the pose that gets applied.
            self.assertLess(cfp.rotation_between(last, axes(want))[1], 1e-4)

    def test_the_old_splitter_is_what_this_replaces(self):
        # Not a test of correctness elsewhere — evidence that the fix is needed.
        # If this ever stops finding an over-cap step, the fast splitter was fixed
        # too and this whole class can be reconsidered.
        rng = random.Random(3)
        over_cap = 0
        for _ in range(4000):
            start, want = random_basis(rng), random_basis(rng)
            total = cfp.rotation_between(axes(start), axes(want))[1]
            if total < 100.0:
                continue
            old = fg.orientation_steps({"approach": start["approach"],
                                        "opening": start["opening"]},
                                       {"approach": want["approach"],
                                        "opening": want["opening"]})
            if old is None:
                continue
            angles = cfp.step_angles_deg(axes(start), old)
            if any(a is not None and a > MAX_ROTATION_DEG for a in angles):
                over_cap += 1
        self.assertGreater(over_cap, 0,
                           "the defect this splitter exists to fix was not "
                           "reproduced, so the replacement is unjustified")

    def test_a_half_turn_is_split_rather_than_refused(self):
        start = cfp.basis_from_axes([0, 0, -1], [1, 0, 0])
        want = cfp.basis_from_axes([0, 0, 1], [1, 0, 0])
        steps = cfp.orientation_steps(axes(start), axes(want))
        self.assertIsNotNone(steps)
        angles = cfp.step_angles_deg(axes(start), steps)
        self.assertTrue(all(a <= MAX_ROTATION_DEG for a in angles), angles)
        self.assertGreaterEqual(len(steps), 3)


class AngleConstructionTests(unittest.TestCase):
    def test_tilt_zero_approaches_along_minus_up_at_any_azimuth(self):
        for azimuth in (-180, -37, 0, 45, 90, 270, 360):
            basis = cfp.axes_from_angles([0, 0, 1], azimuth, 0.0)
            self.assertIsNotNone(basis)
            for got, expect in zip(basis["approach"], [0, 0, -1]):
                self.assertAlmostEqual(got, expect, places=9)

    def test_opening_stays_perpendicular_to_approach_off_axis(self):
        for up in ([0, 0, 1], [1, 0, 0], [0.3, -0.4, 0.9]):
            for azimuth in (0, 45, 90, -130, 270):
                for tilt in (0, 15, 40, -25, 75):
                    basis = cfp.axes_from_angles(up, azimuth, tilt)
                    self.assertIsNotNone(basis, (up, azimuth, tilt))
                    self.assertAlmostEqual(
                        cfp._dot(basis["approach"], basis["opening"]), 0.0, places=9)

    def test_azimuth_is_continuous_not_snapped_to_candidates(self):
        # Two azimuths one degree apart must give two different, close orientations.
        a = cfp.axes_from_angles([0, 0, 1], 31.0, 30.0)
        b = cfp.axes_from_angles([0, 0, 1], 32.0, 30.0)
        angle = cfp.rotation_between(axes(a), axes(b))[1]
        self.assertGreater(angle, 0.1)
        self.assertLess(angle, 2.0)

    def test_an_unusable_up_axis_is_refused(self):
        self.assertIsNone(cfp.axes_from_angles([0, 0, 0], 0, 0))
        self.assertIsNone(cfp.axes_from_angles(None, 0, 0))


class ArrivalTests(unittest.TestCase):
    """`within_tolerance` decides whether a later step runs, so it must be earned.

    A pose step commands an orientation as well as a position. A reading that
    checked only the position would call a gripper arrived while it faced 90 deg
    away, and the executor would then run the next step — often a close — on a
    pose that never arrived. Unverifiable must be None, not True.
    """

    def target(self, **over):
        base = {"position": [0.1, 0.2, 1.0],
                "approach": [0.0, 0.0, -1.0], "opening": [0.0, 1.0, 0.0],
                "tolerance_m": 0.02,
                "orientation_tolerance_deg": cfp.DEFAULT_ORIENTATION_TOLERANCE_DEG}
        base.update(over)
        return base

    def measured(self, position, approach=(0.0, 0.0, -1.0), opening=(0.0, 1.0, 0.0)):
        out = {"status": "ok",
               "fingertip_position": dict(zip("xyz", position))}
        if approach is not None:
            out["approach"] = dict(zip("xyz", approach))
        if opening is not None:
            out["opening"] = dict(zip("xyz", opening))
        return out

    def test_a_pose_on_target_in_both_respects_is_arrival(self):
        error = cfp.pose_error(self.target(), self.measured([0.1, 0.2, 1.0]))
        self.assertTrue(error["within_tolerance"])
        self.assertEqual(error["failed"], [])

    def test_a_ninety_degree_orientation_error_is_not_arrival(self):
        # The exact case the acceptance probe caught: position perfect, gripper
        # turned a quarter turn, and the old reading called it arrived.
        error = cfp.pose_error(self.target(),
                               self.measured([0.1, 0.2, 1.0],
                                             approach=(1.0, 0.0, 0.0),
                                             opening=(0.0, 1.0, 0.0)))
        self.assertEqual(error["position_error_m"], 0.0)
        self.assertAlmostEqual(error["orientation_error_deg"], 90.0, places=3)
        self.assertFalse(error["within_tolerance"])
        self.assertEqual(error["failed"], ["orientation"])

    def test_an_unmeasurable_orientation_is_unknown_not_arrival(self):
        for approach, opening in ((None, (0.0, 1.0, 0.0)),
                                  ((0.0, 0.0, -1.0), None),
                                  ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
            with self.subTest(approach=approach, opening=opening):
                error = cfp.pose_error(
                    self.target(),
                    self.measured([0.1, 0.2, 1.0], approach=approach, opening=opening))
                self.assertIsNone(error["within_tolerance"])
                self.assertIsNone(error["orientation_within_tolerance"])
                self.assertEqual(error["reason"], "orientation_unverifiable")

    def test_no_measured_pose_at_all_is_unknown_not_arrival(self):
        for measured in (None, {}, {"status": "unknown", "reason": "no telemetry"}):
            with self.subTest(measured=measured):
                error = cfp.pose_error(self.target(), measured)
                self.assertIsNone(error["within_tolerance"])
                self.assertEqual(error["status"], "unknown")

    def test_a_small_orientation_residual_still_counts_as_arrival(self):
        # The controller settles a fraction of a degree off; that is not a miss.
        approach = cfp.rotate_about([0.0, 1.0, 0.0], 2.0, [0.0, 0.0, -1.0])
        error = cfp.pose_error(self.target(),
                               self.measured([0.1, 0.2, 1.0], approach=approach,
                                             opening=(0.0, 1.0, 0.0)))
        self.assertLess(error["orientation_error_deg"], 3.0)
        self.assertTrue(error["within_tolerance"])

    def test_the_orientation_tolerance_is_validated_like_the_position_one(self):
        for bad in (0.0, 0.5, 46.0, "10", None):
            with self.subTest(value=bad):
                with self.assertRaises(cfp.Rejected):
                    cfp.validate_step({"pose": {"frame": "robot",
                                                "offset": [0, 0, -0.05],
                                                "orientation_tolerance_deg": bad}})
        step = cfp.validate_step({"pose": {"frame": "robot", "offset": [0, 0, -0.05],
                                          "orientation_tolerance_deg": 20.0}})
        self.assertEqual(step["orientation_tolerance_deg"], 20.0)


class ExecutorGateTests(unittest.TestCase):
    """The one line that decides whether the next step runs at all."""

    def test_only_an_explicit_true_continues_a_program(self):
        # Reading the source is the honest check here: the executor's gate is a
        # single comparison, and the defect was that it read `is False`, so an
        # unverifiable arrival continued into the next step.
        import inspect

        from spatial_interface import coarse_fine_tools as cft
        source = inspect.getsource(cft.run_program)
        self.assertIn('record["within_tolerance"] is not True', source)
        self.assertNotIn('record["within_tolerance"] is False', source)

    def test_an_abort_keeps_the_steps_that_already_ran(self):
        # The records list is the caller's, so unwinding the Aborted must not take
        # the evidence with it. Driven with a stub frame rather than a browser: the
        # property under test is only that what ran survives the exception.
        import asyncio

        from spatial_interface import coarse_fine_tools as cft

        async def stub_read_frame(ctx):
            return {"version": "e1-s1-c1", "measured": {"status": "unknown"},
                    "pairing": {"paired": True}}

        program = cfp.validate_program(
            {"steps": [{"observe": {"radius_m": 0.05}},
                       {"pose": {"frame": "robot", "offset": [0, 0, -0.05]}}]})
        records: list = []

        async def drive():
            with patch.object(cft.fgt, "read_frame", stub_read_frame), \
                 patch.object(cft, "fine_observation", _stub_observation):
                await cft.run_program(None, program, _StubBudget(), records=records)

        with self.assertRaises(cft.fgt.Aborted):
            asyncio.run(drive())
        # Both the observe that succeeded and the pose that refused are still here,
        # and the refused one says why — returning only the committed actions would
        # have dropped both.
        self.assertEqual([r["kind"] for r in records], ["observe", "pose"])
        self.assertIn("observation", records[0])
        self.assertEqual(records[1]["status"], "refused")
        self.assertEqual(records[1]["physical"], False)

    def test_this_calls_actions_are_sliced_out_of_the_episode_history(self):
        # The accounting defect: the reply returned the whole episode's committed
        # list, so a second call looked like it had re-executed the first call's
        # waypoints. fg_run's committed_before pattern is what fixes it.
        import inspect

        from spatial_interface import coarse_fine_tools as cft
        source = inspect.getsource(cft.CfPolicyTool.__call__)
        self.assertIn("committed_before = len(STATE.committed)", source)
        self.assertIn("STATE.committed[committed_before:]", source)

    def test_the_waypoint_count_has_exactly_one_set_of_books(self):
        # Two counters incremented on different paths under-reported real motion.
        from spatial_interface import coarse_fine_tools as cft
        self.assertIsInstance(cft.CoarseFineState.waypoints, property)
        self.assertIsInstance(cft.CoarseFineState.committed, property)
        self.assertIn("fgt.STATE.waypoints += 1",
                      inspect.getsource(cft.run_dwell))


async def _stub_observation(ctx, **kwargs):
    return {"frame": "e1-s1-c1", "paired": True}


class _StubBudget:
    """Just enough of fast_geometry_tools.Budget for the executor's own checks."""

    def report(self):
        return {"waypoints_used": 0}

    def remaining_s(self):
        return 60.0

    def check(self, *a, **k):
        return None

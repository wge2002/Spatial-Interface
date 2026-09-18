"""Producer-side observation snapshot lifetime.

The defect these cover: while the sim is idle, `SimEnv.observe()` renders again
from `self.obs`, which only `apply_action` replaces. The old stream loop called it
once per period and the broadcast child numbered every *message*, so an
observation the model had just bound a reference to was retired a few tens of
milliseconds later by a re-read that measured nothing. Bind-then-use across two
tool calls could not succeed at real model latency.

These are behavioural: they drive the real snapshot/message helpers and the real
`frame_meta` reader with a fake env whose step counter they control, and assert on
identity, on what got recomputed, and on what a consumer would conclude.
"""

import unittest
from unittest.mock import patch

from spatial_interface import geometry_ref as geom


class FakeEnv:
    """Enough of SimEnv to exercise snapshot lifetime: `observe()` renders from
    cached state and never steps; only `step()` advances `num_step`."""

    def __init__(self):
        self.num_step = 0
        self.observe_calls = 0
        self.cached = "state-0"
        self.cfg = type("cfg", (), {"image_size": 224, "crop_table": True})()

    def observe(self):
        self.observe_calls += 1
        # A fresh dict each call, like the real renderer: identity of the *object*
        # is never what makes an observation new.
        return {"agentview_image": self.cached, "read": self.observe_calls}

    def step(self):
        self.num_step += 1
        self.cached = f"state-{self.num_step}"


def recorder(mode="geometry"):
    """An InteractiveBot with the snapshot machinery live and nothing else built."""
    from spatial_interface.record_sim import InteractiveBot

    rec = InteractiveBot.__new__(InteractiveBot)
    rec.env = FakeEnv()
    rec._observation_id = 0
    rec._snapshot = None
    rec._preserve_idle_snapshot = mode == "geometry"
    rec.collect_cam_info = lambda obs: {"agentview": {"read": obs["read"]}}
    return rec


class IdleSnapshotLifetimeTests(unittest.TestCase):
    def test_idle_reads_do_not_mint_new_observations(self):
        rec = recorder()
        first = rec._current_snapshot()
        for _ in range(5):
            again = rec._current_snapshot()
            self.assertEqual(again["observation_id"], first["observation_id"])
        # And the sensors were read exactly once: the later ticks did not even
        # re-render, because there was nothing new to render.
        self.assertEqual(rec.env.observe_calls, 1)

    def test_the_preserved_snapshot_stays_whole(self):
        # Not just the id: the cloud/camera source and the calibration collected
        # with it must be the same objects, or a consumer could pair this id's
        # geometry with a different read's calibration.
        rec = recorder()
        first = rec._current_snapshot()
        again = rec._current_snapshot()
        self.assertIs(again["obs"], first["obs"])
        self.assertIs(again["cam_info"], first["cam_info"])

    def test_a_sim_step_retires_the_snapshot(self):
        rec = recorder()
        first = rec._current_snapshot()
        rec.env.step()
        after = rec._current_snapshot()
        self.assertNotEqual(after["observation_id"], first["observation_id"])
        self.assertEqual(after["obs"]["agentview_image"], "state-1")
        self.assertEqual(rec.env.observe_calls, 2)

    def test_post_waypoint_observation_is_always_new(self):
        # `_new_snapshot` is what the waypoint branch calls. It must never return
        # the preserved one even if it were somehow called without a step.
        rec = recorder()
        first = rec._current_snapshot()
        forced = rec._new_snapshot()
        self.assertNotEqual(forced["observation_id"], first["observation_id"])

    def test_non_geometry_modes_keep_re_observing(self):
        # The repair must not change what legacy/compact stream.
        rec = recorder(mode="compact")
        a = rec._current_snapshot()
        b = rec._current_snapshot()
        self.assertNotEqual(a["observation_id"], b["observation_id"])
        self.assertEqual(rec.env.observe_calls, 2)

    def test_mode_is_read_from_the_control_interface(self):
        from spatial_interface.record_sim import InteractiveBot

        with patch.dict("os.environ", {"VIA_CONTROL_INTERFACE": "geometry"}):
            self.assertTrue(InteractiveBot._preserve_idle_snapshot_for_env())
        for mode in ("legacy", "compact"):
            with patch.dict("os.environ", {"VIA_CONTROL_INTERFACE": mode}):
                self.assertFalse(InteractiveBot._preserve_idle_snapshot_for_env())

    def test_the_message_carries_the_observation_number(self):
        rec = recorder()
        snap = rec._current_snapshot()
        msg = rec._stream_message(snap, update_ui=False)
        self.assertEqual(msg["observation_id"], snap["observation_id"])
        self.assertIs(msg["cam_info"], snap["cam_info"])
        self.assertFalse(msg["update_ui"])

    def test_redelivery_does_not_ask_the_browser_to_update(self):
        # update_ui is what increments the browser's waypoint_done_count and snaps
        # the virtual target onto the real gripper. An idle redelivery that set it
        # would re-count a completion that never happened and move the model's
        # target out from under it.
        rec = recorder()
        for _ in range(3):
            msg = rec._stream_message(rec._current_snapshot(), update_ui=False)
            self.assertFalse(msg["update_ui"])


class DeliveryVersusObservationTests(unittest.TestCase):
    """`frame_meta`'s transport fields must stay out of observation identity."""

    def meta(self, **kw):
        base = {"epoch": "e1", "seq": 5, "cloud_seq": 5, "cam_labels": ["agentview"],
                "cam_info_seq": 5, "proprio_seq": 5, "delivery_seq": 40,
                "redelivery": False}
        base.update(kw)
        return {"frame_meta": base, "cloud_points": 100}

    def test_a_redelivered_observation_keeps_its_version(self):
        first = self.meta(delivery_seq=40, redelivery=False)
        again = self.meta(delivery_seq=41, redelivery=True)
        self.assertEqual(geom.observation_version(first),
                         geom.observation_version(again))
        self.assertEqual(geom.cloud_version(first), geom.cloud_version(again))

    def test_a_redelivery_is_not_an_advance(self):
        # The execution check requires a strictly later frame afterwards. A
        # redelivery must not satisfy it.
        first = self.meta(delivery_seq=40)
        again = self.meta(delivery_seq=99, redelivery=True)
        # frame_advance returns the reason it is not an advance, None when it is.
        self.assertEqual(geom.frame_advance(geom.observation_version(first),
                                            geom.observation_version(again)),
                         "not_a_later_frame")

    def test_a_new_observation_still_advances(self):
        self.assertIsNone(geom.frame_advance(
            geom.observation_version(self.meta(seq=5, cloud_seq=5)),
            geom.observation_version(self.meta(seq=6, cloud_seq=6))))

    def test_transport_fields_are_reported_but_not_trusted(self):
        meta = geom.frame_meta(self.meta(delivery_seq=41, redelivery=True))
        self.assertEqual(meta["delivery_seq"], 41)
        self.assertTrue(meta["redelivery"])
        # Absent on a producer that does not send them: reported as unknown
        # rather than defaulted to a number that would look like a real count.
        older = geom.frame_meta({"frame_meta": {"epoch": "e1", "seq": 5,
                                                "cloud_seq": 5}})
        self.assertIsNone(older["delivery_seq"])
        self.assertFalse(older["redelivery"])

    def test_a_redelivered_frame_is_still_paired(self):
        # The cloud and the rest of it were measured together; redelivering that
        # pair does not make the cloud stale.
        state = geom.pairing_state(self.meta(delivery_seq=41, redelivery=True))
        self.assertEqual(state["status"], "paired")

    def test_a_cached_cloud_motion_frame_stays_unpaired(self):
        # Motion during a waypoint: new tip and new images, old cloud. This must
        # keep reporting unpaired -- the repair only preserves idle observations.
        state = geom.pairing_state(self.meta(seq=9, cloud_seq=7, cam_info_seq=9,
                                             proprio_seq=9))
        self.assertNotEqual(state["status"], "paired")


if __name__ == "__main__":
    unittest.main()

"""Frame-pairing contracts for the geometry tools, with a mocked browser.

Every case here is a frame the producer really sends (a streamed skip_pcl frame,
a JPEG still decoding, a new frame arriving between two page.evaluate calls) and
asserts the tool reports unknown and stores nothing, rather than reading a mix of
two observations as one look at the scene.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from spatial_interface import geometry_ref as geom
from spatial_interface import mcp_server as server
from tests.test_geometry_ref import blob, frame


def context(observations, *, points=None):
    """A ctx whose observe_geometry() returns `observations` in order.

    The last entry repeats once exhausted, so a test only has to state the frames
    it cares about.
    """
    seen = list(observations)

    async def observe():
        return seen.pop(0) if len(seen) > 1 else seen[0]

    cloud = points if points is not None else blob(300)
    flat = [c for p in cloud for c in geom.robot_to_ui(p, 0.8)]
    return SimpleNamespace(
        ui_z_offset=0.8, timing_id="test",
        observe_geometry=AsyncMock(side_effect=observe),
        uv_to_xy=AsyncMock(return_value=(100, 100, 640, 480)),
        show_cursor=AsyncMock(return_value=None),
        preview_cloud_point=AsyncMock(return_value={
            "surface": "pointcloud", "point": {"x": 0.0, "y": 2.0, "z": 0.0}}),
        sample_local_cloud=AsyncMock(return_value={
            "support": len(cloud), "stride": 1, "points": flat,
            "cloud_points": 100000}),
        corridor_clearance=AsyncMock(return_value={
            "min_distance": [1.0] * geom.PATH_SAMPLES,
            "body_support": [0] * geom.PATH_SAMPLES,
            "observed_support": [5] * geom.PATH_SAMPLES,
            "excluded_self_points": 3, "cloud_points": 100000}),
        mark_ui_point=AsyncMock(return_value={"inside": True}),
        gripper_pose=AsyncMock(return_value={
            "robot_position": dict(x=0, y=0, z=1.05),
            "ui_position": dict(x=0, y=2.05, z=0),
            "robot_approach": dict(x=0, y=0, z=-1),
            "robot_opening": dict(x=0, y=-1, z=0), "gripper_open": True}),
        snap=AsyncMock(return_value=[]),
        geometry_invalid=0,
    )


def payload_of(result):
    return json.loads(result[0].text)


class BindFrameTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server._geometry_reset_episode_state()

    async def test_a_paired_frame_binds_and_records_both_versions(self):
        ctx = context([frame(seq=4, cloud_seq=4)])
        out = payload_of(await server.GeometryBindReferenceTool()(ctx, {"u": 0.5, "v": 0.5}))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["pairing"]["status"], "paired")
        self.assertEqual(out["observation_version"],
                         geom.observation_version(frame(seq=4, cloud_seq=4)))
        self.assertEqual(len(server.GEOMETRY_REFERENCES), 1)

    async def test_cached_cloud_with_newer_proprioception_binds_nothing(self):
        # record_sim's skip_pcl stream: this frame's tip and images are new, its
        # cloud is the previous frame's.
        ctx = context([frame(seq=9, cloud_seq=7)])
        out = payload_of(await server.GeometryBindReferenceTool()(ctx, {"u": 0.5, "v": 0.5}))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["pairing"]["reason"],
                         "stale_cloud_reused_by_streamed_frame")
        self.assertEqual(server.GEOMETRY_REFERENCES, {})
        ctx.sample_local_cloud.assert_not_awaited()

    async def test_pending_camera_decode_blocks_a_camera_feed_click_only(self):
        # Clicking the wrist feed reads pixels that are still decoding; the same
        # frame is fine for a canvas click, which reads no image at all.
        obs = frame(seq=9, displayed=8, pending=9)
        ctx = context([obs])
        ctx.preview_cloud_point.return_value = {
            "surface": "wrist", "point": {"x": 0.0, "y": 2.0, "z": 0.0}}
        out = payload_of(await server.GeometryBindReferenceTool()(ctx, {"u": 0.5, "v": 0.5}))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["pairing"]["reason"], "camera_image_decode_pending")
        self.assertEqual(server.GEOMETRY_REFERENCES, {})

        canvas = context([obs])
        self.assertEqual(
            payload_of(await server.GeometryBindReferenceTool()(canvas, {"u": 0.5, "v": 0.5}))["status"],
            "ok")

    async def test_ui_without_metadata_is_unknown_not_a_binding(self):
        ctx = context([{"cloud_points": 100, "cloud_hash": "abc",
                        "waypoint_done_count": 1, "fingertip_ui": [1, 2, 3]}])
        out = payload_of(await server.GeometryBindReferenceTool()(ctx, {"u": 0.5, "v": 0.5}))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["pairing"]["reason"], "no_frame_metadata")
        self.assertEqual(server.GEOMETRY_REFERENCES, {})

    async def test_a_new_frame_while_the_click_is_resolved_stores_nothing(self):
        # preview_cloud_point ray-casts against the cloud in the browser, so it is
        # an extraction too. A frame arriving between it and the first frame read
        # leaves a point from the old cloud beside the new frame's version, and
        # every later same-version check would agree.
        ctx = context([frame(seq=4, cloud_seq=4), frame(seq=5, cloud_seq=5)])
        out = payload_of(await server.GeometryBindReferenceTool()(ctx, {"u": 0.5, "v": 0.5}))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(server.GEOMETRY_REFERENCES, {})
        self.assertNotEqual(out["observation_version_before_locating"],
                            out["observation_version"])
        ctx.sample_local_cloud.assert_not_awaited()

    async def test_a_new_frame_mid_extraction_stores_no_reference(self):
        # Sampling the neighbourhood is a separate page.evaluate from reading the
        # frame; if the producer delivers a new frame in between, the summary
        # describes neither frame. Two reads happen before sampling (locate, then
        # pair), so the change is staged after both.
        ctx = context([frame(seq=4, cloud_seq=4), frame(seq=4, cloud_seq=4),
                       frame(seq=5, cloud_seq=5)])
        out = payload_of(await server.GeometryBindReferenceTool()(ctx, {"u": 0.5, "v": 0.5}))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(server.GEOMETRY_REFERENCES, {})
        self.assertNotEqual(out["observation_version"],
                            out["observation_version_after_extraction"])
        self.assertNotIn("reference_id", out)

    async def test_an_unchanging_frame_is_stable_across_the_extraction(self):
        # Screenshots and the marker drawn by this very tool must not look like a
        # new observation, or nothing could ever be bound.
        ctx = context([frame(seq=6, cloud_seq=6)] * 4)
        self.assertEqual(
            payload_of(await server.GeometryBindReferenceTool()(ctx, {"u": 0.5, "v": 0.5}))["status"],
            "ok")


class RefreshAndProxyFrameTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server._geometry_reset_episode_state()

    async def bind(self, seq=4):
        ctx = context([frame(seq=seq, cloud_seq=seq)])
        out = payload_of(await server.GeometryBindReferenceTool()(ctx, {"u": 0.5, "v": 0.5}))
        self.assertEqual(out["status"], "ok")
        return out["reference_id"]

    async def test_refresh_on_an_unpaired_frame_leaves_only_stale_coordinates(self):
        ref = await self.bind()
        stored = dict(server.GEOMETRY_REFERENCES[ref]["geometry"])
        ctx = context([frame(seq=11, cloud_seq=7)])
        out = payload_of(await server.GeometryRefreshReferenceTool()(ctx, {"reference_id": ref}))
        self.assertEqual(out["status"], "unknown")
        self.assertIsNone(out.get("geometry"))
        self.assertIn("stale_stored_center", out)
        # No extraction ran, and nothing came out of this call that could stand as
        # current geometry, so the old numbers survive under stale names only.
        self.assertIsNone(server.GEOMETRY_REFERENCES[ref]["geometry"])
        self.assertEqual(server.GEOMETRY_REFERENCES[ref]["stale_geometry"], stored)
        ctx.sample_local_cloud.assert_not_awaited()

    async def test_a_new_frame_mid_refresh_writes_no_geometry_to_the_reference(self):
        ref = await self.bind()
        before = dict(server.GEOMETRY_REFERENCES[ref])
        history_len = len(before["history"])
        ctx = context([frame(seq=8, cloud_seq=8), frame(seq=9, cloud_seq=9)])
        out = payload_of(await server.GeometryRefreshReferenceTool()(ctx, {"reference_id": ref}))
        self.assertEqual(out["status"], "unknown")
        # The extraction spanned two frames, so it summarizes neither: no new
        # geometry is stored and no history entry is written. The reference is
        # retired, its previous numbers readable as stale only.
        self.assertIsNone(server.GEOMETRY_REFERENCES[ref]["geometry"])
        self.assertEqual(server.GEOMETRY_REFERENCES[ref]["stale_geometry"],
                         before["geometry"])
        self.assertEqual(len(server.GEOMETRY_REFERENCES[ref]["history"]), history_len)
        self.assertIsNone(out.get("geometry"))

    async def test_proxy_on_an_unpaired_frame_reports_no_corridor_and_no_prediction(self):
        ref = await self.bind()
        ctx = context([frame(seq=12, cloud_seq=7)])
        out = payload_of(await server.GeometryProxyTool()(ctx, {
            "frame": "robot", "delta_position": [0, 0, -0.02], "reference_id": ref}))
        self.assertEqual(out["status"], "unknown")
        self.assertNotIn("corridor", out)
        self.assertEqual(server.GEOMETRY_PREDICTIONS, {})
        ctx.corridor_clearance.assert_not_awaited()

    async def test_a_new_frame_mid_proxy_check_records_no_prediction(self):
        ref = await self.bind(seq=8)
        # Same frame as the binding, so the reference is still valid; the change
        # lands between the corridor pass and the confirming read.
        ctx = context([frame(seq=8, cloud_seq=8), frame(seq=9, cloud_seq=9)])
        out = payload_of(await server.GeometryProxyTool()(ctx, {
            "frame": "robot", "delta_position": [0, 0, -0.02], "reference_id": ref}))
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(server.GEOMETRY_PREDICTIONS, {})
        self.assertNotIn("prediction", out)

    async def test_a_stable_frame_still_produces_a_corridor_and_a_prediction(self):
        ref = await self.bind(seq=8)
        ctx = context([frame(seq=8, cloud_seq=8)])
        out = payload_of(await server.GeometryProxyTool()(ctx, {
            "frame": "robot", "delta_position": [0, 0, -0.02], "reference_id": ref}))
        self.assertEqual(out["status"], "ok")
        self.assertIn("corridor", out)
        self.assertEqual(len(server.GEOMETRY_PREDICTIONS), 1)


if __name__ == "__main__":
    unittest.main()

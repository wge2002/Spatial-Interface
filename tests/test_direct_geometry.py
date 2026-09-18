"""Checks at the new transport/geometry boundaries, independent of UI actions."""
import asyncio
import unittest
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from spatial_interface import direct_control as dc, direct_geometry_tools as dg, geometry_workspace as gw
from spatial_interface import coarse_fine_policy as cfp


class DirectGeometryTests(unittest.TestCase):
    def setUp(self):
        dg.reset_episode()

    def test_pose_axes_agree_with_robot_quaternion_for_arbitrary_rotations(self):
        for r in Rotation.random(200, random_state=12).as_matrix():
            pose = dc.pose_from_axes([0.1, 0.2, 1.1], r[:, 2], r[:, 1])
            np.testing.assert_allclose(np.asarray(pose)[:3, :3], r, atol=1e-9)

    def test_wire_is_exact_and_rejects_non_rigid_pose(self):
        for command in (gw.Move(gw.IDENTITY), gw.Gripper("close"), gw.Hold(0.5)):
            request = dc.encode("p1/0", command)
            self.assertEqual(type(dc.decode(request)), type(command))
            with self.assertRaises(ValueError):
                dc.decode(dict(request, object_gt=[1, 2, 3]))
        pose = np.eye(4); pose[0, 0] = 2
        with self.assertRaises(gw.BoundaryError):
            dc.decode({"protocol": dc.PROTOCOL, "id": "a", "kind": "pose", "value": pose.tolist()})

    def test_actual_sensor_width_and_pose_are_not_ui_target_fields(self):
        quat = Rotation.from_euler("xyz", [0.3, -0.8, 1.2]).as_quat()
        state = dc.sensor_state({"ee_pos": [0.2, 0.1, 1.1], "ee_quat": quat,
                                 "gripper_open": np.array([0.5]), "sim_state": [123]},
                                seq=5, sim_steps=10, width_max_m=0.08,
                                commanded_open=0, control_freq=20)
        self.assertAlmostEqual(state["gripper_width_m"], 0.04)
        self.assertNotIn("sim_state", state)
        np.testing.assert_allclose(np.asarray(state["pose"])[:3, 3], [0.2, 0.1, 1.1])

    def test_proxy_check_never_executes_and_compile_preserves_reference(self):
        dg.STATE.workspace.publish(gw.Observation("f1", "robot", ((0, 0, 1),), ("cloud",), 0))
        pose = np.eye(4); pose[2, 3] = 1
        card = {"valid": True, "center": [0, 0, 1], "up": [0, 0, 1]}
        with patch.object(dc.RpcBackend, "execute", side_effect=AssertionError("unexpected actuation")):
            dg.propose("item", pose.tolist(), "f1", card)
            dg.STATE.workspace.check(dg.STATE.refs["item"].ref, 0.1)
            program = cfp.validate_program({"steps": [
                {"pose": {"frame": "proxy:item", "offset": [0.02, 0, 0.05],
                          "approach": [0, 0, -1], "opening": [0, 1, 0]}},
                {"pose": {"frame": "robot", "offset": [0, 0, 0.1]}}]})
            robot = {"pose": pose.tolist(), "gripper_width_m": .08, "commanded_gripper_open": 1}
            commands, targets = dg.compile_program(program, robot, "f1")
            committed = dg.STATE.workspace.commit("p1", "f1", commands)
            self.assertEqual(len(committed.bindings), 1)
            np.testing.assert_allclose(np.asarray(committed.resolved[1].pose)[:3, 3], [.02, 0, 1.15])

    def test_registered_enabled_listed_surface_and_legacy_agree(self):
        from spatial_interface import codex_harness as h, mcp_server as s
        from spatial_interface.target_edit import control_interface
        expected = ("dg_look", "dg_policy", "dg_state", "end_episode")
        self.assertEqual(control_interface({"VIA_CONTROL_INTERFACE": "direct_geometry"}), "direct_geometry")
        self.assertEqual(h.enabled_tools_for_mode("direct_geometry"), expected)
        self.assertEqual(s.surface_for_mode("direct_geometry"), expected)
        with patch.dict("os.environ", VIA_CONTROL_INTERFACE="direct_geometry"):
            self.assertEqual(tuple(t.name for t in asyncio.run(s.list_tools())), expected)
        self.assertEqual(len(s.surface_for_mode("legacy")), 18)
        self.assertEqual(len(s.surface_for_mode("coarse_fine_policy")), 4)
        self.assertIn("direct_geometry", h.INTERFACE_REPLACEMENT_GUIDE_MODES)

    def test_model_view_drops_candidate_hints_recursively(self):
        self.assertEqual(dg.public({"proxy": {"grasp_candidates": ["a"], "center": [0, 1, 2]}}),
                         {"proxy": {"center": [0, 1, 2]}})

    def test_inline_binding_uses_normalized_object_name_before_commit(self):
        """Regression for native attempt 01: validate_bind nests name in objects."""
        dg.STATE.workspace.publish(gw.Observation("f1", "robot", ((0, 0, 1),), ("cloud",), 0))
        pose = np.eye(4); pose[2, 3] = 1
        robot = {"pose": pose.tolist(), "gripper_width_m": .08, "commanded_gripper_open": 1}
        view = {"frame": "f1", "paired": True}
        ctx = SimpleNamespace(timing_id="bind-test", snap=AsyncMock(return_value=[]))
        async def bind(context, normalized):
            self.assertEqual(normalized["objects"][0]["name"], "chosen")
            dg.cf.STATE.proxies["chosen"] = {"valid": True, "center": [0, 0, 1], "up": [0, 0, 1]}
            return {"status": "ok", "proxy": {"name": "chosen", "valid": True}}
        with patch.object(dg, "observe", AsyncMock(return_value=(view, robot))), \
             patch.object(dg.cf, "bind_proxy", side_effect=bind), \
             patch.object(dc.RpcBackend, "execute", side_effect=AssertionError("unexpected actuation")):
            result = asyncio.run(dg.DgPolicyTool()(ctx, {
                "bind": {"name": "chosen", "shape": "blob", "region": {
                    "kind": "box", "u0": .2, "v0": .2, "u1": .6, "v1": .6}},
                "steps": [{"observe": {}}]}))
        import json
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "geometry_only", payload)
        self.assertEqual(dg.STATE.refs["chosen"].ref.name, "chosen")
        self.assertEqual(dg.STATE.executed, [])


if __name__ == "__main__":
    unittest.main()

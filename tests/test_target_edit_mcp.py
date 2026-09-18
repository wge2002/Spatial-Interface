"""MCP failure/preview contracts using the DSW environment, without a model API."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from spatial_interface import mcp_server as server


def pose():
    return {"robot_position": dict(x=0, y=0, z=1), "ui_position": dict(x=0, y=2, z=0),
            "robot_approach": dict(x=0, y=0, z=-1), "robot_opening": dict(x=0, y=-1, z=0),
            "gripper_open": True}


def context():
    return SimpleNamespace(
        page=SimpleNamespace(evaluate=AsyncMock(return_value={"ok": True})),
        ensure_translation_mode=AsyncMock(return_value=True),
        gripper_pose=AsyncMock(return_value=pose()), ui_z_offset=0.8, timing_id="test",
        snap=AsyncMock(return_value=[server.types.ImageContent(type="image", data="eA==",
                                                             mimeType="image/jpeg")]),
    )


class TargetMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_has_one_final_preview(self):
        ctx = context()
        result = await server.EditTargetTool()(ctx, {"frame": "robot", "delta_position": [0, 0, 0]})
        self.assertEqual(json.loads(result[0].text)["status"], "ok")
        self.assertEqual(sum(c.type == "image" for c in result), 1)
        ctx.snap.assert_awaited_once()
        ctx.page.evaluate.assert_awaited_once()

    async def test_invalid_orientation_prevents_valid_translation(self):
        ctx = context()
        result = await server.EditTargetTool()(ctx, {"frame": "robot", "delta_position": [0.03, 0, 0],
                                                    "approach": [0, 0, -1]})
        self.assertEqual(json.loads(result[0].text)["status"], "rejected")
        ctx.page.evaluate.assert_not_awaited()
        ctx.ensure_translation_mode.assert_not_awaited()
        ctx.snap.assert_not_awaited()

    async def test_absolute_limit_rejects_before_mutation(self):
        ctx = context()
        result = await server.EditTargetTool()(ctx, {"frame": "robot", "position": [1, 0, 1]})
        self.assertEqual(json.loads(result[0].text)["status"], "rejected")
        ctx.page.evaluate.assert_not_awaited()

    async def test_stale_target_returns_actual_state(self):
        ctx = context()
        ctx.page.evaluate.return_value = {"ok": False, "reason": "Target changed"}
        result = await server.EditTargetTool()(ctx, {"frame": "robot", "delta_position": [0.03, 0, 0]})
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["target"]["position"], pose()["robot_position"])

    async def test_exception_after_mutation_does_not_claim_rollback(self):
        ctx = context()
        actual = copy.deepcopy(pose())
        actual["robot_position"]["x"] = 0.03
        ctx.gripper_pose.side_effect = [pose(), actual]
        ctx.page.evaluate.side_effect = RuntimeError("curve update failed")
        result = await server.EditTargetTool()(ctx, {"frame": "robot", "delta_position": [0.03, 0, 0]})
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["target"]["position"]["x"], 0.03)
        self.assertFalse(payload["robot_execution_requested"])

    async def test_missing_readback_is_unknown_not_old_pose(self):
        ctx = context()
        ctx.gripper_pose.side_effect = [pose(), RuntimeError("disconnected"), RuntimeError("disconnected")]
        result = await server.EditTargetTool()(ctx, {"frame": "robot", "delta_position": [0, 0, 0]})
        self.assertEqual(json.loads(result[0].text)["status"], "failed")
        self.assertIsNone(json.loads(result[0].text)["target"])

    async def test_readback_mismatch_and_preview_failure_are_reported(self):
        ctx = context()
        result = await server.EditTargetTool()(ctx, {"frame": "robot", "delta_position": [0.03, 0, 0]})
        self.assertEqual(json.loads(result[0].text)["status"], "failed")
        ctx.snap.side_effect = RuntimeError("screenshot unavailable")
        result = await server.EditTargetTool()(ctx, {"frame": "robot", "delta_position": [0, 0, 0]})
        self.assertEqual(json.loads(result[0].text)["status"], "preview_failed")

    async def test_original_schemas_preserved_and_new_tool_gated(self):
        with patch.dict("os.environ", {"VIA_CONTROL_INTERFACE": "legacy"}):
            original = await server.list_tools()
        with patch.dict("os.environ", {"VIA_CONTROL_INTERFACE": "compact"}):
            compact = await server.list_tools()
        self.assertEqual(len(original), 18)
        self.assertEqual(compact[:-1], original)
        self.assertEqual(compact[-1].name, "edit_target")
        with patch.dict("os.environ", {"VIA_CONTROL_INTERFACE": "legacy"}), patch.object(server, "_ctx", context()):
            result = await server.call_tool("edit_target", {"frame": "robot", "delta_position": [0, 0, 0]})
        self.assertIn("Unknown tool", result[0].text)


if __name__ == "__main__":
    unittest.main()

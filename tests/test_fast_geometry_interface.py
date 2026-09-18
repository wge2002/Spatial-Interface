"""The fast-geometry interface must agree with itself in four independent places.

A tool is only usable if the server lists it, the server's dispatch gate accepts
it, and the native harness enables it. Any two of the three agreeing is the
failure mode worth testing: a listed-but-not-enabled tool is invisible to a
Codex agent, and an enabled-but-not-listed one fails at call time with a message
the agent cannot act on. The fourth place is the observation pipeline, which must
preserve paired idle frames for this interface as it does for geometry.

These tests also pin the frozen surfaces. `fast_geometry` is a REPLACEMENT, and
the whole point of keeping it opt-in is that a run recorded under legacy,
compact or geometry stays comparable with every historical run.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from spatial_interface import codex_harness as harness
from spatial_interface import fast_geometry_tools as fgt
from spatial_interface import mcp_server as server
from spatial_interface import record_sim
from spatial_interface.target_edit import control_interface

MODES = ("legacy", "compact", "geometry", "fast_geometry", "coarse_fine_policy")


def listed(mode: str) -> list[str]:
    with patch.dict("os.environ", {"VIA_CONTROL_INTERFACE": mode}):
        import asyncio

        return [t.name for t in asyncio.run(server.list_tools())]


class SurfaceAgreementTests(unittest.TestCase):
    def test_every_mode_lists_exactly_what_the_harness_enables(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertEqual(sorted(listed(mode)),
                                 sorted(harness.enabled_tools_for_mode(mode)))

    def test_the_listing_reads_the_same_table_dispatch_does(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertEqual(listed(mode), list(server.surface_for_mode(mode)))

    def test_the_fast_surface_is_six_tools(self):
        self.assertEqual(listed("fast_geometry"),
                         ["fg_look", "fg_bind", "fg_check", "fg_run", "fg_state",
                          "end_episode"])

    def test_the_fast_surface_offers_no_per_nudge_or_toggle_tool(self):
        # Not an accident of ordering: a fast-geometry episode expresses a grasp
        # through fg_run's own stages, so exposing gripper_toggle or a nudge tool
        # would let the model actuate outside the bounded executor.
        surface = set(listed("fast_geometry"))
        for name in ("gripper_toggle", "gripper_translate", "gripper_rotate",
                     "gripper_advance_or_retreat", "gripper_teleport_via_click",
                     "edit_target", "execute_waypoint", "screenshot"):
            self.assertNotIn(name, surface)

    def test_the_three_older_surfaces_are_unchanged(self):
        self.assertEqual(listed("legacy"), list(server.ORIGINAL_TOOLS))
        self.assertEqual(len(listed("legacy")), 18)
        self.assertEqual(listed("compact"), list(server.ORIGINAL_TOOLS) + ["edit_target"])
        self.assertEqual(listed("geometry"),
                         list(server.ORIGINAL_TOOLS) + [
                             "edit_target", "geometry_bind_reference",
                             "geometry_refresh_reference", "geometry_proxy_check"])

    def test_the_coarse_fine_surface_is_four_tools(self):
        self.assertEqual(listed("coarse_fine_policy"),
                         ["cf_look", "cf_policy", "cf_state", "end_episode"])

    def test_the_two_replacement_surfaces_do_not_reach_each_other(self):
        # Two experimental replacement surfaces now exist. Each must be exactly
        # itself: a cf run that could call fg_run would have the named skills this
        # interface deliberately does not have, and the comparison between the two
        # would be meaningless.
        self.assertEqual(set(listed("fast_geometry"))
                         & set(server.COARSE_FINE_TOOLS), set())
        self.assertEqual(set(listed("coarse_fine_policy"))
                         & set(server.FAST_GEOMETRY_TOOLS), set())

    def test_no_older_surface_can_reach_a_fast_tool(self):
        for mode in ("legacy", "compact", "geometry"):
            for name in server.FAST_GEOMETRY_TOOLS + server.COARSE_FINE_TOOLS:
                with self.subTest(mode=mode, tool=name):
                    self.assertNotIn(name, listed(mode))
                    self.assertNotIn(name, harness.enabled_tools_for_mode(mode))

    def test_an_unknown_mode_is_rejected_rather_than_defaulted(self):
        with patch.dict("os.environ", {"VIA_CONTROL_INTERFACE": "fastgeometry"}):
            with self.assertRaises(ValueError):
                control_interface()


class FastDispatchGateTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, mode: str, name: str, args: dict | None = None):
        # A context has to be present: the browser-readiness reply comes first, so
        # without one every call would look rejected for the wrong reason.
        ctx = SimpleNamespace(
            page=SimpleNamespace(evaluate=AsyncMock(return_value={"ok": True})),
            ensure_translation_mode=AsyncMock(return_value=True),
            gripper_pose=AsyncMock(return_value=None), ui_z_offset=0.8,
            timing_id="test", geometry_invalid=0, browser=None,
            screenshots_dir=None, snap=AsyncMock(return_value=[]),
        )
        with patch.dict("os.environ", {"VIA_CONTROL_INTERFACE": mode}), \
                patch.object(server, "_ctx", ctx):
            return await server.call_tool(name, args or {})

    async def test_a_hidden_tool_is_unknown_in_the_fast_mode(self):
        # The listing hiding it is not enough — dispatch is what a model that
        # guesses a tool name actually hits.
        for name in ("execute_waypoint", "gripper_toggle", "edit_target",
                     "geometry_proxy_check", "screenshot"):
            with self.subTest(tool=name):
                result = await self.call("fast_geometry", name)
                self.assertIn("Unknown tool", result[0].text)

    async def test_a_fast_tool_is_unknown_outside_the_fast_mode(self):
        for mode in ("legacy", "compact", "geometry"):
            result = await self.call(mode, "fg_state")
            self.assertIn("Unknown tool", result[0].text)

    async def test_a_hidden_tool_is_unknown_in_the_coarse_fine_mode(self):
        for name in ("execute_waypoint", "gripper_toggle", "edit_target",
                     "geometry_proxy_check", "screenshot", "fg_run", "fg_bind"):
            with self.subTest(tool=name):
                result = await self.call("coarse_fine_policy", name)
                self.assertIn("Unknown tool", result[0].text)

    async def test_a_coarse_fine_tool_is_unknown_outside_its_mode(self):
        for mode in ("legacy", "compact", "geometry", "fast_geometry"):
            result = await self.call(mode, "cf_state")
            self.assertIn("Unknown tool", result[0].text)

    async def test_a_coarse_fine_tool_is_reachable_in_its_own_mode(self):
        # cf_state is the one cf tool that needs no browser, the same role fg_state
        # plays above: anything other than the gate's message proves dispatch let it
        # through.
        result = await self.call("coarse_fine_policy", "cf_state")
        self.assertNotIn("Unknown tool", result[0].text)

    async def test_a_malformed_program_moves_nothing(self):
        # The fail-closed gate that matters most: validation happens before any
        # browser call, so a bad step cannot cost a waypoint. A pose with both
        # position and offset is refused, and the target is never even read.
        result = await self.call("coarse_fine_policy", "cf_policy",
                                {"steps": [{"pose": {"frame": "robot",
                                                     "position": [0.1, 0.0, 1.0],
                                                     "offset": [0.0, 0.0, 0.1]}}]})
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["steps"], [])

    async def test_an_unresolvable_pose_stops_before_executing_anything(self):
        # The program is well formed, so validation passes and resolution is what
        # refuses. With no telemetry behind this mock the reason is
        # no_measured_end_effector rather than the unbound proxy, but the property
        # under test is the same one either way: a pose that cannot be resolved must
        # stop the program with nothing executed, never fall back to a default frame
        # origin and move the robot somewhere the program did not ask for.
        result = await self.call("coarse_fine_policy", "cf_policy",
                                {"steps": [{"pose": {"frame": "proxy:nothing",
                                                     "offset": [0, 0, 0.1]}}]})
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "stopped")
        self.assertEqual(payload["reason"], "pose_not_resolvable")
        self.assertEqual(payload["executed"], [])
        self.assertEqual(payload["budget"]["waypoints_used"], 0)

    async def test_a_fast_tool_is_reachable_in_the_fast_mode(self):
        # fg_state is the one fast tool that needs no browser, so reaching the
        # handler is observable without a page: anything other than the gate's
        # own message proves dispatch let it through.
        result = await self.call("fast_geometry", "fg_state")
        self.assertNotIn("Unknown tool", result[0].text)


class HarnessWiringTests(unittest.TestCase):
    def test_the_fast_mode_replaces_the_tool_list_rather_than_extending_it(self):
        enabled = harness.enabled_tools_for_mode("fast_geometry")
        self.assertNotIn("execute_waypoint", enabled)
        self.assertEqual(len(enabled), 6)
        # ...while the additive modes still carry all 18.
        for mode in ("legacy", "compact", "geometry"):
            self.assertTrue(set(harness.ROBOT_TOOLS) <=
                            set(harness.enabled_tools_for_mode(mode)))

    def test_the_overrides_carry_the_fast_mode_and_its_log_file(self):
        cfg = {"mcpServers": {"sphinx2": {"command": "spatial_interface/run_mcp.sh", "args": [],
                                          "env": {}}}}
        env = {"VIA_CONTROL_INTERFACE": "fast_geometry",
               "VIA_FAST_GEOMETRY_LOG_FILE": "/tmp/fg.jsonl",
               "VIA_GEOMETRY_LOG_FILE": "/tmp/geom.jsonl"}
        with patch.object(harness, "get_json_file", lambda _p: cfg), \
                patch.dict("os.environ", env):
            flags = harness.codex_mcp_overrides(9100)
        table = flags[1]
        self.assertIn('VIA_CONTROL_INTERFACE="fast_geometry"', table)
        self.assertIn('VIA_FAST_GEOMETRY_LOG_FILE="/tmp/fg.jsonl"', table)
        # The other interface's log variable must not ride along: forwarding it
        # would point a fast run's diagnostics at a geometry run's file.
        self.assertNotIn("VIA_GEOMETRY_LOG_FILE", table)
        self.assertIn(json.dumps(harness.enabled_tools_for_mode("fast_geometry")),
                      table)

    def test_the_geometry_log_variable_still_only_follows_geometry(self):
        cfg = {"mcpServers": {"sphinx2": {"command": "spatial_interface/run_mcp.sh", "args": [],
                                          "env": {}}}}
        env = {"VIA_CONTROL_INTERFACE": "geometry",
               "VIA_GEOMETRY_LOG_FILE": "/tmp/geom.jsonl",
               "VIA_FAST_GEOMETRY_LOG_FILE": "/tmp/fg.jsonl"}
        with patch.object(harness, "get_json_file", lambda _p: cfg), \
                patch.dict("os.environ", env):
            table = harness.codex_mcp_overrides(9100)[1]
        self.assertIn('VIA_GEOMETRY_LOG_FILE="/tmp/geom.jsonl"', table)
        self.assertNotIn("VIA_FAST_GEOMETRY_LOG_FILE", table)

    def test_the_fast_mode_may_ship_its_own_guide(self):
        self.assertIn("fast_geometry", harness.INTERFACE_GUIDE_MODES)
        self.assertIn("geometry", harness.INTERFACE_GUIDE_MODES)
        for mode in ("legacy", "compact"):
            self.assertNotIn(mode, harness.INTERFACE_GUIDE_MODES)
        # Membership only says a guide is allowed. Whether it REPLACES CLAUDE.md is a
        # different fact, and it is the one that matters — see the assembly tests.
        self.assertEqual(harness.INTERFACE_REPLACEMENT_GUIDE_MODES,
                         ("fast_geometry", "coarse_fine_policy", "direct_geometry"))

    def test_the_coarse_fine_mode_has_its_own_log_variable(self):
        cfg = {"mcpServers": {"sphinx2": {"command": "spatial_interface/run_mcp.sh", "args": [],
                                          "env": {}}}}
        env = {"VIA_CONTROL_INTERFACE": "coarse_fine_policy",
               "VIA_COARSE_FINE_LOG_FILE": "/tmp/cf.jsonl",
               "VIA_FAST_GEOMETRY_LOG_FILE": "/tmp/fg.jsonl"}
        with patch.object(harness, "get_json_file", lambda _p: cfg), \
                patch.dict("os.environ", env):
            table = harness.codex_mcp_overrides(9100)[1]
        self.assertIn('VIA_COARSE_FINE_LOG_FILE="/tmp/cf.jsonl"', table)
        self.assertNotIn("VIA_FAST_GEOMETRY_LOG_FILE", table)
        self.assertIn(json.dumps(harness.enabled_tools_for_mode("coarse_fine_policy")),
                      table)


class AssembledGuideTests(unittest.TestCase):
    """What the agent is actually handed, read back off disk.

    The earlier test asserted only that `fast_geometry` was in
    INTERFACE_GUIDE_MODES, which is true of `geometry` as well and says nothing
    about whether the old manual-waypoint guide is still prepended. It was, so a
    fast run opened with a page describing execute_waypoint, gripper_toggle and the
    camera tools — none of which that surface answers. These tests read the
    assembled AGENTS.md instead of the table that decides it.
    """

    REPO = Path(harness.__file__).resolve().parent.parent
    TASK = REPO / "prompts/bowl_plate_min.md"
    # Sentences that only make sense with the frozen 18 tools.
    MANUAL_WAYPOINT_MARKERS = (
        "Repeat the following to complete a task, one waypoint at a time",
        "gripper_teleport_via_click",
        "execute_waypoint",
        "camera_reset",
        "gripper_toggle",
    )

    def assemble(self, mode, *, guide=None, task=None):
        """Assemble AGENTS.md for one mode and return its text."""
        env = {"VIA_CONTROL_INTERFACE": mode}
        if guide is not None:
            env["VIA_EXTRA_GUIDE"] = str(guide)
        with tempfile.TemporaryDirectory(prefix="via_assemble_") as home, \
                patch.dict("os.environ", env):
            cwd = harness.write_codex_agents_md(home, str(task or self.TASK))
            try:
                return (Path(cwd) / "AGENTS.md").read_text()
            finally:
                shutil.rmtree(cwd, ignore_errors=True)

    def test_the_fast_guide_replaces_the_manual_waypoint_guide(self):
        text = self.assemble("fast_geometry",
                             guide=self.REPO / "docs/FAST_GEOMETRY_GUIDE.md")
        for marker in self.MANUAL_WAYPOINT_MARKERS:
            self.assertNotIn(marker, text,
                             f"{marker!r} describes a tool this surface answers "
                             f"'Unknown tool' to")
        # And the guide it is supposed to carry IS there, with its six tools.
        self.assertIn("fg_bind", text)
        self.assertIn("fg_run", text)
        for tool in harness.enabled_tools_for_mode("fast_geometry"):
            self.assertIn(tool, text)
        self.assertIn("# Task-specific instructions", text)
        self.assertIn(self.TASK.read_text().strip(), text)

    def test_the_additive_modes_still_open_with_claude_md(self):
        legacy_md = (self.REPO / "CLAUDE.md").read_text()
        for mode, guide in (("legacy", None), ("compact", None),
                            ("geometry", self.REPO / "docs/GEOMETRY_GUIDE.md")):
            with self.subTest(mode=mode):
                text = self.assemble(mode, guide=guide)
                self.assertTrue(text.startswith(legacy_md),
                                f"{mode} must still be CLAUDE.md then its additions")
                # Byte for byte: these three surfaces did not change, so neither may
                # the guidance describing them.
                expected = [legacy_md]
                if guide is not None:
                    expected.append(guide.read_text())
                expected.append("# Task-specific instructions\n\n"
                                + self.TASK.read_text())
                self.assertEqual(text, "\n\n".join(expected))

    def test_the_coarse_fine_guide_also_replaces_the_manual_waypoint_guide(self):
        text = self.assemble("coarse_fine_policy",
                             guide=self.REPO / "docs/COARSE_FINE_GUIDE.md")
        for marker in self.MANUAL_WAYPOINT_MARKERS:
            self.assertNotIn(marker, text)
        for tool in harness.enabled_tools_for_mode("coarse_fine_policy"):
            self.assertIn(tool, text)
        self.assertIn("# Task-specific instructions", text)
        self.assertIn(self.TASK.read_text().strip(), text)

    def test_a_missing_coarse_fine_guide_fails_loudly(self):
        with self.assertRaises(ValueError) as caught:
            self.assemble("coarse_fine_policy")
        self.assertIn("VIA_EXTRA_GUIDE", str(caught.exception))

    def test_a_missing_fast_guide_fails_loudly_rather_than_shipping_nothing(self):
        # The failure mode this replaces: no guide variable in a replacement mode used
        # to assemble a task file alone, handing the agent six undocumented tools.
        with self.assertRaises(ValueError) as caught:
            self.assemble("fast_geometry")
        self.assertIn("VIA_EXTRA_GUIDE", str(caught.exception))
        with tempfile.NamedTemporaryFile("w", suffix=".md") as empty:
            empty.write("   \n")
            empty.flush()
            with self.assertRaises(ValueError):
                self.assemble("fast_geometry", guide=empty.name)
        with self.assertRaises(OSError):
            self.assemble("fast_geometry", guide="/nonexistent/fast_guide.md")

    def test_the_replacement_mode_does_not_read_claude_md_at_all(self):
        # Not merely "the markers are absent": CLAUDE.md's own title must be gone too,
        # so a future edit to that file cannot leak back into a fast run through some
        # section that happens not to name a tool.
        text = self.assemble("fast_geometry",
                             guide=self.REPO / "docs/FAST_GEOMETRY_GUIDE.md")
        first_line = (self.REPO / "CLAUDE.md").read_text().splitlines()[0]
        self.assertNotIn(first_line, text)


class EpisodeResetTests(unittest.IsolatedAsyncioTestCase):
    """end_episode must not leak one episode's geometry into the next.

    Before this wiring, both reset functions existed and neither had a call site,
    so a second episode in the same server process would have started holding a
    reference bound against the previous scene.
    """

    def setUp(self):
        fgt.reset_episode()
        server._geometry_reset_episode_state()

    def dirty_state(self):
        fgt.STATE.refs["bowl#1"] = {"anchor": [0.1, 0.2, 0.3]}
        fgt.STATE.attachment = {"offset": [0.0, 0.0, 0.01]}
        fgt.STATE.attachment_ref = "bowl#1"
        fgt.STATE.committed.append({"kind": "move"})
        server.GEOMETRY_REFERENCES["ref1"] = {"center": [0.0, 0.0, 0.0]}

    async def end(self, closed: bool):
        ctx = SimpleNamespace(
            page=SimpleNamespace(evaluate=AsyncMock(return_value={"ok": True})),
            timing_id="test", snap=AsyncMock(return_value=[]),
        )
        with patch.object(server, "_looks_closed", lambda *_a: closed), \
                patch.object(server, "terminal_reply", lambda *a, **k: []), \
                patch.object(server, "_fetch_sim_json", AsyncMock(return_value=None)):
            await server.EndEpisodeTool()(ctx, {})

    async def test_ending_an_episode_clears_both_interfaces_state(self):
        self.dirty_state()
        await self.end(closed=False)
        self.assertEqual(fgt.STATE.refs, {})
        self.assertIsNone(fgt.STATE.attachment)
        self.assertIsNone(fgt.STATE.attachment_ref)
        self.assertEqual(fgt.STATE.committed, [])
        self.assertEqual(dict(server.GEOMETRY_REFERENCES), {})

    async def test_an_already_closed_episode_still_clears_state(self):
        # The auto-close path ends the episode without ever clicking #btn-end. It
        # is the common case when a goal-satisfying waypoint closes the UI first.
        self.dirty_state()
        await self.end(closed=True)
        self.assertEqual(fgt.STATE.refs, {})
        self.assertIsNone(fgt.STATE.attachment_ref)


class ObservationPipelineTests(unittest.TestCase):
    def test_the_fast_mode_preserves_the_paired_idle_snapshot(self):
        preserve = record_sim.InteractiveBot._preserve_idle_snapshot_for_env
        for mode, expected in (("legacy", False), ("compact", False),
                               ("geometry", True), ("fast_geometry", True),
                               ("coarse_fine_policy", True)):
            with self.subTest(mode=mode), \
                    patch.dict("os.environ", {"VIA_CONTROL_INTERFACE": mode}):
                self.assertEqual(preserve(), expected)


if __name__ == "__main__":
    unittest.main()

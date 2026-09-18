"""The advertised bind shapes must be the ones the fitter actually accepts.

Regression for the grounded Qwen campaign of 2026-09-17, which recorded 17
``cf_policy`` bind rejections across its 21 episodes -- affected, not failed:
5 of the affected episodes still ended in success after the model worked around
the refusal.  The refusal originates in the interface, not in the request:
``cf_policy``'s schema described the field as
"Fit shape, e.g. 'blob', 'cylinder'" while ``fast_geometry.validate_bind`` has
only ever accepted ``fg.SHAPES``, so a caller following the advertised contract
was refused for doing so.  ``fg_bind`` had it right (``enum: list(fg.SHAPES)``);
only the policy tools carried a hand-written example, so the two drifted
silently.  What this says about anything else the campaign produced is out of
scope here.

These tests deliberately assert against the *advertised* schema as an MCP client
receives it — not against the source text — so that a future edit that reworded
the description, or added a fitter without exposing it, is caught as a contract
break rather than a string diff.
"""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from spatial_interface import coarse_fine_policy as cfp
from spatial_interface import coarse_fine_tools as cf
from spatial_interface import direct_control as dc
from spatial_interface import direct_geometry_tools as dg
from spatial_interface import fast_geometry as fg
from spatial_interface import geometry_workspace as gw
from spatial_interface import mcp_server as s

REGION = {"kind": "box", "u0": .2, "v0": .2, "u1": .6, "v1": .6}

# Shapes a model could plausibly infer from the old description or from ordinary
# geometric vocabulary.  None of them has a fitter; all must be refused.
UNSUPPORTED = ("cylinder", "cuboid", "box", "sphere", "")


def advertised_shape_schema(mode, tool_name):
    """The bind.shape subschema exactly as an MCP client would receive it."""
    with patch.dict("os.environ", VIA_CONTROL_INTERFACE=mode):
        tools = asyncio.run(s.list_tools())
    tool = next(t for t in tools if t.name == tool_name)
    return tool.inputSchema["properties"]["bind"]["properties"]["shape"]


class ShapeContract(unittest.TestCase):

    def setUp(self):
        """Both tool modules keep process-global state; start every test clean.

        Use each module's own reset rather than rebuilding State by hand, so this
        keeps working if either gains a field. ``dg.reset_episode`` rebinds the
        module global, so always read ``dg.STATE`` through the module.
        """
        cf.reset_episode()
        dg.reset_episode()

    def test_policy_tools_advertise_exactly_the_shapes_the_fitter_accepts(self):
        """The enum is the backend's own constant, for every policy surface."""
        for mode, tool_name in (("coarse_fine_policy", "cf_policy"),
                                ("direct_geometry", "dg_policy")):
            with self.subTest(tool=tool_name):
                schema = advertised_shape_schema(mode, tool_name)
                self.assertEqual(schema["enum"], list(fg.SHAPES))
                # The description must not contradict the enum it sits next to.
                for bogus in UNSUPPORTED:
                    if bogus:
                        self.assertNotIn(bogus, schema["description"])

    def test_every_advertised_shape_survives_bind_validation(self):
        """Advertising a shape is a promise that validate_bind will take it."""
        for mode, tool_name in (("coarse_fine_policy", "cf_policy"),
                                ("direct_geometry", "dg_policy")):
            for shape in advertised_shape_schema(mode, tool_name)["enum"]:
                with self.subTest(tool=tool_name, shape=shape):
                    normalized = cfp.validate_bind(
                        {"name": "target", "region": REGION, "shape": shape})
                    self.assertEqual(normalized["objects"][0]["shape"], shape)

    def test_omitted_shape_still_defaults_to_blob(self):
        normalized = cfp.validate_bind({"name": "target", "region": REGION})
        self.assertEqual(normalized["objects"][0]["shape"], "blob")

    def test_unsupported_shapes_are_refused_before_anything_moves(self):
        # fast_geometry.Rejected is a ValueError and is NOT a subclass of
        # coarse_fine_policy.Rejected; the two hierarchies are unrelated.
        for shape in UNSUPPORTED:
            with self.subTest(shape=shape):
                with self.assertRaises(fg.Rejected) as caught:
                    cfp.validate_bind(
                        {"name": "target", "region": REGION, "shape": shape})
                self.assertIn("must be one of", str(caught.exception))

    def test_a_cylinder_program_is_rejected_without_binding_or_actuating(self):
        """The campaign's exact failure: refused, zero actions, nothing bound."""
        dg.STATE.workspace.publish(
            gw.Observation("f1", "robot", ((0, 0, 1),), ("cloud",), 0))
        ctx = SimpleNamespace(timing_id="shape-contract", snap=AsyncMock(return_value=[]))
        with patch.object(dg.cf, "bind_proxy",
                          side_effect=AssertionError("bind ran despite a bad shape")), \
             patch.object(dc.RpcBackend, "execute",
                          side_effect=AssertionError("unexpected actuation")):
            result = asyncio.run(dg.DgPolicyTool()(ctx, {
                "bind": {"name": "target", "shape": "cylinder", "region": REGION},
                "steps": [{"observe": {}}]}))
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "rejected", payload)
        self.assertIn("shape", payload["reason"])
        self.assertEqual(payload["commands"], [])  # model view of `executed`
        self.assertEqual(dg.STATE.executed, [])
        self.assertNotIn("target", cf.STATE.proxies)

    def test_cf_policy_lets_a_bad_shape_escape_instead_of_reporting_it(self):
        """Documents a live asymmetry this change does NOT fix.

        ``dg_policy`` catches plain ``ValueError`` (direct_geometry_tools.py:309)
        and so turns a bad shape into ``status: rejected``.  ``cf_policy`` catches
        only ``cfp.Rejected`` (coarse_fine_tools.py:800), so ``fg.Rejected``
        propagates out of the tool as an unhandled exception instead of a refusal
        the model can read and correct.  Fixing the enum removes the trigger — no
        model is told 'cylinder' any more — but not this fault path, which any
        other invalid bind field would still reach.  Pinned here so the asymmetry
        is visible and the day it is fixed, this test fails loudly rather than
        the behaviour changing unnoticed.
        """
        ctx = SimpleNamespace(timing_id="shape-contract", snap=AsyncMock(return_value=[]))
        with patch.object(cf, "bind_proxy",
                          side_effect=AssertionError("bind ran despite a bad shape")), \
             patch.object(dc.RpcBackend, "execute",
                          side_effect=AssertionError("unexpected actuation")):
            with self.assertRaises(fg.Rejected):
                asyncio.run(cf.CfPolicyTool()(ctx, {
                    "bind": {"name": "target", "shape": "cylinder", "region": REGION},
                    "steps": [{"observe": {}}]}))
        self.assertNotIn("target", cf.STATE.proxies)


if __name__ == "__main__":
    unittest.main()

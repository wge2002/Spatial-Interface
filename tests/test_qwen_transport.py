"""Transport and lifecycle tests: no model keys, browser or simulator required."""
import asyncio
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_interface.qwen_agent import Agent, ModelOutputError, StreamReply, feedback_messages
from spatial_interface.experiment import classify_result as classify
from spatial_interface.ports import require_ports_free


def reply(name="look", args="{}", ident="call1"):
    r = StreamReply()
    for delta in ({"reasoning_content": "inspect"},
                  {"tool_calls": [{"index": 0, "id": ident, "function": {"name": name, "arguments": args[:1]}}]},
                  {"tool_calls": [{"index": 0, "function": {"arguments": args[1:]}}]}):
        r.add({"id": "response1", "choices": [{"delta": delta}]})
    r.add({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    r.add({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 4}})
    r.done = True
    return r


class Transport(unittest.TestCase):
    def test_fragmented_native_calls_and_usage(self):
        r = reply(args='{"x":0.35}')
        self.assertEqual(json.loads(r.complete({"look"})[0]["function"]["arguments"]), {"x": 0.35})
        self.assertEqual(r.reasoning, "inspect")
        self.assertEqual(r.usage["completion_tokens"], 4)

    def test_refuse_truncated_stream_or_partial_tool(self):
        r = reply()
        r.done = False
        with self.assertRaises(RuntimeError):
            r.complete({"look"})
        r.done = True
        r.finish_reason = "length"
        with self.assertRaises(ModelOutputError):
            r.complete({"look"})
        for args in ('{"x":', '[]', 'null'):
            with self.assertRaises(ModelOutputError):
                reply(args=args).complete({"look"})

    def test_unknown_and_duplicate_calls_rejected_before_execution(self):
        with self.assertRaises(ModelOutputError):
            reply(name="shell").complete({"look"})
        r = reply()
        r.calls[1] = copy.deepcopy(r.calls[0])
        with self.assertRaises(ModelOutputError):
            r.complete({"look"})

    def test_image_bytes_and_text_are_preserved(self):
        raw = b"original png bytes"
        encoded = base64.b64encode(raw).decode()
        message, images, evidence = feedback_messages("id9", [
            {"type": "text", "text": "metric x=0.00001\nraw"},
            {"type": "image", "mimeType": "image/png", "data": encoded}])
        self.assertIn("metric x=0.00001\nraw", message["content"])
        self.assertEqual(message["tool_call_id"], "id9")
        self.assertEqual(images[1]["image_url"]["url"], "data:image/png;base64," + encoded)
        self.assertEqual(evidence[0]["sha256"], hashlib.sha256(raw).hexdigest())

    def test_queue_never_masks_infrastructure_failure_with_saved_verdict(self):
        self.assertEqual(classify({"status": "ok", "success": True,
                                   "agent_status": "infrastructure_error"}), "infrastructure_error")
        self.assertEqual(classify({"status": "ok", "success": False,
                                   "agent_status": "model_output_invalid"}), "completed")
        self.assertEqual(classify({"status": "ok", "agent_status": "model_end_episode", "reused": True}),
                         "infrastructure_error")

    def test_claiming_busy_ports_fails_without_killing_owner(self):
        import socket
        with socket.socket() as owner:
            owner.bind(("0.0.0.0", 0))
            owner.listen()
            with self.assertRaises(RuntimeError):
                require_ports_free([owner.getsockname()[1]])
            self.assertGreater(owner.fileno(), 0)


class Lifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        key = root / "key"
        key.write_text("test-only-key")
        self.agent = Agent({"output": str(root / "output"), "timeout": 20,
            "server": {"key_file": str(key)}, "model": "fake", "seed": 1, "mode": "test",
            "demo_folder": str(root), "expected_tools": ["look", "end_episode"],
            "guide": "guide bytes", "prompt": "goal"})
        self.executed = []
        class Tool:
            def __init__(self, name): self.name = name
            def model_dump(self, **_): return {"name": self.name, "inputSchema": {"type": "object"}}
        async def listed(): return SimpleNamespace(tools=[Tool("look"), Tool("end_episode")])
        async def call(name, arguments):
            self.executed.append(name)
            data = {"content": [{"type": "text", "text": "observation"},
                {"type": "image", "mimeType": "image/png", "data": base64.b64encode(b"pixels").decode()}]}
            return SimpleNamespace(model_dump=lambda **_: data)
        self.session = SimpleNamespace(list_tools=listed, call_tool=call)

    async def asyncTearDown(self):
        self.agent.events.close()
        self.temp.cleanup()

    async def test_actual_loop_delivers_image_then_ends_without_extra_request(self):
        seen = []
        async def request(client, messages, tools):
            seen.append(copy.deepcopy(messages))
            r = reply("look" if len(seen) == 1 else "end_episode")
            return r, r.complete({"look", "end_episode"})
        self.agent.request = request
        self.assertEqual(await self.agent.loop(self.session, None), "model_end_episode")
        self.assertEqual(self.executed, ["look", "end_episode"])
        self.assertEqual(len(seen), 2)
        self.assertEqual([m["role"] for m in seen[1]], ["system", "user", "assistant", "tool", "user"])
        self.assertTrue(seen[1][-1]["content"][1]["image_url"]["url"].endswith("cGl4ZWxz"))

    async def test_terminal_call_suppresses_later_calls_in_same_response(self):
        async def request(*_):
            r = reply("end_episode")
            r.calls[1] = reply("look", ident="later").calls[0]
            return r, r.complete({"look", "end_episode"})
        self.agent.request = request
        await self.agent.loop(self.session, None)
        self.assertEqual(self.executed, ["end_episode"])

    async def test_uncertain_robot_ack_is_infrastructure_error_not_episode_timeout(self):
        async def request(*_):
            r = reply()
            return r, r.complete({"look"})
        async def call(*_):
            raise asyncio.TimeoutError()
        self.agent.request = request
        self.session.call_tool = call
        with self.assertRaisesRegex(RuntimeError, "uncertain_ack"):
            await self.agent.loop(self.session, None)

    async def test_surface_mismatch_precedes_any_model_request(self):
        self.agent.context["expected_tools"] = ["another_interface"]
        with self.assertRaisesRegex(RuntimeError, "surface mismatch"):
            await self.agent.loop(self.session, None)

    async def test_allowlist_order_does_not_change_live_tool_order(self):
        self.agent.context["expected_tools"] = ["end_episode", "look"]
        async def request(client, messages, tools):
            self.assertEqual([t["function"]["name"] for t in tools], ["look", "end_episode"])
            r = reply("end_episode")
            return r, r.complete({"look", "end_episode"})
        self.agent.request = request
        self.assertEqual(await self.agent.loop(self.session, None), "model_end_episode")


if __name__ == "__main__":
    unittest.main()

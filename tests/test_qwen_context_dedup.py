"""Lossless reference de-duplication of repeated images and observations.

Every case checks two things: the projection actually stops re-sending identical
payloads, and `expand_request_context` restores the original context exactly. A
test that only counted savings would pass on a lossy projection.
"""
import asyncio
import base64
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_interface.qwen_agent import (Agent, bound_image_history, dedupe_request_context,
                            expand_request_context, feedback_messages)


def observation(frame="e1-s5-c5", **overrides):
    """Roughly the size and shape of a real dg observation (~1.5 kB of numbers)."""
    view = {"what": "observation", "frame": frame, "cloud_frame": "e1-c5", "paired": True,
            "sim_steps_used": 778, "sim_steps_budget": 5000,
            "measured_end_effector": {
                "status": "ok", "source": "robot_proprioception", "frame": "robot",
                "fingertip_position": {"x": 0.03814650309709075, "y": 0.1451596125421057,
                                       "z": 0.968358460513443},
                "approach": {"x": -0.061160612266991696, "y": 0.016321328723498268,
                             "z": -0.9979944858243587},
                "opening": {"x": 0.9026489059831522, "y": 0.42765615440420734,
                            "z": -0.048323556653287164},
                "gripper_width_m": 0.07964179382159484, "gripper_state_class": "open",
                "gripper_state_note": "command state and measured width; neither proves a grasp"},
            "measured_gripper": "open",
            "local_depth": {"status": "ok", "source": "wrist_depth_reprojected",
                            "radius_m": 0.08, "support_points": 9535,
                            "residual_rms_m": 0.01611, "centroid": [0.0338, 0.1596, 1.01026],
                            "z_min": 0.95739, "z_max": 1.02573, "nearest_return_m": 0.0382,
                            "samples_xyz": [[0.06283 + n / 1000, 0.17418 - n / 1000,
                                             1.02264 + n / 10000] for n in range(18)]}}
    view.update(overrides)
    return view


def payload(program="p1", obs=None, **extra):
    """A dg_policy-shaped payload: the fields the projection must never touch."""
    body = {"status": "ok", "executed": [{"step": "translate", "ok": True}],
            "program_id": program, "arrival_checks": [{"name": "reached", "ok": True}],
            "observation": obs if obs is not None else observation()}
    body.update(extra)
    return body


def tool_message(call_id, blocks):
    message, images, _ = feedback_messages(call_id, blocks)
    return message, images


def text_block(value):
    return {"type": "text", "text": json.dumps(value) if isinstance(value, dict) else value}


def image_block(raw, mime="image/jpeg"):
    # Pad to a realistic screenshot size: a citation is only sent when it is
    # actually shorter than the image it replaces.
    return {"type": "image", "mimeType": mime, "data": base64.b64encode(raw.ljust(4096, b"\0")).decode()}


def conversation(groups):
    """Build [system, user, (assistant, tool.., user-images)..] like Agent.loop does."""
    messages = [{"role": "system", "content": "guide"}, {"role": "user", "content": "goal"}]
    for index, calls in enumerate(groups):
        ids = [f"call_{index}_{n}" for n in range(len(calls))]
        messages.append({"role": "assistant", "content": "plan",
                         "tool_calls": [{"id": i, "type": "function",
                                         "function": {"name": "dg_policy", "arguments": "{}"}}
                                        for i in ids]})
        images = []
        for call_id, blocks in zip(ids, calls):
            message, group = tool_message(call_id, blocks)
            messages.append(message)
            images.extend(group)
        if images:
            messages.append({"role": "user", "content": images})
    return messages


def images_in(messages):
    return [part["image_url"]["url"] for m in messages
            if isinstance(m.get("content"), list) for part in m["content"]
            if part.get("type") == "image_url"]


class Images(unittest.TestCase):
    def check_lossless(self, messages):
        original = copy.deepcopy(messages)
        sent, stats = dedupe_request_context(messages)
        self.assertEqual(messages, original, "the caller's history must not be mutated")
        self.assertEqual(expand_request_context(sent), original)
        return sent, stats

    def test_repeats_within_one_batch_of_calls_are_sent_once(self):
        scene, first, second = b"scene-pixels", b"overlay-1", b"overlay-2"
        messages = conversation([[
            [text_block(payload("a")), image_block(scene), image_block(first)],
            [text_block(payload("b")), image_block(scene), image_block(second)]]])
        sent, stats = self.check_lossless(messages)
        self.assertEqual((stats["images_total"], stats["images_unique"],
                          stats["images_referenced"]), (4, 3, 1))
        self.assertEqual(len(images_in(sent)), 3)
        # Both distinct overlays survive; only the second copy of the scene is cited.
        for raw in (scene, first, second):
            self.assertIn("data:image/jpeg;base64," + image_block(raw)["data"],
                          images_in(sent))

    def test_repeat_across_the_two_retained_groups_is_sent_once(self):
        scene = b"scene-pixels"
        messages = conversation([[[text_block(payload("a")), image_block(scene)]],
                                 [[text_block(payload("b")), image_block(scene)]]])
        sent, stats = self.check_lossless(messages)
        self.assertEqual(stats["images_referenced"], 1)
        self.assertEqual(len(images_in(sent)), 1)

    def test_surviving_copy_is_kept_in_full_once_the_first_is_pruned(self):
        scene = b"scene-pixels"
        messages = conversation([[[text_block(payload(str(n))), image_block(scene)]]
                                 for n in range(3)])
        bounded, removed = bound_image_history(messages, 2)
        self.assertEqual(removed, 1)
        sent, stats = self.check_lossless(bounded)
        # The pruned group can no longer be cited, so the oldest *surviving* copy
        # carries the bytes and nothing points at the dropped one.
        self.assertEqual((stats["images_total"], stats["images_referenced"]), (2, 1))
        self.assertEqual(len(images_in(sent)), 1)
        cited = [part["text"] for m in sent if isinstance(m.get("content"), list)
                 for part in m["content"] if "Identical image already sent" in part.get("text", "")]
        self.assertEqual(len(cited), 1)
        self.assertIn("tool_call_id=call_1_0", cited[0])
        self.assertNotIn("call_0_0", cited[0])

    def test_different_bytes_or_mime_are_never_merged(self):
        raw = b"scene-pixels"
        messages = conversation([[
            [text_block(payload("a")), image_block(raw, "image/jpeg")],
            [text_block(payload("b")), image_block(raw, "image/png")],
            [text_block(payload("c")), image_block(b"scene-pixels!", "image/jpeg")]]])
        sent, stats = self.check_lossless(messages)
        self.assertEqual((stats["images_total"], stats["images_unique"],
                          stats["images_referenced"]), (3, 3, 0))
        self.assertEqual(len(images_in(sent)), 3)

    def test_user_text_and_non_data_urls_are_left_alone(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "my own photo"},
            {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}},
            {"type": "text", "text": "my own photo"},
            {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}]}]
        sent, stats = self.check_lossless(messages)
        self.assertEqual((stats["images_total"], stats["images_referenced"]), (0, 0))
        self.assertEqual(sent, messages)

    def test_a_label_naming_two_different_images_is_never_used_as_an_anchor(self):
        """The label is model-visible text, so its uniqueness has to be checked."""
        scene = image_block(b"scene-pixels")["data"]
        other = image_block(b"other-pixels")["data"]
        def group(data):
            return [{"type": "text", "text": "Observation image from tool_call_id=x, block=1."},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + data}}]
        messages = [{"role": "user", "content": group(scene) + group(other) + group(scene)}]
        sent, stats = self.check_lossless(messages)
        # All three copies stay: the shared label cannot address the first alone.
        self.assertEqual((stats["images_total"], stats["images_unique"],
                          stats["images_referenced"]), (3, 2, 0))
        self.assertEqual(sent, messages)

    def test_a_unique_label_still_anchors_when_another_label_is_ambiguous(self):
        scene, other = image_block(b"scene")["data"], image_block(b"other")["data"]
        def group(label, data):
            return [{"type": "text", "text": label},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + data}}]
        messages = [{"role": "user", "content":
                     group("shared", scene) + group("shared", other)
                     + group("unique", scene) + group("also", scene)}]
        sent, stats = self.check_lossless(messages)
        # "shared" is ambiguous, so the scene is first anchored at "unique"; the
        # fourth copy cites that one and the two "shared" copies are left intact.
        self.assertEqual(stats["images_referenced"], 1)
        cited = [p["text"] for p in sent[0]["content"] if p["type"] == "text"
                 and "Identical image" in p["text"]]
        self.assertEqual(len(cited), 1)
        self.assertIn('under: "unique"', cited[0])

    def test_presentation_options_are_part_of_the_identity(self):
        data = "data:image/jpeg;base64," + image_block(b"scene-pixels")["data"]
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "low detail"},
            {"type": "image_url", "image_url": {"url": data, "detail": "low"}},
            {"type": "text", "text": "high detail"},
            {"type": "image_url", "image_url": {"url": data, "detail": "high"}},
            {"type": "text", "text": "low detail again"},
            {"type": "image_url", "image_url": {"url": data, "detail": "low"}}]}]
        sent, stats = self.check_lossless(messages)
        self.assertEqual((stats["images_total"], stats["images_unique"],
                          stats["images_referenced"]), (3, 2, 1))
        # The surviving parts keep their own detail setting untouched.
        self.assertEqual([p["image_url"]["detail"] for p in sent[0]["content"]
                          if p["type"] == "image_url"], ["low", "high"])


class Observations(unittest.TestCase):
    def check_lossless(self, messages):
        original = copy.deepcopy(messages)
        sent, stats = dedupe_request_context(messages)
        self.assertEqual(messages, original)
        self.assertEqual(expand_request_context(sent), original)
        return sent, stats

    def test_equal_observations_collapse_and_expand_field_for_field(self):
        shared = observation()
        messages = conversation([[[text_block(payload("a", shared))],
                                  [text_block(payload("b", shared))],
                                  [text_block(payload("c", shared))]]])
        sent, stats = self.check_lossless(messages)
        self.assertEqual((stats["observations_total"], stats["observations_unique"],
                          stats["observations_referenced"]), (3, 1, 2))
        bodies = [json.loads(m["content"]) for m in sent if m["role"] == "tool"]
        self.assertEqual(bodies[0]["observation"], shared)
        for body in bodies[1:]:
            self.assertEqual(body["observation"]["what"], "observation_repeat")
            self.assertEqual(body["observation"]["identical_to"],
                             {"tool_call_id": "call_0_0", "field": "observation",
                              "text_block": 0})
        # Non-observation fields, including the receipts, stay exactly as sent.
        for body, program in zip(bodies, "abc"):
            self.assertEqual(body["program_id"], program)
            self.assertEqual(body["executed"], [{"step": "translate", "ok": True}])
            self.assertEqual(body["arrival_checks"], [{"name": "reached", "ok": True}])
            self.assertEqual(body["status"], "ok")

    def test_same_frame_with_different_depth_or_samples_is_not_merged(self):
        base = observation()
        other_radius = observation()
        other_radius["local_depth"] = dict(base["local_depth"], radius_m=0.12)
        other_samples = observation()
        other_samples["local_depth"] = dict(base["local_depth"],
                                           samples_xyz=[[0.1, 0.2, 1.0], [0.9, 0.9, 0.9]])
        self.assertEqual(base["frame"], other_radius["frame"], "same frame id by construction")
        messages = conversation([[[text_block(payload("a", base))],
                                  [text_block(payload("b", other_radius))],
                                  [text_block(payload("c", other_samples))]]])
        sent, stats = self.check_lossless(messages)
        self.assertEqual((stats["observations_unique"], stats["observations_referenced"]), (3, 0))
        for body, expected in zip((json.loads(m["content"]) for m in sent if m["role"] == "tool"),
                                  (base, other_radius, other_samples)):
            self.assertEqual(body["observation"], expected)

    def test_last_observation_is_deduped_against_an_equal_observation(self):
        shared = observation()
        messages = conversation([[
            [text_block(payload("a", shared))],
            [text_block({"status": "ok", "policy_calls": 3, "last_observation": shared})]]])
        sent, stats = self.check_lossless(messages)
        self.assertEqual(stats["observations_referenced"], 1)
        state = json.loads([m for m in sent if m["role"] == "tool"][1]["content"])
        self.assertEqual(state["policy_calls"], 3)
        self.assertEqual(state["last_observation"]["identical_to"]["field"], "observation")

    def test_several_text_blocks_of_one_call_are_addressed_separately(self):
        shared = observation()
        other = observation(frame="e1-s9-c9")
        messages = conversation([[[text_block(payload("a", other)),
                                   text_block(payload("b", shared)),
                                   text_block(payload("c", shared))]]])
        sent, stats = self.check_lossless(messages)
        self.assertEqual((stats["observations_total"], stats["observations_unique"],
                          stats["observations_referenced"]), (3, 2, 1))
        blocks = [m for m in sent if m["role"] == "tool"][0]["content"].split("\n\n")
        self.assertEqual(len(blocks), 3)
        self.assertEqual(json.loads(blocks[2])["observation"]["identical_to"],
                         {"tool_call_id": "call_0_0", "field": "observation", "text_block": 1})

    def test_prose_and_malformed_json_blocks_are_preserved_verbatim(self):
        shared = observation()
        messages = conversation([[
            [text_block(payload("a", shared)), text_block("Saved screenshot to screenshots/x.jpg"),
             text_block('{"observation": {"what": "observation",'),
             text_block("[1, 2, 3]"), text_block('"just a json string"')],
            [text_block(payload("b", shared))]]])
        sent, stats = self.check_lossless(messages)
        self.assertEqual(stats["observations_referenced"], 1)
        blocks = [m for m in sent if m["role"] == "tool"][0]["content"].split("\n\n")
        self.assertEqual(blocks[1:], ["Saved screenshot to screenshots/x.jpg",
                                      '{"observation": {"what": "observation",',
                                      "[1, 2, 3]", '"just a json string"'])

    def test_non_canonical_payload_text_is_refused_rather_than_reformatted(self):
        shared = observation()
        indented = json.dumps(payload("a", shared), indent=2)
        messages = [{"role": "tool", "tool_call_id": "c1", "content": indented},
                    {"role": "tool", "tool_call_id": "c2",
                     "content": json.dumps(payload("b", shared))}]
        sent, stats = self.check_lossless(messages)
        self.assertEqual(stats["payloads_not_canonical"], 1)
        self.assertEqual(sent[0]["content"], indented)
        # c1 was never registered as a source, so c2 still carries the full values.
        self.assertEqual(json.loads(sent[1]["content"])["observation"], shared)
        self.assertEqual(stats["observations_referenced"], 0)

    def test_an_ambiguous_tool_call_id_is_never_used_as_a_reference(self):
        shared = observation()
        messages = [{"role": "tool", "tool_call_id": "dup",
                     "content": json.dumps(payload("a", shared))},
                    {"role": "tool", "tool_call_id": "dup",
                     "content": json.dumps(payload("b", shared))},
                    {"role": "tool", "tool_call_id": "fine",
                     "content": json.dumps(payload("c", shared))}]
        sent, stats = self.check_lossless(messages)
        self.assertEqual(stats["repeated_tool_call_ids"], ["dup"])
        self.assertEqual(stats["observations_referenced"], 0)
        for message, sent_message in zip(messages, sent):
            self.assertEqual(sent_message["content"], message["content"])

    def test_a_user_message_shaped_like_an_observation_is_not_rewritten(self):
        body = json.dumps(payload("a"))
        messages = [{"role": "user", "content": body},
                    {"role": "assistant", "content": body, "tool_calls": []},
                    {"role": "user", "content": body}]
        sent, stats = self.check_lossless(messages)
        self.assertEqual((stats["observations_total"], stats["observations_referenced"]), (0, 0))
        self.assertEqual(sent, messages)


class Accounting(unittest.TestCase):
    def test_saving_is_reported_in_characters_and_matches_the_projection(self):
        shared, scene = observation(), b"scene-pixels"
        messages = conversation([[[text_block(payload("a", shared)), image_block(scene)],
                                  [text_block(payload("b", shared)), image_block(scene)]]])
        sent, stats = dedupe_request_context(messages)
        before = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))
        after = len(json.dumps(sent, ensure_ascii=False, separators=(",", ":")))
        self.assertEqual((stats["context_chars_before"], stats["context_chars_after"]),
                         (before, after))
        self.assertEqual(stats["context_chars_saved"], before - after)
        self.assertGreater(stats["context_chars_saved"], 0)
        self.assertEqual(stats["images_referenced"], 1)
        self.assertEqual(stats["observations_referenced"], 1)


class Integration(unittest.IsolatedAsyncioTestCase):
    """Drive Agent.loop and Agent.request and read the payload actually written."""

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "key").write_text("test-only-key")
        self.scene = b"scene-pixels"
        self.observation = observation()
        self.agents = []

    async def asyncTearDown(self):
        for agent in self.agents:
            agent.events.close()
        self.temp.cleanup()

    def build(self, output, **extra):
        agent = Agent({"output": str(self.root / output), "timeout": 60, "model": "fake",
                       "seed": 1, "mode": "direct_geometry", "sampling": {},
                       "server": {"key_file": str(self.root / "key"),
                                  "base_url": "http://unused.invalid/v1"},
                       "demo_folder": str(self.root / output),
                       "expected_tools": ["dg_policy", "end_episode"],
                       "guide": "guide", "prompt": "goal", **extra})
        self.agents.append(agent)
        return agent

    def session(self, calls):
        """Two dg_policy calls per turn, both returning the same scene and observation."""
        class Tool:
            def __init__(self, name): self.name = name
            def model_dump(self, **_): return {"name": self.name, "inputSchema": {"type": "object"}}

        async def listed():
            return SimpleNamespace(tools=[Tool("dg_policy"), Tool("end_episode")])

        async def call_tool(name, arguments):
            calls.append(name)
            if name == "end_episode":
                return SimpleNamespace(model_dump=lambda **_: {"content": []})
            data = {"content": [text_block(payload(f"p{len(calls)}", self.observation)),
                                image_block(self.scene)]}
            return SimpleNamespace(model_dump=lambda **_: data)
        return SimpleNamespace(list_tools=listed, call_tool=call_tool)

    def stream(self, agent, turns):
        """Reply with two parallel dg_policy calls, then end_episode."""
        sent = []

        class Response:
            status_code = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def aiter_lines(inner):
                index = len(sent) - 1
                names = ["end_episode"] if index >= turns else ["dg_policy", "dg_policy"]
                deltas = [{"tool_calls": [{"index": n, "id": f"r{index}c{n}",
                                           "function": {"name": name, "arguments": "{}"}}]}
                          for n, name in enumerate(names)]
                for delta in deltas:
                    yield "data: " + json.dumps({"id": f"resp{index}",
                                                 "choices": [{"delta": delta}]})
                yield "data: " + json.dumps({"choices": [{"delta": {},
                                                          "finish_reason": "tool_calls"}]})
                yield "data: [DONE]"

        class Client:
            def stream(inner, method, url, content=None, headers=None):
                sent.append(json.loads(content))
                return Response()
        return Client(), sent

    async def test_sent_payload_is_deduped_while_raw_events_keep_every_byte(self):
        calls = []
        agent = self.build("on")
        client, sent = self.stream(agent, turns=2)
        self.assertEqual(await agent.loop(self.session(calls), client), "model_end_episode")
        self.assertEqual(calls, ["dg_policy", "dg_policy", "dg_policy", "dg_policy",
                                 "end_episode"])
        self.assertEqual(len(sent), 3)

        # The last request carried four identical observations and four identical
        # images; exactly one copy of each went on the wire.
        last = sent[-1]["messages"]
        on_disk = json.loads((Path(agent.output) / "request_003" / "request.json").read_text())
        self.assertEqual(on_disk["messages"], last)
        self.assertEqual(len(images_in(last)), 1)
        bodies = [json.loads(m["content"].split("\n\n")[0]) for m in last if m["role"] == "tool"]
        self.assertEqual(len(bodies), 4)
        self.assertEqual(bodies[0]["observation"], self.observation)
        self.assertEqual([b["observation"]["what"] for b in bodies[1:]],
                         ["observation_repeat"] * 3)
        self.assertEqual([b["program_id"] for b in bodies], ["p1", "p2", "p3", "p4"])

        # The projection is reversible and the recorded stats describe it.
        stats = json.loads((Path(agent.output) / "request_003" / "timing.json").read_text())
        self.assertEqual(stats["context_dedup"]["observations_referenced"], 3)
        self.assertEqual(stats["context_dedup"]["images_referenced"], 3)
        self.assertGreater(stats["context_dedup"]["context_chars_saved"], 0)
        events = [json.loads(line) for line in
                  (Path(agent.output) / "events.jsonl").read_text().splitlines()]
        dedup_events = [e for e in events if e["type"] == "context_dedup"]
        self.assertEqual([e["sequence"] for e in dedup_events], [2, 3])

        # Raw tool events and image-delivery evidence keep the full original bytes.
        encoded = image_block(self.scene)["data"]
        finished = [e for e in events if e["type"] == "tool_finished"
                    and e["tool_call"]["function"]["name"] == "dg_policy"]
        self.assertEqual(len(finished), 4)
        for event in finished:
            blocks = event["result"]["content"]
            self.assertEqual(blocks[1]["data"], encoded)
            self.assertEqual(json.loads(blocks[0]["text"])["observation"], self.observation)
        delivery = [e for e in events if e["type"] == "image_delivery"]
        self.assertEqual(len({e["images"][0]["sha256"] for e in delivery}), 1)
        self.assertEqual([e["images"][0]["bytes"] for e in delivery],
                         [len(base64.b64decode(encoded))] * 4)

    async def test_disabling_the_projection_restores_the_previous_payload(self):
        calls = []
        agent = self.build("off", context_dedup=False)
        client, sent = self.stream(agent, turns=2)
        await agent.loop(self.session(calls), client)
        self.assertEqual(len(images_in(sent[-1]["messages"])), 4)
        self.assertFalse(agent.summary["context_dedup"])
        events = [json.loads(line) for line in
                  (Path(agent.output) / "events.jsonl").read_text().splitlines()]
        self.assertEqual([e for e in events if e["type"] == "context_dedup"], [])
        stats = json.loads((Path(agent.output) / "request_003" / "timing.json").read_text())
        self.assertIsNone(stats["context_dedup"])

    async def test_a_second_episode_starts_with_no_carried_over_references(self):
        """Each Agent, and each request, cites only what that request still holds."""
        first, second = [], []
        agent_a = self.build("ep1")
        client_a, sent_a = self.stream(agent_a, turns=1)
        await agent_a.loop(self.session(first), client_a)
        agent_b = self.build("ep2")
        client_b, sent_b = self.stream(agent_b, turns=1)
        await agent_b.loop(self.session(second), client_b)
        for sent in (sent_a, sent_b):
            payload_messages = sent[-1]["messages"]
            # Same bytes as the other episode, yet this request still sends a copy.
            self.assertEqual(len(images_in(payload_messages)), 1)
            restored = expand_request_context(payload_messages)
            bodies = [json.loads(m["content"].split("\n\n")[0])
                      for m in restored if m["role"] == "tool"]
            self.assertEqual(len(bodies), 2)
            self.assertEqual([b["observation"] for b in bodies], [self.observation] * 2)
            self.assertEqual(len(images_in(restored)), 2)

    async def test_invalid_switch_is_refused_before_any_request(self):
        with self.assertRaises(ValueError):
            self.build("bad", context_dedup="yes")


if __name__ == "__main__":
    unittest.main()

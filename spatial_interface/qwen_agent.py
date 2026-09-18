"""Qwen Chat Completions client for the existing robot MCP, with raw evidence."""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time


class ModelOutputError(ValueError):
    pass


class ContextBudgetError(RuntimeError):
    """The request cannot fit; no new robot command has been submitted."""


def bound_image_history(messages, keep):
    """Retain recent image-message groups; keep all text and tool associations.

    Original images remain in the immutable tool events and past requests. This
    only changes the explicitly configured next-request context, never its logs.
    """
    if keep is None:
        return messages, 0
    if type(keep) is not int or keep < 1:
        raise ValueError("image_history_messages must be a positive integer")
    groups = [i for i, m in enumerate(messages) if isinstance(m.get("content"), list)
              and any(p.get("type") == "image_url" for p in m["content"])]
    old = set(groups[:-keep])
    result, removed = [], 0
    for i, message in enumerate(messages):
        if i not in old:
            result.append(message)
            continue
        content = []
        for part in message["content"]:
            if part.get("type") == "image_url":
                removed += 1
                content.append({"type": "text", "text":
                    "[Older observation image omitted from this request by the fixed "
                    "image-history limit; original evidence is preserved in the run log.]"})
            else:
                content.append(part)
        result.append(dict(message, content=content))
    return result, removed


IMAGE_REPEAT_TEXT = ('[Identical image already sent in this request under: "{label}" '
                     'Same bytes, not repeated; sha256 {digest}.]')
OBSERVATION_REPEAT_NOTE = ("identical full content to the observation named in identical_to, "
                          "which is still present in this request; read it there. Nothing was "
                          "dropped, rounded or summarized")
DEDUPED_OBSERVATION_FIELDS = ("observation", "last_observation")


def _data_url_identity(url):
    """Identify inline image data by its real decoded bytes, not by its frame or text."""
    if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
        return None
    head, encoded = url.split(";base64,", 1)
    try:
        blob = base64.b64decode(encoded, validate=True)
    except ValueError:
        return None
    return head[len("data:"):], hashlib.sha256(blob).hexdigest()


def _image_identity(part):
    """Two image parts share an identity only if every byte AND every presentation
    option matches: same mime, same pixels, same `detail` and any other key.
    """
    if part.get("type") != "image_url" or not isinstance(part.get("image_url"), dict):
        return None
    identity = _data_url_identity(part["image_url"].get("url"))
    if identity is None:
        return None
    skeleton = dict(part, image_url=dict(part["image_url"], url="%s#%s" % identity))
    return json.dumps(skeleton, sort_keys=True, separators=(",", ":")), identity


def _image_anchors(messages):
    """Label text that addresses exactly one image part in this request.

    The label in front of an image names its tool_call_id and content block, but
    nothing guarantees a model-visible string is unique -- a reused tool_call_id
    or a repeated label would make a citation point at two different images. Only
    labels that resolve to a single image part are usable, in either direction.
    """
    found = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            previous = content[index - 1] if index else None
            if (_image_identity(part) is None or not previous
                    or previous.get("type") != "text"):
                continue
            label = previous["text"]
            if label in found and found[label] != part:
                found[label] = None  # ambiguous: leave every copy under it intact
            else:
                found.setdefault(label, part)
    return {label: part for label, part in found.items() if part is not None}


def _repeated_tool_call_ids(messages):
    """A tool_call_id used twice cannot address one copy, so never reference it."""
    seen, repeated = set(), set()
    for message in messages:
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        (repeated if call_id in seen else seen).add(call_id)
    return repeated


def _dedupe_images(message, kept, anchors, stats):
    """Send each distinct image once; later copies cite the copy still in this request."""
    parts, result, changed = message["content"], [], False
    for index, part in enumerate(parts):
        identified = _image_identity(part)
        if identified is None:
            result.append(part)
            continue
        key, (mime, digest) = identified
        stats["images_total"] += 1
        stats["_image_keys"].add(key)
        previous = parts[index - 1] if index else None
        label = previous["text"] if previous and previous.get("type") == "text" else None
        source = kept.get(key)
        if source is None:
            # An image is citable only through a label that addresses it alone; the
            # label block naming its tool_call_id stays in front of every copy.
            if label is not None and anchors.get(label) == part:
                kept[key] = label
                stats["images"].append({"mime_type": mime, "sha256": digest,
                                        "label": label, "repeats": 0})
            result.append(part)
            continue
        citation = IMAGE_REPEAT_TEXT.format(label=source, digest=digest)
        if len(citation) >= len(part["image_url"]["url"]):
            result.append(part)  # a citation longer than the image saves nothing
            continue
        result.append({"type": "text", "text": citation})
        stats["images_referenced"] += 1
        changed = True
        for entry in stats["images"]:
            if entry["label"] == source:
                entry["repeats"] += 1
    return dict(message, content=result) if changed else message


def _dedupe_observations(message, kept, repeated_ids, stats):
    """Replace a byte-identical observation object with a citation of the first copy.

    Only whole observation objects of a known tool payload shape are touched, and
    only when re-serializing the payload reproduces the server's own text exactly,
    so every other field (executed, arrival_checks, proxies, ...) is untouched.
    """
    call_id = message.get("tool_call_id")
    parts = message["content"].split("\n\n")
    result, changed = list(parts), False
    for index, part in enumerate(parts):
        try:
            payload = json.loads(part)
        except ValueError:
            continue  # ordinary prose or a non-JSON block: keep it verbatim
        if not isinstance(payload, dict) or json.dumps(payload) != part:
            stats["payloads_not_canonical"] += 1
            continue
        updated, referenced = dict(payload), 0
        for field in DEDUPED_OBSERVATION_FIELDS:
            value = payload.get(field)
            if not isinstance(value, dict) or value.get("what") != "observation":
                continue
            stats["observations_total"] += 1
            key = json.dumps(value, sort_keys=True, separators=(",", ":"))
            stats["_observation_keys"].add(key)
            source = kept.get(key)
            if source is None:
                if call_id is not None and call_id not in repeated_ids:
                    kept[key] = {"tool_call_id": call_id, "field": field, "text_block": index}
                continue
            updated[field] = {"what": "observation_repeat", "identical_to": source,
                              "note": OBSERVATION_REPEAT_NOTE}
            referenced += 1
        if referenced:
            replacement = json.dumps(updated)
            if len(replacement) >= len(part):
                continue  # a citation no shorter than the observation saves nothing
            stats["observations_referenced"] += referenced
            result[index] = replacement
            changed = True
    return dict(message, content="\n\n".join(result)) if changed else message


def context_size(messages):
    """Characters of the serialized context. Characters, deliberately not tokens."""
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))


def dedupe_request_context(messages):
    """Project one request so exactly equal images and observations are sent once.

    Purely model-facing and request-local: the caller's own history, the tool
    events, the image-delivery evidence and every recorded action and receipt keep
    all of their original bytes. A repeat only ever cites a copy that is still
    present in the SAME projection, so no reference can dangle and no state
    carries across requests or episodes.
    """
    stats = {"images_total": 0, "images_referenced": 0, "observations_total": 0,
             "observations_referenced": 0, "payloads_not_canonical": 0, "images": [],
             "_image_keys": set(), "_observation_keys": set()}
    repeated_ids = _repeated_tool_call_ids(messages)
    anchors = _image_anchors(messages)
    kept_images, kept_observations, result = {}, {}, []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            result.append(_dedupe_observations(message, kept_observations, repeated_ids, stats))
        elif isinstance(content, list):
            result.append(_dedupe_images(message, kept_images, anchors, stats))
        else:
            result.append(message)
    stats["images_unique"] = len(stats.pop("_image_keys"))
    stats["observations_unique"] = len(stats.pop("_observation_keys"))
    stats["repeated_tool_call_ids"] = sorted(i for i in repeated_ids if i is not None)
    stats["context_chars_before"] = context_size(messages)
    stats["context_chars_after"] = context_size(result)
    stats["context_chars_saved"] = stats["context_chars_before"] - stats["context_chars_after"]
    return result, stats


def _image_citation(text):
    prefix, middle = IMAGE_REPEAT_TEXT.split("{label}")[0], IMAGE_REPEAT_TEXT.split("{label}")[1]
    middle = middle.split("{digest}")[0]
    if not (text.startswith(prefix) and text.endswith(".]") and middle in text):
        return None
    label, digest = text[len(prefix):-len(".]")].rsplit(middle, 1)
    return label, digest


def expand_request_context(messages):
    """Invert dedupe_request_context, proving the projection dropped nothing.

    Every citation must resolve inside the same projection; an unresolvable or
    mismatching one raises instead of silently producing a plausible context.
    """
    images, observations = _image_anchors(messages), {}
    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            for index, part in enumerate(content.split("\n\n")):
                try:
                    payload = json.loads(part)
                except ValueError:
                    continue
                if not isinstance(payload, dict):
                    continue
                for field in DEDUPED_OBSERVATION_FIELDS:
                    value = payload.get(field)
                    if not isinstance(value, dict) or value.get("what") != "observation":
                        continue
                    address = (message.get("tool_call_id"), field, index)
                    if address in observations and observations[address] != value:
                        observations[address] = None  # ambiguous address, never citable
                    else:
                        observations.setdefault(address, value)
    result = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            parts = content.split("\n\n")
            for index, part in enumerate(parts):
                try:
                    payload = json.loads(part)
                except ValueError:
                    continue
                if not isinstance(payload, dict) or json.dumps(payload) != part:
                    continue
                restored, changed = dict(payload), False
                for field in DEDUPED_OBSERVATION_FIELDS:
                    value = payload.get(field)
                    if not isinstance(value, dict) or value.get("what") != "observation_repeat":
                        continue
                    source = value["identical_to"]
                    address = (source["tool_call_id"], source["field"], source["text_block"])
                    if observations.get(address) is None:
                        raise ValueError(f"observation citation {address} does not resolve")
                    restored[field] = observations[address]
                    changed = True
                if changed:
                    parts[index] = json.dumps(restored)
            result.append(dict(message, content="\n\n".join(parts)))
        elif isinstance(content, list):
            parts = []
            for part in content:
                citation = (_image_citation(part["text"])
                            if part.get("type") == "text" else None)
                if citation is None:
                    parts.append(part)
                    continue
                label, digest = citation
                if label not in images:
                    raise ValueError(f"image citation {label!r} does not resolve to one image")
                source = images[label]
                if _data_url_identity(source["image_url"]["url"])[1] != digest:
                    raise ValueError(f"cited image {label!r} does not match sha256 {digest}")
                parts.append(source)
            result.append(dict(message, content=parts))
        else:
            result.append(message)
    return result


def exception_details(exc):
    """Keep nested transport errors visible instead of only TaskGroup wrappers."""
    children = getattr(exc, "exceptions", None)
    if children:
        return "; ".join(exception_details(child) for child in children)
    return type(exc).__name__ + ": " + str(exc)


class StreamReply:
    def __init__(self):
        self.content = ""
        self.reasoning = ""
        self.calls = {}
        self.finish_reason = None
        self.usage = None
        self.response_id = None
        self.done = False

    def add(self, event):
        if event.get("error"):
            raise RuntimeError(str(event["error"]))
        self.response_id = event.get("id") or self.response_id
        self.usage = event.get("usage") or self.usage
        for choice in event.get("choices", []):
            self.finish_reason = choice.get("finish_reason") or self.finish_reason
            delta = choice.get("delta") or {}
            self.content += delta.get("content") or ""
            self.reasoning += delta.get("reasoning_content") or ""
            for part in delta.get("tool_calls") or []:
                call = self.calls.setdefault(part["index"], {
                    "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                call["id"] += part.get("id") or ""
                for field in ("name", "arguments"):
                    call["function"][field] += (part.get("function") or {}).get(field) or ""

    def complete(self, allowed):
        if not self.done or not self.finish_reason:
            raise RuntimeError("SSE ended without a complete terminal event")
        if self.finish_reason == "length":
            raise ModelOutputError("generation_length_limit")
        calls = [self.calls[k] for k in sorted(self.calls)]
        seen = set()
        for call in calls:
            if not call["id"] or call["id"] in seen:
                raise ModelOutputError("missing_or_duplicate_tool_call_id")
            seen.add(call["id"])
            if call["function"]["name"] not in allowed:
                raise ModelOutputError("tool_not_in_current_interface")
            try:
                args = json.loads(call["function"]["arguments"])
            except ValueError as exc:
                raise ModelOutputError("malformed_tool_arguments") from exc
            if not isinstance(args, dict):
                raise ModelOutputError("tool_arguments_are_not_an_object")
        return calls

    def record(self):
        return {"id": self.response_id, "content": self.content,
                "reasoning_content": self.reasoning,
                "tool_calls": [self.calls[k] for k in sorted(self.calls)],
                "finish_reason": self.finish_reason, "usage": self.usage, "done": self.done}


def feedback_messages(call_id, blocks):
    """Keep original text verbatim; carry image bytes in associated user blocks."""
    text, images, evidence = [], [], []
    for index, block in enumerate(blocks):
        kind = block.get("type")
        if kind == "text":
            text.append(block["text"])
        elif kind == "image":
            blob = base64.b64decode(block["data"], validate=True)
            digest = hashlib.sha256(blob).hexdigest()
            label = f"Observation image from tool_call_id={call_id}, content block={index}."
            text.append(f"[Image content block {index} follows in the next observation message.]")
            images += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {
                "url": f"data:{block['mimeType']};base64,{block['data']}"}}]
            evidence.append({"tool_call_id": call_id, "block": index, "sha256": digest,
                             "mime_type": block["mimeType"], "bytes": len(blob)})
        else:
            raise RuntimeError(f"Unsupported MCP result block: {kind!r}")
    return {"role": "tool", "tool_call_id": call_id, "content": "\n\n".join(text)}, images, evidence


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class Agent:
    def __init__(self, context):
        self.context = context
        self.output = Path(context["output"])
        self.output.mkdir(exist_ok=False)
        self.started = time.monotonic()
        self.deadline = self.started + context["timeout"]
        self.requests = []
        self.tool_count = 0
        self.key = Path(context["server"]["key_file"]).read_text().strip()
        self.events = (self.output / "events.jsonl").open("x")
        self.terminal = False
        self.summary = {"status": "starting", "model": context["model"], "seed": context["seed"],
                        "mode": context["mode"], "requests": self.requests}
        self.image_history_messages = context.get("image_history_messages")
        bound_image_history([], self.image_history_messages)
        self.summary["image_history_messages"] = self.image_history_messages
        self.context_dedup = context.get("context_dedup", True)
        if not isinstance(self.context_dedup, bool):
            raise ValueError("context_dedup must be a boolean")
        self.summary["context_dedup"] = self.context_dedup

    def event(self, kind, **data):
        value = {"type": kind, "elapsed_s": time.monotonic() - self.started, **data}
        self.events.write(json.dumps(value, ensure_ascii=False) + "\n")
        self.events.flush()

    def environment_ended(self):
        # Only detect the normal terminal lifecycle; never send saved GT to Qwen.
        return (Path(self.context["demo_folder"]) / "verdict.json").exists()

    async def request(self, client, messages, tools, dedup=None):
        number = len(self.requests) + 1
        directory = self.output / f"request_{number:03d}"
        directory.mkdir(exist_ok=False)
        settings = self.context["sampling"]
        # Project here so request.json is always exactly the bytes that were sent.
        if self.context_dedup:
            messages, dedup = dedupe_request_context(messages)
            if dedup["images_referenced"] or dedup["observations_referenced"]:
                self.event("context_dedup", sequence=number, **dedup)
        payload = {"model": self.context["model"], "messages": messages, "tools": tools,
                   "tool_choice": "auto", "parallel_tool_calls": False,
                   "stream": True, "stream_options": {"include_usage": True}, **settings}
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        (directory / "request.json").write_bytes(body)
        record = {"sequence": number, "path": str(directory),
                  "request_sha256": hashlib.sha256(body).hexdigest(), "usage": None,
                  "context_dedup": dedup}
        self.requests.append(record)
        self.event("request_started", sequence=number, path=str(directory))
        reply = StreamReply()
        started = time.monotonic()
        try:
            async with client.stream("POST", self.context["server"]["base_url"].rstrip("/") + "/chat/completions",
                                     content=body, headers={"Authorization": "Bearer " + self.key,
                                                            "Content-Type": "application/json"}) as response:
                record.update(http_status=response.status_code, headers_s=time.monotonic() - started)
                if response.status_code != 200:
                    error = (await response.aread()).decode(errors="replace")
                    if response.status_code == 400 and "maximum context length" in error:
                        raise ContextBudgetError(error[:4000])
                    raise RuntimeError(f"HTTP {response.status_code}: {error[:4000]}")
                with (directory / "response.sse").open("x") as raw:
                    async for line in response.aiter_lines():
                        raw.write(line + "\n")
                        raw.flush()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            reply.done = True
                            break
                        event = json.loads(data)
                        record.setdefault("first_event_s", time.monotonic() - started)
                        reply.add(event)
                        if reply.reasoning:
                            record.setdefault("first_reasoning_s", time.monotonic() - started)
                        if reply.content.strip():
                            record.setdefault("first_content_s", time.monotonic() - started)
                        if reply.calls:
                            record.setdefault("first_tool_delta_s", time.monotonic() - started)
            calls = reply.complete({t["function"]["name"] for t in tools})
            return reply, calls
        except BaseException as exc:
            record["error"] = (type(exc).__name__ + ": " + str(exc)).replace(self.key, "<redacted>")
            raise
        finally:
            record.update(wall_s=time.monotonic() - started, usage=reply.usage,
                          response_id=reply.response_id, finish_reason=reply.finish_reason)
            write_json(directory / "response.json", reply.record())
            write_json(directory / "timing.json", record)
            self.event("request_finished", **record)

    async def loop(self, session, client):
        listed = await session.list_tools()
        live = [t.model_dump(mode="json", exclude_none=True) for t in listed.tools]
        names = [t["name"] for t in live]
        # The historical Codex enabled-tools allowlist has a different order
        # from the legacy MCP list. Check membership/multiplicity, and retain
        # the live MCP order when presenting the actual schemas to Qwen.
        if sorted(names) != sorted(self.context["expected_tools"]):
            raise RuntimeError(f"Tool surface mismatch: {names}")
        write_json(self.output / "mcp_tools.json", live)
        tools = [{"type": "function", "function": {"name": t["name"],
                  "description": t.get("description", ""), "parameters": t["inputSchema"]}} for t in live]
        messages = [{"role": "system", "content": self.context["guide"]},
                    {"role": "user", "content": self.context["prompt"]}]
        while not self.terminal and not self.environment_ended():
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            messages, removed = bound_image_history(messages, self.image_history_messages)
            if removed:
                self.event("image_history_bounded", removed_images=removed,
                           retained_image_message_limit=self.image_history_messages)
            # `request` projects this list for the wire; the history keeps its own
            # full copies, so a citation can never outlive the copy it points at.
            reply, calls = await asyncio.wait_for(self.request(client, messages, tools), remaining)
            if self.environment_ended():
                return "environment_terminal"
            if not calls:
                return "model_stopped"
            messages.append({"role": "assistant", "content": reply.content or None, "tool_calls": calls})
            observation_images = []
            for call in calls:
                if self.environment_ended() or self.terminal:
                    self.event("tool_skipped_after_terminal", tool_call=call)
                    continue
                name = call["function"]["name"]
                arguments = json.loads(call["function"]["arguments"])
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                self.tool_count += 1
                self.event("tool_started", sequence=self.tool_count, tool_call=call, origin="model")
                started = time.monotonic()
                try:
                    result = await asyncio.wait_for(session.call_tool(name, arguments),
                        min(180, remaining))
                except asyncio.TimeoutError as exc:
                    # A timed-out robot command may still be executing. Never send
                    # another action (including end_episode) with an unknown ack.
                    raise RuntimeError("robot_tool_timeout_with_uncertain_ack") from exc
                raw = result.model_dump(mode="json", exclude_none=True)
                self.event("tool_finished", sequence=self.tool_count, tool_call=call,
                           result=raw, wall_s=time.monotonic() - started, origin="model")
                if name == "end_episode":
                    if raw.get("isError"):
                        raise RuntimeError("end_episode returned an MCP error")
                    self.terminal = True
                    continue
                message, images, evidence = feedback_messages(call["id"], raw.get("content", []))
                messages.append(message)
                observation_images.extend(images)
                self.event("image_delivery", images=evidence)
            if observation_images and not self.terminal:
                messages.append({"role": "user", "content": observation_images})
        return "model_end_episode" if self.terminal else "environment_terminal"

    async def run(self):
        import httpx
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        status = "infrastructure_error"
        try:
            env = {k: v for k, v in os.environ.items()
                   if not any(s in k.upper() for s in ("API_KEY", "ACCESS_TOKEN", "PASSWORD", "SECRET"))}
            server = StdioServerParameters(command=sys.executable,
                args=[str(Path(__file__).resolve().parent / "mcp_server.py")], env=env)
            with (self.output / "mcp.stderr").open("x") as stderr:
                async with stdio_client(server, errlog=stderr) as (reader, writer):
                    async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=180)) as session:
                        await asyncio.wait_for(session.initialize(), 90)
                        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10),
                                                     follow_redirects=False, trust_env=False) as client:
                            try:
                                status = await self.loop(session, client)
                            except ModelOutputError as exc:
                                status = "model_output_invalid"
                                self.summary["error"] = str(exc)
                            except asyncio.TimeoutError:
                                status = "timeout"
                            except ContextBudgetError as exc:
                                status = "context_budget_exhausted"
                                self.summary["error"] = str(exc).replace(self.key, "<redacted>")
                            # Save an abandoned model episode without another model call.
                            # This is recording cleanup, explicitly not a model action.
                            if status in {"model_stopped", "model_output_invalid", "timeout", "context_budget_exhausted"} and not self.environment_ended():
                                self.event("cleanup_end_started", origin="harness_cleanup")
                                try:
                                    result = await asyncio.wait_for(session.call_tool("end_episode", {}), 30)
                                    self.event("cleanup_end_finished", origin="harness_cleanup",
                                               result=result.model_dump(mode="json", exclude_none=True))
                                except Exception as exc:
                                    self.event("cleanup_end_error", error=str(exc).replace(self.key, "<redacted>"))
        except BaseException as exc:
            self.summary["error"] = exception_details(exc).replace(self.key, "<redacted>")
            status = "infrastructure_error"
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                status = "interrupted"
        finally:
            self.summary.update(status=status, wall_s=time.monotonic() - self.started,
                                model_requests=len(self.requests), model_tool_calls=self.tool_count)
            write_json(self.output / "summary.json", self.summary)
            self.event("agent_finished", status=status)
            self.events.close()
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context", type=Path)
    args = parser.parse_args()
    context = json.loads(args.context.read_text())
    status = asyncio.run(Agent(context).run())
    return 2 if status in {"infrastructure_error", "interrupted"} else 0


if __name__ == "__main__":
    raise SystemExit(main())

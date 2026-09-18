"""Register the opt-in Qwen harness without changing historical model backends."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from spatial_interface.agent_common import register_agent, unregister_agent, terminate
from spatial_interface.codex_harness import write_codex_agents_md, enabled_tools_for_mode
from spatial_interface.harness import Harness, DriveOutcome, register_harness


class QwenHarness(Harness):
    key = "qwen"
    guide_filename = "AGENTS.md"
    supported_efforts = frozenset({"low"})

    def drive(self, ctx):
        config = json.loads(Path(os.environ["VIA_QWEN_SERVER_CONFIG"]).read_text())
        guide_dir = Path(write_codex_agents_md(ctx.demo_folder, ctx.instruct_file))
        try:
            guide = (guide_dir / "AGENTS.md").read_text()
        finally:
            shutil.rmtree(guide_dir)
        demo = Path(ctx.demo_folder)
        output = demo / "qwen"
        mode = ctx.env.get("VIA_CONTROL_INTERFACE", "legacy")
        context = {"server": config["server"], "sampling": config["sampling"],
                   "model": ctx.model, "seed": ctx.seed, "mode": mode,
                   "expected_tools": list(enabled_tools_for_mode(mode)),
                   "guide": guide, "prompt": ctx.prompt, "timeout": ctx.timeout,
                   "demo_folder": str(demo), "output": str(output)}
        if os.environ.get("VIA_QWEN_IMAGE_HISTORY_MESSAGES") is not None:
            context["image_history_messages"] = int(os.environ["VIA_QWEN_IMAGE_HISTORY_MESSAGES"])
            if context["image_history_messages"] < 1:
                raise ValueError("VIA_QWEN_IMAGE_HISTORY_MESSAGES must be positive")
        # On by default; the opt-out exists so a frozen protocol can be replayed.
        if os.environ.get("VIA_QWEN_CONTEXT_DEDUP") is not None:
            value = os.environ["VIA_QWEN_CONTEXT_DEDUP"]
            if value not in {"0", "1"}:
                raise ValueError("VIA_QWEN_CONTEXT_DEDUP must be 0 or 1")
            context["context_dedup"] = value == "1"
        path = demo / "qwen_context.json"
        if path.exists() or output.exists():
            raise RuntimeError("Qwen episode artifacts already exist")
        path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
        proc = None
        status = "infrastructure_error"
        try:
            with open(ctx.log_path, "x") as log:
                proc = subprocess.Popen([sys.executable, "-m", "spatial_interface.qwen_agent", str(path)],
                    cwd=Path(__file__).resolve().parents[1], env=ctx.env,
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                register_agent(proc)
                try:
                    proc.wait(timeout=ctx.timeout + 60)
                except subprocess.TimeoutExpired:
                    status = "timeout"
                    terminate(proc)
                summary = output / "summary.json"
                if summary.exists():
                    status = json.loads(summary.read_text())["status"]
        finally:
            if proc is not None:
                terminate(proc)
                unregister_agent(proc)
        return DriveOutcome(status=status, session_id=str(output))


register_harness(QwenHarness())

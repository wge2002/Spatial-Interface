#!/usr/bin/env python3
"""Model-free simulator/browser/MCP look-end smoke; no robot motion commands."""
import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from spatial_interface.identity import ROOT
from spatial_interface.ports import require_ports_free
from spatial_interface.run_eval import wait_for_sim_ready, terminate_group

async def look_end(out, env):
    params = StdioServerParameters(command=sys.executable,
        args=["-m", "spatial_interface.mcp_server"], env=env, cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [t.name for t in (await session.list_tools()).tools]
            if names != ["dg_look", "dg_policy", "dg_state", "end_episode"]:
                raise RuntimeError("Unexpected tool surface: " + str(names))
            look = await asyncio.wait_for(session.call_tool("dg_look", {}), 90)
            if look.isError:
                raise RuntimeError("dg_look returned an error")
            images = []
            for block in look.content:
                if block.type == "image":
                    suffix = ".jpg" if block.mimeType == "image/jpeg" else ".png"
                    name = f"observation_{len(images)}{suffix}"
                    (out / name).write_bytes(base64.b64decode(block.data))
                    images.append(name)
            if not images:
                raise RuntimeError("No real images returned by dg_look")
            end = await asyncio.wait_for(session.call_tool("end_episode", {}), 60)
            if end.isError:
                raise RuntimeError("end_episode returned an error")
            return {"tools": names, "images": images, "calls": ["dg_look", "end_episode"],
                    "motion_programs_submitted": 0}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="stack")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--base-port", type=int, default=8500)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    require_ports_free(range(args.base_port, args.base_port + 4))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, SPHINX_BASE_PORT=str(args.base_port),
               VIA_CONTROL_INTERFACE="direct_geometry", VIA_DG_FEEDBACK="grounded",
               VIA_EXTRA_GUIDE=str(ROOT / "docs/DIRECT_GEOMETRY_GUIDE.md"))
    started = time.monotonic()
    with (out / "sim.log").open("w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "spatial_interface.record_sim",
            "--task", args.task, "--seed", str(args.seed), "--render", "0",
            "--demo_folder", str(out / "episodes" / args.task.replace('/', '_') / f"seed{args.seed}")],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            info = wait_for_sim_ready(proc, args.base_port, 180, 3)
            if not info:
                raise RuntimeError("Simulator/browser did not become ready; inspect sim.log")
            report = asyncio.run(look_end(out, env))
            proc.wait(timeout=30)
            if proc.returncode:
                raise RuntimeError(f"Simulator exited with {proc.returncode}")
            report.update(status="passed", kind="setup-smoke-not-scored", task=args.task,
                          seed=args.seed, elapsed_s=time.monotonic()-started)
            (out / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(report, indent=2))
        finally:
            terminate_group(proc)

if __name__ == "__main__":
    main()

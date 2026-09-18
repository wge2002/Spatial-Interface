"""Run one fresh, identified episode using the existing robot harness."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess

from spatial_interface.identity import ROOT, digest, snapshot
from spatial_interface.ports import require_ports_free

GUIDES = {"direct_geometry": "DIRECT_GEOMETRY_GUIDE.md", "coarse_fine_policy": "COARSE_FINE_GUIDE.md",
          "fast_geometry": "FAST_GEOMETRY_GUIDE.md", "geometry": "GEOMETRY_GUIDE.md"}
TASKS = {row["key"]: row for row in json.loads((ROOT / "config/tasks.json").read_text())}

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--task", choices=sorted(TASKS), default="stack")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, help="New output directory; existing paths are refused")
    p.add_argument("--base-port", type=int, default=8300)
    p.add_argument("--harness", choices=["codex", "qwen"], default="codex")
    p.add_argument("--model", help="Default: gpt-6-astra for Codex; required for Qwen")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh"])
    p.add_argument("--interface", choices=["legacy", "compact", *GUIDES], default="direct_geometry")
    p.add_argument("--feedback", choices=["baseline", "grounded", "paired"], default="grounded")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--dry-run", action="store_true", help="Print planned configuration without running")
    p.add_argument("--smoke", action="store_true", help="DG look/end only, no task solving; timeout <= 180s")
    return p

def plan(args):
    if args.seed < 0 or args.timeout <= 0 or not 1024 <= args.base_port <= 65532:
        raise ValueError("Use a nonnegative seed, positive timeout, and base port 1024..65532")
    if args.smoke and args.interface != "direct_geometry":
        raise ValueError("The look/end smoke uses direct_geometry")
    if args.harness == "qwen" and not args.model:
        raise ValueError("Qwen requires --model matching your configured server")
    effort = args.effort or ("medium" if args.harness == "codex" else "low")
    if args.harness == "qwen" and effort != "low":
        raise ValueError("The retained Qwen client supports low effort")
    return {"kind": "integration-smoke" if args.smoke else "experiment", "task": TASKS[args.task],
            "seed": args.seed, "interface": args.interface, "feedback": args.feedback,
            "harness": args.harness, "model": args.model or "gpt-6-astra", "effort": effort,
            "base_port": args.base_port, "timeout_s": min(args.timeout, 180) if args.smoke else args.timeout,
            "release": "si-r1", "client_profile": "si-codex-r1" if args.harness == "codex" else "si-qwen-r1",
            "attempt": 1, "automatic_retries": 0,
            "out": str(args.out.resolve()) if args.out else None}

def configure_runtime(record):
    os.environ["VIA_CONTROL_INTERFACE"] = record["interface"]
    os.environ["VIA_DG_FEEDBACK"] = record["feedback"]
    if record["interface"] in GUIDES:
        os.environ["VIA_EXTRA_GUIDE"] = str(ROOT / "docs" / GUIDES[record["interface"]])
    else:
        os.environ.pop("VIA_EXTRA_GUIDE", None)
    if record["harness"] == "codex":
        binary = os.environ.get("VIA_CODEX_BIN", str(ROOT / "data/env/codex/bin/codex"))
        home = Path(os.environ.get("CODEX_HOME", ROOT / "data/env/codex/home"))
        config = home / "config.toml"
        if not config.is_file() or config.read_bytes() != (ROOT / "config/codex-jkwl.toml").read_bytes():
            raise ValueError("Run scripts/setup_codex.py and source set_env.sh; client config must match JKWL")
        if not os.environ.get("SPATIAL_JKWL_API_KEY"):
            raise ValueError("Set SPATIAL_JKWL_API_KEY in your shell")
        version = subprocess.check_output([binary, "--version"], text=True).strip()
        if version != "codex-cli 0.153.4":
            raise ValueError("Expected codex-cli 0.153.4; got " + version)
        os.environ.update(VIA_CODEX_BIN=binary, CODEX_HOME=str(home))
        record["client_runtime"] = {"version": version, "config_sha256": digest(config),
                                    "client_entrypoint_sha256": digest(binary),
                                    "provider": "jkwl", "base_url": "https://jkwl.dmxapi.cn/v1"}
    else:
        config = Path(os.environ.get("VIA_QWEN_SERVER_CONFIG", ""))
        if not config.is_file():
            raise ValueError("Set VIA_QWEN_SERVER_CONFIG to your private server configuration")
        os.environ["VIA_QWEN_CONTEXT_DEDUP"] = "1"
        os.environ["VIA_QWEN_IMAGE_HISTORY_MESSAGES"] = "2"
        record["client_runtime"] = {"config_sha256": digest(config), "image_history_messages": 2, "dedup": True}

def write_json(path, record):
    with path.open("x") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")

def classify_result(result):
    """A saved environment grade must not mask an agent infrastructure error."""
    normal = {"ok", "model_end_episode", "environment_terminal", "model_stopped", "model_output_invalid", "timeout"}
    if result.get("agent_status") not in normal or result.get("reused"):
        return "infrastructure_error"
    if result.get("status") not in {"ok", "timeout"}:
        return "infrastructure_error"
    return "completed"

def main():
    p = parser()
    args = p.parse_args()
    if args.list_tasks:
        for key, row in TASKS.items():
            print(f"{key:18s} {row['language']} ({row['sim_step_budget']} simulation steps)")
        return
    try:
        record = plan(args)
        if args.dry_run:
            print(json.dumps(record, ensure_ascii=False, indent=2))
            return
        if not args.out:
            raise ValueError("An explicit fresh --out directory is required")
        if args.out.exists():
            raise ValueError("Output already exists; use a new --out directory")
        identity = snapshot(record["release"], record["client_profile"])
        configure_runtime(record)
        require_ports_free(range(args.base_port, args.base_port + 4))
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        p.exit(2, str(exc) + "\n")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    record["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(out / "method_identity.json", identity)
    record["method_identity_sha256"] = digest(out / "method_identity.json")
    write_json(out / "manifest.json", record)
    for name, filename in {"VIA_DIRECT_GEOMETRY_LOG_FILE": "direct_geometry.jsonl",
                           "VIA_COARSE_FINE_LOG_FILE": "coarse_fine.jsonl",
                           "VIA_FAST_GEOMETRY_LOG_FILE": "fast_geometry.jsonl",
                           "VIA_GEOMETRY_LOG_FILE": "geometry.jsonl"}.items():
        os.environ[name] = str(out / filename)
    if args.harness == "qwen":
        import spatial_interface.qwen_harness  # noqa: F401
    from spatial_interface import run_eval as runner
    from spatial_interface.agent_common import active_agents, terminate
    from spatial_interface.build_video_index import build_video_index
    from spatial_interface.utils import setup_logging
    setup_logging()
    # Keep timeout accounting separate from any official environment verdict.
    runner.write_timeout_verdict = lambda demo, exp: write_json(out / "timeout.json", {
        "origin": "harness_wall_timeout", "official_verdict": None, "timeout_s": exp.episode_timeout})
    language = record["task"]["language"]
    prompt = record["task"]["prompt"]
    if args.smoke:
        language = "Integration smoke only: call dg_look exactly once, inspect its images, then call end_episode. Do not move, solve the task, or call dg_policy."
        prompt = None
    exp = runner.Experiment(task=args.task, language=language, instruct_file=prompt,
        n_seeds=1, seed_start=args.seed, data_root=str(out / "episodes"),
        base_port=args.base_port, harness=args.harness, model=record["model"],
        reasoning_effort=record["effort"], episode_timeout=record["timeout_s"], grace=20)
    result = None
    try:
        result = runner.run_seed(exp, args.seed, args.base_port, str(out / "logs"))
        result["attempt_status"] = classify_result(result)
        result["grading"] = "smoke-not-scored" if args.smoke else ("official" if record["task"]["detector"] else "manual-required")
        write_json(out / "result.json", result)
        build_video_index(str(out / "episodes"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        for proc in active_agents():
            terminate(proc)
        for proc in list(runner._active_sims):
            runner.terminate_group(proc)
    if result and (result.get("attempt_status") != "completed" or result.get("status") == "timeout"):
        raise SystemExit(2)

if __name__ == "__main__":
    main()

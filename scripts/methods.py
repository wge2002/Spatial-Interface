#!/usr/bin/env python3
"""List, validate, or snapshot the new repository's release identities."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_interface.identity import ROOT, snapshot

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "validate", "snapshot"])
    parser.add_argument("--release", default="si-r1")
    parser.add_argument("--profile", default="si-codex-r1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "list":
        print((ROOT / "methods/registry.json").read_text())
        return
    try:
        record = snapshot(args.release, args.profile)
        if args.command == "snapshot":
            if not args.out:
                parser.error("snapshot requires --out")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x") as f:
                json.dump(record, f, indent=2)
                f.write("\n")
            print(args.out)
        else:
            print("Validated " + args.release + " at " + record["source_commit"])
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, str(exc) + "\n")

if __name__ == "__main__":
    main()

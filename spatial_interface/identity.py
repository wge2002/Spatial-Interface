"""Immutable release content plus exact Git identity for new experiments."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if p.returncode:
        raise ValueError("Git identity unavailable: " + p.stderr.strip())
    return p.stdout.strip()

def snapshot(release="si-r1", profile="si-codex-r1", root=ROOT):
    root = Path(root)
    path = root / "methods/registry.json"
    registry = json.loads(path.read_text())
    if release not in registry["releases"]:
        raise ValueError("Unknown release: " + release)
    if profile not in registry["client_profiles"]:
        raise ValueError("Unknown client profile: " + profile)
    spec = registry["releases"][release]
    if spec["anchor"] != "unique-root-commit":
        raise ValueError("Unsupported source anchor")
    roots = git(root, "rev-list", "--max-parents=0", "HEAD").splitlines()
    if len(roots) != 1:
        raise ValueError("Expected exactly one root commit")
    # The initial release definition itself is pinned to the root's registry.
    initial = json.loads(git(root, "show", roots[0] + ":methods/registry.json"))
    if spec != initial["releases"].get(release):
        raise ValueError("Initial release definition differs from its root anchor")
    client = registry["client_profiles"][profile]
    if client != initial["client_profiles"].get(profile):
        raise ValueError("Client definition differs from its root anchor")
    files = {**spec["files"], **client["files"]}
    dirty = git(root, "status", "--porcelain", "--", "methods/registry.json", *files)
    if dirty:
        raise ValueError("Pinned source/registry has uncommitted changes")
    for name, expected in files.items():
        p = root / name
        if not p.is_file() or digest(p) != expected:
            raise ValueError("Release content mismatch: " + name)
        blob = subprocess.check_output(["git", "-C", str(root), "show", roots[0] + ":" + name])
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError("Root anchor mismatch: " + name)
    for name, expected in spec.get("dependencies", {}).items():
        entry = git(root, "ls-tree", "HEAD", name).split()
        if len(entry) < 3 or entry[0] != "160000" or entry[2] != expected:
            raise ValueError("Dependency gitlink mismatch: " + name)
        if git(root / name, "rev-parse", "HEAD") != expected:
            raise ValueError("Dependency checkout mismatch: " + name)
        if git(root / name, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Dependency has modified tracked files: " + name)
    return {"schema_version": 1, "release": release, "client_profile": profile,
            "source_commit": git(root, "rev-parse", "HEAD"), "release_anchor": roots[0],
            "registry_sha256": digest(path), "files": files,
            "submodule": git(root, "ls-tree", "HEAD", "third_party/LIBERO")}

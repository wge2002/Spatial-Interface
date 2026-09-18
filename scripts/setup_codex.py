"""Configure an installed native Codex 0.153.4 without storing API credentials."""
import argparse
import filecmp
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def copy_native_runtime(binary, target):
    """Copy the complete 0.153.4 runtime, including its tool execution host."""
    with binary.open("rb") as f:
        magic = f.read(4)
    if magic not in (b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe"):
        raise ValueError("--copy-native requires the native runtime, not an npm wrapper")
    sources = [binary, *(binary.parent / name for name in
                        ("codex-code-mode-host", "codex-resources", "rg"))]
    missing = [p.name for p in sources if not p.exists()]
    if missing:
        raise ValueError("Incomplete native installation; missing: " + ", ".join(missing))
    entries = [(binary, target)]
    for source in sources[1:]:
        if source.is_dir():
            entries.extend((p, target.parent / p.relative_to(binary.parent))
                           for p in source.rglob("*") if p.is_file())
        else:
            entries.append((source, target.parent / source.name))
    for source, dest in entries:
        if dest.exists() and not filecmp.cmp(source, dest, shallow=False):
            raise ValueError("Refusing to replace a different runtime file: " + str(dest))
    for source, dest in entries:
        if source.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=shutil.which("codex"), help="Installed codex executable")
    parser.add_argument("--copy-native", action="store_true", help="Copy the native binary and its companion runtime files")
    args = parser.parse_args()
    if not args.binary:
        parser.error("Install @openai/codex@0.153.4 or pass --binary /path/to/codex")
    binary = Path(args.binary).expanduser().resolve()
    version = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if version != "codex-cli 0.153.4":
        parser.error("Expected codex-cli 0.153.4, got " + version)
    target = ROOT / "data/env/codex/bin/codex"
    home = ROOT / "data/env/codex/home"
    target.parent.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.copy_native:
        try:
            copy_native_runtime(binary, target)
        except ValueError as exc:
            parser.error(str(exc))
    elif target.resolve() != binary:
        if target.exists() or target.is_symlink():
            parser.error("Runtime executable already exists; use that executable or a fresh checkout")
        target.symlink_to(binary)
    config = (ROOT / "config/codex-jkwl.toml").read_bytes()
    config_path = home / "config.toml"
    if config_path.exists() and config_path.read_bytes() != config:
        parser.error("Refusing to overwrite a different client configuration")
    config_path.write_bytes(config)
    config_path.chmod(0o600)
    print(version + "; project runtime configured; credentials supplied by SPATIAL_JKWL_API_KEY.")

if __name__ == "__main__":
    main()

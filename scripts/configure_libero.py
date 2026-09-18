"""Write a project-local LIBERO config without touching ~/.libero."""
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[1]
libero = root / "third_party/LIBERO/libero/libero"
if not (libero / "__init__.py").is_file():
    raise SystemExit("Initialize the pinned LIBERO submodule first")
config = root / ".libero/config.yaml"
config.parent.mkdir(exist_ok=True)
config.write_text(yaml.safe_dump({
    "assets": str(libero / "assets"),
    "bddl_files": str(libero / "bddl_files"),
    "benchmark_root": str(libero),
    "datasets": str(root / "data/libero_datasets"),
    "init_states": str(libero / "init_files"),
}))
print("Configured project-local LIBERO paths.")

"""Regression checks for a complete native Codex runtime installation."""
from pathlib import Path
import tempfile
import unittest

from scripts.setup_codex import copy_native_runtime


class NativeRuntimeCopy(unittest.TestCase):
    def test_missing_tool_host_fails_before_copying_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "source" / "codex"
            binary.parent.mkdir()
            binary.write_bytes(b"\x7fELFfixture")
            target = root / "target" / "codex"
            with self.assertRaisesRegex(ValueError, "codex-code-mode-host"):
                copy_native_runtime(binary, target)
            self.assertFalse(target.exists())

    def test_partial_install_is_repaired_with_all_runtime_components(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            (source / "codex-resources" / "nested").mkdir(parents=True)
            for name in ("codex", "codex-code-mode-host", "rg"):
                (source / name).write_bytes(b"\x7fELFfixture-" + name.encode())
            (source / "codex-resources" / "nested" / "resource.js").write_text("resource")
            target = root / "target" / "codex"
            target.parent.mkdir()
            target.write_bytes((source / "codex").read_bytes())
            copy_native_runtime(source / "codex", target)
            for name in ("codex", "codex-code-mode-host", "rg", "codex-resources/nested/resource.js"):
                self.assertEqual((source / name).read_bytes(), (target.parent / name).read_bytes())
            copy_native_runtime(source / "codex", target)


if __name__ == "__main__":
    unittest.main()

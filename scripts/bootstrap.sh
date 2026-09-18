#!/usr/bin/env bash
set -euo pipefail
SI_SETUP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SI_SETUP_ROOT"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV_BIN" ]]; then
  echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null || ! ffmpeg -hide_banner -encoders 2>/dev/null | grep libx264 >/dev/null; then
  echo 'Install FFmpeg with libx264 and put ffmpeg on PATH before setup.' >&2
  exit 1
fi
git submodule update --init --recursive
test "$(git -C third_party/LIBERO rev-parse HEAD)" = 8f1084e3132a39270c3a13ebe37270a43ece2a01
if [[ ! -x .venv/bin/python ]]; then
  "$UV_BIN" venv --python 3.10 --prompt spatial-interface .venv
fi
UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4 "$UV_BIN" sync --locked
.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/configure_libero.py
if [[ "${1:-}" == --headless ]]; then
  bash scripts/headless_bootstrap.sh
elif [[ -n "${1:-}" ]]; then
  echo "Unknown option: $1" >&2
  exit 2
fi
echo 'Environment ready. Next: source set_env.sh'

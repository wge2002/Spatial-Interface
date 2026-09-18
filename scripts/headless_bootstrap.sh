#!/usr/bin/env bash
# Optional Ubuntu/Debian x86_64 support for containers without Chromium libraries.
# Extract dependencies into this project; no sudo or global package installation.
set -euo pipefail
SI_SETUP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SI_ENV="$SI_SETUP_ROOT/data/env"
SI_APT="$SI_ENV/aptroot"
SI_SYSROOT="$SI_ENV/sysroot"
mkdir -p "$SI_APT/lists/partial" "$SI_APT/archives/partial" "$SI_APT/debs" "$SI_SYSROOT" "$SI_ENV/egl_vendor.d"
if command -v apt-get >/dev/null && command -v dpkg >/dev/null; then
  si_apt_opts=(-o "Dir::State::Lists=$SI_APT/lists" -o "Dir::Cache::archives=$SI_APT/archives" -o Debug::NoLocking=1 -o Acquire::Languages=none)
  if ! compgen -G "$SI_APT/debs/libnss3_*.deb" >/dev/null || ! compgen -G "$SI_APT/debs/libnspr4_*.deb" >/dev/null; then
    apt-get "${si_apt_opts[@]}" update
    (cd "$SI_APT/debs" && apt-get "${si_apt_opts[@]}" download libnss3 libnspr4)
  fi
  for si_deb in "$SI_APT/debs"/libnss3_*.deb "$SI_APT/debs"/libnspr4_*.deb; do
    dpkg -x "$si_deb" "$SI_SYSROOT"
  done
else
  echo 'apt-get/dpkg unavailable; supply Chromium system libraries for your distribution.' >&2
fi
if command -v nvidia-smi >/dev/null; then
  cat > "$SI_ENV/egl_vendor.d/10_nvidia.json" <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
JSON
fi
echo 'Headless overlay prepared. Source scripts/headless_env.sh after set_env.sh.'

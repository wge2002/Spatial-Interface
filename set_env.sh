# Source from bash or zsh; paths resolve from this file, not the caller's cwd.
if [ -n "${BASH_VERSION:-}" ]; then
  _si_script="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  _si_script="${(%):-%N}"
else
  echo 'Use bash or zsh to source set_env.sh.' >&2
  return 1
fi
export SPATIAL_ROOT="$(cd "$(dirname "$_si_script")" && pwd)"
unset _si_script
if [ ! -x "$SPATIAL_ROOT/.venv/bin/python" ]; then
  echo 'Run bash scripts/bootstrap.sh from the repository first.' >&2
  return 1
fi
source "$SPATIAL_ROOT/.venv/bin/activate"
export PYTHONPATH="$SPATIAL_ROOT:$SPATIAL_ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$SPATIAL_ROOT/.libero"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VIA_CONTROL_INTERFACE="${VIA_CONTROL_INTERFACE:-direct_geometry}"
export VIA_DG_FEEDBACK="${VIA_DG_FEEDBACK:-grounded}"
case "$VIA_CONTROL_INTERFACE" in
  direct_geometry) _si_guide=DIRECT_GEOMETRY_GUIDE.md ;;
  coarse_fine_policy) _si_guide=COARSE_FINE_GUIDE.md ;;
  fast_geometry) _si_guide=FAST_GEOMETRY_GUIDE.md ;;
  geometry) _si_guide=GEOMETRY_GUIDE.md ;;
  *) _si_guide='' ;;
esac
if [ -n "$_si_guide" ]; then
  export VIA_EXTRA_GUIDE="${VIA_EXTRA_GUIDE:-$SPATIAL_ROOT/docs/$_si_guide}"
fi
unset _si_guide
export CODEX_HOME="${SPATIAL_CODEX_HOME:-$SPATIAL_ROOT/data/env/codex/home}"
export VIA_CODEX_BIN="${SPATIAL_CODEX_BIN:-$SPATIAL_ROOT/data/env/codex/bin/codex}"
export PATH="$(dirname "$VIA_CODEX_BIN"):$PATH"

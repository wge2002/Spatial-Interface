# Source after set_env.sh on Linux servers without a desktop.
if [ -z "${SPATIAL_ROOT:-}" ]; then
  echo 'Source set_env.sh first.' >&2
  return 1
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
for _si_d in "$SPATIAL_ROOT/data/env/sysroot/usr/lib/x86_64-linux-gnu" "$SPATIAL_ROOT/data/env/sysroot/lib/x86_64-linux-gnu"; do
  if [ -d "$_si_d" ]; then
    export LD_LIBRARY_PATH="$_si_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
unset _si_d
if [ -d "$SPATIAL_ROOT/data/env/egl_vendor.d" ]; then
  export __EGL_VENDOR_LIBRARY_DIRS="$SPATIAL_ROOT/data/env/egl_vendor.d:/usr/share/glvnd/egl_vendor.d"
fi
export VIA_CHROMIUM_EXTRA_ARGS="${VIA_CHROMIUM_EXTRA_ARGS:---headless=new --no-sandbox --disable-dev-shm-usage --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader}"

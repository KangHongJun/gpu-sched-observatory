#!/usr/bin/env bash
# Build the CUPTI injection library against the CUPTI/CUDA headers shipped in the project venv.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
nv="$(echo "$root"/.venv/lib/python3*/site-packages/nvidia/cu13)"
[ -f "$nv/include/cupti.h" ] || { echo "cupti.h not found under $nv (run uv sync)" >&2; exit 1; }
# cu13 headers omit crt/host_defines.h; the venv triton wheel ships it (host-only macros).
tr="$(echo "$root"/.venv/lib/python3*/site-packages/triton/backends/nvidia/include)"
mkdir -p "$here/build"
cmd=(g++ -std=c++17 -O2 -fPIC -shared -I"$nv/include" -idirafter "$tr" "$here/inject.cpp" -o "$here/build/libgso_inject.so"
     -L"$nv/lib" -l:libcupti.so.13 -Wl,-rpath,"$nv/lib")
echo "${cmd[@]}"
"${cmd[@]}"
echo "built: $here/build/libgso_inject.so"
echo "cupti lib dir: $nv/lib"

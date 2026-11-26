#!/usr/bin/env bash
# G2 검증: nsys 전역 GPU context-switch 캡처.
#   capture <name> [duration]   docs/verification/raw/<name>.nsys-rep 생성 (기본 10초)
#   export <name>               같은 위치에 <name>.sqlite 생성
# 이 스크립트는 sudo를 호출하지 않는다. 루트 캡처가 필요하면 사용자가 앞에 붙인다:
#   sudo NSYS="$NSYS" scripts/verify/04_nsys_ctxsw.sh capture ctx_baseline 10
# 환경 변수: NSYS(nsys 경로), TARGET(캡처 대상 명령, 기본 "sleep <duration>")
set -euo pipefail

NSYS="${NSYS:-$HOME/opt/nsys/opt/nvidia/nsight-systems-cli/2026.6.1/target-linux-x64/nsys}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="$ROOT/docs/verification/raw"
mkdir -p "$RAW"

cmd="${1:-}"
name="${2:-}"
[ -n "$name" ] || { echo "usage: $0 capture <name> [duration] | export <name>" >&2; exit 2; }

case "$cmd" in
  capture)
    dur="${3:-10}"
    # CPU 샘플링/OS 스레드 추적은 끄고 GPU context switch만 본다(paranoid 4에서 비루트는 허용되지 않음).
    # 대상 명령이 필요하면 TARGET으로 준다. 기본은 sleep.
    # shellcheck disable=SC2086
    "$NSYS" profile --gpuctxsw=true --sample=none --cpuctxsw=none \
      --duration="$dur" --force-overwrite=true -o "$RAW/$name" \
      ${TARGET:-sleep $dur}
    ;;
  export)
    "$NSYS" export --type=sqlite --force-overwrite=true -o "$RAW/$name.sqlite" "$RAW/$name.nsys-rep"
    ;;
  *)
    echo "usage: $0 capture <name> [duration] | export <name>" >&2; exit 2 ;;
esac

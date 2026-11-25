#!/usr/bin/env bash
# 검증 환경 정보를 docs/verification/raw/00_env.txt 에 기록한다.
# sudo 불필요. 없거나 실패하는 도구는 "not available"로 기록하고 계속 진행한다.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW="$ROOT/docs/verification/raw"
OUT="$RAW/00_env.txt"
mkdir -p "$RAW"

# 명령을 실행해 섹션으로 기록한다. 실패해도 스크립트는 중단하지 않는다.
section() {
    local title="$1"
    shift
    {
        echo "=== $title ==="
        echo "\$ $*"
        if command -v "$1" >/dev/null 2>&1; then
            if ! "$@" 2>&1; then
                echo "(command failed: not available)"
            fi
        else
            echo "not available"
        fi
        echo
    } >> "$OUT"
}

: > "$OUT"
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT"
echo >> "$OUT"

section "nvidia-smi" nvidia-smi
section "nvidia-smi pmon" nvidia-smi pmon -s um -c 1
section "driver version" nvidia-smi --query-gpu=driver_version --format=csv,noheader
section "dcgmi version" dcgmi --version
section "nsys version" nsys --version

{
    echo "=== perf_event_paranoid ==="
    if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
        cat /proc/sys/kernel/perf_event_paranoid
    else
        echo "not available"
    fi
} >> "$OUT"

echo "wrote $OUT"
exit 0

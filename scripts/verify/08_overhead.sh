#!/usr/bin/env bash
# G6 검증: 수집 오버헤드 측정(GPU 1 전용, MPS OFF).
# 조건 none / dcgm / nsys를 번갈아 3회씩(30초, small_kernels 2-process), 마지막에 DCGM+nsys 동시 1회.
# 결과: docs/verification/raw/08_*.json|txt (git 제외). 표는 08_summarize.py가 만든다.
# 사용: [PREFIX=08] 08_overhead.sh [reps]   (재실행 안전: 같은 이름 파일을 덮어쓴다)
# PREFIX: 출력 파일 이름 접두어(기본 08). 시험 실행은 다른 값을 줘서 측정 데이터를 덮어쓰지 않게 한다.
# 종료 시(오류, Ctrl-C, SIGTERM 포함) 이 스크립트가 띄운 프로세스를 모두 종료하고 DCGM 컨테이너를 지운다.
# sudo, MPS 데몬은 쓰지 않는다.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="$ROOT/docs/verification/raw"
HERE="$ROOT/scripts/verify"
REPS="${1:-3}"
PREFIX="${PREFIX:-08}"
DUR=30
GPU=1
mkdir -p "$RAW"
cd "$ROOT" || exit 1

# 이 스크립트가 띄운 프로세스의 PID만 기록하고, 그 자손까지 종료한다(이름 패턴으로 찾지 않는다).
WL_PIDS=""
killtree() {
    local c
    for c in $(pgrep -P "$1" 2>/dev/null); do killtree "$c"; done
    kill "$1" 2>/dev/null
}
cleanup() {
    trap - EXIT INT TERM
    local p
    for p in $WL_PIDS; do killtree "$p"; done
    wait 2>/dev/null
    # 컨테이너 안의 dcgmi dmon은 컨테이너를 지우면 함께 끝난다(docker exec 클라이언트는 위에서 종료).
    "$HERE/03_dcgm.sh" stop >/dev/null 2>&1
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

temp() { nvidia-smi -i "$GPU" --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' '; }
util() { nvidia-smi -i "$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' '; }

if pgrep -x nvidia-cuda-mps >/dev/null; then
    echo "MPS 프로세스가 있다. 중단." >&2
    exit 3
fi
if [ "$(util)" != "0" ]; then
    echo "GPU $GPU가 유휴가 아니다. 중단." >&2
    exit 3
fi
BASE_T="$(temp)"
echo "base_temp=$BASE_T" | tee "$RAW/${PREFIX}_order.txt"

# GPU가 유휴이고 시작 온도 +3도 이내로 돌아올 때까지 최대 60초 대기
cool() {
    for _ in $(seq 1 60); do
        t="$(temp)"
        if [ "$(util)" = "0" ] && [ "$t" -le $((BASE_T + 3)) ]; then
            return 0
        fi
        sleep 1
    done
    echo "  (cool-down 시간 초과, 온도 $(temp))" | tee -a "$RAW/${PREFIX}_order.txt"
}

start_workloads() { # $1 = 이름 접두어
    for n in a b; do
        uv run python -m gso.bench.workloads.small_kernels --gpu "$GPU" --duration "$DUR" \
            --out "$RAW/$1_$n.json" >"$RAW/$1_$n.log" 2>&1 &
        WL_PIDS="$WL_PIDS $!"
    done
}

dmon_plain() { docker exec gso-dcgm dcgmi dmon -i 0 -e 1001,1002,1003,1004,1005 -d 100 -c 300; }
# 충돌 실행용: 각 줄 앞에 벽시계(epoch 초)를 붙인다
dmon_stamped() {
    dmon_plain 2>&1 | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done
}

run_one() { # $1 조건 $2 반복
    cond="$1"
    r="$2"
    name="${PREFIX}_${cond}_r${r}"
    cool
    echo "run $name start_temp=$(temp) at $(date +%H:%M:%S)" | tee -a "$RAW/${PREFIX}_order.txt"
    WL_PIDS=""
    start_workloads "$name"
    case "$cond" in
    none) ;;
    dcgm) dmon_plain >"$RAW/${name}_dmon.txt" 2>&1 &
        WL_PIDS="$WL_PIDS $!" ;;
    nsys)
        (
            sleep 10
            "$HERE/04_nsys_ctxsw.sh" capture "${name}_ctx" 10 >"$RAW/${name}_nsys.txt" 2>&1
        ) &
        WL_PIDS="$WL_PIDS $!" ;;
    conflict)
        # 이 실행만 시작 시각을 기록한다
        date +%s.%N >"$RAW/${name}_t_launch.txt"
        dmon_stamped >"$RAW/${name}_dmon.txt" &
        WL_PIDS="$WL_PIDS $!"
        (
            sleep 10
            date +%s.%N >"$RAW/${name}_t_nsys_start.txt"
            "$HERE/04_nsys_ctxsw.sh" capture "${name}_ctx" 10 >"$RAW/${name}_nsys.txt" 2>&1
            date +%s.%N >"$RAW/${name}_t_nsys_end.txt"
        ) &
        WL_PIDS="$WL_PIDS $!" ;;
    esac
    # shellcheck disable=SC2086
    wait $WL_PIDS
    for n in a b; do
        if [ ! -s "$RAW/${name}_$n.json" ]; then
            echo "실패: $name 프로세스 $n의 JSON이 없거나 비어 있다" | tee -a "$RAW/${PREFIX}_order.txt" >&2
            exit 1
        fi
    done
}

need_dcgm=1
"$HERE/03_dcgm.sh" start || { echo "DCGM 컨테이너 시작 실패" >&2; need_dcgm=0; }

for r in $(seq 1 "$REPS"); do
    for cond in none dcgm nsys; do
        if [ "$cond" = dcgm ] && [ "$need_dcgm" = 0 ]; then continue; fi
        run_one "$cond" "$r"
    done
done

if [ "$need_dcgm" = 1 ]; then
    run_one conflict 1
    "$HERE/04_nsys_ctxsw.sh" export "${PREFIX}_conflict_r1_ctx" >"$RAW/${PREFIX}_conflict_r1_export.txt" 2>&1
fi
echo "done" | tee -a "$RAW/${PREFIX}_order.txt"

#!/usr/bin/env bash
# G6 검증: 수집 오버헤드 측정(GPU 1 전용, MPS OFF).
# 조건 none / dcgm / nsys를 번갈아 3회씩(30초, small_kernels 2-process), 마지막에 DCGM+nsys 동시 1회.
# 결과: docs/verification/raw/08_*.json|txt (git 제외). 표는 08_summarize.py가 만든다.
# 사용: 08_overhead.sh [reps]   (재실행 안전: 같은 이름 파일을 덮어쓴다)
# 종료 시(오류 포함) DCGM 컨테이너를 trap으로 정리한다. sudo, MPS 데몬은 쓰지 않는다.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="$ROOT/docs/verification/raw"
HERE="$ROOT/scripts/verify"
REPS="${1:-3}"
DUR=30
GPU=1
mkdir -p "$RAW"
cd "$ROOT" || exit 1

cleanup() {
    "$HERE/03_dcgm.sh" stop >/dev/null 2>&1
}
trap cleanup EXIT

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
echo "base_temp=$BASE_T" | tee "$RAW/08_order.txt"

# GPU가 유휴이고 시작 온도 +3도 이내로 돌아올 때까지 최대 60초 대기
cool() {
    for _ in $(seq 1 60); do
        t="$(temp)"
        if [ "$(util)" = "0" ] && [ "$t" -le $((BASE_T + 3)) ]; then
            return 0
        fi
        sleep 1
    done
    echo "  (cool-down 시간 초과, 온도 $(temp))" | tee -a "$RAW/08_order.txt"
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
    name="08_${cond}_r${r}"
    cool
    echo "run $name start_temp=$(temp) at $(date +%H:%M:%S)" | tee -a "$RAW/08_order.txt"
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
    "$HERE/04_nsys_ctxsw.sh" export 08_conflict_r1_ctx >"$RAW/08_conflict_r1_export.txt" 2>&1
fi
echo "done" | tee -a "$RAW/08_order.txt"

#!/usr/bin/env bash
# G4 검증: GSO_GPU 하나 전용 사용자 범위 CUDA MPS 데몬 관리.
# 사용: GSO_GPU=<nvidia-smi GPU 인덱스> 06_mps.sh start|stop|status|env   (기본값 없음: 호출자가 고른다)
#   start는 그 GPU에 compute 프로세스가 하나라도 있으면 거부한다(다른 사용자의 작업일 수 있다).
#   start    데몬 시작 (이미 nvidia-cuda-mps-control 프로세스가 있으면 거부)
#   stop     우리 pipe 디렉터리로만 quit을 보내고 이 사용자의 nvidia-cuda-mps 프로세스가 없는지 확인
#   status   우리 pipe 디렉터리 기준 get_server_list와 프로세스 목록
#   env      클라이언트가 쓸 export 줄 출력 (eval "$(scripts/verify/06_mps.sh env)")
# gso.bench.mps.MpsSession과 이 스크립트는 같은 /tmp/gso-mps-$UID를 쓰므로 동시에 쓰지 않는다
#   (start는 MpsSession의 session.lock이 잡혀 있으면 거부한다).
# 시스템 기본 pipe 디렉터리(/tmp/nvidia-mps)와 GPU compute mode는 건드리지 않는다. sudo 없음.
set -euo pipefail

usage() { echo "usage: GSO_GPU=<gpu index> $0 start|stop|status|env" >&2; exit 2; }
GPU="${GSO_GPU:-}"
case "$GPU" in
  ''|*[!0-9]*) echo "error: set GSO_GPU to the GPU index (nvidia-smi numbering); there is no default" >&2; usage ;;
esac
# CUDA_VISIBLE_DEVICES=$GPU 가 nvidia-smi와 같은 보드를 가리키도록 PCI 순서를 고정한다.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

BASE="/tmp/gso-mps-$(id -u)"
PIPE="$BASE/pipe"
LOG="$BASE/log"
CTL="${NVIDIA_MPS_CONTROL:-nvidia-cuda-mps-control}"

ours() { CUDA_MPS_PIPE_DIRECTORY="$PIPE" CUDA_MPS_LOG_DIRECTORY="$LOG" "$@"; }
# pgrep의 기본 매칭은 15자로 잘린 comm("nvidia-cuda-mps")이라 "nvidia-cuda-mps-control"로는 아무것도 못 찾는다.
# 그래서 comm 형태와 전체 명령줄 형태를 둘 다 본다(오탐은 허용, 누락은 불허).
CMDRE='(^|/)nvidia-cuda-mps-(control|server)( |$)'
# 모든 사용자의 MPS 제어 데몬/서버 (pid와 명령줄)
any_mps() { { pgrep -a -x nvidia-cuda-mps; pgrep -a -f "$CMDRE"; } 2>/dev/null | sort -u || true; }
# 이 사용자의 MPS 프로세스
mine() { { pgrep -u "$(id -u)" -a -x nvidia-cuda-mps; pgrep -u "$(id -u)" -a -f "$CMDRE"; } 2>/dev/null | sort -u || true; }

# BASE는 mode 700, 현재 사용자 소유 디렉터리여야 하고 심볼릭 링크면 거부한다.
prepare_base() {
  if [ -L "$BASE" ] || { [ -e "$BASE" ] && { [ ! -d "$BASE" ] || [ ! -O "$BASE" ]; }; }; then
    echo "refuse: $BASE exists but is a symlink or not a directory owned by this user" >&2
    exit 1
  fi
  mkdir -p -m 700 "$BASE"
  chmod 700 "$BASE"
  mkdir -p "$PIPE" "$LOG"
}

case "${1:-}" in
  start)
    existing="$(any_mps)"
    if [ -n "$existing" ]; then
      echo "refuse: an MPS control daemon or server already exists on this machine:" >&2
      echo "$existing" >&2
      exit 1
    fi
    # 이 GPU에 이미 compute 프로세스가 있으면(다른 사용자 포함) 시작하지 않는다. 조회 실패도 거부.
    if ! apps="$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name --format=csv,noheader 2>&1)"; then
      echo "refuse: cannot query compute processes on GPU $GPU: $apps" >&2
      exit 1
    fi
    if [ -n "$(printf '%s' "$apps" | tr -d '[:space:]')" ]; then
      echo "refuse: GPU $GPU has compute processes; not starting an MPS daemon for it:" >&2
      echo "$apps" >&2
      exit 1
    fi
    prepare_base
    if [ -e "$BASE/session.lock" ] && ! flock -n "$BASE/session.lock" true; then
      echo "refuse: a gso MPS session (MpsSession) holds $BASE/session.lock" >&2
      exit 1
    fi
    # 데몬은 GSO_GPU 하나만 볼 수 있다.
    CUDA_VISIBLE_DEVICES="$GPU" CUDA_MPS_PIPE_DIRECTORY="$PIPE" CUDA_MPS_LOG_DIRECTORY="$LOG" "$CTL" -d
    sleep 1
    mine
    ;;
  stop)
    if [ -d "$PIPE" ]; then
      echo quit | ours "$CTL" || true
    fi
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      [ -z "$(mine)" ] && break
      sleep 1
    done
    if [ -n "$(mine)" ]; then
      echo "ERROR: nvidia-cuda-mps still running for this user:" >&2
      mine >&2
      exit 1
    fi
    echo "stopped: no nvidia-cuda-mps process for this user"
    ;;
  status)
    echo "processes (this user):"; mine
    if [ -d "$PIPE" ]; then
      echo "get_server_list:"; echo get_server_list | ours "$CTL" 2>&1 || echo "(exit $?)"
    fi
    ;;
  env)
    echo "export CUDA_MPS_PIPE_DIRECTORY=$PIPE"
    echo "export CUDA_MPS_LOG_DIRECTORY=$LOG"
    ;;
  *)
    usage ;;
esac

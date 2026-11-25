#!/usr/bin/env bash
# G4 검증: GPU 1 전용 사용자 범위 CUDA MPS 데몬 관리.
#   start    데몬 시작 (이미 nvidia-cuda-mps-control 프로세스가 있으면 거부)
#   stop     우리 pipe 디렉터리로만 quit을 보내고 이 사용자의 nvidia-cuda-mps 프로세스가 없는지 확인
#   status   우리 pipe 디렉터리 기준 get_server_list와 프로세스 목록
#   env      클라이언트가 쓸 export 줄 출력 (eval "$(scripts/verify/06_mps.sh env)")
# 시스템 기본 pipe 디렉터리(/tmp/nvidia-mps)와 GPU compute mode는 건드리지 않는다. sudo 없음.
set -euo pipefail

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
    prepare_base
    # 데몬은 GPU 1만 볼 수 있다.
    CUDA_VISIBLE_DEVICES=1 CUDA_MPS_PIPE_DIRECTORY="$PIPE" CUDA_MPS_LOG_DIRECTORY="$LOG" "$CTL" -d
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
    echo "usage: $0 start|stop|status|env" >&2; exit 2 ;;
esac

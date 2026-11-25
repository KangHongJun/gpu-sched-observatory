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
mine() { pgrep -u "$(id -u)" -a nvidia-cuda-mps || true; }

case "${1:-}" in
  start)
    if pgrep -a nvidia-cuda-mps-control >/dev/null; then
      echo "refuse: nvidia-cuda-mps-control already running:" >&2
      pgrep -a nvidia-cuda-mps-control >&2
      exit 1
    fi
    mkdir -p "$PIPE" "$LOG"
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

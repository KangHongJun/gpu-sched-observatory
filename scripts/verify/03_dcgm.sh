#!/usr/bin/env bash
# DCGM host engine 컨테이너(호스트 GPU 하나 전용)를 올리고 내리는 도우미.
# 사용: GSO_DCGM_GPU=<호스트 GPU 인덱스> 03_dcgm.sh start | stop | status | dmon [count]
# start와 dmon은 GSO_DCGM_GPU가 필요하다(stop, status는 컨테이너 이름만 쓴다).
# sudo 불필요. docker nvidia 런타임이 필요하다. 기본값 없음: 호출자가 GPU를 골라야 한다.
set -u

IMAGE_TAG="4.8.0-ubuntu24.04"
IMAGE="nvcr.io/nvidia/cloud-native/dcgm:${IMAGE_TAG}"
NAME="gso-dcgm"
HOST_GPU=${GSO_DCGM_GPU:-}
need_gpu() { : "${GSO_DCGM_GPU:?set GSO_DCGM_GPU to the host GPU index}"; }
# 컨테이너 안에서 GPU 하나만 보이므로 인덱스는 0으로 열거된다(실측은 RESULTS.md 참고).
CONT_GPU=0
FIELDS="1001,1002,1003,1004,1005"

cmd="${1:-status}"
case "$cmd" in
start)
    need_gpu
    if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
        echo "already exists: $NAME"
        exit 0
    fi
    docker run -d --name "$NAME" --gpus "device=${HOST_GPU}" --cap-add SYS_ADMIN \
        "$IMAGE" >/dev/null || exit 1
    # host engine 기동 대기
    for _ in $(seq 1 20); do
        if docker exec "$NAME" dcgmi discovery -l >/dev/null 2>&1; then
            echo "ready: $NAME ($IMAGE)"
            exit 0
        fi
        sleep 1
    done
    echo "engine not ready" >&2
    exit 1
    ;;
stop)
    docker rm -f "$NAME" >/dev/null 2>&1 && echo "removed: $NAME" || echo "not present: $NAME"
    ;;
status)
    docker ps -a --filter "name=^${NAME}$" --format '{{.Names}} {{.Status}} {{.Image}}'
    ;;
dmon)
    need_gpu
    count="${2:-50}"
    docker exec "$NAME" dcgmi dmon -i "$CONT_GPU" -e "$FIELDS" -d 100 -c "$count"
    ;;
*)
    echo "usage: $0 start|stop|status|dmon [count]" >&2
    exit 2
    ;;
esac

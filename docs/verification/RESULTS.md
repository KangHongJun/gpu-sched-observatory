# 기술 검증 결과 (1주차)

각 게이트의 판정과 근거를 이 문서에 기록한다. 원본 출력은 `docs/verification/raw/`에 두며 git에 올리지 않는다.

## 게이트 표

| 게이트 | 질문 | 통과 시 | 실패 시 | 판정 |
|---|---|---|---|---|
| G1 | A5000에서 DCGM 1001~1005가 나오는가 | Live에 5개 막대 표시 | 해당 필드는 `N/A`, Live는 NVML 중심 | 통과: 1001~1005 전부 값이 나옴(1006만 미지원). 단 부하 시 sm_active는 0.08 수준 |
| G2 | nsys가 외부 프로세스의 Context 전환을 PID와 함께 잡는가 | 3주차를 nsys 캡처로 진행 | 3주차를 "직접 띄운 프로세스의 커널 타임라인"으로 축소 | 미정 |
| G3 | SQLite export에서 전환·커널 테이블을 읽을 수 있는가 | Task 18을 실제 스키마로 구현 | nsys의 다른 export 형식(json)을 검토 | 미정 |
| G4 | MPS ON에서 전환 이벤트와 커널 중첩이 어떻게 달라지는가 | 결과대로 표시 규칙 확정 | 전환은 `N/A`, overlap과 앱 성능 우선 | 미정 |
| G5 | CUPTI를 외부 프로세스에 붙일 수 있는가 | MVP 이후 collector 설계 입력 | MVP 이후 과제로만 기록 | 미정 |
| G6 | 수집 오버헤드가 얼마인가 | README에 수치 기재 | 기본 수집 주기를 낮춘다 | 미정 |

## Task 1: 프로젝트 스캐폴드와 검증 기록 틀

(결과 없음)

## Task 2: 2-process 재현 워크로드

명령 (GPU 1 전용, `--size` 기본 256, 워밍업 10회 제외):

```bash
# 단독
uv run python -m gso.bench.workloads.small_kernels --gpu 1 --duration 20 --out docs/verification/raw/small_single.json
# 2-process 동시
for n in a b; do uv run python -m gso.bench.workloads.small_kernels --gpu 1 --duration 20 --out docs/verification/raw/small_$n.json & done; wait
```

| 실행 | throughput (iter/s) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| 단독 | 53,792 | 0.0171 | 0.0220 | 0.0269 |
| 2-process, 프로세스 a | 6,029 | 0.1662 | 0.1670 | 0.1830 |
| 2-process, 프로세스 b | 6,033 | 0.1661 | 0.1671 | 0.1830 |

- 처리량 변화: 프로세스당 53,792 -> 약 6,030 iter/s (단독 대비 약 11.2%, 약 -89%). 두 프로세스 합계는 약 12,060 iter/s로 단독의 약 22%.
- p50 지연은 0.0171 ms -> 0.166 ms로 약 9.7배 늘었다. 동시 실행 중 GPU 1 사용률은 99%였고 nvidia-smi에 두 프로세스가 GPU 1 컨텍스트로 보였다.
- 두 프로세스의 수치가 거의 같아, 시분할이 대칭적으로 일어난 것으로 보인다. 원인(Context 전환 여부)은 이후 Task의 nsys 캡처로 확인한다.
- 원본 JSON은 `docs/verification/raw/`(git 제외)에 있다.

## Task 3: DCGM 필드 지원 확인 (G1)

**판정: 통과.** 1001~1005 다섯 필드가 모두 A5000에서 지원되고 값이 나온다.

### 환경
- 경로 A(컨테이너)만 사용했다. 이미지: `nvcr.io/nvidia/cloud-native/dcgm:4.8.0-ubuntu24.04` (실행 시점 NGC의 최신 4.x 태그). 호스트의 dcgmi 3.3.9와 sudo는 쓰지 않았다.
- 실행: `docker run -d --name gso-dcgm --gpus device=1 --cap-add SYS_ADMIN <image>` (이미지 기본 CMD가 `nv-hostengine -n -b 0.0.0.0 -f -`). 포트 게시 없음, 조회는 `docker exec gso-dcgm dcgmi ...`.
- 컨테이너 안에서는 GPU가 **인덱스 0**으로 열거된다. 컨테이너의 `nvidia-smi` UUID가 호스트 GPU 1의 UUID와 일치함을 확인했다. 따라서 호스트 `-i 1`은 컨테이너 안 `-i 0`이다. 이후 Task는 `-i 0`을 써야 한다.
- 기동/정지/조회는 `scripts/verify/03_dcgm.sh start|stop|status|dmon [count]`. 태그는 스크립트 맨 위 `IMAGE_TAG` 변수 하나.

### `dcgmi profile -l -i 0` 지원 목록 (11개)
1001 gr_engine_active, 1002 sm_active, 1003 sm_occupancy, 1004 tensor_active, 1005 dram_active, 1007 fp32_active, 1008 fp16_active, 1009 pcie_tx_bytes_per_second, 1010 pcie_rx_bytes_per_second, 1011 prof_nvlink_tx_bytes_per_second, 1012 prof_nvlink_rx_bytes_per_second.
(1006은 목록에 없다. 1011/1012는 목록에는 있으나 값은 항상 0이다. A5000 단일 GPU 구성에서 NVLink 트래픽이 없기 때문으로 보이나, 브리지 장착 여부는 확인하지 않았다: 미확인.)

### 필드별 결과 (`dmon -i 0 -e <필드> -d 100 -c 2`, 필드를 하나씩 조회)
| 필드 | 결과 |
|---|---|
| 1001 gr_engine_active | 지원. 부하 시 0.988~0.992 |
| 1002 sm_active | 지원. 부하 시 0.076~0.079 |
| 1003 sm_occupancy | 지원. 부하 시 0.006~0.007 |
| 1004 tensor_active | 지원. 부하 시 0.000 (워크로드가 텐서코어를 쓰지 않음) |
| 1005 dram_active | 지원. 부하 시 0.021 |
| 1006 | **미지원**. `Error setting watches. Result: -6: Feature not supported` |
| 1007, 1008 | 지원(유휴 0.000만 확인, 부하 값은 미확인) |
| 1009, 1010 (PCIe 바이트/초) | 지원. 유휴에서도 바이트 단위 정수가 찍힘 |
| 1011, 1012 | 항상 0 |

### 출력 형식 (파서 입력)
- 헤더 2줄: `#Entity` 뒤에 필드 ID 열, 다음 줄은 `ID`. 그다음 데이터 줄은 `GPU 0` + 값 열들이며, 열은 공백으로 정렬된다(구분자 없음, 줄 끝에 공백이 붙는다). 엔티티 열은 `GPU <index>` 두 토큰이다.
- 헤더 2줄이 **약 15행마다 다시 출력된다**(50샘플에서 1, 18, 34, 50번째 줄). 파서는 `#Entity`/`ID` 줄을 건너뛰어야 한다.
- 1001~1005 값은 **0~1 비율**이다(% 아님). 1009/1010은 바이트/초 정수.
- 미지원 필드는 `N/A` 표기가 나오지 않았다. 목록에 없는 필드(1006)는 `dmon`이 오류 한 줄(`Error setting watches ... -6`)을 내고 **다른 필드가 섞여 있어도 명령 전체가 실패**해 데이터 줄이 하나도 안 나온다. 존재하지 않는 ID(9999)는 `Got invalid field ID`로 실패한다. 따라서 "필드 하나가 N/A로 찍히는" 경우는 이번에 관찰하지 못했다. 지원 여부는 `profile -l`로 먼저 걸러야 한다.
- fixture에 `N/A`가 없는 이유가 이것이다(Brief의 "N/A 표기 그대로"는 해당 없음).

### 갱신 간격
- `-d 100 -c 50`의 벽시계 시간은 유휴 5.26초, 부하 5.21초(docker exec 오버헤드 포함) → 100ms가 그대로 지켜진다. DCGM이 간격을 늘리지 않았다.
- 반대로 `-d 1`은 `-68: Watching this field at high frequencies causes undefined behavior`로 거부된다. 하한이 있다(정확한 값은 미확인, 100ms는 허용).
- 연속된 샘플 값이 거의 같아(예: 0.992가 반복) 실제 측정 갱신이 100ms마다 일어나는지는 값 변화만으로는 확인하지 못했다. 출력 행 수 기준 간격만 확인했다.

### 부하 시 값 (2-process small_kernels, GPU 1, 25초, 각 프로세스 약 6.1k iter/s)
- `nvidia-smi` GPU 사용률은 약 99%였는데 DCGM sm_active는 0.08, sm_occupancy는 0.006이다. gr_engine_active(0.99)는 nvidia-smi 사용률과 같은 의미에 가깝고, sm_active는 SM이 실제로 일하는 비율이라 작은 커널 다발에서는 매우 낮다. 막대 표시에서 gr_engine_active와 sm_active를 구분해 보여줄 근거가 된다.
- 유휴: 1001~1005 모두 0.000.
- 부하 구간 최소~최대는 위 표 참고. 변동폭이 매우 작다.

### 산출물
- fixture: `tests/fixtures/dcgmi_dmon_a5000.txt` (부하 시 실제 출력 30줄, 헤더 포함, 호스트명/UUID/PID 없음)
- 원본: `docs/verification/raw/03_*.txt` (git 제외)

## Task 4: nsys 전역 Context Switch 캡처 (G2)

(결과 없음)

## Task 5: SQLite 스키마 기록과 fixture (G3)

(결과 없음)

## Task 6: MPS OFF/ON 수집 차이 (G4)

(결과 없음)

## Task 7: CUPTI 적용 범위 스파이크 (G5, 반나절 제한)

(결과 없음)

## Task 8: 오버헤드 측정과 게이트 판정 (G6)

(결과 없음)

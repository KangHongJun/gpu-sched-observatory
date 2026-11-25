# 기술 검증 결과 (1주차)

각 게이트의 판정과 근거를 이 문서에 기록한다. 원본 출력은 `docs/verification/raw/`에 두며 git에 올리지 않는다.

## 게이트 표

| 게이트 | 질문 | 통과 시 | 실패 시 | 판정 |
|---|---|---|---|---|
| G1 | A5000에서 DCGM 1001~1005가 나오는가 | Live에 5개 막대 표시 | 해당 필드는 `N/A`, Live는 NVML 중심 | 통과: 1001~1005 전부 지원되고 0이 아닌 값이 관측됨(1004는 FP16 행렬곱 부하에서 0.951~0.959, small_kernels에서는 0.000). 1006만 미지원 |
| G2 | nsys가 외부 프로세스의 Context 전환을 PID와 함께 잡는가 | 3주차를 nsys 캡처로 진행 | 3주차를 "직접 띄운 프로세스의 커널 타임라인"으로 축소 | **잠정: 통과(비루트 기준).** nsys 2026.6.1이 sudo 없이, 대상 프로세스를 띄우지 않고도 외부 워크로드 A/B의 GPU 컨텍스트 전환 이벤트를 PID와 함께 기록했다. root 실행과의 차이(특히 타 사용자 프로세스의 PID 마스킹)는 미확인: 아래 "사용자 실행 대기" 참고 |
| G3 | SQLite export에서 전환·커널 테이블을 읽을 수 있는가 | Task 18을 실제 스키마로 구현 | nsys의 다른 export 형식(json)을 검토 | 미정 |
| G4 | MPS ON에서 전환 이벤트와 커널 중첩이 어떻게 달라지는가 | 결과대로 표시 규칙 확정 | 전환은 `N/A`, overlap과 앱 성능 우선 | 미정 |
| G5 | CUPTI를 외부 프로세스에 붙일 수 있는가 | MVP 이후 collector 설계 입력 | MVP 이후 과제로만 기록 | 미정 |
| G6 | 수집 오버헤드가 얼마인가 | README에 수치 기재 | 기본 수집 주기를 낮춘다 | 미정 |

## 사용자 실행 대기 (sudo)

에이전트는 sudo를 쓸 수 없어 아래는 사용자가 직접 실행해야 한다.

### Task 4 (G2): root 전역 캡처 `ctx_baseline`

터미널 1 (nsys 밖에서 외부 워크로드 두 개, GPU 1, 60초):

```bash
cd <repo>
for n in a b; do uv run python -m gso.bench.workloads.small_kernels --gpu 1 --duration 60 --out docs/verification/raw/04_wl_root_$n.json & done; wait
```

터미널 2 (워크로드가 돈 지 몇 초 뒤, 10초 캡처. sudo 환경에서 `$HOME`이 달라질 수 있어 NSYS를 명시한다):

```bash
cd <repo>
NSYS="$HOME/opt/nsys/opt/nvidia/nsight-systems-cli/2026.6.1/target-linux-x64/nsys"
sudo env NSYS="$NSYS" scripts/verify/04_nsys_ctxsw.sh capture ctx_baseline 10
sudo chown "$USER" docs/verification/raw/ctx_baseline.nsys-rep
scripts/verify/04_nsys_ctxsw.sh export ctx_baseline
```

- 생성되어야 하는 파일: `docs/verification/raw/ctx_baseline.nsys-rep`, `ctx_baseline.sqlite`.
- 확인할 것 (비루트 `ctx_nonroot.sqlite`와 같은 질의로 비교):
  1. `GPU_CONTEXT_SWITCH_EVENTS`의 GPU별(`gpuId`) 이벤트 수와 `globalPid` 종류 수. 비루트에서는 GPU 1에 워크로드 A/B 두 개, GPU 0에 `pid=0`으로 마스킹된 한 종류(3,357건)였다.
  2. root에서 GPU 0 이벤트의 `globalPid`가 실제 PID로 풀리는지(마스킹 해제 여부).
  3. 이벤트 태그 종류(비루트는 7, 8 두 종류뿐), 타임스탬프 정밀도(비루트는 ns 단위 상대 시각).
  4. 워크로드 A/B 이벤트 수가 비루트와 비슷한지.
- G2의 남은 질문: 타 사용자 프로세스 PID가 root에서만 보이는지. 이것은 비루트 데이터만으로는 답할 수 없다. 단 워크로드 A/B(같은 사용자)에 대한 판정은 비루트로 이미 얻었다.

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

**판정: 통과.** 1001~1005 다섯 필드가 모두 A5000에서 지원된다. 1001/1002/1003/1005는 small_kernels 부하에서 0이 아닌 값이 나왔고, 1004는 small_kernels에서는 0.000이었으나 FP16 행렬곱 부하에서 0.951~0.959로 0이 아닌 값이 관측됐다(`raw/03_dmon_fp16.txt`). 어느 하드웨어 경로를 쓴 결과인지는 확인하지 않았다.

### 환경
- 경로 A(컨테이너)만 사용했다. 이미지: `nvcr.io/nvidia/cloud-native/dcgm:4.8.0-ubuntu24.04` (실행 시점 NGC의 최신 4.x 태그). 호스트의 dcgmi 3.3.9와 sudo는 쓰지 않았다.
- 실행: `docker run -d --name gso-dcgm --gpus device=1 --cap-add SYS_ADMIN <image>` (이미지 기본 CMD가 `nv-hostengine -n -b 0.0.0.0 -f -`). 포트 게시 없음, 조회는 `docker exec gso-dcgm dcgmi ...`.
- 컨테이너 안에서는 GPU가 **인덱스 0**으로 열거된다. 컨테이너의 `nvidia-smi` UUID가 호스트 GPU 1의 UUID와 일치함을 확인했다(`raw/03_uuid_compare.txt`). 따라서 호스트 `-i 1`은 컨테이너 안 `-i 0`이다. 이후 Task는 `-i 0`을 써야 한다.
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
| 1004 tensor_active | 지원. small_kernels 부하에서 0.000. FP16 행렬곱 부하(4096x4096 float16 matmul, 약 18초, `raw/03_dmon_fp16.txt`)에서 관측: 0.951~0.959. 0.000의 원인은 확인하지 않았다 |
| 1005 dram_active | 지원. small_kernels 부하에서 0.007(재측정, `raw/03_dmon_load2.txt`; 첫 측정은 0.021). FP16 행렬곱에서 0.200~0.208 |
| 1006 | **미지원**. `Error setting watches. Result: -6: Feature not supported` |
| 1007, 1008 | 지원(유휴 0.000만 확인, 부하 값은 미확인) |
| 1009, 1010 (PCIe 바이트/초) | 지원. 유휴에서도 바이트 단위 정수가 찍힘 |
| 1011, 1012 | 항상 0 |

### 출력 형식 (파서 입력)
- 헤더 2줄: `#Entity` 뒤에 필드 ID 열, 다음 줄은 `ID`. 그다음 데이터 줄은 `GPU 0` + 값 열들이며, 열은 공백으로 정렬된다(구분자 없음, 줄 끝에 공백이 붙는다). 엔티티 열은 `GPU <index>` 두 토큰이다.
- 헤더 2줄이 **약 15행마다 다시 출력된다**(50샘플에서 1, 18, 34, 50번째 줄). 파서는 `#Entity`/`ID` 줄을 건너뛰어야 한다.
- 1001~1005 값은 **0~1 비율**이다(% 아님). 1009/1010은 바이트/초 정수.
- 미지원 필드는 `N/A` 표기가 나오지 않았다. 목록에 없는 필드(1006)는 `dmon`이 오류 한 줄(`Error setting watches ... -6`)을 내고 **다른 필드가 섞여 있어도 명령 전체가 실패**해 데이터 줄이 하나도 안 나온다. 존재하지 않는 ID(9999)는 `Got invalid field ID`로 실패한다(`raw/03_dmon_9999.txt`). 1006의 오류는 `raw/03_dmon_perfield.txt`. 따라서 "필드 하나가 N/A로 찍히는" 경우는 이번에 관찰하지 못했다. 지원 여부는 `profile -l`로 먼저 걸러야 한다.
- fixture에 `N/A`가 없는 이유가 이것이다(Brief의 "N/A 표기 그대로"는 해당 없음).

### 갱신 간격
- `-d 100 -c 50`의 벽시계 시간(`time` 측정, docker exec 오버헤드 포함)은 유휴 5.48초(`raw/03_timing_idle.txt`), small_kernels 부하 5.17초(`raw/03_timing_load.txt`), FP16 부하 5.20초(`raw/03_timing_fp16.txt`) → 100ms가 그대로 지켜진다. DCGM이 간격을 늘리지 않았다.
- 반대로 `-d 1`은 `-68: Watching this field at high frequencies causes undefined behavior`로 거부된다(`raw/03_dmon_d1.txt`). 하한이 있다(정확한 값은 미확인, 100ms는 허용).
- 연속된 샘플 값이 거의 같아(예: 0.992가 반복) 실제 측정 갱신이 100ms마다 일어나는지는 값 변화만으로는 확인하지 못했다. 출력 행 수 기준 간격만 확인했다.

### 부하 시 값 (2-process small_kernels, GPU 1, 25초, 각 프로세스 약 6.1k iter/s)
- small_kernels 2프로세스 부하(재측정): nvidia-smi GPU 1 사용률 99~100%(`raw/03_smi_load.txt`)였고, 같은 시간 DCGM은 gr_engine_active 0.982~0.992, sm_active 0.075~0.080, sm_occupancy 0.006~0.007, tensor_active 0.000, dram_active 0.007(`raw/03_dmon_load2.txt`). sm_active가 0.08인 원인은 확인하지 않았다. 필드 간 의미 관계도 이번에 검증하지 않았다.
- FP16 행렬곱 부하: nvidia-smi 사용률 100%(`raw/03_smi_fp16.txt`), gr_engine_active 0.996~1.000, sm_active 0.992~0.999, sm_occupancy 0.165~0.166, tensor_active 0.951~0.959, dram_active 0.199~0.208(`raw/03_dmon_fp16.txt`).
- 유휴: 1001~1005 모두 0.000.
- fixture의 원본 캡처(`raw/03_dmon_load.txt`)는 첫 측정이며 값 범위는 위 재측정과 같은 수준이다(1005만 0.021).

### 산출물
- fixture: `tests/fixtures/dcgmi_dmon_a5000.txt` (부하 시 실제 출력 30줄, 헤더 포함, 호스트명/UUID/PID 없음)
- 원본: `docs/verification/raw/03_*.txt` (git 제외)

## Task 4: nsys 전역 Context Switch 캡처 (G2)

**판정: 잠정.** 비루트로 외부 프로세스의 컨텍스트 전환 이벤트가 PID와 함께 캡처됐다. root 실행은 사용자 몫이라 "사용자 실행 대기" 절에 두었고, 그 결과가 나올 때까지 확정하지 않는다.

### 설치와 명령
- 패키지 `nsight-systems-cli-2026.6.1`, 버전 문자열 `2026.6.1.251-266139333969v0`. NVIDIA devtools 저장소(ubuntu2204/amd64)의 `Packages` 색인에서 가장 새 항목(같은 2026.6.1에 빌드 218과 251이 있어 251)을 골랐고 SHA256이 색인과 일치함을 확인했다. `dpkg -x`로 `~/opt/nsys`에 풀었다(apt/루트 없음, 누락 공유 라이브러리 없음). 바이너리: `~/opt/nsys/opt/nvidia/nsight-systems-cli/2026.6.1/target-linux-x64/nsys`.
- 이 버전의 옵션 철자는 `--gpuctxsw=true`, `--sample=none`, `--cpuctxsw=none`, `--duration`, `-o`, `--force-overwrite=true`이다(`nsys profile --help`). `perf_event_paranoid=4`라 CPU 샘플링과 OS 스레드 추적은 끄고 GPU 컨텍스트 전환만 본다.
- 스크립트: `scripts/verify/04_nsys_ctxsw.sh capture <name> [duration]`, `export <name>`. 대상 명령은 `TARGET` 환경 변수, 기본은 `sleep <duration>`. nsys는 대상 명령 없이는 실행하지 않아 `sleep`을 대상으로 했다(프로그램 자체의 GPU 활동 없음).

### 캡처 1: 비루트, 전역, 외부 워크로드 (`ctx_nonroot`)
- 워크로드 A/B(`small_kernels --gpu 1 --duration 60`, 2프로세스)를 nsys 밖에서 먼저 띄우고 15초 뒤 `capture ctx_nonroot 10`. 성공(종료 코드 0, `.nsys-rep` 4.1 MB, export 200,089 events). 워크로드 원본: `raw/04_wl_a.json`, `04_wl_b.json`.
- 테이블: `GPU_CONTEXT_SWITCH_EVENTS` 197,949행이 있다. `CUPTI_ACTIVITY_KIND_*`(커널/런타임) 테이블은 **없다**(외부 프로세스의 CUDA 커널은 기록되지 않음). 전체 테이블 목록: `raw/04_sqlite_look_nonroot.txt`.
- 열: `tag, vmId, seqNo, contextId, timestamp, globalPid, gpuId` (`raw/04_sqlite_look2_nonroot.txt`). `tag`는 `ENUM_GPU_CTX_SWITCH`의 값이며 이번 캡처에서는 7(SAVE_END)과 8(RESTORE_START)만 나왔다(각 약 99,000건).
- GPU별 구분: 된다. `gpuId`가 이벤트에 있고 `TARGET_INFO_GPU`의 PCI 버스 ID로 nvidia-smi 인덱스와 일치한다(nsys gpuId 1 = nvidia-smi GPU 1; `raw/04_smi_bus.txt`). GPU 1: 194,592건, GPU 0: 3,357건, GPU 2: 0건. GPU 0 이벤트는 다른 사용자의 워크로드로 보이며 우리 것이 아니다.
- PID 식별: **된다.** GPU 1 이벤트의 `globalPid`는 정확히 두 종류이고, `(globalPid >> 24) & 0xFFFFFF`가 워크로드 A/B의 PID(워크로드 JSON의 `pid`)와 각각 일치한다. `PROCESSES` 테이블(`globalPid`, `pid`, `name`)에도 두 PID가 있다(`raw/04_sqlite_look3_nonroot.txt`, `04_sqlite_look4_nonroot.txt`). 이 디코딩은 관측에서 맞춘 것이며 공식 문서로 확인하지 않았다.
- A/B 이벤트 수: 각각 약 97,000건(A 97,299, B 97,293). 10초 캡처에서 GPU 1 전체 194,592건. 시간순으로 A의 7,8 다음 B의 8,7이 번갈아 나온다(첫 12건 확인). `contextId`도 프로세스별로 하나씩이다.
- 마스킹: GPU 0의 다른 사용자 프로세스는 `globalPid`가 `1<<48`(PID 필드 0) 한 종류로만 나온다(3,357건, contextId 0). 같은 사용자의 워크로드는 마스킹되지 않았다. 다른 사용자 프로세스가 여러 개여도 하나로 합쳐 보이는지는 별도 확인하지 않았다(GPU 0 `nvidia-smi`에는 여러 PID가 있었다).
- 타임스탬프: 정수, 캡처 시작 기준 ns로 보인다(최소 259,841, 최대 약 10.117e9로 10초 캡처와 일치). 워크로드 A/B 교대 주기는 약 177 µs(첫 이벤트 쌍들의 간격에서). 2-process 시 p50 지연 0.166 ms(Task 2)와 같은 크기다.
- 부수 관찰: `PROCESSES` 테이블에 시스템 전체 2,110개 프로세스(이름 포함)가 들어간다. 캡처 파일은 반드시 git 밖(`raw/`)에만 둔다.

### 캡처 2: 비루트, nsys가 직접 띄운 대상 (`ctx_launched`)
- `TARGET="uv run python -m gso.bench.workloads.small_kernels --gpu 1 --duration 12 ..." capture ctx_launched 15` (워크로드 1개, 기본 trace 옵션 유지). 성공, `.nsys-rep` 42.6 MB.
- `CUPTI_ACTIVITY_KIND_KERNEL` 435,120행, `CUPTI_ACTIVITY_KIND_RUNTIME` 1,306,317행. 커널의 `globalPid`는 모두 대상 프로세스의 PID와 일치하고 `deviceId`는 전부 0이다(`gpuId`/GPU 번호와 어떻게 대응하는지는 이번에 확인하지 않았다: 미확인).
- 컨텍스트 전환: GPU 1 3건, GPU 0 4건뿐이다(`GPU_CONTEXT_SWITCH_EVENTS` 7행). 대상이 혼자라 전환이 거의 없는 것과 일치하지만 원인은 검증하지 않았다. 따라서 폴백(직접 띄운 프로세스의 커널 타임라인)은 비루트로 동작하며, 그 경우 2-process 경합 전환은 이 한 번의 캡처로는 보이지 않는다.
- 원본: `raw/ctx_launched.*`, `raw/04_sqlite_look_launched.txt`.

### 미확인
- root 캡처(`ctx_baseline`)와 비루트의 차이, 특히 타 사용자 프로세스의 PID 마스킹 해제 여부: 사용자 실행 대기.
- 7/8 외 태그가 부하 조건에서 나오는지, 전환 간격의 분포, MPS 상태에서의 거동(Task 6).
- 캡처 오버헤드가 워크로드에 미친 영향(G6, Task 8).

## Task 5: SQLite 스키마 기록과 fixture (G3)

(결과 없음)

## Task 6: MPS OFF/ON 수집 차이 (G4)

(결과 없음)

## Task 7: CUPTI 적용 범위 스파이크 (G5, 반나절 제한)

(결과 없음)

## Task 8: 오버헤드 측정과 게이트 판정 (G6)

(결과 없음)

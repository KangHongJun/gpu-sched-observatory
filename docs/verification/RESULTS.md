# 기술 검증 결과 (1주차)

각 게이트의 판정과 근거를 이 문서에 기록한다. 원본 출력은 `docs/verification/raw/`에 두며 git에 올리지 않는다.

## 게이트 표

| 게이트 | 질문 | 통과 시 | 실패 시 | 판정 |
|---|---|---|---|---|
| G1 | A5000에서 DCGM 1001~1005가 나오는가 | Live에 5개 막대 표시 | 해당 필드는 `N/A`, Live는 NVML 중심 | 통과: 1001~1005 전부 지원되고 0이 아닌 값이 관측됨(1004는 FP16 행렬곱 부하에서 0.951~0.959, small_kernels에서는 0.000). 1006만 미지원 |
| G2 | nsys가 외부 프로세스의 Context 전환을 PID와 함께 잡는가 | 3주차를 nsys 캡처로 진행 | 3주차를 "직접 띄운 프로세스의 커널 타임라인"으로 축소 | **잠정: 통과(비루트 기준).** nsys 2026.6.1이 sudo 없이, 대상 프로세스를 띄우지 않고도 외부 워크로드 A/B의 GPU 컨텍스트 전환 이벤트를 PID와 함께 기록했다. root 실행과의 차이(특히 타 사용자 프로세스의 PID 마스킹)는 미확인: 아래 "사용자 실행 대기" 참고 |
| G3 | SQLite export에서 전환·커널 테이블을 읽을 수 있는가 | Task 18을 실제 스키마로 구현 | nsys의 다른 export 형식(json)을 검토 | **통과(캡처 방식별로 범위가 다름, 비루트 기준).** 전역 캡처의 SQLite에서 컨텍스트 전환(`GPU_CONTEXT_SWITCH_EVENTS`)을 PID·GPU별로 읽을 수 있고, 직접 띄운 캡처의 SQLite에서 커널(`CUPTI_ACTIVITY_KIND_KERNEL`)을 읽을 수 있다. 한 파일에서 둘을 함께 얻는 것은 이번 캡처로는 안 된다. 전환 이벤트에서 만드는 구간은 두 구성(`restore_save` 41%, `hold` 약 100%)이 가능하고 데이터는 어느 쪽이 "실행 중"인지 정하지 못한다. 명세의 제출~실행 시작 시간은 얻을 수 없고(커널 테이블에 `start`/`end`뿐), API 호출 시작~커널 시작만 얻는다. 상세: `nsys-schema.md` |
| G4 | MPS ON에서 전환 이벤트와 커널 중첩이 어떻게 달라지는가 | 결과대로 표시 규칙 확정 | 전환은 `N/A`, overlap과 앱 성능 우선 | **잠정: `N/A` 쪽(비루트 기준).** MPS ON에서 클라이언트 2개를 돌린 10초 전역 캡처의 GPU 1에 컨텍스트 전환 이벤트가 0건이었다(OFF는 205,545건). 프로세스당 처리량은 약 5,660 -> 약 49,800~54,400 iter/s(단일 검증 실행(OFF 1회, ON 1회), 반복 없음, 벤치마크 결과 아님). 두 MPS 클라이언트의 커널 겹침은 미확인(이 게이트의 전환 쪽 질문에만 답했다). 표시 규칙은 아래 Task 6 절. root 캡처로는 같은지 미확인: 아래 "사용자 실행 대기" 참고 |
| G5 | CUPTI를 외부 프로세스에 붙일 수 있는가 | MVP 이후 collector 설계 입력 | MVP 이후 과제로만 기록 | 부분 통과 (Task 7: launch injection 동작, 외부 프로세스 부착은 미확인) |
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

### Task 6 (G4): root 전역 캡처 `ctx_mps_on_root` (MPS ON)

비루트로는 MPS ON의 GPU 1 전환 이벤트가 0건이었다. root에서도 0건인지(root에서만 MPS 서버나 클라이언트 이벤트가 보이는지)는 비루트 데이터로 답할 수 없다. 데몬은 에이전트 스크립트로 사용자 범위(GPU 1 전용)로 띄우고, 캡처만 root로 한다.

터미널 1 (데몬 시작과 클라이언트 두 개, 30초. 클라이언트 쪽 GPU 번호는 `--gpu 0`이다, Task 6 참고):

```bash
cd <repo>
scripts/verify/06_mps.sh start
eval "$(scripts/verify/06_mps.sh env)"
for n in a b; do uv run python -m gso.bench.workloads.small_kernels --gpu 0 --duration 30 --out docs/verification/raw/06_root_on_$n.json & done; wait
```

터미널 2 (클라이언트가 돈 지 몇 초 뒤, 10초 캡처):

```bash
cd <repo>
NSYS="$HOME/opt/nsys/opt/nvidia/nsight-systems-cli/2026.6.1/target-linux-x64/nsys"
sudo env NSYS="$NSYS" scripts/verify/04_nsys_ctxsw.sh capture ctx_mps_on_root 10
sudo chown "$USER" docs/verification/raw/ctx_mps_on_root.nsys-rep
scripts/verify/04_nsys_ctxsw.sh export ctx_mps_on_root
```

터미널 1에서 워크로드가 끝난 뒤 데몬 종료와 확인:

```bash
scripts/verify/06_mps.sh stop
pgrep -u "$USER" -a nvidia-cuda-mps   # 비어 있어야 한다
```

- 확인할 것: `GPU_CONTEXT_SWITCH_EVENTS`의 `gpuId=1` 이벤트 수와 `globalPid` 종류(클라이언트, MPS 서버, 마스킹). 비루트 `ctx_mps_on`은 GPU 1이 0건, GPU 0만 다른 사용자의 마스킹된 이벤트였다.

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
- A/B 이벤트 수: 각각 약 97,000건(A 97,299, B 97,293). 10초 캡처에서 GPU 1 전체 194,592건. 시간순으로 두 프로세스 모두 8(RESTORE_START) 다음 7(SAVE_END) 순이고, 첫 이벤트는 B의 8이며 이후 `B8, B7, A8, A7, B8, B7, A8, A7`로 프로세스가 번갈아 나온다(Task 5에서 원본 전체로 재확인). `contextId`도 프로세스별로 하나씩이다.
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

**판정: 통과(비루트, 캡처 방식별로 범위가 다름).** 스키마, 열 의미, 구간 규칙, 미확인 목록은 `docs/verification/nsys-schema.md`에 있다. 이번 작업은 root 캡처 없이 Task 4의 두 export(`ctx_nonroot`, `ctx_launched`)로 했다(Step 1 export는 Task 4에서 완료).

### 읽을 수 있는 것
- **전역 캡처**: `GPU_CONTEXT_SWITCH_EVENTS`(tag, vmId, seqNo, contextId, timestamp, globalPid, gpuId). `ENUM_GPU_CTX_SWITCH`가 태그 -1, 0~11을 정의한다(7 `SAVE_END`, 8 `RESTORE_START`가 관측된 전부이고 나머지 값의 의미/출현은 미확인). 시각은 캡처 시작 기준 ns. PID는 `(globalPid >> 24) & 0xFFFFFF`: 워크로드 JSON의 PID와 일치하고, `PROCESSES`의 2,110행 전부에서도 성립하지만(교차 검증) 공식 문서로 확인하지 않은 경험식이다. 다른 사용자는 `globalPid == 1<<48`(PID 0)로 합쳐져 프로세스별로 구분되지 않는다.
- **직접 띄운 캡처**: `CUPTI_ACTIVITY_KIND_KERNEL`(`start`, `end`, `deviceId`, `contextId`, `streamId`, `correlationId`, `globalPid`, 이름은 `StringIds` 참조). **큐/제출 시각 열은 없다**(`qmd*Done` 열은 있으나 전 행 NULL). API 호출 시작 시각(`CUPTI_ACTIVITY_KIND_RUNTIME`, `correlationId` 조인)과의 차이 `kernel.start - runtime.start`만 얻는다(p50 4.9 µs, 제출 시각이 아님). `deviceId`는 프로세스 안의 CUDA 번호(CUDA_VISIBLE_DEVICES=1이라 0)이고 실제 GPU는 `TARGET_INFO_CUDA_DEVICE(gpuId, cudaId, pid)`로 얻는다. 두 테이블의 `contextId`는 서로 다른 번호 공간이다.
- **유실 표시**: 전용 테이블/열은 없고 `DIAGNOSTIC_EVENT.text`의 문장("Number of GPU context switch events collected: N", CUPTI 이벤트 누락/폐기 경고)과 `seqNo`의 연속성이 전부다. 이번 전역 캡처는 수집 개수가 행 수와 같고 `seqNo` 틈이 0건이다.

### 전환 이벤트에서 구간 만들기 (전역 캡처 GPU 1, 워크로드 A/B)
- 문자 그대로: 8=`RESTORE_START`, 7=`SAVE_END`. 해석은 두 가지이고 데이터는 고르지 못한다: `restore_save`(같은 컨텍스트의 8→다음 7)는 시간폭의 41.1%(컨텍스트당 약 20%, 구간 97,200개, 짝 없는 7 136건, 재개방 8 56건), `hold`(다른 컨텍스트의 다음 8까지 쥠)는 100.0%(컨텍스트당 약 50%). 경합하는 두 프로세스가 GPU를 채워도 `restore_save`로는 약 20%가 나오므로, 이 비율은 "점유"가 아니라 "RESTORE_START→SAVE_END 구간 비율"이라 부르고 실제보다 약 2.5배 낮을 수 있다. 서로 다른 컨텍스트 구간이 겹치지 않는 것은 두 구성 모두에서 성립하므로 어느 쪽의 근거도 아니다.
- 캡처 창은 하나로 정한다: `[0, w1]`, `w1` = max("Profiling has stopped." 진단 행의 timestamp, 이벤트 timestamp 최댓값). 열린 구간은 버리지 않고 이 창으로 자른다. 단독 프로세스 캡처에서는 열린 구간을 버리면 `restore_save`가 약 0.15%(닫힌 8→7 한 쌍 23 ms/15 s)이고, 창으로 자르면 두 구성 모두 76.1%(11.4 s/15.001 s)다. 첫 이벤트가 7인 컨텍스트(캡처 전에 이미 복원됨)는 `[0, 그 7]`로 처리하는 정책을 택했고 GPU 1 전역 캡처에서는 0건이다. nvidia-smi 사용률은 다른, 더 거친 측정이며 이 구간과 대조해 보지 않았다(미확인).
- 구간 사이 간격 p50 52.6 µs(7→8)를 전환 비용이라 부르지 않는다. 한계: 한 쌍의 프로세스·GPU 1에서만 확인했고 경합 중 구간을 커널 실행과 대조하지 못했다.
- 커널 쪽: 큐/제출 시각이 없어 "제출~실행 시작"은 얻을 수 없고 API 호출 시작~커널 시작(p50 4.9 µs)만 얻는다. 이 워크로드는 매 실행 뒤 동기화하므로 큐가 빈 상태에서 잰 값이며, 비동기/경합 시에는 앞선 커널의 시간이 섞인다.

### 비루트 전역 캡처의 마지막 이벤트
- 마지막 이벤트(10.117 s)가 "Profiling has stopped"(10.000 s)보다 늦다. 원인 미확인.

### fixture
- `tests/fixtures/nsys_ctxsw_min.sqlite`(393 KB): 전역 캡처에서 워크로드 A/B 이벤트(캡처 시작 후 0.6초 미만, 11,440행)와 마스킹 행 50개. 지시한 2초는 500 KB 한도를 넘어(1초가 약 630 KB) 0.6초로 줄였다.
- `tests/fixtures/nsys_launched_min.sqlite`(352 KB): 직접 띄운 캡처에서 앞 2,000 커널, 대응하는 런타임 행, 참조된 `StringIds`, 컨텍스트 전환 7행.
- 둘 다 `PROCESSES` 없음, UUID는 NULL, 호스트/사용자/명령줄 정보 없음. `scripts/verify/make_nsys_fixture.py`가 끝에서 개인정보 검사를 수행한다.

### 미확인
- root 캡처에서 PID 마스킹이 풀리는지, 태그 7/8 외 값의 출현, 경합 중 구간과 커널 실행의 대응, 3개 이상 컨텍스트·MPS에서의 규칙(Task 6), 구간 사이 간격의 정체. 전체 목록은 `nsys-schema.md` 11절.


## Task 6: MPS OFF/ON 수집 차이 (G4)

**판정: 잠정.** 비루트 전역 캡처에서 MPS ON의 GPU 1 컨텍스트 전환 이벤트는 클라이언트 2개가 돌 때 0건이었다(`raw/06_derive.txt`). root 캡처로도 같은지는 미확인이라 "사용자 실행 대기"에 두었다. 모든 수치는 같은 날, 같은 스크립트, 같은 워크로드(2프로세스, 30초, 캡처 10초)로 얻었다.

### 데몬과 명령
- `scripts/verify/06_mps.sh start|stop|status|env`: 사용자 범위, `CUDA_VISIBLE_DEVICES=1`로 GPU 1 전용, pipe/log 디렉터리는 `/tmp/gso-mps-<uid>/` 아래. `start`는 어떤 사용자의 MPS 제어 데몬/서버든 이미 있으면 거부한다: 15자로 잘린 comm(`pgrep -x nvidia-cuda-mps`)과 전체 명령줄(`pgrep -f`, `nvidia-cuda-mps-(control|server)`)을 둘 다 검사한다(오탐은 허용). 처음 쓴 `pgrep nvidia-cuda-mps-control`은 comm이 15자로 잘려 실제 데몬을 못 찾는 것이 리뷰에서 확인되어 고쳤고, 실제 데몬 없이 이름만 같은 대역 프로세스로 거부 동작을 확인했다(`raw/06_fix1_guard.txt`). 첫 실행의 시작 전 확인(MPS 프로세스 없음)은 손으로 했다. `/tmp/gso-mps-<uid>`는 mode 700으로 만들고, 심볼릭 링크이거나 현재 사용자 소유 디렉터리가 아니면 거부한다. `stop`은 우리 pipe 디렉터리에만 `quit`을 보낸 뒤 이 사용자의 `nvidia-cuda-mps` 프로세스가 없는지 확인한다. 시작 전 MPS 프로세스 없음, GPU 1 유휴(0%, 1 MiB)를 확인했다. 시스템 기본 pipe 디렉터리, compute mode, sudo는 쓰지 않았다(compute mode는 전 GPU Default 그대로).
- MPS OFF: `small_kernels --gpu 1 --duration 30` 두 개(`raw/06_off_a.json`, `06_off_b.json`)를 띄우고 14초 뒤 `capture ctx_mps_off 10`, `export ctx_mps_off`.
- MPS ON: `eval "$(scripts/verify/06_mps.sh env)"` 뒤 `small_kernels --gpu 0 --duration 30` 두 개(`raw/06_on_a.json`, `06_on_b.json`)를 띄우고 12초 뒤 `nvidia-smi`와 `get_*` 명령을 실행, 이어서 `capture ctx_mps_on 10`, `export ctx_mps_on`.

### 클라이언트에서 보이는 GPU 인덱스 (워크로드 `--gpu` 값)
- 데몬을 `CUDA_VISIBLE_DEVICES=1`로 띄우면 클라이언트에게 보이는 장치는 1개이고 번호는 **0**이다. 클라이언트 환경이 `CUDA_VISIBLE_DEVICES=1`이면 CUDA 초기화가 실패한다: torch는 "No CUDA GPUs are available"(`raw/06_client_index.txt`), MPS 서버 로그는 "Client requested 1 which is not a valid GPU ID in the MPS visible set of devices ... less than 1 or consider using UUIDs"(`raw/06_mps_server.log`). GPU UUID를 주면 장치 1개(`count 1`)로 열리고 PCI 버스 171(=0xAB, nvidia-smi GPU 1)이었다. 이 장치 수/버스 출력은 터미널에서만 확인했고 저장하지 않았다. 같은 실행 중의 `nvidia-smi`는 `raw/06_smi_probe_uuid.txt`에 있고 GPU 1에 `M+C` 클라이언트가 보인다.
- `select_gpu(N)`은 프로세스 안에서 `CUDA_VISIBLE_DEVICES=N`을 덮어쓴다. 따라서 **MPS ON에서는 `--gpu 0`이 호스트 GPU 1에서 돈다**(`raw/06_probe_gpu0.json`, `06_smi_probe_gpu0arg.txt`: 해당 프로세스가 GPU 1 항목에만 `M+C`로 보이고 GPU 0 항목에는 없음). `--gpu 1`은 실패한다. 커밋된 워크로드 코드는 바꾸지 않았다. 호스트 번호를 그대로 넘기는 MPS 벤치 러너는 이 호환 문제를 처리해야 한다(`--gpu` 값을 MPS ON일 때 0으로 매핑하거나 `select_gpu`가 이미 설정된 `CUDA_VISIBLE_DEVICES`를 존중하게 하는 둘 중 하나 필요. 어느 쪽인지는 미결정).
- nsys 쪽: 클라이언트를 nsys로 띄운 캡처의 `TARGET_INFO_CUDA_DEVICE`는 `cudaId=0`이 `gpuId=1`로 대응했다(`raw/06_look_launched1.txt`). 전역 캡처의 `gpuId`는 OFF/ON 모두 nvidia-smi 인덱스와 같은 쪽으로 읽혔다(GPU 1에 우리 이벤트, GPU 0에 다른 사용자 이벤트).

### nvidia-smi와 MPS 명령 출력
- OFF(`raw/06_smi_off.txt`): GPU 1에 워크로드 프로세스 2개, Type `C`.
- ON(`raw/06_smi_on.txt`): GPU 1에 `nvidia-cuda-mps-server` 1개(Type `C`, 30 MiB)와 클라이언트 2개(Type `M+C`, 각 236 MiB). 클라이언트 PID는 워크로드 JSON의 `pid`와 `get_client_list` 출력의 PID와 같다.
- fixture: `tests/fixtures/mps_server_list.txt`(`get_server_list`: 서버 PID 한 줄), `tests/fixtures/mps_client_list.txt`(`get_client_list <서버 PID>`: 클라이언트 PID 두 줄). `get_active_thread_percentage <서버 PID>`는 `100.0`(`raw/06_atp_server.txt`). 이 명령들은 서버 PID를 인자로 받는다.
- 데몬이 없을 때: `tests/fixtures/mps_server_list_empty.txt`에 `Cannot find MPS control daemon process` 한 줄(우리 pipe 디렉터리, stdout+stderr), 종료 코드 1. 데몬이 떠 있을 때 명령의 종료 코드는 0이다.

### OFF 대비 ON (같은 캡처 창 `[0, w1]`, GPU 1)

**단일 검증 실행(OFF 1회, ON 1회), 반복 없음, 벤치마크 결과 아님.**

| 항목 | MPS OFF (`ctx_mps_off`) | MPS ON (`ctx_mps_on`) |
|---|---|---|
| GPU 1 전환 이벤트 | 205,545건 (태그 7: 102,818, 태그 8: 102,727) | 0건 |
| GPU 1 `globalPid` 종류 | 2개: 워크로드 A, B로 디코드 | 없음 |
| GPU 0 이벤트(타 사용자, 마스킹) | 6,284건 | 6,264건 |
| `restore_save` | 38.8% (A 19.5%, B 19.3%) | 구간 없음 |
| `hold` | 100.0% (A 50.1%, B 49.9%) | 구간 없음 |
| 창 길이 `w1` | 10.138 s | 10.056 s |
| 처리량 A / B (iter/s) | 5,655 / 5,667 | 49,779 / 54,388 |
| p50 A / B (ms) | 0.1661 / 0.1661 | 0.0190 / 0.0176 |
| p95 A / B (ms) | 0.1799 / 0.1799 | 0.0238 / 0.0219 |
| p99 A / B (ms) | 0.1836 / 0.1834 | 0.0310 / 0.0288 |
| 합계 처리량 | 11,322 | 104,167 |

출처: 이벤트와 구간 `raw/06_derive.txt`(`scripts/verify/05_fix1_derive.py`의 `report`/`build` 재사용), 처리량과 지연 `raw/06_off_*.json`, `raw/06_on_*.json`. 참고로 단독 실행은 Task 2에서 53,792 iter/s(non-MPS), MPS ON 단독 한 번은 53,811 iter/s(`raw/06_probe_gpu0.json`, 12초)였다.
- OFF 결과는 Task 4/5(`ctx_nonroot`: 194,592건, `restore_save` 41.1%, `hold` 100.0%)와 같은 모양이다.
- ON 캡처에서 `PROCESSES`에는 `nvidia-cuda-mps` 이름의 프로세스가 두 개(제어 데몬, 서버) 있었으나 GPU 1 이벤트가 없어 이벤트 PID는 클라이언트, 서버, 마스킹 중 어느 것도 아니다(0건). GPU 0의 이벤트 6,264건은 전부 마스킹된 한 종류다.
- 두 클라이언트의 커널 구간이 겹치는지: **미확인**. 전역 캡처에는 커널 테이블이 없고(OFF와 같음), 아래 nsys 실행 캡처는 한 클라이언트의 커널만 담는다.

### nsys가 직접 띄운 MPS 클라이언트
- 단일 클라이언트(`ctx_mps_on_launched`, `capture ... 15`, 대상은 `--gpu 0 --duration 12`): 성공. `CUPTI_ACTIVITY_KIND_KERNEL` 496,333행, `..._RUNTIME` 1,489,956행. 커널의 `globalPid`는 클라이언트 PID 하나이고 MPS 서버 PID가 아니다. `deviceId`는 0. GPU 1 전환 이벤트는 태그 8 한 건이다(`raw/06_look_launched1.txt`).
- 클라이언트 2개 중 하나만 nsys로 띄움(`ctx_mps_on_launched2`: 나머지 하나는 nsys 밖에서 30초, 5초 먼저 시작): 성공. 커널 502,847행이 전부 nsys로 띄운 클라이언트의 것(PID 한 종류). 다른 클라이언트의 커널은 기록되지 않았다. 이 동안 두 클라이언트 처리량은 각각 44,388(nsys 아래), 50,910(밖) iter/s(`raw/06_on_launched2.json`, `06_on_other.json`). nsys 아래 값은 오버헤드를 포함하므로 위 표와 비교하지 않는다(G6).
- 같은 캡처의 GPU 1 전환 이벤트는 13건(태그 7: 6, 8: 7)이고 캡처 시작 후 3.43~3.45 s에 몰려 있다(nsys로 띄운 클라이언트의 첫 커널은 3.53 s). 13건의 `globalPid`는 한 종류이며 nsys 밖에서 돌던 클라이언트로 디코드됐다(`raw/06_look_launched2.txt`). 이후 12초 동안 GPU 1 이벤트는 없다. 원인은 미확인.
- 즉 MPS 클라이언트에 대해서도 직접 띄운 캡처는 커널 테이블을 만든다. 동시에 돌아가는 다른 클라이언트의 커널 구간은 얻지 못한다(둘 다 nsys로 띄우는 실험은 하지 않음).

### MPS ON일 때 CTX SW/s 표시 규칙
1. **MPS 감지.** 대상 GPU의 `nvidia-smi` 프로세스 목록에 `nvidia-cuda-mps-server`가 있거나 어떤 프로세스의 Type에 `M`이 들어 있으면(`M+C`) 그 GPU는 MPS ON이다. MPS 제어 명령(`get_server_list`)은 해당 pipe 디렉터리를 아는 같은 사용자만 쓸 수 있어 감지 근거로 쓰지 않는다(보조 정보일 뿐).
2. **표시.** MPS ON인 GPU의 CTX SW/s 칸은 숫자 대신 `N/A (MPS)`로 쓴다. `0/s`로 쓰지 않는다: 이벤트 수 0이 "전환이 없다"인지 "기록되지 않는다"인지 이번 데이터로 가를 수 없고, 같은 조건에서 시작 시점의 이벤트 몇 건이 나온 경우도 있었다. 어느 쪽이든 GPU가 MPS ON이면 이벤트 수와 상관없이 `N/A (MPS)`다.
3. **MPS OFF**(감지 안 됨)에서는 기존대로 이벤트 수에서 계산한 값을 쓴다.
4. **보조 표시.** MPS ON에서는 클라이언트별 처리량·지연(앱 성능)을 우선 보여 주고, 도움말에 "MPS ON: 컨텍스트 전환 이벤트가 수집되지 않음"을 붙인다. 커널 겹침은 표시하지 않는다(전역 캡처에 커널이 없고 둘 다 nsys로 띄우는 경우는 검증하지 않음).
5. 이 규칙은 비루트 캡처 기준이다. root 캡처에서 MPS ON의 이벤트가 달라지면 규칙 2를 다시 본다.

### 미확인
- root 캡처에서 MPS ON의 GPU 1 이벤트(0건인지, 서버 PID가 보이는지): 사용자 실행 대기.
- 두 클라이언트의 커널 구간이 겹치는지. MPS ON에서 이벤트가 0건인 원인. `launched2`의 시작 시점 13건의 원인.
- 클라이언트 2개를 초과하거나 다른 워크로드(큰 커널)일 때의 거동. MPS 서버가 활성 스레드 비율 제한(`get_active_thread_percentage` 100.0 외 값)을 가질 때.
- 데몬 정리: 실험 끝에 `06_mps.sh stop`으로 종료했고 `pgrep -u "$USER" -a nvidia-cuda-mps`가 비었다. 서버/제어 로그 원본은 `raw/06_mps_control.log`, `06_mps_server.log`.

## Task 7: CUPTI 적용 범위 스파이크 (G5, 반나절 제한)

CUPTI 13.x(헤더·라이브러리 모두 API 버전 130202). 프로젝트 venv의 PyTorch 의존 휠(`nvidia/cu13`)에 `cupti.h`와 `libcupti.so.13`이 있어 컨테이너 없이 g++로 빌드했다. 단 `crt/host_defines.h`는 cu13 휠에 없어 같은 venv의 triton 휠 헤더를 `-idirafter`로 추가했다(`scripts/verify/07_cupti/build.sh`가 컴파일 명령을 출력). 모든 실행은 GPU 1, root 아님, 워크로드는 `small_kernels` 5초. 아래 수치는 모두 n=1 단일 실행이다.
- **launch injection 동작.** `CUDA_INJECTION64_PATH`로 `InitializeInjection`이 호출되었고 `CONCURRENT_KERNEL`, `COMPUTE_ENGINE_CTX_SWITCH` 두 종류 모두 enable 성공(오류 없음). 종료 시 flush로 합계 출력, dropped 0 (`raw/07_inj.stderr`, `raw/07_pair_inj.stderr`).
- **(a) 주입 프로세스 단독:** 커널 레코드 269793건, 전환 레코드 2건(START 1, END 1), 전환 레코드의 contextId는 1종(`raw/07_inj.stderr`).
- **(b) 주입 1 + 비주입 1이 GPU 1을 공유:** 주입 프로세스에 전환 레코드 24478건(START 12237, END 12241) 도착. contextId는 1종(커널 레코드의 contextId와 같은 값 1)이고 레코드에 다른 프로세스를 구분할 필드가 없다(`raw/07_pair_inj.stderr`). 다른 프로세스의 컨텍스트가 별도 ID로 보이지는 않았다. 컨텍스트 ID는 프로세스 내 번호라 다른 프로세스 것과 같은 값일 수 있어, "자기 것만 온다"고 단정하지는 않고 "ID로 상대를 구별할 수 없다"로 기록한다. 레코드 필드는 `contextId`, `timestamp`, `operationType`뿐이고 PID·상대 컨텍스트는 없다(`cupti_activity.h`).
- **(c) 처리량(n=1):** 주입 단독 53956/s, 비주입 단독 55048/s(`raw/07_inj.json`, `raw/07_base.json`). 공유 실행 쌍에서는 주입 6902/s, 비주입 6807/s(`raw/07_pair_inj.json`, `raw/07_pair_plain.json`). 단독 비교 차이 약 2%는 단일 실행이라 잡음과 구분하지 못한다.
- **실행 중인 프로세스에 붙이기(문서 기반 결론, 실험하지 않음).** 주입 라이브러리는 CUDA 초기화 때 읽히므로 이미 실행 중인 프로세스에는 환경 변수 방식으로 붙일 수 없다. 헤더에는 늦게 붙이는 경우의 제약(일부 레코드는 CUDA 초기화 전에 붙어야 얻음)이 적혀 있어 CUPTI 자체의 늦은 활성화는 가능하지만 대상 프로세스 안에서 코드가 실행되어야 한다. 외부 프로세스에 코드를 넣는 방법(ptrace 등)은 시도하지 않았고 **미확인**.
- 필요한 권한: 일반 사용자 프로세스에서 위 레코드가 도착했다(root 불필요). 단 이는 내가 직접 띄운 프로세스에 한한다.
- **판정 G5: 부분 통과.** 새로 띄우는 프로세스에는 launch injection으로 붙일 수 있고, 시분할 중 전환 레코드(START/END)가 일반 사용자 프로세스에 도착한다. 이미 실행 중인 외부 프로세스에 붙이는 것과 상대 프로세스 식별은 확인되지 않았다.
- post-MVP collector가 의존할 수 있는 것: 자기가 띄운 프로세스의 커널/전환 시각. 의존할 수 없는 것: 전환 상대의 PID, 남이 띄운 프로세스 관측(전환 레코드에 PID가 없음).

## Task 8: 오버헤드 측정과 게이트 판정 (G6)

(결과 없음)

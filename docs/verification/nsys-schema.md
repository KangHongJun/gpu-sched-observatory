# nsys SQLite export 스키마 기록 (G3)

`gso`의 로더(`load_ctx_events`, `load_kernels`, `ctx_intervals`)가 이 문서와 `tests/fixtures/`의 fixture만 보고 짜질 수 있도록, Nsight Systems가 내보낸 SQLite의 실제 스키마와 관측 결과를 적는다. 모든 문장은 아래 두 캡처에서 확인한 것이며, 확인하지 못한 것은 "미확인"으로 표시했다. 식별자(테이블/열 이름)는 영어 그대로 쓴다.

## 0. 출처와 한계

| 항목 | 값 |
|---|---|
| nsys | 2026.6.1 (`EXPORT_PRODUCT_VERSION` = `2026.6.1.251`), export 스키마 버전 `EXPORT_SCHEMA_VERSION` = `3.31.4` (`META_DATA_EXPORT`) |
| 플랫폼 | `EXPORT_PLATFORM` = `linux-desktop`, `EXPORT_ARCH` = `x64` |
| GPU | RTX A5000 3장. nsys `gpuId` 1 = nvidia-smi 인덱스 1 (§2.3) |
| 캡처 1 `ctx_nonroot` | 비루트, 전역(`--gpuctxsw=true`), 외부 워크로드 A/B 2프로세스(GPU 1), 10초. `GPU_CONTEXT_SWITCH_EVENTS` 197,949행. CUDA 커널 테이블 없음 |
| 캡처 2 `ctx_launched` | 비루트, nsys가 워크로드 1개를 직접 띄움(GPU 1), 15초. `CUPTI_ACTIVITY_KIND_KERNEL` 435,120행. 컨텍스트 전환은 7행뿐 |
| 미수행 | root 캡처(`ctx_baseline`). 아래 결과는 모두 비루트에서 얻은 것이고, root에서 다른 사용자의 PID 마스킹이 풀리는지는 미확인 |

원본 export는 `docs/verification/raw/`(git 제외)에만 있고, 같은 머신 모든 프로세스를 담은 `PROCESSES` 테이블이 들어 있어 커밋하지 않는다. 이 문서의 표본 행은 모두 fixture에서 가져왔다. fixture 안의 `globalPid`/`globalTid`는 우리 워크로드의 실제 값이지만, 문서에는 PID를 적지 않고 `(1<<48)|(PID_A<<24)`처럼 기호로 적었다.

## 1. 전역 캡처 vs 직접 띄운 캡처

| 테이블 | 전역 캡처 (`ctx_nonroot`) | 직접 띄운 캡처 (`ctx_launched`) |
|---|---|---|
| `GPU_CONTEXT_SWITCH_EVENTS` | 있음, 197,949행. 외부 프로세스 A/B 이벤트 포함 | 있음, 7행뿐 (대상 혼자라 전환이 거의 없음. 원인은 검증하지 않음) |
| `CUPTI_ACTIVITY_KIND_KERNEL` / `_RUNTIME` / `_SYNCHRONIZATION` | **없음** (외부 프로세스의 CUDA 커널은 기록되지 않음) | 있음 (커널 435,120행, 런타임 1,306,317행) |
| `TARGET_INFO_CUDA_DEVICE` / `_CONTEXT_INFO` / `_STREAM` | 없음 | 있음 |
| `TARGET_INFO_GPU`, `ENUM_GPU_CTX_SWITCH`, `StringIds`, `DIAGNOSTIC_EVENT`, `META_DATA_EXPORT` | 있음 | 있음 |
| `PROCESSES` | 있음 (2,110행, 시스템 전체. 커밋 금지) | 있음 (2,328행. 커밋 금지) |

결론: **커널 타임라인은 직접 띄운 프로세스에서만, 외부 프로세스 간 컨텍스트 전환은 전역 캡처에서만** 얻는다. 한 파일에서 "경합하는 두 프로세스의 전환 + 각자의 커널"을 함께 얻는 방법은 이번 캡처로는 확인하지 못했다(미확인). 이 구분 때문에 fixture도 둘로 나뉜다: `nsys_ctxsw_min.sqlite`(전역), `nsys_launched_min.sqlite`(직접).

## 2. 공통 규약

### 2.1 타임스탬프
- 모든 시각 열(`timestamp`, `start`, `end`)은 정수 **ns**이고, 캡처 시작을 0으로 한 상대값이다(`DIAGNOSTIC_EVENT`의 "Profiling has started."가 `timestamp` 0이고, 10초 캡처의 "Profiling has stopped."가 10,000,366,207). export 옵션 `EXPORT_PARAM_TIME_NORMALIZE=false`, `TIME_SHIFT=0`(`META_DATA_EXPORT`, 원본에서 확인).
- 캡처 시작의 UTC 시각은 `TARGET_INFO_SESSION_START_TIME.utcEpochNs`(예: 전역 캡처 fixture 한 행)이다. 이 값이 `timestamp` 0에 해당한다고 보지만 서브 ms 정밀도로 검증하지는 않았다(미확인).
- 컨텍스트 전환 이벤트와 CUPTI 커널 시각이 **같은 시간축**인지는 한 파일에 둘이 함께 있는 직접 띄운 캡처에서 간접 확인만 했다: 단독 컨텍스트의 마지막 `RESTORE_START`(3.606 s) 이후 첫 커널이 3.629 s에 나온다. 정밀한 정렬(오프셋)은 미확인. 두 시계가 다른 도메인이면 `TIME_DOMAIN`/`TIME_SYNCHRONIZATION`(실험적 테이블, 문서에 "의존하지 말 것"이라 쓰여 있음)이 있으나 읽어보지 않았다.
- 전역 캡처에서 마지막 이벤트(10.117 s)가 "Profiling has stopped."(10.000 s)보다 117 ms 늦다. 원인은 미확인이다. 구간을 캡처 길이로 자르면 마지막 이벤트가 사라진다.

### 2.2 PID: `globalPid`
- `globalPid`는 "Serialized GlobalId"(열 주석)이다. **PID = `(globalPid >> 24) & 0xFFFFFF`** (SQLite에서도 같은 식, 음수에서도 동일).
- 이 식은 **관측으로 맞춘 경험식**이다. 공식 문서로는 확인하지 못했다. 확인 근거:
  1. 전역 캡처 GPU 1 이벤트의 디코딩 PID가 정확히 두 값이고, 각각 `raw/04_wl_a.json`, `04_wl_b.json`의 `pid`와 일치한다. 직접 띄운 캡처의 커널 435,120행 전부가 `04_wl_launched.json`의 `pid`와 일치한다.
  2. `PROCESSES(globalPid, pid, name)`의 2,110행 전부에서 `(globalPid>>24)&0xFFFFFF = pid`가 성립한다(불일치 0행). `PROCESSES`가 이 식의 교차 검증 근거다.
- 구조: 우리 프로세스의 `globalPid` = `(1<<48) | (PID<<24)`, 하위 24비트는 0. 스레드용 `globalTid`(런타임 테이블)는 같은 구조에 하위 24비트가 TID(메인 스레드는 TID=PID)이다. 비트 48 이상의 `1`의 의미(VM/호스트 ID로 보임)는 미확인.
- `DIAGNOSTIC_EVENT.globalPid`에는 `-144115188075855872` 같은 음수 값도 나온다. 디코딩하면 PID 0이다. 이 열은 fixture에서 NULL로 비웠다.

### 2.3 마스킹된 PID
- 같은 머신의 **다른 사용자** 프로세스의 이벤트는 `globalPid == 1<<48`(= 281474976710656, PID 필드 0)로만 나오고 `contextId`도 0이다. 전역 캡처의 GPU 0에서 3,357건(GPU 0 전체)이 모두 이 값이다. 같은 사용자의 워크로드는 마스킹되지 않는다.
- 다른 사용자 프로세스가 여러 개여도 하나의 값으로 합쳐지므로, **마스킹된 행은 프로세스별로 구분할 수 없다**. 마스킹된 행끼리의 `RESTORE_START`/`SAVE_END` 짝은 여러 프로세스가 섞인 것이다(GPU 0에서 짝 1,678개, 길이 10.9 µs ~ 111 ms — 의미 없음). 로더는 `globalPid == 1<<48` 행을 "외부(식별 불가)"로 분류하고 구간 규칙(§4.3)을 적용하지 말아야 한다.
- root 캡처에서 마스킹이 풀리는지는 미확인.

### 2.4 GPU 번호
- 이벤트의 `gpuId`는 `TARGET_INFO_GPU.id`와 같다. `TARGET_INFO_GPU(id, busLocation)`는 `0→0000:3e:00.0`, `1→0000:ab:00.0`, `2→0000:be:00.0`이고, nvidia-smi 버스 ID(`00000000:3E:00.0` 등, `raw/04_smi_bus.txt`)와 대소문자만 달라 **nsys gpuId N = nvidia-smi 인덱스 N**이다(이 호스트에서). 조인은 버스 ID를 소문자로 맞추고 앞 `0000`(nsys) / `00000000`(nvidia-smi) 자릿수를 정규화해야 한다. 이 대응은 `CUDA_DEVICE_ORDER=PCI_BUS_ID`일 때 nvidia-smi 순서와 같다.
- `TARGET_INFO_GPU.uuid`는 원본에 있으나 fixture에서는 NULL로 비웠다. UUID는 문서/fixture에 쓰지 않는다.

## 3. 컨텍스트 전환 테이블

### 3.1 `GPU_CONTEXT_SWITCH_EVENTS`
```sql
CREATE TABLE GPU_CONTEXT_SWITCH_EVENTS (
    tag                         INTEGER   NOT NULL,                    -- REFERENCES ENUM_GPU_CTX_SWITCH(id)
    vmId                        INTEGER   NOT NULL,                    -- VM ID.
    seqNo                       INTEGER   NOT NULL,                    -- Sequential event number.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    timestamp                   INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    gpuId                       INTEGER                                -- GPU index.
);
```
| 열 | 의미 |
|---|---|
| `tag` | 이벤트 종류. `ENUM_GPU_CTX_SWITCH.id` (§3.2) |
| `vmId` | VM ID. 관측한 모든 행에서 0 |
| `seqNo` | 순번. **GPU별로** 시간순에서 1씩 증가하며 65536에서 0으로 돌아간다 (§6) |
| `contextId` | GPU 컨텍스트 ID(하드웨어 쪽 번호). 프로세스당 하나였다(워크로드 A/B에 각각 하나). **CUDA 커널 테이블의 `contextId`와 다른 번호 공간**이다: 직접 띄운 캡처에서 이벤트는 65194, 같은 프로세스의 커널 행은 1. 둘의 대응표는 찾지 못했다(미확인). 프로세스 단위 연결은 `globalPid`로 한다 |
| `timestamp` | ns, 캡처 시작 기준 (§2.1). 이 이벤트의 시각(주석: "Event start timestamp") |
| `globalPid` | §2.2/§2.3 |
| `gpuId` | §2.4 |

행 순서: rowid 순서가 시간순이었다(GPU 1의 194,592행 전부). 그래도 로더는 `ORDER BY gpuId, timestamp, seqNo`로 정렬한다. GPU 1에서 `timestamp` 중복은 없었다.

### 3.2 `ENUM_GPU_CTX_SWITCH` (태그 의미)
```sql
CREATE TABLE ENUM_GPU_CTX_SWITCH (
    -- GPU context switch labels

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
    name                        TEXT,                                  -- Enum symbol name.
    label                       TEXT                                   -- Enum human name.
);
```
export가 정의한 태그 전체(fixture와 원본이 같다):

| id | name | label |
|---|---|---|
| -1 | `UnknownGpuCtxswTag` | Unknown |
| 0 | `SOF` | SO |
| 1 | `REQ_BY_HOST` | Request by host |
| 2 | `FE_ACK` | Front-end ACK |
| 3 | `FE_ACK_WFI` | Front-end ACK WFI |
| 4 | `FE_ACK_GFXP` | Front-end ACK GFXP |
| 5 | `FE_ACK_CTAP` | Front-end ACK CTAP |
| 6 | `FE_ACK_CILP` | Front-end ACK CILP |
| 7 | `SAVE_END` | Save end |
| 8 | `RESTORE_START` | Restore start |
| 9 | `CONTEXT_START` | Context start |
| 10 | `INVALID_TIMESTAMP` | Invalid timestamp |
| 11 | `ENGINE_RESET` | Engine reset |

**실제로 나온 값은 7과 8뿐이다.** 전역 캡처 197,949행 중 7이 99,015, 8이 98,934(두 캡처 모두 다른 값 없음). 나머지 11개 값(0~6, 9~11, -1)의 의미나 출현 조건은 관측하지 못했다(미확인). 로더는 7/8 외 태그를 만나면 버리지 말고 건너뛰되 개수를 센다.

문자 그대로의 의미는 이름뿐이다("Restore start" = 컨텍스트 복원 시작, "Save end" = 컨텍스트 저장 종료). 어느 컨텍스트가 "들어오는/나가는" 쪽인지, `globalPid`가 그중 어느 쪽을 가리키는지는 열 주석에 적혀 있지 않다. 아래 §4의 해석은 데이터에서 맞춘 것이다.

## 4. 활성 구간 규칙 (가장 중요)

### 4.1 관측 (전역 캡처, GPU 1, 워크로드 A/B)
- 시간순 이벤트 194,592건. 컨텍스트는 둘(A, B). 첫 이벤트는 B의 `RESTORE_START`(8)이고, 처음 8건은 `B8, B7, A8, A7, B8, B7, A8, A7`(8 다음 7, 같은 컨텍스트가 연달아, 컨텍스트끼리 교대). 첫 이벤트 시각 259,841 ns.
- 각 프로세스 안에서는 8 다음 7이 대부분이다. 프로세스별 8→7 주기는 약 177 µs, 두 프로세스를 합친 이벤트열은 `B8 B7 A8 A7` 4개가 177 µs마다 반복된다.
- 인접 이벤트쌍 분포(전체 194,591쌍): `B7→A8` 48,629, `A7→B8` 48,626, `B8→B7` 48,602, `A8→A7` 48,598, 나머지 이탈 136쌍(`A8→B7` 31, `A7→A7` 31, `B8→A7` 25, `B7→B7` 21, `B7→A7` 16, `A7→B7` 12). 이탈은 모두 한 컨텍스트의 `7`이 짝 `8` 없이 나타나거나, `8`이 `7` 없이 다음 `8`을 맞는 경우다.
- 이탈은 캡처 전반에 흩어져 있고(첫 4초 안에 60건 이상), 확인한 예시 몇 건에서 이탈 지점 앞에 ms 단위 공백(약 2~8 ms)이 있었다. 원인(프로세스가 잠시 멈춤 등)은 검증하지 않았다(미확인).

### 4.2 문자 그대로의 의미 vs 해석
- **문자 그대로**: `tag=8`은 `RESTORE_START`, `tag=7`은 `SAVE_END`. 이것이 export가 정의한 전부다.
- **해석(데이터에서 맞춘 것)**: 한 컨텍스트의 `RESTORE_START`로 시작해 같은 컨텍스트의 다음 `SAVE_END`로 끝나는 구간 `[t(8), t(7)]`을 "그 컨텍스트가 GPU에 올라와 있던 구간"으로 본다. 근거:
  1. 짝 구간끼리는 컨텍스트가 다르면 겹치지 않는다(아래 0건). 반대 방향 짝 `[SAVE_END(X), RESTORE_START(Y)]`는 서로 다른 컨텍스트를 잇기 때문에 한 컨텍스트의 구간이 될 수 없다.
  2. 직접 띄운 캡처(단독 컨텍스트)에서 마지막 `RESTORE_START`(3.606 s) 뒤에 `SAVE_END`가 더 나오지 않고, 커널이 그 뒤부터 캡처 끝(15.0 s)까지 계속 실행된다. 즉 8 이후 7이 오기 전까지 컨텍스트는 올라가 있다.
- 이 해석이 **검증되지 않은** 부분: 두 프로세스 경합 중 이벤트 구간을 각 프로세스의 커널 실행 구간과 대조한 적이 없다(전역 캡처에는 커널이 없다). 따라서 "구간 = 실제 SM 점유"가 아니라 "컨텍스트 복원 시작부터 저장 종료까지"일 뿐이다.

### 4.3 규칙 (`ctx_intervals`가 구현할 것)
GPU 1의 비마스킹 행(`globalPid != 1<<48`)을 `(gpuId, globalPid)`별로 `(timestamp, seqNo)` 순으로 훑는다.
1. `tag == 8`: 구간을 연다. 이미 열린 구간이 있으면(= 7 없이 8이 다시 옴) **이전 8은 버리고 새 8로 다시 연다**.
2. `tag == 7`: 열린 구간이 있으면 `[열린 t, 이 t]`를 닫아 내보낸다. 열린 구간이 없으면 이 7은 짝 없음으로 버린다.
3. 훑기가 끝났을 때 열린 8이 남아 있으면(트레일링): 구간을 내보내지 않고 "열린 채 끝남"으로 센다. 이번 전역 캡처에서는 0건이었다. 직접 띄운 캡처의 단독 컨텍스트에서는 마지막 8이 캡처 끝까지 열려 있었으므로(§4.2), 필요하면 로더가 `[t(8), 캡처 끝]`으로 닫는 선택지를 줄 수 있다. 어떻게 처리할지는 정책이지 데이터가 정한 것이 아니다.
4. 7 없이 다시 연 8(규칙 1의 버려진 8)과 짝 없는 7은 개수를 센다. 구간 길이는 `t(7) - t(8)`(ns).
5. 마스킹 행은 구간으로 만들지 않는다(§2.3).

같은 규칙을 SQL로 쓰면(8 바로 다음이 같은 프로세스의 7인 경우와 동일, §7 Q5):
```sql
WITH e AS (SELECT gpuId, globalPid, tag, timestamp, LEAD(tag) OVER w AS nt, LEAD(timestamp) OVER w AS nts
           FROM GPU_CONTEXT_SWITCH_EVENTS WHERE globalPid != 281474976710656
           WINDOW w AS (PARTITION BY gpuId, globalPid ORDER BY timestamp, seqNo))
SELECT gpuId, globalPid, timestamp AS t0, nts AS t1 FROM e WHERE tag = 8 AND nt = 7;
```
(`8 8 7`이면 마지막 8만 짝이 되어 규칙 1과 같다.)

### 4.4 규칙을 지지하는 수치 (전역 캡처 GPU 1)
| 항목 | 값 |
|---|---|
| 이벤트 | 194,592 (7: 97,336, 8: 97,256) |
| 만들어진 구간 | **97,200** (A 48,598, B 48,602) |
| 짝 없는 7 | 136 (7 전체 97,336 = 97,200 + 136) |
| 7 없이 다시 열린 8 | 56 (8 전체 97,256 = 97,200 + 56) |
| 트레일링 열린 8 | 0 |
| 선두의 짝 없는 7 | 0 (첫 이벤트가 8) |
| **서로 다른 컨텍스트 구간 간 겹침** | **0건** (총 겹침 시간 0 ns) |
| 구간 안에 다른 컨텍스트 이벤트가 끼어든 구간 | 0 / 97,200 |
| 구간 길이 p50 / p95 / p99 / max / min | 37.4 / 37.4 / 37.5 / 27,482.5 / 35.7 µs |
| 구간 길이 35~40 µs | 97,085건 (99.88%). 100 µs 이상 97건(0.10%), 다수가 약 4.2 ms 또는 약 7.3 ms 근처에 몰림 |
| 연속 구간 사이 간격 p50 / p95 / p99 / max / min | 52.6 / 52.7 / 52.7 / 65,676.8 / 50.6 µs (음수 0) |
| 구간 합 / 이벤트 시간폭 | 41.1% (워크로드 A/B 두 개) |

- **"한 번에 한 컨텍스트"와의 정합성**: 위 규칙으로 만든 서로 다른 컨텍스트 구간은 GPU 1에서 한 번도 겹치지 않았다. 구간 사이는 항상 양의 간격이다.
- **규칙 1(마지막 8 채택)을 고른 이유**: 첫 8을 채택하는 대안도 겹침은 0건이지만, 그 경우 구간 56개가 중간에 다른 컨텍스트의 이벤트를 포함하게 된다. 마지막 8 채택은 포함 0건이다.
- 간격(위 52 µs, 구간 사이)은 "전환 비용"이라고 부르지 않는다. 그 시간 동안 GPU가 무엇을 했는지(저장/복원, 유휴, 다른 컨텍스트)는 이벤트만으로는 알 수 없다(미확인). 구간 합이 시간폭의 41%뿐인데 nvidia-smi 사용률은 99~100%였다(Task 3): 구간은 "활성 시간"의 **하한**일 수 있고, 어느 쪽인지는 미확인.
- 이 규칙을 **GPU 1 + 워크로드 A/B** 한 쌍에서만 확인했다. 3개 이상의 컨텍스트, MPS, 다른 부하 형태는 미확인. 짝 없는 7·열린 8이 생기는 정확한 원인도 미확인(이탈은 ms 단위 공백 근처에 몰린다는 관측만 있음).

## 5. 커널 테이블 (직접 띄운 캡처)

### 5.1 `CUPTI_ACTIVITY_KIND_KERNEL`
```sql
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    demangledName               INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Kernel function name w/ templates
    shortName                   INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Base kernel function name
    mangledName                 INTEGER,                               -- REFERENCES StringIds(id) -- Raw C++ mangled kernel function name
    launchType                  INTEGER,                               -- REFERENCES ENUM_CUDA_KERNEL_LAUNCH_TYPE(id)
    cacheConfig                 INTEGER,                               -- REFERENCES ENUM_CUDA_FUNC_CACHE_CONFIG(id)
    registersPerThread          INTEGER   NOT NULL,                    -- Number of registers required for each thread executing the kernel.
    gridX                       INTEGER   NOT NULL,                    -- X-dimension grid size.
    gridY                       INTEGER   NOT NULL,                    -- Y-dimension grid size.
    gridZ                       INTEGER   NOT NULL,                    -- Z-dimension grid size.
    blockX                      INTEGER   NOT NULL,                    -- X-dimension block size.
    blockY                      INTEGER   NOT NULL,                    -- Y-dimension block size.
    blockZ                      INTEGER   NOT NULL,                    -- Z-dimension block size.
    staticSharedMemory          INTEGER   NOT NULL,                    -- Static shared memory allocated for the kernel (B).
    dynamicSharedMemory         INTEGER   NOT NULL,                    -- Dynamic shared memory reserved for the kernel (B).
    localMemoryPerThread        INTEGER   NOT NULL,                    -- Amount of local memory reserved for each thread (B).
    localMemoryTotal            INTEGER   NOT NULL,                    -- Total amount of local memory reserved for the kernel (B).
    gridId                      INTEGER   NOT NULL,                    -- Unique grid ID of the kernel assigned at runtime.
    sharedMemoryExecuted        INTEGER,                               -- Shared memory size set by the driver.
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
    sharedMemoryLimitConfig     INTEGER,                               -- REFERENCES ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG(id)
    qmdBulkReleaseDone          INTEGER,                               -- QMD bulk release done timestamp from CWD events.
    qmdPreexitDone              INTEGER,                               -- QMD pre-exit done timestamp from CWD events.
    qmdLastCtaDone              INTEGER,                               -- QMD last CTA done timestamp from CWD events.
    graphId                     INTEGER,                               -- Kernel graph ID.
    correlationIdQmd            INTEGER,                               -- Correlation ID of the QMD associated with the kernel.
    clusterX                    INTEGER,                               -- Cluster X dimension.
    clusterY                    INTEGER,                               -- Cluster Y dimension.
    clusterZ                    INTEGER,                               -- Cluster Z dimension.
    clusterSchedulingPolicy     INTEGER,                               -- Cluster scheduling policy.
    maxPotentialClusterSize     INTEGER,                               -- Maximum potential cluster size.
    maxActiveClusters           INTEGER,                               -- Maximum active clusters.
    sharedMemoryRequestedPercentage   INTEGER,                         -- Shared memory requested percentage.
    tensorSizeMinusOneElements   TEXT      NOT NULL,                   -- TMA descriptor tensor size minus one elements array.
    sourceGraphNodeId           INTEGER,                               -- Node used for graph instantiation or the last node update.
    sourceGraphId               INTEGER                                -- Graph containing the node used for instantiation or the last node update.
);
```
로더가 읽을 열:

| 열 | 의미 |
|---|---|
| `start`, `end` | 커널의 **GPU 실행 시작/종료**, ns, §2.1 시간축. 실행 시간 = `end - start` (이번 워크로드 p50 8.0 µs) |
| `deviceId` | **CUDA 장치 번호**(프로세스 안에서의 번호). `gpuId`가 아니다. §5.4 |
| `contextId` | CUPTI 컨텍스트 ID(전부 1). `GPU_CONTEXT_SWITCH_EVENTS.contextId`와 다른 번호 공간 (§3.1) |
| `streamId` | 스트림 ID. 이번 캡처는 7 하나 |
| `correlationId` | `CUPTI_ACTIVITY_KIND_RUNTIME.correlationId`와 조인(§5.3) |
| `globalPid` | §2.2. 이번 캡처의 435,120행 전부 대상 프로세스의 PID로 디코딩됨 |
| `demangledName`, `shortName`, `mangledName` | `StringIds.id` 참조 (§5.2) |
| 그 외 (`gridX/Y/Z`, `blockX/Y/Z`, 레지스터, 공유 메모리, `gridId` 등) | 이름과 주석 그대로. `gridId`는 런타임이 매긴 고유 번호 |

원본 행은 `start` 오름차순으로 rowid가 매겨진 것으로 보이나(검사한 앞 2만 행에서 확인) 보장은 확인하지 못했다. 로더는 `ORDER BY start`를 쓴다. 같은 스트림 안 커널 겹침은 앞 2만 행에서 0건.

**queued/submitted/start/end 열 유무**: 커널 테이블에는 `start`와 `end`뿐이다. **큐 대기(queued)나 제출(submitted) 시각 열은 없다.** 열 목록에서 확인했고, `qmdBulkReleaseDone`/`qmdPreexitDone`/`qmdLastCtaDone` 열은 존재하지만 이 캡처에서 435,120행 전부 NULL이다(같은 캡처의 진단 메시지에 "CUDA hardware tracing is not supported on this system. A legacy (software instrumented) trace was collected instead."가 있다. 이 NULL의 원인이 그것인지는 미확인). 따라서 "큐 지연"을 **커널 테이블만으로는 계산할 수 없다**. 대신 근사로 런타임 API 호출 시각을 쓸 수 있다(§5.3): `kernel.start - runtime.start`(API 호출 시작에서 GPU 실행 시작까지).

### 5.2 `StringIds`
```sql
CREATE TABLE StringIds (
    -- Consolidation of repetitive string values.

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID reference value.
    value                       TEXT      NOT NULL                     -- String value.
);
```
커널 이름: `SELECT value FROM StringIds WHERE id = <demangledName|shortName|mangledName>`. fixture에는 참조된 id만 들어 있다(6행: 커널 이름 5 + 런타임 함수 이름 1). 원본 캡처에는 서로 다른 커널이 2종류(demangledName 2개) 있고, fixture 앞 2,000행에는 난수 생성 커널(`distribution_elementwise_grid_stride_kernel`)과 `ampere_sgemm_...` 커널이 나온다. 이름 문자열이 매우 길 수 있다(맹글된 이름 수백 자).

### 5.3 `CUPTI_ACTIVITY_KIND_RUNTIME` (제출 시각의 근사)
```sql
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- ID used to identify events that this function call has triggered.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
    callchainId                 INTEGER                                -- REFERENCES CUDA_CALLCHAINS(id)
);
```
- `start`/`end`: CUDA API 호출(`cudaLaunchKernel_v7000` 등)의 시작/종료, ns, 같은 시간축. `nameId`는 `StringIds` 참조, `globalTid`는 §2.2.
- 커널의 `correlationId`와 `RUNTIME.correlationId`가 1:1로 조인된다(435,120행 전부 매칭, 커널의 `correlationId`는 전부 서로 다름).
- 시작 앞 10만 커널로 잰 관계: `kernel.start - runtime.start` p50 4.9 µs / p95 6.2 / p99 8.5 / max 1,377 µs / min 4.1 µs. `kernel.start - runtime.end` p50 1.4 µs, **최솟값 -55 µs**(API가 반환하기 전에 커널이 시작하는 경우가 있다). 이 값이 "제출 대기"인지 단지 API 오버헤드인지는 구분하지 못했다(미확인).

### 5.4 `deviceId`와 GPU의 관계 (`CUDA_VISIBLE_DEVICES`)
워크로드는 `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=1`로 GPU 1을 골랐다. 그 프로세스의 커널 `deviceId`는 **전부 0**이었다(프로세스 안의 CUDA 번호). 실제 GPU는 `TARGET_INFO_CUDA_DEVICE`로 얻는다:
```sql
CREATE TABLE TARGET_INFO_CUDA_DEVICE (
    gpuId                       INTEGER,                               -- GPU ID.
    cudaId                      INTEGER   NOT NULL,                    -- CUDA device ID.
    pid                         INTEGER   NOT NULL,                    -- Process ID.
    uuid                        TEXT,                                  -- Device UUID.
    numMultiprocessors          INTEGER                                -- Number of SMs available on the device.
);
```
`(pid, cudaId)` → `gpuId`: 행 `(gpuId=1, cudaId=0, pid=<워크로드 PID>)`가 "그 프로세스의 CUDA 장치 0 = nsys GPU 1"을 말한다. 이 `gpuId`가 §2.4의 `TARGET_INFO_GPU.id`와 같다. 이 테이블의 `pid`는 디코딩된 PID가 아니라 원래 PID 정수이고, `uuid` 열은 fixture에서 NULL로 비웠다. 이 대응은 `CUDA_VISIBLE_DEVICES`로 GPU 하나만 보이게 한 프로세스 한 개에서만 확인했다. 여러 GPU를 보는 프로세스(미확인)에서도 같은 조인이 성립하는지는 모른다.

### 5.5 그 밖에 fixture에 들어 있는 테이블
- `TARGET_INFO_CUDA_CONTEXT_INFO`(1행): CUPTI 컨텍스트 ↔ 장치/스트림 정보(`contextId`=1, `deviceId`=0, `processId`=원래 PID).
- `TARGET_INFO_CUDA_STREAM`(7행): 스트림별 `streamId`, `contextId`, `priority`, `flag`.
- `ENUM_CUDA_KERNEL_LAUNCH_TYPE`: 이번 캡처의 `launchType`은 전부 0(Unknown).

## 6. 유실/누락 이벤트 표시
- 유실 수를 담은 **전용 테이블이나 숫자 열은 찾지 못했다.** 다음 두 가지 간접 표시가 있다.
  1. `DIAGNOSTIC_EVENT.text`(문장). 전역 캡처: `Number of GPU context switch events collected: 197949.` — 이 수는 `GPU_CONTEXT_SWITCH_EVENTS` 행 수와 같다. 직접 띄운 캡처: `Number of GPU context switch events collected: 7.`, `Number of CUDA events collected: 1741437.`, `135 CudaApi [152] events were discarded because they were outside the report's time range.`, `45 CudaDeviceKernel [79] events were discarded because ...`, `Number of CUPTI events produced: 2176981, CUPTI buffers: 49 of 50. Some CUPTI events might be missing. ...`, `Number of incomplete CUPTI events dropped: 1. ...`. 열: `timestamp, timestampType, source, severity(1 Info, 2 Warning, 3 Error), text, globalPid`. 숫자는 문장 안에 있어 정규식으로 꺼내야 한다. fixture에는 이 문장들만 남기고(경로가 든 문장과 `globalPid`는 제거) 담았다.
  2. `GPU_CONTEXT_SWITCH_EVENTS.seqNo`: GPU별로 시간순에서 1씩 증가하고 65536에서 돌아간다. 전역 캡처의 GPU 0(3,357행)과 GPU 1(194,592행)에서 `(seqNo - 이전 seqNo) mod 65536 != 1`인 곳은 **0건**이어서 이번 캡처에서는 컨텍스트 전환 이벤트 유실의 징후가 없다(§7 Q6). 유실이 있을 때 실제로 틈이 생기는지는 관측하지 못했다(미확인).
- 위 "events were discarded ... outside the report's time range"는 캡처 구간 밖이라 버려진 수이지 버퍼 유실 수가 아니다.

## 7. 테이블 관계
```
GPU_CONTEXT_SWITCH_EVENTS.tag          -> ENUM_GPU_CTX_SWITCH.id
GPU_CONTEXT_SWITCH_EVENTS.gpuId        =  TARGET_INFO_GPU.id  (busLocation -> nvidia-smi 버스 ID)
GPU_CONTEXT_SWITCH_EVENTS.globalPid    ~~ CUPTI_ACTIVITY_KIND_KERNEL.globalPid   (프로세스 단위 연결, PID 디코딩 값이 같음)
CUPTI_ACTIVITY_KIND_KERNEL.deviceId    -> TARGET_INFO_CUDA_DEVICE.cudaId (같은 pid) -> .gpuId = TARGET_INFO_GPU.id
CUPTI_ACTIVITY_KIND_KERNEL.correlationId = CUPTI_ACTIVITY_KIND_RUNTIME.correlationId
CUPTI_ACTIVITY_KIND_KERNEL.{demangled,short,mangled}Name -> StringIds.id
CUPTI_ACTIVITY_KIND_RUNTIME.nameId     -> StringIds.id
```
`GPU_CONTEXT_SWITCH_EVENTS.contextId`와 `CUPTI_ACTIVITY_KIND_KERNEL.contextId`는 같은 번호가 아니다(§3.1). 두 테이블의 연결 키는 `globalPid`(그리고 GPU)다.

## 8. 읽는 쪽이 필요한 테이블 (로더 기준)
| 로더 | 전역 fixture | 직접 fixture |
|---|---|---|
| `load_ctx_events` | `GPU_CONTEXT_SWITCH_EVENTS`, `ENUM_GPU_CTX_SWITCH` | 같음 (7행) |
| `ctx_intervals` | 위 + §4.3 | — |
| `load_kernels` | — | `CUPTI_ACTIVITY_KIND_KERNEL`, `StringIds`, (`CUPTI_ACTIVITY_KIND_RUNTIME`), `TARGET_INFO_CUDA_DEVICE` |
| 보조 | `TARGET_INFO_GPU`, `TARGET_INFO_SESSION_START_TIME`, `META_DATA_EXPORT`(버전), `DIAGNOSTIC_EVENT` | 같음 |

`PROCESSES`는 fixture에 없고 로더도 필요로 하지 않는다(PID는 `globalPid`에서 디코딩).

## 9. fixture 두 개
`scripts/verify/make_nsys_fixture.py`가 원본에서 만든다(우리 워크로드 PID는 `raw/04_wl_*.json`에서 읽음). 원본의 `CREATE TABLE` 문을 그대로 복사했다. 끝에서 개인정보 검사(디코딩 PID ⊂ {워크로드 PID, 0}, 호스트명/사용자명/UUID/명령줄/경로/이메일 문자열 부재, `PROCESSES` 부재)를 돌리고 크기 한도(500 KB)를 확인한다.

| 파일 | 원본 | 내용 |
|---|---|---|
| `tests/fixtures/nsys_ctxsw_min.sqlite` | `ctx_nonroot` | GPU 1의 워크로드 A/B 이벤트 중 캡처 시작 후 **0.6초 미만**(11,440행) + 마스킹 행 50(GPU 0, 같은 구간). `ENUM_GPU_CTX_SWITCH`, `TARGET_INFO_GPU`(uuid NULL), `TARGET_INFO_SESSION_START_TIME`, `META_DATA_EXPORT`(버전 4행), `DIAGNOSTIC_EVENT`(3행) |
| `tests/fixtures/nsys_launched_min.sqlite` | `ctx_launched` | 시작 시각 기준 앞 2,000 커널 + 그 `correlationId`의 런타임 2,000행 + 참조된 `StringIds` 6행 + 컨텍스트 전환 7행 + `TARGET_INFO_CUDA_*` + 위와 같은 보조 테이블 |

지시한 "앞 2초"가 아닌 0.6초인 이유: 2초면 약 4만 행으로 500 KB를 넘고, 1초도 약 630 KB였다. 0.6초가 393 KB다. 0.6초 안에 §4.3 규칙을 적용하면 GPU 1에서 구간 5,716개, 짝 없는 7 6건, 7 없이 다시 열린 8 2건, 트레일링 0건이 나와 예외 처리를 시험할 수 있다. 마스킹 행은 GPU 0의 것이라 GPU 1 구간과 시간대만 같다.

### 9.1 표본 (fixture에서, PID는 기호로 대체)
`GPU_CONTEXT_SWITCH_EVENTS` (`nsys_ctxsw_min`, 시간순 앞 4행. `A`=`(1<<48)|(PID_A<<24)`, `B`는 B의 값):

| tag | vmId | seqNo | contextId | timestamp | globalPid | gpuId |
|---|---|---|---|---|---|---|
| 8 | 0 | 1245 | 64787 | 259841 | B | 1 |
| 7 | 0 | 1246 | 64787 | 296960 | B | 1 |
| 8 | 0 | 1247 | 64788 | 348384 | A | 1 |
| 7 | 0 | 1248 | 64788 | 385375 | A | 1 |

마스킹 행 (`nsys_ctxsw_min`, `WHERE globalPid = 281474976710656`, 앞 3행): `(7, 0, 597, 0, 5569299, 281474976710656, 0)`, `(8, 0, 598, 0, 5634334, 281474976710656, 0)`, `(7, 0, 599, 0, 12074816, 281474976710656, 0)` (`tag, vmId, seqNo, contextId, timestamp, globalPid, gpuId`). 여기서는 GPU 0, `contextId` 0이고 선두가 7이다(여러 프로세스가 섞인 값이라 짝을 해석하지 않는다).

`GPU_CONTEXT_SWITCH_EVENTS` (`nsys_launched_min`, 전체 7행 중 GPU 1의 3행. 나머지 4행은 GPU 0의 마스킹 행 `(8,0,0,0,2094874329,1<<48,0)`, `(7,0,1,0,2115939497,1<<48,0)`, `(8,0,2,0,2116614521,1<<48,0)`, `(7,0,3,0,3614062755,1<<48,0)`; `P`=`(1<<48)|(PID<<24)`):

| tag | vmId | seqNo | contextId | timestamp | globalPid | gpuId |
|---|---|---|---|---|---|---|
| 8 | 0 | 0 | 65194 | 3582317880 | P | 1 |
| 7 | 0 | 1 | 65194 | 3605455057 | P | 1 |
| 8 | 0 | 2 | 65194 | 3606172585 | P | 1 |

`CUPTI_ACTIVITY_KIND_KERNEL` (`nsys_launched_min`, 앞 3행, 일부 열만):

| start | end | deviceId | contextId | streamId | correlationId | demangledName | shortName |
|---|---|---|---|---|---|---|---|
| 3629263075 | 3629267267 | 0 | 1 | 7 | 193 | 1810 | 1811 |
| 3629443905 | 3629448033 | 0 | 1 | 7 | 218 | 1810 | 1811 |
| 3693382794 | 3693391178 | 0 | 1 | 7 | 1503 | 1814 | 1818 |

`CUPTI_ACTIVITY_KIND_RUNTIME` (앞 2행): `(3627886443, 3629281325, 0, <globalTid>, 193, 1812, 0, NULL)`, `(3629432056, 3629444971, 0, <globalTid>, 218, 1812, 0, NULL)` (`start, end, eventClass, globalTid, correlationId, nameId, returnValue, callchainId`). 첫 행은 커널 `correlationId=193`과 조인된다.

`StringIds`: `1811 → 'distribution_elementwise_grid_stride_kernel'`, `1812 → 'cudaLaunchKernel_v7000'`, `1814`는 `ampere_sgemm_32x32_sliced1x4_nn`으로 시작하는 이름(`1818`은 그 짧은 이름), `1810`은 템플릿이 풀린 긴 이름, `1809`는 맹글된 이름. 세 번째 커널 행(1814/1818)은 앞의 두 행과 다른 커널이다.

`TARGET_INFO_CUDA_DEVICE`: `(gpuId=1, cudaId=0, pid=<워크로드 PID>, uuid=NULL, numMultiprocessors=64)`.

`DIAGNOSTIC_EVENT` (`nsys_ctxsw_min`): `(0, 1, 3, 1, 'Profiling has started.', NULL)`, `(141455417, 2, 3, 1, 'Number of GPU context switch events collected: \t197949.', NULL)`, `(10000366207, 1, 3, 1, 'Profiling has stopped.', NULL)`. 이 fixture의 "197949"는 **원본** 캡처의 행 수이며 fixture의 행 수(11,490)와 다르다.

`META_DATA_EXPORT`: `EXPORT_PRODUCT_VERSION = 2026.6.1.251`, `EXPORT_SCHEMA_VERSION = 3.31.4`, `EXPORT_PLATFORM = linux-desktop`, `EXPORT_ARCH = x64`.

## 10. 비교 SQL (Task 4와 같은 질문, root 캡처에서도 그대로 실행)
읽기 전용으로 열어 실행한다(`sqlite3` CLI가 없으면 `python3 -c 'import sqlite3; ...'`). 마스킹 값은 `281474976710656 = 1<<48`.

```sql
-- Q1 GPU별 행 수
SELECT gpuId, count(*) FROM GPU_CONTEXT_SWITCH_EVENTS GROUP BY gpuId ORDER BY gpuId;
-- Q2 GPU별 디코딩된 PID 종류와 행 수 (root 캡처에서 PID 0 외 값이 늘어나는지 보는 질의)
SELECT gpuId, (globalPid >> 24) & 16777215 AS pid, count(*) FROM GPU_CONTEXT_SWITCH_EVENTS GROUP BY gpuId, pid ORDER BY gpuId, pid;
-- Q3 태그별 행 수 (이름 포함)
SELECT e.tag, n.name, count(*) FROM GPU_CONTEXT_SWITCH_EVENTS e LEFT JOIN ENUM_GPU_CTX_SWITCH n ON n.id = e.tag GROUP BY e.tag ORDER BY e.tag;
-- Q4 마스킹된 행 수
SELECT gpuId, count(*) FROM GPU_CONTEXT_SWITCH_EVENTS WHERE globalPid = 281474976710656 GROUP BY gpuId;
-- Q5 8 다음 7 짝 수와 구간 길이 범위(ns) -- 마스킹 행은 여러 프로세스가 섞여 있으니 해석하지 않는다
WITH e AS (SELECT gpuId, globalPid, tag, timestamp, LEAD(tag) OVER w AS nt, LEAD(timestamp) OVER w AS nts
           FROM GPU_CONTEXT_SWITCH_EVENTS WINDOW w AS (PARTITION BY gpuId, globalPid ORDER BY timestamp, seqNo))
SELECT gpuId, count(*), min(nts - timestamp), max(nts - timestamp) FROM e WHERE tag = 8 AND nt = 7 GROUP BY gpuId;
-- Q6 seqNo 틈 (결과가 비어 있으면 틈 없음)
SELECT gpuId, count(*) FROM (SELECT gpuId, (seqNo - LAG(seqNo) OVER (PARTITION BY gpuId ORDER BY timestamp) + 65536) % 65536 AS d
                             FROM GPU_CONTEXT_SWITCH_EVENTS) WHERE d IS NOT NULL AND d != 1 GROUP BY gpuId;
```
원본 `ctx_nonroot`에서의 결과: Q1 `(0, 3357), (1, 194592)`. Q2 GPU 0은 PID 0 한 종류 3,357행, GPU 1은 두 종류(A 97,299, B 97,293). Q3 `(7, SAVE_END, 99015), (8, RESTORE_START, 98934)`. Q4 `(0, 3357)`. Q5 GPU 1은 `97200`행, 최소 35,744 ns, 최대 27,482,506 ns(GPU 0 행은 마스킹된 여러 프로세스가 섞인 값). Q6 빈 결과. root 캡처에서 Q2의 GPU 0 값이 PID 0 한 종류에서 여러 종류로 바뀌면 마스킹이 풀린 것이다.

원본 전체 테이블/스키마/표본(PROCESSES 표본 제외)은 `scripts/verify/05_schema.sh`가 `docs/verification/raw/05_schema_*.txt`에 쓴다.

## 11. 미확인 목록
- root 캡처에서의 차이(PID 마스킹 해제 여부, 이벤트 종류 추가).
- 태그 7/8 외 11개 값이 나오는 조건.
- 두 프로세스 경합 중 활성 구간과 실제 커널 실행의 대응(전역 캡처에는 커널이 없다). 한 캡처에서 둘을 함께 얻는 방법.
- 구간 사이 간격(~52 µs)의 정체, 구간 합이 시간폭의 41%인 이유, 짝 없는 7·열린 8의 원인, 마지막 이벤트가 "Profiling has stopped" 뒤 117 ms에 있는 이유.
- 컨텍스트 전환 이벤트의 `contextId`와 CUPTI `contextId`의 대응.
- 전환 이벤트 시계와 CUPTI 시계의 서브 ms 정렬.
- 3개 이상 컨텍스트, MPS, 다른 GPU에서의 규칙 성립.
- 이벤트 유실이 있을 때 `seqNo`에 틈이 생기는지.
- `PID = (globalPid >> 24) & 0xFFFFFF`는 경험식(공식 문서 미확인). 비트 48 이상의 의미 미확인.
- 여러 GPU를 보는 프로세스에서의 `deviceId` → `gpuId` 대응.

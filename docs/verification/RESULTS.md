# 기술 검증 결과 (1주차)

각 게이트의 판정과 근거를 이 문서에 기록한다. 원본 출력은 `docs/verification/raw/`에 두며 git에 올리지 않는다.

## 게이트 표

| 게이트 | 질문 | 통과 시 | 실패 시 | 판정 |
|---|---|---|---|---|
| G1 | A5000에서 DCGM 1001~1005가 나오는가 | Live에 5개 막대 표시 | 해당 필드는 `N/A`, Live는 NVML 중심 | 미정 |
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

(결과 없음)

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

"""G6 오버헤드 요약: 08_overhead.sh가 만든 JSON에서 조건별 표를 출력한다(표준 라이브러리만)."""

import json
import statistics
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "docs" / "verification" / "raw"
CONDS = ("none", "dcgm", "nsys")


def ms(values):
    """평균과 표본 표준편차(n<2이면 0)."""
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), sd


def load(cond):
    reps = []
    for path in sorted(RAW.glob(f"08_{cond}_r[0-9]_a.json")):
        a = json.loads(path.read_text())
        b = json.loads(path.with_name(path.name.replace("_a.json", "_b.json")).read_text())
        reps.append((a, b))
    return reps


def main():
    data = {c: load(c) for c in CONDS}
    base_sum = None
    base_sd = None
    print("조건 | n | A iter/s | B iter/s | 합계 iter/s | 합계 변화 | p99 ms (A,B 평균 per rep)")
    for c in CONDS:
        reps = data[c]
        if not reps:
            print(f"{c} | 0 | 미측정")
            continue
        ta = [a["throughput_per_s"] for a, _ in reps]
        tb = [b["throughput_per_s"] for _, b in reps]
        ts = [x + y for x, y in zip(ta, tb)]
        p99 = [(a["latency_ms"]["p99"] + b["latency_ms"]["p99"]) / 2 for a, b in reps]
        sm, ss = ms(ts)
        if c == "none":
            base_sum, base_sd = sm, ss
        chg = "-" if c == "none" or base_sum is None else f"{(sm / base_sum - 1) * 100:+.2f}%"
        am, asd = ms(ta)
        bm, bsd = ms(tb)
        pm, psd = ms(p99)
        print(
            f"{c} | {len(reps)} | {am:.0f} ± {asd:.0f} | {bm:.0f} ± {bsd:.0f} | "
            f"{sm:.0f} ± {ss:.0f} | {chg} | {pm:.4f} ± {psd:.4f}"
        )
        print("   per-rep sum:", ", ".join(f"{x:.0f}" for x in ts))
    if base_sum is not None:
        print(
            f"none 합계의 반복 간 표준편차 = {base_sd:.0f} iter/s "
            f"({base_sd / base_sum * 100:.2f}% of mean)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

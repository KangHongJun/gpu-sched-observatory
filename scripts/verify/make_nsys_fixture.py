#!/usr/bin/env python3
"""nsys SQLite export에서 테스트 fixture 두 개를 만든다 (Task 5, G3).

  tests/fixtures/nsys_ctxsw_min.sqlite    <- raw/ctx_nonroot.sqlite  (전역 캡처, 컨텍스트 전환)
  tests/fixtures/nsys_launched_min.sqlite <- raw/ctx_launched.sqlite (직접 띄운 캡처, 커널)

원본 CREATE TABLE 문을 그대로 복사하고, 우리 워크로드 PID(raw/04_wl_*.json의 pid)의 행만 담는다.
PROCESSES 테이블은 만들지 않는다. 끝에서 개인정보 검사를 수행하고 결과를 출력한다.

사용: uv run python scripts/verify/make_nsys_fixture.py
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "docs" / "verification" / "raw"
FIX = ROOT / "tests" / "fixtures"

WINDOW_NS = 600_000_000  # ctxsw fixture: 캡처 시작 후 이 시각 미만의 이벤트만 (500 KB 한도 때문에 2초가 아닌 0.6초)
MAX_MASKED = 50
MAX_KERNELS = 2000
MASKED = 1 << 48
SIZE_LIMIT = 500 * 1024


def ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def pid_of(global_pid: int | None) -> int | None:
    return None if global_pid is None else (global_pid >> 24) & 0xFFFFFF


def create_like(
    src: sqlite3.Connection, dst: sqlite3.Connection, table: str
) -> list[str]:
    """원본의 CREATE TABLE 문을 그대로 실행하고 열 이름 목록을 돌려준다."""
    sql = src.execute(
        "select sql from sqlite_master where type='table' and name=?", (table,)
    ).fetchone()[0]
    dst.execute(sql)
    return [r[1] for r in src.execute(f'pragma table_info("{table}")')]


def copy_rows(src, dst, table, rows, cols, null_cols=()):
    create_done = dst.execute(
        "select 1 from sqlite_master where name=?", (table,)
    ).fetchone()
    assert create_done, table
    idx = [i for i, c in enumerate(cols) if c in null_cols]
    out = []
    for r in rows:
        r = list(r)
        for i in idx:
            r[i] = None
        out.append(r)
    dst.executemany(f'insert into "{table}" values ({",".join("?" * len(cols))})', out)


def copy_table(src, dst, table, where="", params=(), null_cols=(), order=""):
    cols = create_like(src, dst, table)
    rows = src.execute(f'select * from "{table}" {where} {order}', params).fetchall()
    copy_rows(src, dst, table, rows, cols, null_cols)
    return len(rows)


def copy_diag(src, dst):
    """진단 메시지 중 유실/개수 정보와 시작/끝만. 경로가 든 메시지와 PID(globalPid)는 제외/NULL."""
    cols = create_like(src, dst, "DIAGNOSTIC_EVENT")
    rows = src.execute(
        "select * from DIAGNOSTIC_EVENT where (text like 'Profiling has%' or text like 'Number of%'"
        " or text like '%discarded%' or text like '%dropped%' or text like '%missing%')"
        " and text not like '%/%' order by timestamp"
    ).fetchall()
    copy_rows(src, dst, "DIAGNOSTIC_EVENT", rows, cols, null_cols=("globalPid",))


def copy_meta_export(src, dst):
    """EXPORT_HOST/EXPORT_USER/경로가 든 META_DATA_EXPORT는 버전 행만 남긴다."""
    cols = create_like(src, dst, "META_DATA_EXPORT")
    rows = src.execute(
        "select * from META_DATA_EXPORT where name in "
        "('EXPORT_PRODUCT_VERSION','EXPORT_SCHEMA_VERSION','EXPORT_PLATFORM','EXPORT_ARCH')"
    ).fetchall()
    copy_rows(src, dst, "META_DATA_EXPORT", rows, cols)


def common_tables(src, dst):
    copy_table(src, dst, "ENUM_GPU_CTX_SWITCH")
    copy_table(src, dst, "TARGET_INFO_GPU", null_cols=("uuid",))
    copy_table(src, dst, "TARGET_INFO_SESSION_START_TIME")
    copy_meta_export(src, dst)
    copy_diag(src, dst)


def build_ctxsw(pids: set[int]) -> Path:
    out = FIX / "nsys_ctxsw_min.sqlite"
    out.unlink(missing_ok=True)
    src, dst = ro(RAW / "ctx_nonroot.sqlite"), sqlite3.connect(out)
    cols = create_like(src, dst, "GPU_CONTEXT_SWITCH_EVENTS")
    gi, ti = cols.index("globalPid"), cols.index("timestamp")
    rows = src.execute(
        "select * from GPU_CONTEXT_SWITCH_EVENTS where timestamp < ? order by timestamp, seqNo",
        (WINDOW_NS,),
    ).fetchall()
    mine = [r for r in rows if pid_of(r[gi]) in pids]
    masked = [r for r in rows if r[gi] == MASKED][:MAX_MASKED]
    keep = sorted(mine + masked, key=lambda r: (r[ti], r[cols.index("seqNo")]))
    copy_rows(src, dst, "GPU_CONTEXT_SWITCH_EVENTS", keep, cols)
    common_tables(src, dst)
    dst.commit()
    dst.execute("vacuum")
    dst.close()
    print(
        f"{out.name}: ctxsw rows={len(keep)} (workload={len(mine)}, masked={len(masked)})"
    )
    return out


def build_launched(pids: set[int]) -> Path:
    out = FIX / "nsys_launched_min.sqlite"
    out.unlink(missing_ok=True)
    src, dst = ro(RAW / "ctx_launched.sqlite"), sqlite3.connect(out)
    kcols = create_like(src, dst, "CUPTI_ACTIVITY_KIND_KERNEL")
    krows = src.execute(
        "select * from CUPTI_ACTIVITY_KIND_KERNEL order by start limit ?",
        (MAX_KERNELS,),
    ).fetchall()
    copy_rows(src, dst, "CUPTI_ACTIVITY_KIND_KERNEL", krows, kcols)
    corr = sorted({r[kcols.index("correlationId")] for r in krows})
    rcols = create_like(src, dst, "CUPTI_ACTIVITY_KIND_RUNTIME")
    rrows = []
    for i in range(0, len(corr), 500):
        chunk = corr[i : i + 500]
        rrows += src.execute(
            f"select * from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId in ({','.join('?' * len(chunk))})",
            chunk,
        ).fetchall()
    copy_rows(src, dst, "CUPTI_ACTIVITY_KIND_RUNTIME", rrows, rcols)
    ids = set()
    for name in ("demangledName", "shortName", "mangledName"):
        ids |= {r[kcols.index(name)] for r in krows if r[kcols.index(name)] is not None}
    ids |= {r[rcols.index("nameId")] for r in rrows}
    create_like(src, dst, "StringIds")
    srows = src.execute(
        f"select * from StringIds where id in ({','.join('?' * len(ids))}) order by id",
        sorted(ids),
    ).fetchall()
    copy_rows(src, dst, "StringIds", srows, ["id", "value"])
    copy_table(src, dst, "GPU_CONTEXT_SWITCH_EVENTS", order="order by timestamp, seqNo")
    copy_table(src, dst, "ENUM_CUDA_KERNEL_LAUNCH_TYPE")
    copy_table(src, dst, "TARGET_INFO_CUDA_DEVICE", null_cols=("uuid",))
    copy_table(src, dst, "TARGET_INFO_CUDA_CONTEXT_INFO")
    copy_table(src, dst, "TARGET_INFO_CUDA_STREAM")
    common_tables(src, dst)
    dst.commit()
    dst.execute("vacuum")
    dst.close()
    print(f"{out.name}: kernels={len(krows)} runtime={len(rrows)} strings={len(srows)}")
    return out


def forbidden_strings() -> list[str]:
    """원본에서 읽은 식별 정보. fixture 어디에도 나오면 안 된다."""
    bad: list[str] = []
    for f in ("ctx_nonroot.sqlite", "ctx_launched.sqlite"):
        s = ro(RAW / f)
        for name in ("EXPORT_HOST", "EXPORT_USER"):
            bad += [
                r[0]
                for r in s.execute(
                    "select value from META_DATA_EXPORT where name=?", (name,)
                )
            ]
        for tbl in ("TARGET_INFO_GPU", "TARGET_INFO_CUDA_DEVICE"):
            try:
                bad += [
                    r[0]
                    for r in s.execute(f"select uuid from {tbl} where uuid is not null")
                ]
            except sqlite3.OperationalError:
                pass
        # 명령줄 조각 중 "run" 같은 흔한 단어(6자 미만)는 오탐이라 제외한다.
        bad += [
            r[0]
            for r in s.execute(
                "select value from META_DATA_CAPTURE where name like 'PROCESS_%:COMMAND'"
                " or name like 'PROCESS_%:ARGUMENT_%' or name like 'PROCESS_%:WORKING_DIR'"
                " or name like 'PROCESS_%:USER_NAME'"
            )
            if r[0] and len(r[0]) >= 6
        ]
    bad += [os.environ.get("USER", ""), str(Path.home())]
    return sorted({b for b in bad if b and len(b) >= 3})


def check_privacy(path: Path, allowed_pids: set[int], bad_strings: list[str]) -> None:
    con = ro(path)
    print(f"[privacy] {path.name}  size={path.stat().st_size} B")
    assert path.stat().st_size < SIZE_LIMIT, "fixture too large"
    tables = [
        r[0] for r in con.execute("select name from sqlite_master where type='table'")
    ]
    # 1. PROCESSES / META_DATA_CAPTURE 없음 (또는 비어 있음)
    for t in (
        "PROCESSES",
        "META_DATA_CAPTURE",
        "ThreadNames",
        "TARGET_INFO_SYSTEM_ENV",
    ):
        if t in tables:
            assert con.execute(f"select count(*) from {t}").fetchone()[0] == 0, t
    print("  ok: no PROCESSES/META_DATA_CAPTURE/ThreadNames/SYSTEM_ENV rows")
    # 2. 디코딩된 PID는 우리 워크로드 또는 0
    seen: set[int] = set()
    for t in tables:
        cols = [r[1] for r in con.execute(f'pragma table_info("{t}")')]
        for c in cols:
            if c in ("globalPid", "globalTid"):
                for (v,) in con.execute(
                    f'select distinct "{c}" from "{t}" where "{c}" is not null'
                ):
                    seen.add(pid_of(v))
            if c in ("pid", "processId") and t.startswith("TARGET_INFO"):
                seen |= {v for (v,) in con.execute(f'select distinct "{c}" from "{t}"')}
    assert seen <= allowed_pids | {0}, f"unexpected pids: {seen - allowed_pids - {0}}"
    print(
        f"  ok: decoded PIDs {{workload}} subset check, {len(seen - {0})} non-zero PID(s), 0 present={0 in seen}"
    )
    # 3. serial 열 없음, uuid 값 없음
    for t in tables:
        for r in con.execute(f'pragma table_info("{t}")'):
            assert "serial" not in r[1].lower(), (t, r[1])
    for t in ("TARGET_INFO_GPU", "TARGET_INFO_CUDA_DEVICE"):
        if t in tables:
            assert (
                con.execute(
                    f"select count(*) from {t} where uuid is not null"
                ).fetchone()[0]
                == 0
            ), t
    print("  ok: no serial column, uuid columns are NULL")
    # 4. 원본에서 읽은 호스트명/사용자명/UUID/명령줄/경로 문자열이 파일 바이트 어디에도 없음
    blob = path.read_bytes()
    for s in bad_strings:
        assert s.encode() not in blob, "forbidden string present in fixture"
    for pat in (b"/home/", b"GPU-"):
        assert pat not in blob, f"pattern {pat!r} present"
    assert not re.search(rb"[A-Za-z0-9._+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}", blob), (
        "email-like string present"
    )
    print(
        f"  ok: {len(bad_strings)} source-derived identifiers (host/user/uuid/cmdline/path), '/home/', 'GPU-', email pattern absent in file bytes"
    )


def main() -> int:
    wl = {
        k: json.loads((RAW / f"04_wl_{k}.json").read_text())["pid"]
        for k in ("a", "b", "launched")
    }
    ab = {wl["a"], wl["b"]}
    FIX.mkdir(parents=True, exist_ok=True)
    f1 = build_ctxsw(ab)
    f2 = build_launched({wl["launched"]})
    bad = forbidden_strings()
    check_privacy(f1, ab, bad)
    check_privacy(f2, {wl["launched"]}, bad)
    print("all privacy checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

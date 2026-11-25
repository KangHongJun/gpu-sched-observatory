import sqlite3
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"


def _open(name: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{FIX / name}?mode=ro", uri=True)


def _tables(con: sqlite3.Connection) -> set[str]:
    return {
        r[0] for r in con.execute("select name from sqlite_master where type='table'")
    }


def _count(con: sqlite3.Connection, table: str) -> int:
    return con.execute(f'select count(*) from "{table}"').fetchone()[0]


def test_ctxsw_fixture_tables_and_pids():
    con = _open("nsys_ctxsw_min.sqlite")
    tables = _tables(con)
    for t in ("GPU_CONTEXT_SWITCH_EVENTS", "ENUM_GPU_CTX_SWITCH", "TARGET_INFO_GPU"):
        assert t in tables
    assert "PROCESSES" not in tables or _count(con, "PROCESSES") == 0
    assert _count(con, "GPU_CONTEXT_SWITCH_EVENTS") > 0
    pids = {
        r[0]
        for r in con.execute(
            "select distinct (globalPid >> 24) & 16777215 from GPU_CONTEXT_SWITCH_EVENTS"
        )
    }
    nonzero = pids - {0}
    assert len(nonzero) == 2, pids


def test_launched_fixture_tables():
    con = _open("nsys_launched_min.sqlite")
    tables = _tables(con)
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "StringIds", "GPU_CONTEXT_SWITCH_EVENTS"):
        assert t in tables
    assert "PROCESSES" not in tables or _count(con, "PROCESSES") == 0
    assert _count(con, "CUPTI_ACTIVITY_KIND_KERNEL") > 0

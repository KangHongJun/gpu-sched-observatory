#!/usr/bin/env bash
# G3 검증: nsys SQLite export의 테이블 목록과 관련 테이블의 스키마/행 수/표본을 출력한다.
#   05_schema.sh [sqlite 파일...]   기본: raw/ctx_nonroot.sqlite raw/ctx_launched.sqlite
# 출력은 docs/verification/raw/05_schema_<파일명>.txt 에 저장된다(git 제외).
# sqlite3 CLI가 없어 python3 의 sqlite3 모듈(읽기 전용)을 쓴다. sudo/nsys/GPU 불필요.
# PROCESSES 표본 행은 같은 머신 다른 사용자의 프로세스가 담겨 있어 출력하지 않는다(행 수와 스키마만).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW="$ROOT/docs/verification/raw"
mkdir -p "$RAW"
if [ "$#" -eq 0 ]; then
  set -- "$RAW/ctx_nonroot.sqlite" "$RAW/ctx_launched.sqlite"
fi

for db in "$@"; do
  out="$RAW/05_schema_$(basename "${db%.sqlite}").txt"
  python3 - "$db" > "$out" <<'PY'
import re
import sqlite3
import sys

path = sys.argv[1]
con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
tables = {n: s for n, s in con.execute("select name, sql from sqlite_master where type='table' order by name")}
print("== tables ==")
for n in tables:
    k = con.execute('select count(*) from "%s"' % n).fetchone()[0]
    print(n, k, sep="\t")

pat = re.compile(r"CONTEXT|CTX|KERNEL|PROCESS|StringIds|TARGET_INFO_GPU", re.I)
picked = [n for n in tables if pat.search(n)]
for n in list(picked):  # 선택된 테이블이 REFERENCES 하는 ENUM 테이블도 포함
    for e in re.findall(r"REFERENCES\s+(ENUM_\w+)", tables[n]):
        if e in tables and e not in picked:
            picked.append(e)

for n in sorted(picked):
    print(f"\n== {n} ==")
    print(tables[n])
    print("rows:", con.execute('select count(*) from "%s"' % n).fetchone()[0])
    if n == "PROCESSES":
        print("sample rows: suppressed (other users' processes)")
        continue
    cur = con.execute('select * from "%s" limit 5' % n)
    print("columns:", [d[0] for d in cur.description])
    for row in cur:
        print(tuple(str(v)[:100] if isinstance(v, str) else v for v in row))
PY
  echo "wrote $out"
done

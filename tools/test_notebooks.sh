#!/usr/bin/env bash
# Run every notebook source in quick mode (QUICK=1) and report pass/fail.
# Usage: tools/test_notebooks.sh [prefix] [python]
set -u
cd "$(dirname "$0")/.."
PREFIX="${1:-}"
PY="${2:-python3}"
pass=0; fail=0
for f in notebooks/src/${PREFIX}*.py; do
  name=$(basename "$f" .py)
  start=$(date +%s)
  if (cd "$(mktemp -d)" && QUICK=1 MPLBACKEND=Agg timeout 600 "$PY" -I "$OLDPWD/$f") > "${NB_LOGS:-/tmp}/nb_$name.log" 2>&1; then
    echo "ok    $name ($(( $(date +%s) - start ))s)"; pass=$((pass + 1))
  else
    echo "FAIL  $name, see ${NB_LOGS:-/tmp}/nb_$name.log"; tail -5 "${NB_LOGS:-/tmp}/nb_$name.log"; fail=$((fail + 1))
  fi
done
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]

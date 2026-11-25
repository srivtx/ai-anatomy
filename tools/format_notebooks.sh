#!/usr/bin/env bash
# Format the notebook sources for reading: one import and one statement per line, black at 100 columns,
# then a small pass that packs long print() calls and adds a blank line between steps. Code behaviour is unchanged.
# Usage: tools/format_notebooks.sh   (needs ruff and black on PATH)
set -eu
cd "$(dirname "$0")/.."
ruff check -q --isolated --select E401 --fix notebooks/src
black -q -l 100 notebooks/src
python3 tools/tidy_notebooks.py notebooks/src/*.py
python3 tools/build_notebooks.py

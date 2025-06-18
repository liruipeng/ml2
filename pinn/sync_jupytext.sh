#!/bin/bash

# Usage: ./sync_jupytext.sh update-py
#     or: ./sync_jupytext.sh update-ipynb

MODE="$1"

if [[ "$MODE" != "update-py" && "$MODE" != "update-ipynb" ]]; then
    echo "Usage: $0 update-py | update-ipynb"
    exit 1
fi

# Process .py files
for pyfile in *.py; do
    [[ -e "$pyfile" ]] || continue  # skip if no .py files
    jupytext --set-formats ipynb,py "$pyfile"
    if [[ "$MODE" == "update-py" ]]; then
        jupytext --sync "$pyfile"
    fi
done

# Process .ipynb files
for ipynbfile in *.ipynb; do
    [[ -e "$ipynbfile" ]] || continue  # skip if no .ipynb files
    jupytext --set-formats ipynb,py "$ipynbfile"
    if [[ "$MODE" == "update-ipynb" ]]; then
        jupytext --sync "$ipynbfile"
    fi
done


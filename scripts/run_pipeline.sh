#!/bin/bash
# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    # Check if we're on Windows (Git Bash)
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ -n "$WINDIR" ]]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi
fi

# Build command with optional exp_backend parameter
CMD="uv run launch_discovery.py --task AutoSeg"
if [ -n "$1" ]; then
    CMD="$CMD --exp_backend $1"
fi
CMD="$CMD --gpus 0,1"

eval $CMD

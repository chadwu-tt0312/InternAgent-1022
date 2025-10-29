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

uv run launch_discovery.py \
    --task AutoSeg \
    --exp_backend aider \
    --gpus 0,1

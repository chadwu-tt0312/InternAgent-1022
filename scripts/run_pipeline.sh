#!/bin/bash
# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

uv run launch_discovery.py \
    --task AutoSeg \
    --exp_backend aider \
    --gpus 0,1

#!/bin/bash
set -e

cd /workspace/unity-cs
git checkout dev
git pull origin dev -f
cp -r /workspace/unity-cs/server /workspace/comfystream
cp /workspace/unity-cs/src/comfystream/pipeline.py /workspace/comfystream/src/comfystream/pipeline.py
cp /workspace/unity-cs/nodes/api/__init__.py /workspace/comfystream/nodes/api/__init__.py

cd /workspace/comfystream

exec "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/third_party/calvin_runtime"
PYTHON="$ROOT/.venv/bin/python"
UV="${UV:-uv}"

mkdir -p "$TARGET"
"$UV" pip install --target "$TARGET" --upgrade --python-version 3.10 \
    hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 \
    rich==13.9.4 GitPython==3.1.45
"$UV" pip install --target "$TARGET" --upgrade --python-version 3.10 --no-deps \
    gym==0.26.2 pybullet==3.2.7 numpy-quaternion==2023.0.4

PYTHONPATH="$TARGET:$ROOT/third_party/calvin/calvin_models:$ROOT/third_party/calvin/calvin_env:$ROOT" \
    "$PYTHON" -c 'import gym, hydra, omegaconf, pybullet, quaternion, calvin_env; print("CALVIN runtime imports: PASS")'

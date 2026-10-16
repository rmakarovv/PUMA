#!/bin/bash
# Create the project environment (default: .venv in the repo root) from requirements.txt.
#   bash scripts/setup_env.sh
# Options (environment variables):
#   BASE_PYTHON=/path/to/python  interpreter the venv is built from (Python >= 3.10; default python3)
#   REUSE_TORCH=1                inherit torch from BASE_PYTHON (--system-site-packages) instead of
#                                downloading it; BASE_PYTHON must already have a working CUDA torch
#   VENV_DIR=path                venv location, relative to the repo root (default .venv)
# CUDA itself can only be checked on a compute node: sbatch scripts/cuda_diag.sbatch
set -euo pipefail
cd "$(dirname "$0")/.."
BASE_PYTHON=${BASE_PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}
REUSE_TORCH=${REUSE_TORCH:-0}

"$BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10, got " + sys.version'
if [ -e "$VENV_DIR" ]; then
    echo "$VENV_DIR already exists; remove it or set VENV_DIR" >&2
    exit 1
fi

if [ "$REUSE_TORCH" = 1 ]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
    "$VENV_DIR/bin/python" -c "import torch; print('reusing torch', torch.__version__, 'from', torch.__file__)"
    "$VENV_DIR/bin/python" -m pip install -r <(grep -v '^torch==' requirements.txt)
else
    "$BASE_PYTHON" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install -r requirements.txt
fi

"$VENV_DIR/bin/python" - <<'EOF'
import torch, numpy, tqdm, omegaconf, transformers, datasets, wandb, gdown
print(f"ok: torch {torch.__version__} (CUDA build {torch.version.cuda}), numpy {numpy.__version__}, "
      f"transformers {transformers.__version__}, omegaconf {omegaconf.__version__}")
EOF
echo "environment ready: $VENV_DIR"

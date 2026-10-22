#!/bin/bash
#SBATCH --job-name=puma_sudoku_cache
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR" # submit from the repo root, after mkdir -p runs/logs
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
${PY:-.venv/bin/python} scripts/preprocess_sudoku_cache.py data/sudoku_new "$SLURM_CPUS_PER_TASK"
echo "end=$(date -Is)"

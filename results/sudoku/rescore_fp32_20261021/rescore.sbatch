#!/bin/bash
# fp32 re-scoring of saved EMA checkpoints.
# usage: sbatch --chdir=/home/timur.aisin/intro_to_ml/PUMA-team-D rescore.sbatch <run_dir> [<run_dir> ...]
# Each <run_dir> (…/sudoku_<X>_<stamp>/seed<s>) gets one jsonl in $OUT named <X>_seed<s>.jsonl.
#SBATCH --job-name=rescore_fp32
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu-02,gpu-05,gpu-51,gpu-52,gpu-54,gpu-55
#SBATCH --output=/home/timur.aisin/intro_to_ml/runs/logs/%x_%j.out
set -uo pipefail
PY=/home/timur.aisin/miniconda3/envs/puma/bin/python
OUT=/home/timur.aisin/intro_to_ml/runs/rescore_fp32_20261021
SCRIPT=/home/timur.aisin/intro_to_ml/runs/rescore_fp32_20261021/rescore.py
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) git_head=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader
for run in "$@"; do
    tag=$(basename "$(dirname "$run")" | sed -E 's/^sudoku_([^_]+)_.*/\1/')_$(basename "$run")
    ckpts=$(ls "$run"/ema_step=*.pt | sort -t= -k2 -n)
    echo "[$(date -Is)] $tag"
    $PY "$SCRIPT" "$OUT/$tag.jsonl" $ckpts || echo "FAILED $tag"
done
echo "end=$(date -Is)"

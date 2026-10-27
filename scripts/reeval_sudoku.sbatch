#!/bin/bash
# Re-evaluate Sudoku EMA checkpoints with scripts/reeval_sudoku.py, one process per allocated GPU.
# Usage (from the repo root): the two jobs below cover shards 0-3 of all runs/sudoku_* checkpoints.
#   OUT_DIR=runs/<name> SHARD_BASE=0 NUM_SHARDS=4 sbatch scripts/reeval_sudoku.sbatch --bf16-final
#   OUT_DIR=runs/<name> SHARD_BASE=2 NUM_SHARDS=4 sbatch scripts/reeval_sudoku.sbatch --bf16-final
# Extra arguments go to reeval_sudoku.py. Finished entries are skipped, so a job can be resubmitted.
#SBATCH --job-name=puma_sudoku_reeval
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclude=gpu-02,gpu-05,gpu-51,gpu-52,gpu-54,gpu-55
#SBATCH --output=runs/logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=${PY:-.venv/bin/python}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
SHARD_BASE=${SHARD_BASE:-0}
NUM_SHARDS=${NUM_SHARDS:-2}
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR"

{
    echo "slurm_job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) shards=$SHARD_BASE,$((SHARD_BASE + 1))/$NUM_SHARDS args=$*"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
    echo "git_head=$(git rev-parse HEAD)"
    git status --short
} > "$OUT_DIR/run_info_job$SLURM_JOB_ID.txt"
git diff HEAD > "$OUT_DIR/code_job$SLURM_JOB_ID.diff"
cat "$OUT_DIR/run_info_job$SLURM_JOB_ID.txt"

# preflight: a real CUDA calculation on every allocated GPU
$PY - <<'EOF' || { echo "CUDA preflight failed on $(hostname)"; exit 1; }
import torch
n = torch.cuda.device_count()
assert n == 2, f"expected 2 GPUs, got {n}"
for i in range(n):
    x = torch.randn(2048, 2048, device=f"cuda:{i}")
    print(f"preflight cuda:{i} {torch.cuda.get_device_name(i)} sum={(x @ x).sum().item():.3e}")
EOF

pids=()
for g in 0 1; do
    shard=$((SHARD_BASE + g))
    $PY scripts/reeval_sudoku.py --out-dir "$OUT_DIR" --shard "$shard" --num-shards "$NUM_SHARDS" --gpu "$g" "$@" \
        > "$OUT_DIR/shard$shard.log" 2>&1 &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "end=$(date -Is) exit=$status" | tee -a "$OUT_DIR/run_info_job$SLURM_JOB_ID.txt"
exit $status

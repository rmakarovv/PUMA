#!/bin/bash
# Runs Sudoku condition B for several seeds sequentially on one 2-GPU allocation.
# Usage (from the repo root, after mkdir -p runs/logs):
#   EXP_DIR=<abs path> PY=<python> sbatch scripts/run_sudoku_B.sbatch 1000 2000 ...
# Output: $EXP_DIR/seed<s>/ (no condition subfolder). DONE items are skipped.
#SBATCH --job-name=puma_sudoku_B
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=gpu-02,gpu-05,gpu-51,gpu-52,gpu-54,gpu-55
#SBATCH --output=runs/logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=${PY:?set PY}
EXP_DIR=${EXP_DIR:?set EXP_DIR}
cond=B
cfg=yaml_files/sudoku_B_schedule.yaml
export TQDM_MININTERVAL=60 PYTHONUNBUFFERED=1

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) seeds=$*"

# preflight: a real CUDA calculation on every allocated GPU (as in run_sudoku_AE.sbatch)
$PY - <<'PYEOF' || { echo "CUDA preflight failed on $(hostname)"; exit 1; }
import torch
n = torch.cuda.device_count()
assert n == 2, f"expected 2 GPUs, got {n}"
for i in range(n):
    x = torch.randn(2048, 2048, device=f"cuda:{i}")
    print(f"preflight cuda:{i} {torch.cuda.get_device_name(i)} sum={(x @ x).sum().item():.3e}")
PYEOF

status=0
for seed in "$@"; do
    out=$EXP_DIR/seed$seed
    if [ -f "$out/DONE" ]; then echo "skip $cond:$seed (DONE)"; continue; fi
    mkdir -p "$out"
    {
        echo "condition=$cond seed=$seed config=$cfg"
        echo "slurm_job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
        nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
        echo "git_head=$(git rev-parse HEAD)"
        git status --short
    } > "$out/run_info.txt"
    git diff HEAD > "$out/code.diff"
    cp "$cfg" scripts/run_sudoku_$cond.sbatch "$out/"

    echo "[$(date -Is)] start $cond:$seed -> $out"
    $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
        train.py --cfg "$cfg" training.seed="$seed" training.ckpt_dir="$out" \
        > "$out/train.log" 2>&1
    rc=$?
    echo "end=$(date -Is) exit=$rc" >> "$out/run_info.txt"
    echo "[$(date -Is)] end $cond:$seed exit=$rc"
    if [ $rc -eq 0 ]; then touch "$out/DONE"; else status=1; fi
done
exit $status

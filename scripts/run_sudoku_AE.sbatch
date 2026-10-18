#!/bin/bash
# Runs Sudoku training items sequentially on one 2-GPU allocation (PUMA setup: 2 GPUs x 32).
# Usage (from the repo root, after mkdir -p runs/logs):
#   EXP_DIR=runs/<exp> sbatch scripts/run_sudoku_AE.sbatch A:1000 E:1000 ...
# Each item is COND:SEED with COND in {A, E, F, P, PN} (P = stock PUMA, PN = PUMA without collapse); DONE items are skipped.
#SBATCH --job-name=puma_sudoku_AE
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
PY=${PY:-.venv/bin/python}
EXP_DIR=${EXP_DIR:?set EXP_DIR}
EXTRA=${EXTRA:-} # optional extra dotlist overrides (smoke tests only)
RUN_TIMEOUT=${RUN_TIMEOUT:-0} # per-item time limit for smoke tests; 0 = none
export TQDM_MININTERVAL=60 PYTHONUNBUFFERED=1

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) items=$*"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

# preflight: a real CUDA calculation on every allocated GPU
$PY - <<'EOF' || { echo "CUDA preflight failed on $(hostname)"; exit 1; }
import torch
n = torch.cuda.device_count()
assert n == 2, f"expected 2 GPUs, got {n}"
for i in range(n):
    x = torch.randn(2048, 2048, device=f"cuda:{i}")
    print(f"preflight cuda:{i} {torch.cuda.get_device_name(i)} sum={(x @ x).sum().item():.3e}")
EOF

status=0
for item in "$@"; do
    cond=${item%%:*}; seed=${item##*:}
    case $cond in
        A) cfg=yaml_files/sudoku_A_random.yaml ;;
        E) cfg=yaml_files/sudoku_E_confidence.yaml ;;
        F) cfg=yaml_files/sudoku_F_gtmargin.yaml ;;
        P) cfg=yaml_files/sudoku_PUMA_reference.yaml ;;
        PN) cfg=yaml_files/sudoku_PUMA_nocollapse.yaml ;;
        *) echo "unknown condition in $item"; status=1; continue ;;
    esac
    out=$EXP_DIR/$cond/seed$seed
    if [ -f "$out/DONE" ]; then echo "skip $item (DONE)"; continue; fi
    mkdir -p "$out"
    {
        echo "condition=$cond seed=$seed config=$cfg extra=$EXTRA"
        echo "slurm_job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
        nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
        echo "git_head=$(git rev-parse HEAD)"
        git status --short
    } > "$out/run_info.txt"
    git diff HEAD > "$out/code.diff"
    cp "$cfg" scripts/run_sudoku_AE.sbatch "$out/"

    echo "[$(date -Is)] start $item -> $out"
    timeout "$RUN_TIMEOUT" $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
        train.py --cfg "$cfg" training.seed="$seed" training.ckpt_dir="$out" $EXTRA \
        > "$out/train.log" 2>&1
    rc=$?
    echo "end=$(date -Is) exit=$rc" >> "$out/run_info.txt"
    echo "[$(date -Is)] end $item exit=$rc"
    if [ $rc -eq 0 ]; then touch "$out/DONE"; else status=1; fi
done
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
exit $status

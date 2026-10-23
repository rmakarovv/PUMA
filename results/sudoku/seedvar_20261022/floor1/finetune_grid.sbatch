#!/bin/bash
# Fine-tune grid for the B/D low-mask test (finetune_floor.py), declared before launch:
# ckpts B2000 D2000 B5000 D4000 B3000 (20261019 final EMA) x {sched4, sched5} x steps {20, 200, 1000}, lr 3e-5, seed 0.
#SBATCH --job-name=ft_floor
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/timur.aisin/intro_to_ml/runs/floor1_20261022/ft_grid_%j.out
D=/home/timur.aisin/intro_to_ml/runs/floor1_20261022
R=/home/timur.aisin/intro_to_ml/runs
REPO=/home/timur.aisin/intro_to_ml/PUMA-team-BD
PY=/home/timur.aisin/miniconda3/envs/puma/bin/python
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) git_head=$(git -C $REPO rev-parse HEAD)"
for steps in 1000 200 20; do for tag in B2000 D2000 B5000 D4000 B3000; do for s in sched4 sched5; do
  echo "$tag $s $steps"
done; done; done | xargs -P 8 -L 1 bash -c 'c=${0:0:1}; seed=${0:1}; NT=8 '$PY' -I '$D'/finetune_floor.py '$REPO' '$R'/sudoku_${c}_20261019/seed${seed}/ema_step=53570.pt $c $1 $2 3e-5 0 '$D'/finetune_grid.jsonl ${0}_$1_$2 > '$D'/ft_${0}_$1_$2.log 2>&1'
echo "end=$(date -Is)"

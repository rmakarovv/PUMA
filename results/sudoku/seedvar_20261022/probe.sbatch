#!/bin/bash
#SBATCH --job-name=probe_masklevel
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/timur.aisin/intro_to_ml/runs/seedvar_20261022/probe_%j.out
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) git_head=$(git -C /home/timur.aisin/intro_to_ml/PUMA-team-BD rev-parse HEAD)"
NT=32 /home/timur.aisin/miniconda3/envs/puma/bin/python -I /home/timur.aisin/intro_to_ml/runs/seedvar_20261022/probe_mask_level.py /home/timur.aisin/intro_to_ml/PUMA-team-BD 999 /home/timur.aisin/intro_to_ml/runs/seedvar_20261022/probe_final.jsonl \
  Erepro1000=/home/timur.aisin/intro_to_ml/runs/sudoku_Erepro_20261019/seed1000/ema_step=53570.pt \
  B1000=/home/timur.aisin/intro_to_ml/runs/sudoku_B_20261019/seed1000/ema_step=53570.pt \
  B2000=/home/timur.aisin/intro_to_ml/runs/sudoku_B_20261019/seed2000/ema_step=53570.pt \
  B3000=/home/timur.aisin/intro_to_ml/runs/sudoku_B_20261019/seed3000/ema_step=53570.pt \
  B4000=/home/timur.aisin/intro_to_ml/runs/sudoku_B_20261019/seed4000/ema_step=53570.pt \
  B5000=/home/timur.aisin/intro_to_ml/runs/sudoku_B_20261019/seed5000/ema_step=53570.pt \
  D1000=/home/timur.aisin/intro_to_ml/runs/sudoku_D_20261019/seed1000/ema_step=53570.pt \
  D2000=/home/timur.aisin/intro_to_ml/runs/sudoku_D_20261019/seed2000/ema_step=53570.pt \
  D3000=/home/timur.aisin/intro_to_ml/runs/sudoku_D_20261019/seed3000/ema_step=53570.pt \
  D4000=/home/timur.aisin/intro_to_ml/runs/sudoku_D_20261019/seed4000/ema_step=53570.pt \
  D5000=/home/timur.aisin/intro_to_ml/runs/sudoku_D_20261019/seed5000/ema_step=53570.pt \
  B1000_30k=/home/timur.aisin/intro_to_ml/runs/sudoku_B_20261019/seed1000/ema_step=30000.pt \
  B2000_30k=/home/timur.aisin/intro_to_ml/runs/sudoku_B_20261019/seed2000/ema_step=30000.pt \
  B3000_30k=/home/timur.aisin/intro_to_ml/runs/sudoku_B_20261019/seed3000/ema_step=30000.pt \
  B4000_30k=/home/timur.aisin/intro_to_ml/runs/sudoku_B_20261019/seed4000/ema_step=30000.pt \
  B5000_30k=/home/timur.aisin/intro_to_ml/runs/sudoku_B_20261019/seed5000/ema_step=30000.pt \
  D1000_30k=/home/timur.aisin/intro_to_ml/runs/sudoku_D_20261019/seed1000/ema_step=30000.pt \
  D2000_30k=/home/timur.aisin/intro_to_ml/runs/sudoku_D_20261019/seed2000/ema_step=30000.pt \
  D3000_30k=/home/timur.aisin/intro_to_ml/runs/sudoku_D_20261019/seed3000/ema_step=30000.pt \
  D4000_30k=/home/timur.aisin/intro_to_ml/runs/sudoku_D_20261019/seed4000/ema_step=30000.pt \
  D5000_30k=/home/timur.aisin/intro_to_ml/runs/sudoku_D_20261019/seed5000/ema_step=30000.pt \
  Erepro1000_30k=/home/timur.aisin/intro_to_ml/runs/sudoku_Erepro_20261019/seed1000/ema_step=30000.pt
echo "end=$(date -Is)"

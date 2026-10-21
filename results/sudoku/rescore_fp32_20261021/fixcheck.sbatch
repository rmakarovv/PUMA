#!/bin/bash
#SBATCH --job-name=fixcheck
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu-02,gpu-05,gpu-51,gpu-52,gpu-54,gpu-55
#SBATCH --output=/home/timur.aisin/intro_to_ml/runs/rescore_fp32_20261021/fixcheck_%j.out
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
echo "job=$SLURM_JOB_ID node=$(hostname) git_head=$(git rev-parse HEAD)"
/home/timur.aisin/miniconda3/envs/puma/bin/python /home/timur.aisin/intro_to_ml/runs/rescore_fp32_20261021/fixcheck.py "$@"

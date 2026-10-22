#!/bin/bash
#SBATCH --job-name=check_patch
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu-02,gpu-05,gpu-51,gpu-52,gpu-54,gpu-55
#SBATCH --output=/home/timur.aisin/intro_to_ml/runs/rescore_fp32_20261021/check_patch_%j.out
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
O=/home/timur.aisin/intro_to_ml/runs/rescore_fp32_20261021
echo "job=$SLURM_JOB_ID node=$(hostname) git_head=$(git rev-parse HEAD)"; git diff --stat
/home/timur.aisin/miniconda3/envs/puma/bin/python $O/check_patch.py $O/sampling_orig.py "$@"

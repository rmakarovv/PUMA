#!/bin/bash
# Low-mask probes on Anton's condition C checkpoints (read via ACL):
#  1. probe_mask_level.py on the full C runs (53,570 updates, final EMA), seeds 1000-5000
#  2. probe_mask_level.py on the schedule pilot (30,000 updates, final EMA): C-K2 (floor 41/81), C-K4 (21/81), C-K8 (11/81)
#  3. ctx_probe.py (total masked count vs local context) on full-run C5000, C2000, C3000
#  4. finetune_floor.py: C5000 and C2000, sched4 vs sched5, 200 updates (C pool from PUMA-team-C)
#SBATCH --job-name=probe_C
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/timur.aisin/intro_to_ml/runs/floor1_20261022/probe_C_%j.out
D=/home/timur.aisin/intro_to_ml/runs/floor1_20261022
PY="/home/timur.aisin/miniconda3/envs/puma/bin/python -I"
PROBE=/home/timur.aisin/intro_to_ml/PUMA-team/results/sudoku/seedvar_20261022/probe_mask_level.py
BD=/home/timur.aisin/intro_to_ml/PUMA-team-BD
CREPO=/home/timur.aisin/intro_to_ml/PUMA-team-C
F=/home/anton.egorov/PUMA-C/runs/sudoku_C_20261022/C
P=/home/anton.egorov/PUMA-C/runs/sudoku_C_pilot_20261023
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) git_BD=$(git -C $BD rev-parse HEAD) git_C=$(git -C $CREPO rev-parse HEAD)"

full=""; for s in 1000 2000 3000 4000 5000; do full="$full C$s=$F/seed$s/ema_step=53570.pt"; done
k24=""; for k in K2 K4; do for s in 1000 2000 3000 4000 5000; do k24="$k24 C${k}_$s=$P/C-$k/seed$s/ema_step=30000.pt"; done; done
k8=""; for s in 1000 2000 3000 4000 5000; do k8="$k8 CK8_$s=$P/C-K8/seed$s/ema_step=30000.pt"; done

NT=12 $PY $PROBE $BD 999 $D/probe_C_full.jsonl $full > $D/probe_C_full.log 2>&1 &
NT=12 $PY $PROBE $BD 999 $D/probe_C_pilot.jsonl $k24 > $D/probe_C_pilot_k24.log 2>&1 &
NT=12 $PY $PROBE $BD 999 $D/probe_C_pilot.jsonl $k8 > $D/probe_C_pilot_k8.log 2>&1 &
NT=8 $PY $D/ctx_probe.py $F/seed5000/ema_step=53570.pt $F/seed2000/ema_step=53570.pt $F/seed3000/ema_step=53570.pt > $D/ctx_probe_C.out 2>&1 &
(for s in 5000 2000; do for sch in sched4 sched5; do
  NT=10 $PY $D/finetune_floor.py $CREPO $F/seed$s/ema_step=53570.pt C $sch 200 3e-5 0 $D/finetune_C.jsonl C${s}_${sch}_200 > $D/ft_C${s}_${sch}_200.log 2>&1
done; done) &
wait
echo "end=$(date -Is)"

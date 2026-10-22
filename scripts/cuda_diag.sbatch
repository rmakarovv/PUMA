#!/bin/bash
#SBATCH --job-name=puma_cuda_diag
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=gpu-debug-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=runs/logs/%x_%j.out
# Read-only CUDA diagnostics; CUDA_VISIBLE_DEVICES is only printed, never changed.
cd "$SLURM_SUBMIT_DIR" # submit from the repo root, after mkdir -p runs/logs
echo "node=$(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
env | grep -E "^SLURM_(JOB_GPUS|STEP_GPUS|GPUS_ON_NODE)|GPU_DEVICE_ORDINAL|NVIDIA_VISIBLE" | sort
nvidia-smi | head -12
nvidia-smi --query-gpu=index,uuid,compute_mode,mig.mode.current,pstate,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
ls -l /dev/nvidia* 2>&1 | head
cat /proc/self/cgroup
for PY in ${PY:-.venv/bin/python}; do
  echo "== $PY"
  $PY - <<'PYEOF'
import ctypes
c = ctypes.CDLL("libcuda.so.1")
n = ctypes.c_int(); v = ctypes.c_int()
print("cuInit", c.cuInit(0), "cuDeviceGetCount", c.cuDeviceGetCount(ctypes.byref(n)), n.value,
      "cuDriverGetVersion", c.cuDriverGetVersion(ctypes.byref(v)), v.value)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    x = torch.randn(1024, 1024, device=f"cuda:{i}")
    print("ok", i, torch.cuda.get_device_name(i), (x @ x).sum().item())
PYEOF
done

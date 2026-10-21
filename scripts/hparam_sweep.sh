#!/bin/bash
# One Phoenix task per learning rate. Each task gets its own 8-hour clock.
# %1 runs a single task at a time; delete it to run all four GPUs at once.
# The --array range must stay in sync with CONFIGS below.
#
#   mkdir -p logs
#   sbatch scripts/hparam_sweep.sh
#

#SBATCH --job-name=gemma4-lora-hparam
#SBATCH --account=gts-smussmann3
#SBATCH --partition=gpu-a100
#SBATCH --qos=embers
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

CONFIGS=(
    lr_5e-5
    lr_1e-4
    lr_2e-4
    lr_4e-4
)

module load cuda/12.1 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Submit with sbatch so SLURM_ARRAY_TASK_ID is set." >&2
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} is outside CONFIGS (0-$(( ${#CONFIGS[@]} - 1 )))." >&2
    exit 1
fi
CONFIG_NAME="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
else
    source /storage/project/r-smussmann3-0/shared/shared_venvs/default_venv/bin/activate
fi

echo "===== Job information ====="
echo "Job ID:       ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:         ${SLURMD_NODENAME:-$(hostname)}"
echo "REPO_ROOT:    $REPO_ROOT"
echo "Config:       configs/gemma/${CONFIG_NAME}.yaml"
echo "GPU:"
nvidia-smi
echo "==========================="

if [[ ! -f "$REPO_ROOT/.env" ]]; then
    echo "Missing $REPO_ROOT/.env (WANDB_PROJECT, WANDB_ENTITY, WANDB_API_KEY)." >&2
    exit 1
fi
if [[ ! -f "$REPO_ROOT/data/proofjudge/dev.jsonl" || ! -f "$REPO_ROOT/data/proofjudge/test.jsonl" ]]; then
    echo "Missing data/proofjudge/dev.jsonl or data/proofjudge/test.jsonl." >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$REPO_ROOT/src"
export HF_HOME="${HF_HOME:-/storage/project/r-smussmann3-0/${USER}/hf_home}"
mkdir -p "$HF_HOME"

python -u - <<'PY'
import sys
import torch

print(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available in this Python environment.")
print(f"device={torch.cuda.get_device_name(0)}")
PY

python -u src/proof_quality/train.py --config "configs/gemma/${CONFIG_NAME}.yaml"
echo "Experiment ${CONFIG_NAME} completed."

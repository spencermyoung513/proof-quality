#!/bin/bash
# One training batch and one validation batch of the Gemma LoRA setup.
#   mkdir -p logs
#   sbatch scripts/smoke_test.sh
#
#SBATCH --job-name=gemma4-lora-smoke
#SBATCH --account=gts-smussmann3
#SBATCH --partition=gpu-l40s
#SBATCH --qos=inferno
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load cuda/12.1 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
else
    source /storage/project/r-smussmann3-0/shared/shared_venvs/default_venv/bin/activate
fi

echo "===== Job information ====="
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-$(hostname)}"
echo "REPO_ROOT:    $REPO_ROOT"
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

python -u src/proof_quality/train.py --config configs/gemma/smoke.yaml
echo "Smoke test completed."

#!/bin/bash
# Train one config on a single GPU, then score best_val_loss.ckpt on the test set.
#   mkdir -p logs
#   sbatch scripts/train.sh configs/gemma/final.yaml
#
#
#SBATCH --job-name=gemma4-lora-train
#SBATCH --account=gts-smussmann3
#SBATCH --partition=gpu-l40s
#SBATCH --qos=inferno
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 1 || "$1" == -* ]]; then
    echo "Usage: sbatch scripts/train.sh <config.yaml>" >&2
    exit 1
fi

SUBMIT_DIR="$(pwd)"
if [[ "$1" = /* ]]; then
    CONFIG="$1"
else
    CONFIG="${SUBMIT_DIR}/$1"
fi

module load cuda/12.1 2>/dev/null || true

REPO_ROOT="/storage/project/r-smussmann3-0/syoung328/code/proof-quality"
SCRATCH="/storage/home/hcoda1/1/syoung328/scratch"
VENV="${SCRATCH}/venvs/.proof-quality-venv"
HF_HOME="${SCRATCH}/weights/hf_home"

cd "$REPO_ROOT"

echo "===== Job information ====="
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-$(hostname)}"
echo "REPO_ROOT:    $REPO_ROOT"
echo "Venv:         $VENV"
echo "HF_HOME:      $HF_HOME"
echo "Config:       $CONFIG"
echo "GPU:"
nvidia-smi
echo "==========================="

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$REPO_ROOT/.env" ]]; then
    echo "Missing $REPO_ROOT/.env (WANDB_PROJECT, WANDB_ENTITY, WANDB_API_KEY)." >&2
    exit 1
fi
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Missing a usable interpreter at $VENV/bin/python." >&2
    echo "Rebuild it with scripts/build_or_update_venv.sh." >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$REPO_ROOT/src"
export HF_HOME
mkdir -p "$HF_HOME"

"$VENV/bin/python" -u - "$CONFIG" "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path

import torch
import yaml

print(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available in this Python environment.")
print(f"device={torch.cuda.get_device_name(0)}")

config_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
cfg = yaml.safe_load(config_path.read_text())
root = Path((cfg.get("data") or {}).get("root", "data/mathlibpr"))
if not root.is_absolute():
    root = repo_root / root
missing = [
    path
    for split in ("dev", "test")
    if not (path := root / f"{split}.jsonl").is_file()
]
if missing:
    sys.exit("Missing " + " and ".join(str(path) for path in missing) + ".")
print(f"data_root={root}")
PY

"$VENV/bin/python" -u src/proof_quality/train.py --config "$CONFIG"
echo "Training completed: $CONFIG"

CKPT="$("$VENV/bin/python" -u - "$CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

from proof_quality.configs import TrainingConfig

config = TrainingConfig(**yaml.safe_load(Path(sys.argv[1]).read_text()))
ckpt = config.run.ckpt_dir / config.run.name / "best_val_loss.ckpt"
if not ckpt.is_file():
    sys.exit(f"Checkpoint not found: {ckpt}")
print(ckpt)
PY
)"

echo "Evaluating $CKPT"
"$VENV/bin/python" -u src/proof_quality/evaluate.py --config "$CONFIG" --ckpt "$CKPT"
echo "Evaluation completed: $CKPT"

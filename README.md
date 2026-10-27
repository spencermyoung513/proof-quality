# Proof Quality

## Setup on PACE

To run code from this repository on PACE, you will first need to build a virtual environment in a temporary scratch directory. You can do this via

```bash
bash scripts/build_or_update_venv.sh
```

The SLURM scripts run Python from that environment directly (`/storage/home/hcoda1/1/syoung328/scratch/venvs/.proof-quality-venv/bin/python`).

## Training a Model

To kick off a training run, first fill out a config (using [configs/test.yaml](configs/test.yaml) as a guide). Then, from the terminal:

```bash
uv run train --config <path-to-your-config>
```

Training holds out a small percentage (default: 10%) of the dev set for validation and logs to Weights and Biases.

## Evaluating a Model

To score a saved checkpoint on the test set, use the same config as training and point to the weights checkpoint you want to evaluate. Then:

```bash
uv run evaluate --config <path-to-your-config> --ckpt <path-to-checkpoint>
```

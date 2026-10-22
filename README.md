# Proof Quality

## Setup on PACE

To run code from this repository on PACE, you will first need to build a virtual environment in a temporary scratch directory. You can do this via

```bash
bash scripts/build_or_update_venv.sh
```

All other SLURM scripts activate this venv (located at `/storage/home/hcoda1/1/syoung328/scratch`) before executing any Python code.

## Training a Model

To kick off a training run, first fill out a config (using [configs/test.yaml](configs/test.yaml) as a guide). Then, from the terminal:

```bash
uv run train --config <path-to-your-config>
```

The training run will log to Weights and Biases.

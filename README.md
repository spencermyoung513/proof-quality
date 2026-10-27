# Proof Quality

This project investigates the feasibility of training a neural reward model to score LM-generated proofs in Lean.

## Training

We have assembled a dataset of several hundred proof pairs from Mathlib PRs, where we know which was eventually accepted into the repository and which was rejected.

From this dataset, we use LoRA to fine-tune a language model with a reward head for distinguishing between the chosen/rejected proofs.

### Local training

To kick off a training run locally, first fill out a config (using [configs/example.yaml](configs/example.yaml) as a guide). Then, from the terminal:

```bash
uv run train --config <path-to-your-config>
```

Training holds out a small percentage (default: 10%) of the dev set for validation and logs to Weights and Biases.

### Training on PACE

To kick off a training run on PACE, use [scripts/train.sh](scripts/train.sh):

```bash
mkdir -p logs
sbatch scripts/train.sh <path-to-your-config>
```

The training job requests one L40S for 8 hours by default. You can override Slurm commands as needed from the command line:

```bash
sbatch --time=<new-time-budget> scripts/train.sh <path-to-config>
```

During training, stdout and stderr are written under `logs/`. The model with the lowest validation loss during training gets evaluated on the test split of our data, and results are logged to WandB.

## Evaluation

We have reserved out a small number of declaration-proof-proof triplets from the dataset for the sole purpose of evaluation. This set should not be used to tune hyperparameters.

### Local evaluation

To score a saved checkpoint on the test set locally, use the same config as training and point to the weights checkpoint you want to evaluate. Then:

```bash
uv run evaluate --config <path-to-your-config> --ckpt <path-to-checkpoint>
```

### Remote evaluation

The remote train script automatically runs evaluation on the weights that achieved the lowest validation loss during training.

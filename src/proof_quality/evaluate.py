from argparse import ArgumentParser
from argparse import Namespace
import os
from pathlib import Path

import lightning as L
from dotenv import load_dotenv
import yaml
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer

from proof_quality.configs import TrainingConfig
from proof_quality.datamodule import ProofPreferenceDataModule
from proof_quality.helpers import fix_random_seed
from proof_quality.model import RewardModel
from proof_quality.train import load_text_backbone


def evaluate(config: TrainingConfig, ckpt: Path) -> None:
    """Score a trained checkpoint on the held-out test set.

    Args:
        config: The config used for training.
        ckpt: Path to a saved model (adapter + reward head) checkpoint.
    """
    load_dotenv(override=True)
    fix_random_seed(config.run.seed)

    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_project or not wandb_entity or not wandb_api_key:
        raise ValueError(
            "WANDB_PROJECT, WANDB_ENTITY, and WANDB_API_KEY must be set in the local environment."
        )
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    base_lm = load_text_backbone(config.model.base_lm)
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_lm)
    model = RewardModel.from_checkpoint(base_lm, ckpt)
    datamodule = ProofPreferenceDataModule(
        tokenizer=tokenizer,
        batch_size=config.optimizer.batch_size,
        max_length=config.model.max_length,
        root=config.data.root,
        val_fraction=config.data.val_fraction,
        seed=config.run.seed,
    )
    logger = WandbLogger(
        name=f"{config.run.name}-eval",
        project=wandb_project,
        entity=wandb_entity,
    )
    trainer = L.Trainer(
        accelerator=config.run.device,
        logger=logger,
        precision=config.run.precision,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    metrics = trainer.test(model=model, datamodule=datamodule)[0]
    print(f"test/loss {metrics['test/loss']:.4f}")
    print(f"test/acc {metrics['test/acc']:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    evaluate(TrainingConfig(**config), Path(args.ckpt))


if __name__ == "__main__":
    main()

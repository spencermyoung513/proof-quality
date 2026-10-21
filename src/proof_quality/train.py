from argparse import ArgumentParser
from argparse import Namespace
import os
import lightning as L
from dotenv import load_dotenv
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from proof_quality.configs import TrainingConfig
from proof_quality.datamodule import ProofPreferenceDataModule
from proof_quality.model import RewardModel
from proof_quality.helpers import LogExampleRewards, fix_random_seed, get_ckpt_callbacks

from transformers import AutoModelForCausalLM, AutoTokenizer


def train(config: TrainingConfig):
    load_dotenv(override=True)
    fix_random_seed(config.run.seed)

    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if not wandb_project or not wandb_entity or not wandb_api_key:
        raise ValueError(
            "WANDB_PROJECT, WANDB_ENTITY, and WANDB_API_KEY must be set in the local environment."
        )

    base_lm = AutoModelForCausalLM.from_pretrained(config.model.base_lm)
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_lm)

    datamodule = ProofPreferenceDataModule(
        tokenizer=tokenizer,
        batch_size=config.optimizer.batch_size,
        max_length=config.model.max_length,
        root=config.data.root,
    )
    datamodule.setup("fit")

    model = RewardModel(
        base_lm=base_lm,
        hidden_size=base_lm.config.hidden_size,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        warmup_ratio=config.optimizer.warmup_ratio,
        lora_rank=config.model.lora_rank,
        lora_scaling_factor=config.model.lora_scaling_factor,
        lora_dropout=config.model.lora_dropout,
    )

    logger = WandbLogger(
        name=config.run.name,
        project=wandb_project,
        entity=wandb_entity,
    )
    ckpt_dir = config.run.ckpt_dir / config.run.name
    callbacks = [
        *get_ckpt_callbacks(ckpt_dir),
        LogExampleRewards(),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        accelerator=config.run.device,
        min_epochs=config.optimizer.num_epochs,
        max_epochs=config.optimizer.num_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        enable_model_summary=False,
        callbacks=callbacks,
        logger=logger,
        precision=config.run.precision,
        fast_dev_run=config.run.smoke_test,
    )
    trainer.fit(model=model, datamodule=datamodule)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    train(TrainingConfig(**config))


if __name__ == "__main__":
    main()
